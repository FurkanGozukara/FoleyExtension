from __future__ import annotations

import json
import math
import re
import wave
from pathlib import Path


class _SmartType(str):
    def __ne__(self, other):
        if self == "*" or other == "*":
            return False
        return str.__ne__(self, other)


ANY_TYPE = _SmartType("*")


def _common_upscale(images, width: int, height: int):
    import comfy.utils

    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)


def _intermediate_device():
    import comfy.model_management

    return comfy.model_management.intermediate_device()


def _conditioning_set_values(conditioning, values):
    import node_helpers

    return node_helpers.conditioning_set_values(conditioning, values)


def _audio_vae_model(audio_vae):
    return getattr(audio_vae, "first_stage_model", audio_vae)


def _window_specs(source_frames: int, fps: float, window_frames: int, overlap_seconds: float) -> list[dict[str, object]]:
    if window_frames % 8 != 1:
        raise ValueError("window_frames must satisfy window_frames % 8 == 1")
    if source_frames <= 0:
        raise ValueError("source frame count must be positive")
    if fps <= 0:
        raise ValueError("frame_rate must be positive")
    if overlap_seconds < 0:
        raise ValueError("overlap_seconds must be non-negative")

    frames = window_frames
    if source_frames <= frames:
        return [{"index": 1, "start_frame": 0, "start_seconds": 0.0, "frames": frames, "duration": frames / fps}]

    overlap_frames = max(0, round(overlap_seconds * fps))
    overlap_frames = min(overlap_frames, frames - 8)
    hop_frames = max(8, frames - overlap_frames)
    last_start = max(0, source_frames - frames)

    starts = list(range(0, last_start + 1, hop_frames))
    if starts[-1] != last_start:
        remainder = last_start - starts[-1]
        if len(starts) > 1 and remainder <= overlap_frames:
            # A snapped-to-end window this close to the previous start would
            # show the model nearly the same frames twice, and the second
            # generation can repeat events (e.g. a door close) near the end of
            # the stitched audio. Merging is only safe up to overlap_frames:
            # the window before the replaced start reaches exactly
            # starts[-1] + overlap_frames, so any larger remainder would leave
            # a zero-weight (silent) coverage gap instead.
            print(
                "[LTXFoley] merging final window: "
                f"start {starts[-1]} -> {last_start} "
                f"(remainder {remainder} <= overlap {overlap_frames})",
                flush=True,
            )
            starts[-1] = last_start
        else:
            starts.append(last_start)

    return [
        {
            "index": index,
            "start_frame": start_frame,
            "start_seconds": start_frame / fps,
            "frames": frames,
            "duration": frames / fps,
        }
        for index, start_frame in enumerate(starts, start=1)
    ]


def _slice_or_pad_frames(images, start_frame: int, frames: int):
    import torch

    if images.shape[0] == 0:
        raise ValueError("No video frames were provided")
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if frames <= 0:
        raise ValueError("frames must be positive")

    source = images[start_frame : start_frame + frames]
    if source.shape[0] == 0:
        source = images[-1:]
    if source.shape[0] < frames:
        pad = source[-1:].repeat((frames - source.shape[0], 1, 1, 1))
        source = torch.cat([source, pad], dim=0)
    return source


def _audio_waveform(audio: dict[str, object]):
    waveform = audio.get("waveform")
    if waveform is None:
        raise ValueError("Expected AUDIO dict to contain waveform")
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 3:
        raise ValueError(f"Expected audio waveform to have 2 or 3 dimensions, got shape {tuple(waveform.shape)}")

    if waveform.shape[1] in (1, 2, 6):
        return waveform
    if waveform.shape[2] in (1, 2, 6):
        return waveform.movedim(-1, 1)
    raise ValueError(f"Could not infer audio channel dimension from waveform shape {tuple(waveform.shape)}")


def _audio_stats(audio) -> dict[str, object]:
    import torch

    if audio.numel() == 0:
        return {"samples": 0, "rms": 0.0, "peak": 0.0}
    detached = audio.detach().float()
    return {
        "samples": int(detached.shape[-1]),
        "rms": float(torch.sqrt(torch.mean(torch.square(detached))).cpu()),
        "peak": float(torch.max(torch.abs(detached)).cpu()),
    }


def _accum_record_count(accumulation) -> int:
    if isinstance(accumulation, dict):
        return len(accumulation.get("records", []))
    return 0


def _loop_next_remaining(accumulation) -> int:
    """Windows still to generate, derived from the latest accumulated record.

    The loop countdown is carried inside the accumulation value instead of a
    graph link to the loop-open node so the ephemeral loop nodes never need a
    link back into the visible graph (see LTXFoleyForLoopClose docstring).
    """
    records = accumulation.get("records", []) if isinstance(accumulation, dict) else []
    if not records:
        raise ValueError("The sliding-window loop requires at least one accumulated window record")
    info = records[-1]["window_info"]
    window_count = int(info.get("window_count", info.get("planned_window_count", 0)))
    if window_count <= 0:
        raise ValueError("Accumulated window record does not carry a window count")
    return window_count - int(info["spec"]["index"])


def _safe_filename(value: str, default: str = "ltx_foley_window") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return cleaned or default


def _comfy_output_directory() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path.cwd()


def _write_audio_window(audio: dict[str, object], *, prefix: str, window_index: int) -> str:
    import torch

    sample_rate = int(audio["sample_rate"])
    waveform = _audio_waveform(audio).detach().float().cpu()[0]
    waveform = waveform.clamp(-1.0, 1.0)
    pcm = (waveform.movedim(0, -1) * 32767.0).round().to(torch.int16).contiguous()

    output_dir = _comfy_output_directory() / "ltx_foley_windows"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{_safe_filename(prefix)}_{window_index:03d}.wav"

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(waveform.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.numpy().tobytes())

    return str(path)


def _stitch_audio_windows(
    audio_windows: list[dict[str, object]],
    window_specs: list[dict[str, object]],
    output_duration: float,
    overlap_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    import torch

    if not audio_windows:
        raise ValueError("No window audio was generated")

    sample_rate = int(audio_windows[0]["sample_rate"])
    waveforms = [_audio_waveform(audio).detach().float().cpu() for audio in audio_windows]
    channels = int(waveforms[0].shape[1])
    total_samples = max(1, math.ceil(output_duration * sample_rate))
    accum = torch.zeros((1, channels, total_samples), dtype=torch.float32)
    weights = torch.zeros((1, 1, total_samples), dtype=torch.float32)
    overlap_samples = max(1, round(overlap_seconds * sample_rate))

    stats: list[dict[str, object]] = []
    for index, (waveform, spec) in enumerate(zip(waveforms, window_specs, strict=True)):
        current_rate = int(audio_windows[index]["sample_rate"])
        if current_rate != sample_rate:
            raise ValueError(f"Mismatched audio sample rate {current_rate}; expected {sample_rate}")
        if waveform.shape[1] != channels:
            raise ValueError(f"Mismatched audio channel count {waveform.shape[1]}; expected {channels}")

        start = round(float(spec["start_seconds"]) * sample_rate)
        if start >= total_samples:
            continue
        available = total_samples - start
        waveform = waveform[:, :, :available]
        if waveform.numel() == 0:
            continue

        window_weight = torch.ones((1, 1, waveform.shape[-1]), dtype=torch.float32)
        if index > 0:
            fade = min(overlap_samples, waveform.shape[-1])
            window_weight[:, :, :fade] = torch.linspace(0.0, 1.0, fade, dtype=torch.float32).view(1, 1, -1)
        if index < len(waveforms) - 1:
            fade = min(overlap_samples, waveform.shape[-1])
            fade_out = torch.linspace(1.0, 0.0, fade, dtype=torch.float32).view(1, 1, -1)
            window_weight[:, :, -fade:] = torch.minimum(window_weight[:, :, -fade:], fade_out)

        end = start + waveform.shape[-1]
        accum[:, :, start:end] += waveform * window_weight
        weights[:, :, start:end] += window_weight
        stats.append({"window_index": int(spec["index"]), **_audio_stats(waveform)})

    stitched = accum / torch.clamp(weights, min=1e-6)
    return {"waveform": stitched, "sample_rate": sample_rate}, stats


def _manifest_text(
    *,
    source_frames: int,
    fps: float,
    window_specs: list[dict[str, object]],
    planned_window_count: int | None = None,
    truncated_by_max_windows: bool = False,
    audio_stats: list[dict[str, object]],
    window_audio_paths: list[str],
    warnings: list[str],
) -> str:
    payload = {
        "version": "ltx_foley_loop",
        "source_frames": int(source_frames),
        "frame_rate": float(fps),
        "source_duration": float(source_frames / fps),
        "window_count": len(window_specs),
        "planned_window_count": int(planned_window_count if planned_window_count is not None else len(window_specs)),
        "truncated_by_max_windows": bool(truncated_by_max_windows),
        "window_specs": window_specs,
        "window_audio_stats": audio_stats,
        "window_audio_paths": window_audio_paths,
        "warnings": warnings,
    }
    return json.dumps(payload, indent=2)


class LTXFoleyForLoopOpen:
    """Minimal execution-inversion for-loop.

    Loop mechanics are adapted from akatz-ai/ComfyUI-Execution-Inversion
    under the MIT license, Copyright (c) 2025 akatz-ai. This node deliberately
    only exposes the sockets needed by the Foley workflow.

    This node is a pure pass-through: the visible graph body is the first loop
    iteration. Recursion happens in _LTXFoleyLoopIterator, which clones the
    subgraph between this node and the accumulator and seeds the clone of this
    node with plain values via remaining/initial_value0.
    """

    CATEGORY = "LTX/Foley/Loop"
    RETURN_TYPES = (_SmartType("FLOW_CONTROL"), "INT", "FOLEY_AUDIO_ACCUM")
    RETURN_NAMES = ("flow_control", "remaining", "audio_accumulation")
    FUNCTION = "open"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "remaining": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            },
            "optional": {"audio_accumulation": ("FOLEY_AUDIO_ACCUM",)},
            "hidden": {"initial_value0": (ANY_TYPE,), "unique_id": "UNIQUE_ID"},
        }

    def open(self, remaining: int, **kwargs):
        seeded = "initial_value0" in kwargs
        if seeded:
            remaining = kwargs["initial_value0"]
        remaining = int(remaining)
        print(
            "[LTXFoley][trace] loop open "
            f"node={kwargs.get('unique_id')} remaining={remaining} seeded={seeded} "
            f"accum_records={_accum_record_count(kwargs.get('audio_accumulation'))}",
            flush=True,
        )
        return ("stub", remaining, kwargs.get("audio_accumulation"))


class LTXFoleyForLoopClose:
    """Closes the sliding-window loop.

    The loop-carried audio accumulation is received as a materialized value
    (not a rawLink) and handed to the recursion as an embedded constant, and
    the countdown is derived from the accumulation itself. The ephemeral
    iterator node this expands must never receive graph links as inputs:
    links attached to freshly expanded nodes bypass the per-consumer
    execution cache, so under ComfyUI's default RAM-pressure caching an
    evicted upstream output re-executes the entire visible sampler chain
    (observed as a duplicated first window).
    """

    CATEGORY = "LTX/Foley/Loop"
    RETURN_TYPES = ("FOLEY_AUDIO_ACCUM",)
    RETURN_NAMES = ("audio_accumulation",)
    FUNCTION = "close"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {"flow_control": (_SmartType("FLOW_CONTROL"), {"rawLink": True})},
            "optional": {"audio_accumulation": ("FOLEY_AUDIO_ACCUM",)},
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    def close(self, flow_control, audio_accumulation=None, dynprompt=None, unique_id=None):
        from comfy_execution.graph_utils import GraphBuilder, is_link

        open_node = flow_control[0]
        next_remaining = _loop_next_remaining(audio_accumulation)
        if next_remaining <= 0:
            print(
                "[LTXFoley][trace] loop finished "
                f"node={unique_id} accum_records={_accum_record_count(audio_accumulation)}",
                flush=True,
            )
            return (audio_accumulation,)

        anchor = dynprompt.get_node(unique_id)["inputs"].get("audio_accumulation")
        if not is_link(anchor):
            raise ValueError("LTXFoleyForLoopClose expects audio_accumulation to be linked from the loop body")
        print(
            "[LTXFoley][trace] loop close expanding "
            f"node={unique_id} open_node={open_node} anchor={anchor[0]} next_remaining={next_remaining}",
            flush=True,
        )
        graph = GraphBuilder()
        iterator = graph.node(
            "_LTXFoleyLoopIterator",
            audio_accumulation=audio_accumulation,
            open_node=open_node,
            anchor_node=anchor[0],
            anchor_output=anchor[1],
        )
        return {"result": (iterator.out(0),), "expand": graph.finalize()}


class _LTXFoleyLoopIterator:
    """Runs one recursion step of the sliding-window loop.

    All inputs are constants (values and node-id strings). The loop body is
    discovered by walking dynprompt links upward from the anchor (the
    accumulator feeding the loop close), then cloning every node between the
    loop open and the anchor. The clone's open node is seeded with plain
    values, and the next iterator receives the cloned anchor's output — the
    only link it ever holds, and one that points inside its own expansion.
    """

    CATEGORY = "LTX/Foley/Internal"
    RETURN_TYPES = ("FOLEY_AUDIO_ACCUM",)
    RETURN_NAMES = ("audio_accumulation",)
    FUNCTION = "iterate"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "audio_accumulation": ("FOLEY_AUDIO_ACCUM",),
                "open_node": (ANY_TYPE,),
                "anchor_node": (ANY_TYPE,),
                "anchor_output": ("INT", {"default": 0}),
            },
            "hidden": {"dynprompt": "DYNPROMPT", "unique_id": "UNIQUE_ID"},
        }

    def _explore_dependencies(self, node_id, dynprompt, upstream):
        from comfy_execution.graph_utils import is_link

        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for value in node_info["inputs"].values():
            if is_link(value):
                parent_id = value[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self._explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def _collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self._collect_contained(child_id, upstream, contained)

    def iterate(self, audio_accumulation, open_node, anchor_node, anchor_output, dynprompt=None, unique_id=None):
        next_remaining = _loop_next_remaining(audio_accumulation)
        if next_remaining <= 0:
            print(
                "[LTXFoley][trace] loop finished "
                f"node={unique_id} accum_records={_accum_record_count(audio_accumulation)}",
                flush=True,
            )
            return (audio_accumulation,)

        from comfy_execution.graph_utils import GraphBuilder, is_link

        upstream = {}
        self._explore_dependencies(anchor_node, dynprompt, upstream)
        contained = {}
        self._collect_contained(open_node, upstream, contained)
        contained[open_node] = True
        contained[anchor_node] = True

        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], node_id)
            node.set_override_display_id(node_id)
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node(node_id)
            for key, value in original_node["inputs"].items():
                if is_link(value) and value[0] in contained:
                    parent = graph.lookup_node(value[0])
                    node.set_input(key, parent.out(value[1]))
                else:
                    node.set_input(key, value)

        new_open = graph.lookup_node(open_node)
        # Replace the remaining link with a plain value as well, so the clone
        # does not pull the (possibly evicted) window-plan chain back onto the
        # execution list.
        new_open.set_input("remaining", next_remaining)
        new_open.set_input("initial_value0", next_remaining)
        new_open.set_input("audio_accumulation", audio_accumulation)
        new_anchor = graph.lookup_node(anchor_node)
        recurse = graph.node(
            "_LTXFoleyLoopIterator",
            "Recurse",
            audio_accumulation=new_anchor.out(int(anchor_output)),
            open_node=new_open.id,
            anchor_node=new_anchor.id,
            anchor_output=int(anchor_output),
        )
        recurse.set_override_display_id(unique_id)
        print(
            "[LTXFoley][trace] loop recursing "
            f"node={unique_id} next_remaining={next_remaining} "
            f"accum_records={_accum_record_count(audio_accumulation)} "
            f"cloned_nodes={sorted(contained)}",
            flush=True,
        )
        return {"result": (recurse.out(0),), "expand": graph.finalize()}


class LTXFoleyVideoToAudioLatent:
    CATEGORY = "LTX/Foley"
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("positive", "negative", "av_latent", "preview_images", "frame_rate", "frames")
    FUNCTION = "prepare"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "images": ("IMAGE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "width": ("INT", {"default": 576, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 576, "min": 256, "max": 2048, "step": 32}),
                "frames": ("INT", {"default": 89, "min": 9, "max": 257, "step": 8}),
            }
        }

    def prepare(
        self,
        images,
        positive,
        negative,
        video_vae,
        audio_vae,
        frame_rate: float,
        width: int,
        height: int,
        frames: int,
    ):
        import torch

        if frames % 8 != 1:
            raise ValueError("frames must satisfy frames % 8 == 1")
        if images.shape[0] == 0:
            raise ValueError("No video frames were provided")

        source = _slice_or_pad_frames(images, 0, frames)

        resized = _common_upscale(source, width, height).clamp(0.0, 1.0)
        video_latent = video_vae.encode(resized[:, :, :, :3])
        video_noise_mask = torch.zeros(
            (1, 1, video_latent.shape[2], 1, 1),
            dtype=torch.float32,
            device=video_latent.device,
        )

        audio_model = _audio_vae_model(audio_vae)
        z_channels = getattr(audio_vae, "latent_channels", audio_model.latent_channels)
        audio_freq = audio_model.latent_frequency_bins
        audio_latents = audio_model.num_of_latents_from_frames(frames, int(round(frame_rate)))
        audio_latent = torch.zeros(
            (1, z_channels, audio_latents, audio_freq),
            device=_intermediate_device(),
        )
        audio_noise_mask = torch.ones_like(audio_latent)

        import comfy.nested_tensor

        av_latent = {
            "samples": comfy.nested_tensor.NestedTensor((video_latent, audio_latent)),
            "noise_mask": comfy.nested_tensor.NestedTensor((video_noise_mask, audio_noise_mask)),
        }
        values = {"frame_rate": frame_rate}
        return (
            _conditioning_set_values(positive, values),
            _conditioning_set_values(negative, values),
            av_latent,
            resized,
            float(frame_rate),
            int(frames),
        )


class LTXFoleyTrimImages:
    CATEGORY = "LTX/Foley"
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "frame_rate")
    FUNCTION = "trim"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "frames": ("INT", {"default": 89, "min": 9, "max": 257, "step": 8}),
            }
        }

    def trim(self, images, frame_rate: float, frames: int):
        import torch

        if images.shape[0] == 0:
            raise ValueError("No video frames were provided")
        trimmed = images[:frames]
        if trimmed.shape[0] < frames:
            pad = trimmed[-1:].repeat((frames - trimmed.shape[0], 1, 1, 1))
            trimmed = torch.cat([trimmed, pad], dim=0)
        return (trimmed, float(frame_rate))


class LTXFoleyWindowPlan:
    CATEGORY = "LTX/Foley"
    RETURN_TYPES = ("FOLEY_WINDOW_PLAN", "INT", "STRING")
    RETURN_NAMES = ("window_plan", "window_count", "manifest")
    FUNCTION = "plan"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 0.01}),
                "window_frames": ("INT", {"default": 89, "min": 9, "max": 257, "step": 8}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_windows": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
            }
        }

    def plan(self, images, frame_rate: float, window_frames: int, overlap_seconds: float, max_windows: int):
        fps = float(frame_rate)
        source_frames = int(images.shape[0])
        all_specs = _window_specs(source_frames, fps, int(window_frames), float(overlap_seconds))
        max_windows = int(max_windows)
        specs = all_specs[:max_windows]
        truncated = len(specs) < len(all_specs)
        if not specs:
            raise ValueError("max_windows must allow at least one diagnostic window")
        plan = {
            "version": "ltx_foley_loop_plan",
            "source_frames": source_frames,
            "frame_rate": fps,
            "source_duration": source_frames / fps,
            "window_frames": int(window_frames),
            "overlap_seconds": float(overlap_seconds),
            "planned_window_count": len(all_specs),
            "truncated_by_max_windows": truncated,
            "window_specs": specs,
        }
        start_frames = [spec["start_frame"] for spec in specs]
        if truncated:
            print(
                "[LTXFoley] diagnostic truncation: "
                f"planned {len(all_specs)} windows, running first {len(specs)} because max_windows={max_windows} "
                f"starts={start_frames}",
                flush=True,
            )
        else:
            print(f"[LTXFoley] planned {len(specs)} windows starts={start_frames}", flush=True)
        return (plan, len(specs), json.dumps(plan, indent=2))


class LTXFoleyWindowSelect:
    CATEGORY = "LTX/Foley"
    RETURN_TYPES = ("IMAGE", "FOLEY_WINDOW")
    RETURN_NAMES = ("window_images", "window_info")
    FUNCTION = "select"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "images": ("IMAGE",),
                "window_plan": ("FOLEY_WINDOW_PLAN",),
                "remaining": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def select(self, images, window_plan: dict[str, object], remaining: int, unique_id=None):
        specs = list(window_plan["window_specs"])
        loop_index = len(specs) - int(remaining)
        if loop_index < 0 or loop_index >= len(specs):
            raise ValueError(f"Loop remaining={remaining} is outside the planned {len(specs)} windows")
        spec = dict(specs[loop_index])
        frames = int(window_plan.get("window_frames", spec["frames"]))
        window_images = _slice_or_pad_frames(images, int(spec["start_frame"]), frames)
        window_info = {
            "spec": spec,
            "window_count": len(specs),
            "source_frames": int(window_plan["source_frames"]),
            "frame_rate": float(window_plan["frame_rate"]),
            "source_duration": float(window_plan["source_duration"]),
            "overlap_seconds": float(window_plan["overlap_seconds"]),
            "planned_window_count": int(window_plan.get("planned_window_count", len(specs))),
            "truncated_by_max_windows": bool(window_plan.get("truncated_by_max_windows", False)),
        }
        print(
            "[LTXFoley] selecting window "
            f"{spec['index']}/{len(specs)} start_frame={spec['start_frame']} frames={frames} "
            f"node={unique_id} remaining={remaining}",
            flush=True,
        )
        return (window_images, window_info)


class LTXFoleyWindowAudioSave:
    CATEGORY = "LTX/Foley"
    RETURN_TYPES = ("AUDIO", "FOLEY_WINDOW_RECORD")
    RETURN_NAMES = ("audio", "window_record")
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "audio": ("AUDIO",),
                "window_info": ("FOLEY_WINDOW",),
                "save_audio": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "ltx_foley_window"}),
            }
        }

    def save(self, audio, window_info: dict[str, object], save_audio: bool, filename_prefix: str):
        spec = dict(window_info["spec"])
        path = ""
        if save_audio:
            path = _write_audio_window(audio, prefix=filename_prefix, window_index=int(spec["index"]))
        stats = _audio_stats(_audio_waveform(audio))
        print(
            "[LTXFoley] window "
            f"{spec['index']} audio samples={stats['samples']} rms={stats['rms']:.6f} "
            f"peak={stats['peak']:.6f} path={path or '<not saved>'}",
            flush=True,
        )
        return (audio, {"audio": audio, "window_info": window_info, "path": path})


class LTXFoleyAudioAccumulator:
    CATEGORY = "LTX/Foley/Loop"
    RETURN_TYPES = ("FOLEY_AUDIO_ACCUM",)
    RETURN_NAMES = ("accumulation",)
    FUNCTION = "accumulate"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {"window_record": ("FOLEY_WINDOW_RECORD",)},
            "optional": {"accumulation": ("FOLEY_AUDIO_ACCUM",)},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def accumulate(self, window_record: dict[str, object], accumulation: dict[str, object] | None = None, unique_id=None):
        records = [] if accumulation is None else list(accumulation.get("records", []))
        records.append(window_record)
        print(
            "[LTXFoley][trace] accumulate "
            f"node={unique_id} window={window_record['window_info']['spec']['index']} "
            f"total_records={len(records)}",
            flush=True,
        )
        return ({"records": records},)


class LTXFoleyAudioStitch:
    CATEGORY = "LTX/Foley"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "manifest")
    FUNCTION = "stitch"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "accumulation": ("FOLEY_AUDIO_ACCUM",),
                "window_plan": ("FOLEY_WINDOW_PLAN",),
            }
        }

    def stitch(self, accumulation: dict[str, object], window_plan: dict[str, object]):
        records = list(accumulation.get("records", []))
        if not records:
            raise ValueError("No generated window audio records were provided")

        records.sort(key=lambda item: int(item["window_info"]["spec"]["index"]))
        audio_windows = [item["audio"] for item in records]
        specs = [dict(item["window_info"]["spec"]) for item in records]
        stitched_audio, audio_stats = _stitch_audio_windows(
            audio_windows,
            specs,
            output_duration=float(window_plan["source_duration"]),
            overlap_seconds=float(window_plan["overlap_seconds"]),
        )
        manifest = _manifest_text(
            source_frames=int(window_plan["source_frames"]),
            fps=float(window_plan["frame_rate"]),
            window_specs=specs,
            planned_window_count=int(window_plan.get("planned_window_count", len(specs))),
            truncated_by_max_windows=bool(window_plan.get("truncated_by_max_windows", False)),
            audio_stats=audio_stats,
            window_audio_paths=[str(item.get("path", "")) for item in records],
            warnings=(
                [f"Diagnostic run truncated to {len(specs)} of {int(window_plan.get('planned_window_count', len(specs)))} planned windows"]
                if bool(window_plan.get("truncated_by_max_windows", False))
                else []
            ),
        )
        print(
            "[LTXFoley] stitched "
            f"{len(specs)} window(s); truncated={bool(window_plan.get('truncated_by_max_windows', False))}",
            flush=True,
        )
        return (stitched_audio, manifest)


class LTXFoleyAudioVAEDecode:
    CATEGORY = "LTX/Foley"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, object]:
        return {
            "required": {
                "samples": ("LATENT",),
                "audio_vae": ("VAE",),
            }
        }

    def decode(self, samples, audio_vae):
        audio_latent = samples["samples"]
        if audio_latent.is_nested:
            audio_latent = audio_latent.unbind()[-1]

        output_device = audio_latent.device
        audio = audio_vae.decode(audio_latent).to(output_device)
        if audio.ndim == 2:
            audio = audio.unsqueeze(1)
        elif audio.ndim != 3:
            raise ValueError(f"Expected decoded audio to have 2 or 3 dimensions, got shape {tuple(audio.shape)}")

        if audio.shape[1] in (1, 2, 6):
            waveform = audio
        elif audio.shape[2] in (1, 2, 6):
            waveform = audio.movedim(-1, 1)
        else:
            raise ValueError(f"Could not infer audio channel dimension from decoded shape {tuple(audio.shape)}")

        output_sample_rate = _audio_vae_model(audio_vae).output_sample_rate
        return ({"waveform": waveform, "sample_rate": int(output_sample_rate)},)


NODE_CLASS_MAPPINGS = {
    "_LTXFoleyLoopIterator": _LTXFoleyLoopIterator,
    "LTXFoleyForLoopOpen": LTXFoleyForLoopOpen,
    "LTXFoleyForLoopClose": LTXFoleyForLoopClose,
    "LTXFoleyVideoToAudioLatent": LTXFoleyVideoToAudioLatent,
    "LTXFoleyTrimImages": LTXFoleyTrimImages,
    "LTXFoleyWindowPlan": LTXFoleyWindowPlan,
    "LTXFoleyWindowSelect": LTXFoleyWindowSelect,
    "LTXFoleyWindowAudioSave": LTXFoleyWindowAudioSave,
    "LTXFoleyAudioAccumulator": LTXFoleyAudioAccumulator,
    "LTXFoleyAudioStitch": LTXFoleyAudioStitch,
    "LTXFoleyAudioVAEDecode": LTXFoleyAudioVAEDecode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXFoleyForLoopOpen": "LTX Foley For Loop Open",
    "LTXFoleyForLoopClose": "LTX Foley For Loop Close",
    "LTXFoleyVideoToAudioLatent": "LTX Foley Video To Audio Latent",
    "LTXFoleyTrimImages": "LTX Foley Trim Images",
    "LTXFoleyWindowPlan": "LTX Foley Window Plan",
    "LTXFoleyWindowSelect": "LTX Foley Window Select",
    "LTXFoleyWindowAudioSave": "LTX Foley Window Audio Save",
    "LTXFoleyAudioAccumulator": "LTX Foley Audio Accumulator",
    "LTXFoleyAudioStitch": "LTX Foley Audio Stitch",
    "LTXFoleyAudioVAEDecode": "LTX Foley Audio VAE Decode",
}
