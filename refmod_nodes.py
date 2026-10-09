"""Portable MiniMax H3 reference latents, shared by ComfyUI and SwarmUI.

Reads the MIT-licensed MiniMaxH3Mod / Fantastic RefMod safetensors format.
See THIRD_PARTY_NOTICES.md for provenance. No upstream node pack is required.
"""

from __future__ import annotations

import io as bytes_io
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image
from safetensors import safe_open, SafetensorError
from safetensors.torch import save_file
import torch
import torch.nn.functional as F

import folder_paths
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo, _encode_ref_audio, _resize,
)


REF_PACK = "SECOURSES_REF_PACK"
META_KEY = "refmod_meta"
LABELS = {"image": "Picture", "video": "Video", "audio": "Audio"}


def refmod_roots():
    # Swarm's generated model config registers its VAE roots, but has no refmods
    # category. The sibling folder also works with remote ComfyUI backends.
    candidates = list(folder_paths.get_folder_paths("refmods")) if "refmods" in folder_paths.folder_names_and_paths else []
    candidates += [str(Path(folder_paths.models_dir) / "refmods")]
    candidates += [str(Path(p).parent / "refmods") for p in folder_paths.get_folder_paths("vae")
                   if Path(p).name.lower() == "vae"]
    roots = list(dict.fromkeys(Path(p).resolve() for p in candidates))
    for root in roots:
        folder_paths.add_model_folder_path("refmods", str(root))
    folder_paths.folder_names_and_paths["refmods"][1].add(".safetensors")
    return roots


def relative_name(name):
    name = str(name).strip().replace("\\", "/")
    if not name or name.startswith("/") or ":" in name or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("RefMod names must be relative paths inside models/refmods.")
    if not name.lower().endswith(".safetensors"):
        name += ".safetensors"
    return name


def resolve_refmod(name):
    name = relative_name(name)
    for root in refmod_roots():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("RefMod path leaves models/refmods.")
        if path.is_file():
            return path
    raise FileNotFoundError(f"RefMod '{name}' was not found. Put it in models/refmods and refresh the library.")


def read_metadata(path, header):
    raw = header.get(META_KEY) or header.get("audio_refmod_meta")
    if raw:
        meta = json.loads(raw)
    else:
        sidecar = path.with_suffix(".json")
        if sidecar.resolve().parent != path.parent:
            raise ValueError("RefMod sidecar leaves its model folder.")
        meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
    if not isinstance(meta, dict):
        raise ValueError(f"'{path.name}' has no RefMod metadata; diffusion models and LoRAs are not RefMods.")
    members = meta.get("members") if meta.get("kind") == "bundle" else [meta]
    if meta.get("kind") == "bundle" and meta.get("_format_version") != 5:
        raise ValueError(f"Unsupported RefMod bundle version in '{path.name}'.")
    if not isinstance(members, list) or not members:
        raise ValueError(f"Empty RefMod bundle: {path.name}.")
    if any(not isinstance(m, dict) or m.get("kind") not in LABELS for m in members):
        raise ValueError(f"Unknown RefMod reference kind in '{path.name}'.")
    return meta, members


def parse_selection(selection):
    if not selection or not str(selection).strip():
        return []
    if isinstance(selection, list):
        rows = selection
    elif str(selection).lstrip().startswith("["):
        rows = json.loads(selection)
    else:
        rows = [s.strip() for s in str(selection).splitlines() if s.strip()]
    if not isinstance(rows, list):
        raise ValueError("RefMods must be a JSON list or one filename per line.")
    active = []
    for slot, row in enumerate(rows, 1):
        row = {"file": row} if isinstance(row, str) else row
        if not isinstance(row, dict):
            raise ValueError(f"RefMod row {slot} must contain a filename.")
        if not row.get("enabled", True) or not row.get("file"):
            continue
        weight = float(row.get("strength", 1))
        if not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError(f"RefMod row {slot}: strength must be between 0 and 1.")
        if weight == 0:
            continue
        component = row.get("components", "all").lower()
        if component not in ("all", "visual", "audio"):
            raise ValueError(f"RefMod row {slot}: components must be all, visual, or audio.")
        active.append({**row, "file": relative_name(row["file"]), "strength": weight,
                       "components": component, "slot": row.get("slot", slot)})
    return active


def catalog():
    files, errors, seen = [], [], set()
    for root in refmod_roots():
        if not root.is_dir():
            continue
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()
                       and not (Path(directory) / d).is_junction()]
            for name in sorted(names):
                if not name.lower().endswith(".safetensors"):
                    continue
                path = Path(directory) / name
                rel = path.relative_to(root).as_posix()
                if rel in seen or not path.resolve().is_relative_to(root):
                    continue
                seen.add(rel)
                try:
                    with safe_open(path, framework="pt", device="cpu") as file:
                        meta, members = read_metadata(path, file.metadata() or {})
                        tokens = {"visual": 0, "audio": 0}
                        for index, member in enumerate(members):
                            key = f"ref_{index}" if meta.get("kind") == "bundle" else "latent"
                            shape = file.get_slice(key).get_shape()
                            if member["kind"] == "audio" and len(shape) == 4:
                                tokens["audio"] += 2 * shape[-1]
                            elif len(shape) == 5:
                                tokens["visual"] += shape[2] * (shape[3] // 2) * (shape[4] // 2)
                    files.append({"file": rel, "name": meta.get("name", path.stem),
                                  "kinds": [m["kind"] for m in members], "members": len(members), "tokens": tokens})
                except (ValueError, OSError, RuntimeError, KeyError, SafetensorError) as error:
                    errors.append({"file": rel, "error": str(error)})
    return {"files": files, "errors": errors, "folders": [str(p) for p in refmod_roots()]}


def validate_latent(z, kind, filename):
    if kind == "audio":
        valid = z.ndim == 4 and tuple(z.shape[:3]) == (1, 32, 2) and z.shape[-1] > 0
    else:
        valid = (z.ndim == 5 and tuple(z.shape[:2]) == (1, 24)
                 and all(n > 0 for n in z.shape[2:]) and all(n % 2 == 0 for n in z.shape[-2:])
                 and (kind != "image" or z.shape[2] == 1))
    if not valid or not z.is_floating_point() or not torch.isfinite(z).all():
        raise ValueError(f"Invalid {kind} RefMod latent in '{filename}': {tuple(z.shape)}.")


def weaken(z, weight):
    if weight == 1:
        return z
    # Same low-pass reference-retention convention as MiniMaxH3Mod, not a LoRA scale.
    if z.ndim == 5:
        t, h, w = z.shape[2:]
        blurred = F.interpolate(F.adaptive_avg_pool3d(z.float(), (t, max(1, h // 8), max(1, w // 8))),
                                size=(t, h, w), mode="trilinear", align_corners=False)
    else:
        t = z.shape[-1]
        flat = z.reshape(-1, 1, t).float()
        blurred = F.interpolate(F.adaptive_avg_pool1d(flat, max(1, t // 8)), size=t,
                                mode="linear", align_corners=False).reshape_as(z)
    return (z.float() * weight + blurred * (1 - weight)).to(z.dtype)


def load_members(selection):
    result = []
    for row in parse_selection(selection):
        path = resolve_refmod(row["file"])
        with safe_open(path, framework="pt", device="cpu") as file:
            meta, members = read_metadata(path, file.metadata() or {})
            bundle = meta.get("kind") == "bundle"
            for index, member in enumerate(members):
                kind = member["kind"]
                if row["components"] == "audio" and kind != "audio" or row["components"] == "visual" and kind == "audio":
                    continue
                key = f"ref_{index}" if bundle else "latent"
                z = file.get_tensor(key).clone()
                validate_latent(z, kind, row["file"])
                z = weaken(z, row["strength"])
                prefix = f"ref_{index}_enc_" if bundle else "enc_"
                frames = []
                # Fantastic stores encoder frames as JPEG byte tensors. Our bundles
                # use the same convention with a member prefix; unknown tensors are
                # harmless to third-party readers.
                if row["strength"] == 1:
                    frame_keys = sorted((k for k in file.keys() if k.startswith(prefix)), key=lambda k: int(k[len(prefix):]))
                    for frame_key in frame_keys:
                        image = Image.open(bytes_io.BytesIO(file.get_tensor(frame_key).numpy().tobytes())).convert("RGB")
                        frames.append(torch.from_numpy(np.array(image)).float() / 255)
                result.append({"meta": member, "latent": z, "frames": torch.stack(frames) if frames else None,
                               "slot": row["slot"], "file": row["file"]})
    return result


def member_item(member, vae):
    meta, z = member["meta"], member["latent"]
    kind = meta["kind"]
    if kind == "audio":
        return {"type": "audio"}, {"kind": "audio", "ref_audio_t": z.shape[-1], "audio_latent": z}
    if vae is None:
        raise ValueError("Connect the MiniMax H3 video VAE to use visual RefMods.")
    frames = member["frames"]
    times = meta.get("secourses_enc_times") or meta.get("enc_times")
    if frames is None:
        if meta.get("source") == "stack":
            frames = []
            for i in range(z.shape[2]):
                pixels = vae.decode(z[:, :, i:i + 1])
                frames.append((pixels[0] if pixels.ndim == 5 else pixels)[:1].cpu())
            frames = torch.cat(frames)
            times = [float(i) for i in range(len(frames))]
        else:
            pixels = vae.decode(z)
            pixels = pixels[0] if pixels.ndim == 5 else pixels
            indices = [0] if kind == "image" else list(range(0, len(pixels), 12))
            frames = pixels[indices].cpu()
            times = [i / 24 for i in indices]
    block = {"kind": kind, "latent": z, "latent_h": z.shape[3], "latent_w": z.shape[4]}
    item = {"type": kind, "data": frames}
    if kind == "video":
        block.update(latent_t=z.shape[2], ref_audio_t=0, audio_latent=None)
        item["timestamps"] = times if times and len(times) == len(frames) else [i / 2 for i in range(len(frames))]
    return item, block


class RefModClip:
    """An execution-local adapter of the native tokenizer, with no model patches."""

    def __init__(self, clip, vae, selection):
        self.clip, self.vae, self.selection = clip, vae, selection
        self.blocks = []
        self.mapping = []
        self.prepared = []

    def __getattr__(self, name):
        return getattr(self.clip, name)

    @property
    def apply_hooks_to_conds(self):
        return self.clip.apply_hooks_to_conds

    @apply_hooks_to_conds.setter
    def apply_hooks_to_conds(self, value):
        self.clip.apply_hooks_to_conds = value

    def clone(self, *args, **kwargs):
        clone = RefModClip(self.clip.clone(*args, **kwargs), self.vae, self.selection)
        clone.blocks, clone.mapping, clone.prepared = self.blocks, self.mapping, self.prepared
        return clone

    def tokenize(self, prompt, **kwargs):
        self.blocks.clear()
        self.mapping.clear()
        items = list(kwargs.pop("minimax_ref_items", None) or [])
        # Native keyframe conditioning is retained by ImageToVideo; the tokenizer
        # otherwise ignores its `images` argument whenever ref_items are present.
        images = kwargs.pop("images", [])
        if images is not None:
            items += [{"type": "image", "data": img} for img in images]
        counts = {kind: sum(item["type"] == kind for item in items) for kind in LABELS}
        slot_labels = {}
        if not self.prepared:
            for member in load_members(self.selection):
                item, block = member_item(member, self.vae)
                self.prepared.append((member, item, block))
        for member, item, block in self.prepared:
            kind = item["type"]
            counts[kind] += 1
            label = f"<{LABELS[kind]} {counts[kind]}>"
            slot_labels.setdefault(member["slot"], []).append(label)
            self.mapping.append(f"@refmod{member['slot']} = {label} ({member['meta'].get('name', member['file'])})")
            items.append(item)
            self.blocks.append(block)

        def replace(match):
            slot = int(match.group(1))
            if slot not in slot_labels:
                raise ValueError(f"@refmod{slot} has no enabled reference. Enable that row or remove its prompt token.")
            return " and ".join(slot_labels[slot])

        prompt = re.sub(r"(?<![\w@])@refmod\s*(\d+)(?!\w)", replace, prompt, flags=re.I)
        print("[SECourses H3 RefMods] " + "; ".join(self.mapping), flush=True)
        return self.clip.tokenize(prompt, minimax_ref_items=items, **kwargs)

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        return self.clip.encode_from_tokens_scheduled(tokens, *args, **kwargs)

    def append(self, conditioning):
        return [[embedding, {**meta, "minimax_refs": list(meta.get("minimax_refs", [])) + self.blocks}]
                for embedding, meta in conditioning]


def encode_with_refmods(native, selection, **kwargs):
    if not parse_selection(selection):
        return native.execute(**kwargs)
    proxy = RefModClip(kwargs["clip"], kwargs.get("vae"), selection)
    result = native.execute(**{**kwargs, "clip": proxy})
    return io.NodeOutput(proxy.append(result.args[0]), result.args[1])


class SECoursesH3RefModReferences(MiniMaxH3ReferenceToVideo):
    @classmethod
    def define_schema(cls):
        schema = MiniMaxH3ReferenceToVideo.define_schema()
        schema.node_id = "SECoursesH3RefModReferences"
        schema.display_name = "MiniMax H3 References + Optional RefMods"
        schema.inputs.append(io.String.Input("refmods", default="[]", multiline=True, optional=True))
        return schema

    @classmethod
    def fingerprint_inputs(cls, refmods="[]", **kwargs):
        return SECoursesH3RefModStack.IS_CHANGED(refmods)

    @classmethod
    def execute(cls, refmods="[]", **kwargs):
        return encode_with_refmods(MiniMaxH3ReferenceToVideo, refmods, **kwargs)


class SECoursesH3RefModImageToVideo(MiniMaxH3ImageToVideo):
    @classmethod
    def define_schema(cls):
        schema = MiniMaxH3ImageToVideo.define_schema()
        schema.node_id = "SECoursesH3RefModImageToVideo"
        schema.display_name = "MiniMax H3 Keyframes + Optional RefMods"
        schema.inputs.append(io.String.Input("refmods", default="[]", multiline=True, optional=True))
        return schema

    @classmethod
    def fingerprint_inputs(cls, refmods="[]", **kwargs):
        return SECoursesH3RefModStack.IS_CHANGED(refmods)

    @classmethod
    def execute(cls, refmods="[]", **kwargs):
        return encode_with_refmods(MiniMaxH3ImageToVideo, refmods, **kwargs)


class SECoursesH3RefModStack:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"selection": ("STRING", {"default": "[]", "multiline": True,
                    "tooltip": "Optional RefMods. Add as many rows as needed. Empty or disabled rows cost nothing. Use @refmod1 etc in the prompt."})},
                "optional": {"references": (REF_PACK,)}}

    CATEGORY = "SECourses/references"
    RETURN_TYPES = (REF_PACK,)
    RETURN_NAMES = ("references",)
    FUNCTION = "collect"

    def collect(self, selection="[]", references=None):
        rows = parse_selection(selection)
        return ({**(references or {}), "refmods": rows},)

    @classmethod
    def IS_CHANGED(cls, selection="[]", **kwargs):
        files = []
        for row in parse_selection(selection):
            path = resolve_refmod(row["file"])
            files.append((str(path), path.stat().st_mtime_ns, path.stat().st_size))
            sidecar = path.with_suffix(".json")
            if sidecar.is_file():
                files.append((str(sidecar), sidecar.stat().st_mtime_ns, sidecar.stat().st_size))
        return json.dumps(files)


class SECoursesH3RefModTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        import nodes
        native = nodes.NODE_CLASS_MAPPINGS.get(cls.NATIVE)
        if native is None:
            # Standalone ComfyUI has no SwarmTextEncodeAdvanced. Keep object_info
            # functional; this adapter is only used by a Swarm backend.
            native = nodes.NODE_CLASS_MAPPINGS["CLIPTextEncode"]
        inputs = native.INPUT_TYPES()
        return {**inputs, "optional": {**inputs.get("optional", {}), "vae": ("VAE",),
                                      "refmods": ("STRING", {"default": "[]", "multiline": True})}}

    NATIVE = "CLIPTextEncode"
    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    @classmethod
    def IS_CHANGED(cls, refmods="[]", **kwargs):
        return SECoursesH3RefModStack.IS_CHANGED(refmods)

    def encode(self, clip, vae=None, refmods="[]", **kwargs):
        import nodes
        native = nodes.NODE_CLASS_MAPPINGS[self.NATIVE]()
        if not parse_selection(refmods):
            return native.encode(clip=clip, **kwargs)
        proxy = RefModClip(clip, vae, refmods)
        result = native.encode(clip=proxy, **kwargs)
        return (proxy.append(result[0]),)


class SECoursesH3RefModSwarmTextEncode(SECoursesH3RefModTextEncode):
    NATIVE = "SwarmTextEncodeAdvanced"


def pack_frame(frame):
    buffer = bytes_io.BytesIO()
    Image.fromarray((frame.float().clamp(0, 1) * 255).round().byte().cpu().numpy()).save(buffer, "JPEG", quality=95)
    return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)


def build_refmod(name, description, resolution, image_mode, compression, vae=None, audio_vae=None,
                 ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
    filename = relative_name(name)
    root = refmod_roots()[0]
    destination = (root / filename).resolve()
    if not destination.is_relative_to(root):
        raise ValueError("RefMod output leaves models/refmods.")
    # New files are numbered; a queued builder never replaces a valuable RefMod.
    destination.parent.mkdir(parents=True, exist_ok=True)
    stem = destination.stem
    serial = 1
    while destination.exists():
        destination = destination.with_name(f"{stem}_{serial:03d}.safetensors")
        serial += 1
    members, tensors, image_latents, image_previews = [], {}, [], []
    preview = None

    def add(kind, z, frames=None, times=None, source=""):
        nonlocal preview
        z = z.detach().cpu().contiguous()
        if compression != "none" and kind != "audio":
            size = int(compression)
            z = F.adaptive_avg_pool3d(z.float(), (z.shape[2], min(size, z.shape[3]), min(size, z.shape[4]))).to(z.dtype)
        validate_latent(z, kind, name)
        index = len(members)
        if index >= 256:
            raise ValueError("A portable RefMod bundle supports up to 256 members. Split these sources into multiple files.")
        member = {"_format_version": 4, "name": f"{Path(filename).stem} {LABELS[kind]} {index + 1}",
                  "kind": kind, "source": source, "mode": "encode" if compression == "none" else "training",
                  "description": description, "sample_rate": 32000,
                  "latent_t": z.shape[-1] if kind == "audio" else z.shape[2],
                  "latent_h": 0 if kind == "audio" else z.shape[3],
                  "latent_w": 0 if kind == "audio" else z.shape[4]}
        tensors[f"ref_{index}"] = z
        if frames is not None:
            if preview is None:
                preview = frames[:1].cpu()
            # Fantastic's enc_times contract refers to standalone enc_N tensors.
            # Namespace our bundle previews so that upstream readers decode the
            # latent instead of looking for a non-existent standalone frame.
            member.update(secourses_enc_times=times or [float(i) for i in range(len(frames))], secourses_enc_fps=24.0)
            for j, frame in enumerate(frames):
                tensors[f"ref_{index}_enc_{j}"] = pack_frame(frame)
        members.append(member)

    def visual(frames, kind, canvas=None):
        if vae is None:
            raise ValueError("Connect the MiniMax H3 video VAE to build visual RefMods.")
        h, w = frames.shape[1:3]
        scale = min(1.0, resolution / max(h, w))
        tw, th = canvas or (max(32, round(w * scale / 32) * 32), max(32, round(h * scale / 32) * 32))
        frames = _resize(frames[..., :3], tw, th, "center" if canvas else "disabled")
        if kind == "image":
            frames = frames[:1]
        else:
            n = len(frames) - ((len(frames) - 5) % 17)
            if n < 5:
                raise ValueError("A RefMod video needs at least 5 frames at 24 FPS.")
            frames = frames[:n]
        return vae.encode(frames), frames

    for pixels in (ref_images or {}).values():
        if pixels is None:
            continue
        canvas = (image_previews[0].shape[2], image_previews[0].shape[1]) if image_previews and image_mode == "one character" else None
        z, frames = visual(pixels, "image", canvas)
        if image_mode == "one character":
            image_latents.append(z.cpu())
            image_previews.append(frames.cpu())
        else:
            add("image", z, frames, [0.0], "image")
    if image_latents:
        kind = "image" if len(image_latents) == 1 else "video"
        add(kind, torch.cat(image_latents, dim=2), torch.cat(image_previews), source="stack" if kind == "video" else "image")
    for key, frames in (ref_videos or {}).items():
        if frames is None:
            continue
        z, frames = visual(frames, "video")
        add("video", z, frames[::12], [i / 24 for i in range(0, len(frames), 12)], "video")
        audio = (ref_video_audios or {}).get("ref_video_audio_" + key.rsplit("_", 1)[-1])
        if audio is not None:
            if audio_vae is None:
                raise ValueError("Connect the MiniMax H3 audio VAE to include the video's soundtrack.")
            z, _ = _encode_ref_audio(audio_vae, audio)
            add("audio", z, source="video soundtrack")
    for audio in (ref_audios or {}).values():
        if audio is None:
            continue
        if audio_vae is None:
            raise ValueError("Connect the MiniMax H3 audio VAE to build audio RefMods.")
        z, _ = _encode_ref_audio(audio_vae, audio)
        add("audio", z, source="audio")
    if not members:
        raise ValueError("Add at least one image, video, or audio reference to build a RefMod.")
    metadata = {"_format_version": 5, "kind": "bundle", "name": Path(filename).stem,
                "description": description, "members": members}
    temporary = destination.with_name(f".refmod-{uuid.uuid4().hex}.tmp")
    save_file(tensors, str(temporary), metadata={META_KEY: json.dumps(metadata)})
    os.rename(temporary, destination)
    if preview is None:
        preview = torch.full((1, 256, 256, 3), 0.15)
    rel = destination.relative_to(root).as_posix()
    report = f"Saved {rel}: " + ", ".join(f"{m['kind']} ({m['latent_t']} latent frames)" for m in members)
    print("[SECourses H3 RefMods] " + report, flush=True)
    return rel, preview, report


class SECoursesH3RefModBuild(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = MiniMaxH3ReferenceToVideo.define_schema()
        schema.node_id = "SECoursesH3RefModBuild"
        schema.display_name = "Build MiniMax H3 RefMod"
        schema.inputs = [i for i in schema.inputs if i.id in ("vae", "audio_vae", "ref_images", "ref_videos", "ref_video_audios", "ref_audios")]
        schema.inputs += [io.String.Input("name", default="my_character"),
                          io.String.Input("description", default="", multiline=True),
                          io.Int.Input("resolution", default=512, min=64, max=2048, step=32),
                          io.Combo.Input("image_mode", options=["one character", "separate references"], default="one character"),
                          io.Combo.Input("compression", options=["none", "32", "16", "8"], default="none")]
        schema.outputs = [io.String.Output(display_name="filename"), io.Image.Output(display_name="preview"), io.String.Output(display_name="report")]
        schema.is_output_node = True
        return schema

    @classmethod
    def execute(cls, **kwargs):
        return io.NodeOutput(*build_refmod(**kwargs))


class SECoursesH3RefModBuildGallery:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"references": (REF_PACK,), "name": ("STRING", {"default": "my_character"}),
                    "description": ("STRING", {"default": "", "multiline": True}),
                    "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                    "image_mode": (["one character", "separate references"],), "compression": (["none", "32", "16", "8"],)},
                "optional": {"vae": ("VAE",), "audio_vae": ("VAE",)}}

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("filename", "preview", "report")
    FUNCTION = "build"
    OUTPUT_NODE = True

    def build(self, references, name, description, resolution, image_mode, compression, vae=None, audio_vae=None):
        from .reference_gallery_nodes import (
            _prepare_image_references, _prepare_video_references, _LazyImageReferences,
            _LazyVideoReferences, _LazyVideoAudioReferences, _LazyAudioReferences,
        )
        images = _prepare_image_references(references.get("images", []), resolution, resolution, "match")
        seconds = float(references.get("max_seconds", 15))
        videos = _prepare_video_references(references.get("videos", []), 24, seconds, math.ceil(seconds * 24))
        result = build_refmod(name, description, resolution, image_mode, compression, vae, audio_vae,
                              _LazyImageReferences(images), _LazyVideoReferences(videos, 24),
                              _LazyVideoAudioReferences(videos), _LazyAudioReferences(references.get("audios", []), seconds))
        return {"ui": {"text": [result[2]]}, "result": result}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (
    SECoursesH3RefModStack, SECoursesH3RefModReferences, SECoursesH3RefModImageToVideo,
    SECoursesH3RefModBuild, SECoursesH3RefModBuildGallery, SECoursesH3RefModTextEncode,
    SECoursesH3RefModSwarmTextEncode,
)}
NODE_DISPLAY_NAME_MAPPINGS = {"SECoursesH3RefModStack": "MiniMax H3 Optional RefMods",
                              "SECoursesH3RefModBuildGallery": "Build MiniMax H3 RefMod (Gallery)"}


def register_routes():
    from aiohttp import web
    from server import PromptServer

    @PromptServer.instance.routes.get("/secourses/h3/refmods")
    async def list_refmods(request):
        return web.json_response(catalog())

    @PromptServer.instance.routes.post("/secourses/h3/refmods/upload")
    async def upload_refmod(request):
        reader = await request.multipart()
        part = await reader.next()
        if part is None or not part.filename or not part.filename.lower().endswith(".safetensors"):
            raise web.HTTPBadRequest(text="Choose a RefMod .safetensors file.")
        try:
            filename = relative_name(part.filename)
        except ValueError as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        root = refmod_roots()[0]
        root.mkdir(parents=True, exist_ok=True)
        destination = (root / filename).resolve()
        if not destination.is_relative_to(root):
            raise web.HTTPBadRequest(text="RefMod path leaves models/refmods.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        stem, serial = destination.stem, 1
        while destination.exists():
            destination = destination.with_name(f"{stem}_{serial:03d}.safetensors")
            serial += 1
        temporary = destination.with_name(f".refmod-upload-{uuid.uuid4().hex}.tmp")
        with temporary.open("xb") as target:
            while chunk := await part.read_chunk(1024 * 1024):
                target.write(chunk)
        try:
            with safe_open(temporary, framework="pt", device="cpu") as file:
                meta, members = read_metadata(temporary, file.metadata() or {})
                for index, member in enumerate(members):
                    key = f"ref_{index}" if meta.get("kind") == "bundle" else "latent"
                    validate_latent(file.get_tensor(key), member["kind"], filename)
        except (ValueError, RuntimeError, OSError, KeyError, SafetensorError) as error:
            # Keep rejected bytes outside the library listing for recoverability.
            raise web.HTTPBadRequest(text=f"Invalid RefMod: {error}") from error
        os.rename(temporary, destination)
        return web.json_response({"file": destination.relative_to(root).as_posix()})


register_routes()
