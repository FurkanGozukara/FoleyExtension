import math
import time

import av
import torch


class SwarmLTXFoleyVideoFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "maximum_frames": ("INT", {"default": 169, "min": 1, "max": 4097, "step": 8}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "auto_reencode": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "SwarmUI/video"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "frames")
    FUNCTION = "load"
    DESCRIPTION = "Streams a video, optionally resamples it to the target FPS, stops at the Foley frame cap, and returns a valid 8n+1 frame batch without decoding the entire source into RAM."

    def load(self, video, maximum_frames, target_fps, auto_reencode):
        maximum_frames = max(1, min(int(maximum_frames), 4097))
        target_fps = max(1.0, float(target_fps))
        duration = float(video.get_duration()) if hasattr(video, "get_duration") else 0.0
        source = video.get_stream_source()
        if hasattr(source, "seek"):
            source.seek(0)

        with av.open(source, mode="r") as container:
            if not container.streams.video:
                raise ValueError("The Foley input does not contain a video stream.")
            stream = container.streams.video[0]
            source_fps = float(stream.average_rate) if stream.average_rate else target_fps
            output_fps = target_fps if auto_reencode else source_fps
            estimated = math.ceil(duration * output_fps) + 1 if duration > 0 else maximum_frames
            capacity = max(1, min(maximum_frames, estimated))
            images = torch.empty((capacity, stream.height, stream.width, 3), dtype=torch.float32)

            output_count = 0
            decoded_count = 0
            first_time = None
            next_output_time = 0.0
            output_interval = 1.0 / output_fps

            for frame in container.decode(stream):
                if frame.pts is not None:
                    frame_time = float(frame.pts * frame.time_base)
                else:
                    frame_time = decoded_count / source_fps
                decoded_count += 1
                if first_time is None:
                    first_time = frame_time
                relative_time = max(0.0, frame_time - first_time)

                if auto_reencode and relative_time + 1e-7 < next_output_time:
                    continue

                while output_count < capacity:
                    rgb = frame.to_ndarray(format="rgb24")
                    images[output_count].copy_(torch.from_numpy(rgb).to(torch.float32).div_(255.0))
                    output_count += 1
                    next_output_time += output_interval
                    if not auto_reencode or relative_time + 1e-7 < next_output_time:
                        break
                if output_count >= capacity:
                    break

        if output_count == 0:
            raise ValueError("No decodable frames were found in the Foley input video.")
        valid_frames = ((output_count - 1) // 8) * 8 + 1
        return (images[:valid_frames], valid_frames)


class SwarmLTXValidFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "maximum_frames": ("INT", {"default": 169, "min": 1, "max": 4097, "step": 8}),
            }
        }

    CATEGORY = "SwarmUI/video"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "frames")
    FUNCTION = "trim"
    DESCRIPTION = "Trims a video to the largest LTX-compatible frame count (8n+1) that fits the input and configured maximum."

    def trim(self, image, maximum_frames):
        available = max(1, min(int(image.shape[0]), int(maximum_frames)))
        valid_frames = ((available - 1) // 8) * 8 + 1
        # A view is sufficient here and avoids duplicating multi-GB video tensors.
        return (image[:valid_frames], valid_frames)


class SwarmLTXFoleyVideoWindowPlan:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "maximum_frames": ("INT", {"default": 4097, "min": 1, "max": 4097, "step": 8}),
                "window_frames": ("INT", {"default": 89, "min": 9, "max": 257, "step": 8}),
                "overlap_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "max_windows": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
            }
        }

    CATEGORY = "SwarmUI/video"
    RETURN_TYPES = ("FOLEY_WINDOW_PLAN", "INT", "STRING")
    RETURN_NAMES = ("window_plan", "window_count", "manifest")
    FUNCTION = "plan"
    DESCRIPTION = "Plans low-memory overlapping Foley windows directly from compressed video metadata."

    def plan(self, video, target_fps, maximum_frames, window_frames, overlap_seconds, max_windows):
        import json
        from .fuzzpuppy_nodes import _window_specs

        fps = max(1.0, float(target_fps))
        duration = max(0.0, float(video.get_duration()))
        source_frames = max(1, min(int(maximum_frames), math.ceil(duration * fps)))
        all_specs = _window_specs(source_frames, fps, int(window_frames), float(overlap_seconds))
        specs = all_specs[: max(1, int(max_windows))]
        plan = {
            "version": "swarm_ltx_foley_stream_plan",
            "source_frames": source_frames,
            "frame_rate": fps,
            "source_duration": min(duration, source_frames / fps),
            "window_frames": int(window_frames),
            "overlap_seconds": float(overlap_seconds),
            "planned_window_count": len(all_specs),
            "truncated_by_max_windows": len(specs) < len(all_specs),
            "window_specs": specs,
        }
        print(
            f"[SwarmLTXFoley] planned {len(specs)}/{len(all_specs)} streamed windows "
            f"for {source_frames} frames at {fps:g} FPS",
            flush=True,
        )
        return (plan, len(specs), json.dumps(plan, indent=2))


class SwarmLTXFoleyVideoWindowSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "window_plan": ("FOLEY_WINDOW_PLAN",),
                "remaining": ("INT", {"default": 1, "min": 1, "max": 100000, "step": 1}),
                "width": ("INT", {"default": 576, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 576, "min": 256, "max": 2048, "step": 32}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    CATEGORY = "SwarmUI/video"
    RETURN_TYPES = ("IMAGE", "FOLEY_WINDOW")
    RETURN_NAMES = ("window_images", "window_info")
    FUNCTION = "select"
    DESCRIPTION = "Seeks and decodes only one Foley window, resampling it to the target FPS and conditioning size."

    @staticmethod
    def _resize_frame(frame, width, height):
        import numpy as np
        import torch.nn.functional as functional

        rgb = frame.to_ndarray(format="rgb24")
        source_height, source_width = rgb.shape[:2]
        target_ratio = width / height
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            crop_width = max(1, round(source_height * target_ratio))
            left = max(0, (source_width - crop_width) // 2)
            rgb = rgb[:, left : left + crop_width]
        elif source_ratio < target_ratio:
            crop_height = max(1, round(source_width / target_ratio))
            top = max(0, (source_height - crop_height) // 2)
            rgb = rgb[top : top + crop_height, :]
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        tensor = functional.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False, antialias=True)
        return tensor[0].permute(1, 2, 0)

    def select(self, video, window_plan, remaining, width, height, unique_id=None):
        specs = list(window_plan["window_specs"])
        loop_index = len(specs) - int(remaining)
        if loop_index < 0 or loop_index >= len(specs):
            raise ValueError(f"Loop remaining={remaining} is outside the planned {len(specs)} windows")
        spec = dict(specs[loop_index])
        frames = int(window_plan["window_frames"])
        fps = float(window_plan["frame_rate"])
        start_seconds = float(spec["start_seconds"])

        source = video.get_stream_source()
        if hasattr(source, "seek"):
            source.seek(0)
        images = torch.empty((frames, int(height), int(width), 3), dtype=torch.float32)
        output_count = 0
        next_output_time = start_seconds

        with av.open(source, mode="r") as container:
            if not container.streams.video:
                raise ValueError("The Foley input does not contain a video stream.")
            stream = container.streams.video[0]
            time_base = float(stream.time_base)
            origin_pts = int(stream.start_time or 0)
            seek_pts = origin_pts + max(0, int((start_seconds - 1.0) / time_base))
            if seek_pts > origin_pts:
                container.seek(seek_pts, stream=stream, backward=True, any_frame=False)

            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_time = max(0.0, (int(frame.pts) - origin_pts) * time_base)
                if frame_time + 1e-7 < next_output_time:
                    continue
                resized = self._resize_frame(frame, int(width), int(height))
                while output_count < frames and frame_time + 1e-7 >= next_output_time:
                    images[output_count].copy_(resized)
                    output_count += 1
                    next_output_time += 1.0 / fps
                if output_count >= frames:
                    break

        if output_count == 0:
            raise ValueError(f"No decodable frames were found for Foley window {spec['index']}")
        if output_count < frames:
            images[output_count:frames].copy_(images[output_count - 1 : output_count].expand(frames - output_count, -1, -1, -1))

        window_info = {
            "spec": spec,
            "window_count": len(specs),
            "source_frames": int(window_plan["source_frames"]),
            "frame_rate": fps,
            "source_duration": float(window_plan["source_duration"]),
            "overlap_seconds": float(window_plan["overlap_seconds"]),
            "planned_window_count": int(window_plan.get("planned_window_count", len(specs))),
            "truncated_by_max_windows": bool(window_plan.get("truncated_by_max_windows", False)),
        }
        print(
            f"[SwarmLTXFoley] decoded window {spec['index']}/{len(specs)} "
            f"at {width}x{height}, start={start_seconds:.3f}s, node={unique_id}",
            flush=True,
        )
        return (images, window_info)


class SwarmLTXFoleyMuxVideoAudioWS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "video/LTX23_Foley_AutoLong"}),
                "save_output": ("BOOLEAN", {"default": True}),
            }
        }

    CATEGORY = "SwarmUI/video"
    RETURN_TYPES = ()
    FUNCTION = "mux"
    OUTPUT_NODE = True
    DESCRIPTION = "Muxes generated Foley audio onto the original compressed video without materializing all full-resolution frames."

    def mux(self, video, audio, filename_prefix, save_output):
        import io
        import os
        import struct
        import subprocess
        import tempfile
        import wave

        import folder_paths
        import numpy as np
        from imageio_ffmpeg import get_ffmpeg_exe
        from server import BinaryEventTypes, PromptServer

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        paths = []
        source = video.get_stream_source()
        if isinstance(source, (str, os.PathLike)):
            video_path = os.fspath(source)
        else:
            if hasattr(source, "seek"):
                source.seek(0)
            handle = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir)
            handle.write(source.read())
            handle.close()
            video_path = handle.name
            paths.append(video_path)

        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=temp_dir).name
        if save_output:
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, folder_paths.get_output_directory()
            )
            output_name = f"{filename}_{counter:05}_.mp4"
            output_path = os.path.join(full_output_folder, output_name)
        else:
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir).name
            output_name = os.path.basename(output_path)
            subfolder = ""
            paths.append(output_path)
        paths.append(audio_path)
        try:
            waveform = audio["waveform"]
            if waveform.ndim == 3:
                waveform = waveform[0]
            waveform = waveform.detach().float().cpu().clamp(-1.0, 1.0)
            pcm = (waveform.movedim(0, -1).numpy() * 32767.0).round().astype(np.int16)
            with wave.open(audio_path, "wb") as wav_file:
                wav_file.setnchannels(int(waveform.shape[0]))
                wav_file.setsampwidth(2)
                wav_file.setframerate(int(audio["sample_rate"]))
                wav_file.writeframes(pcm.tobytes())

            ffmpeg = get_ffmpeg_exe()
            common = [ffmpeg, "-v", "error", "-y", "-i", video_path, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-shortest", "-movflags", "+faststart"]
            result = subprocess.run(common + ["-c:v", "copy", output_path], capture_output=True)
            if result.returncode != 0:
                result = subprocess.run(common + ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", output_path], capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(f"Foley video mux failed: {result.stderr.decode('utf-8', errors='replace')}")

            if not save_output:
                with open(output_path, "rb") as output_file:
                    payload = struct.pack(">I", 5) + output_file.read()
                server = PromptServer.instance
                server.send_sync("progress", {"value": 12346, "max": 12346}, sid=server.client_id)
                server.send_sync(BinaryEventTypes.PREVIEW_IMAGE, payload, sid=server.client_id)
        finally:
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass
        if save_output:
            return {
                "ui": {
                    "gifs": [{
                        "filename": output_name,
                        "subfolder": subfolder,
                        "type": "output",
                        "format": "video/mp4",
                        "fullpath": output_path,
                    }]
                }
            }
        return {}

    @classmethod
    def IS_CHANGED(cls, video, audio, filename_prefix, save_output):
        return time.time()


class SwarmPortableVAELoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": ("STRING", {"default": "QwenImage/qwen_image_vae.safetensors"}),
            }
        }

    CATEGORY = "SwarmUI/loaders"
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load"
    DESCRIPTION = "Loads a VAE using slash-normalized matching so the same workflow works on Windows and Linux."

    def load(self, vae_name):
        import folder_paths
        import nodes

        normalized = vae_name.replace("\\", "/").lower()
        available = folder_paths.get_filename_list("vae")
        matches = [name for name in available if name.replace("\\", "/").lower() == normalized]
        if not matches:
            basename = normalized.rsplit("/", 1)[-1]
            matches = [name for name in available if name.replace("\\", "/").lower().rsplit("/", 1)[-1] == basename]
        if len(matches) != 1:
            raise FileNotFoundError(f"Could not uniquely resolve portable VAE path '{vae_name}'. Matches: {matches}")
        return nodes.VAELoader().load_vae(matches[0])


NODE_CLASS_MAPPINGS = {
    "SwarmLTXFoleyVideoFrames": SwarmLTXFoleyVideoFrames,
    "SwarmLTXFoleyVideoWindowPlan": SwarmLTXFoleyVideoWindowPlan,
    "SwarmLTXFoleyVideoWindowSelect": SwarmLTXFoleyVideoWindowSelect,
    "SwarmLTXFoleyMuxVideoAudioWS": SwarmLTXFoleyMuxVideoAudioWS,
    "SwarmLTXValidFrames": SwarmLTXValidFrames,
    "SwarmPortableVAELoader": SwarmPortableVAELoader,
}

from .fuzzpuppy_nodes import NODE_CLASS_MAPPINGS as FUZZPUPPY_NODE_CLASS_MAPPINGS
from .fuzzpuppy_nodes import NODE_DISPLAY_NAME_MAPPINGS as FUZZPUPPY_NODE_DISPLAY_NAME_MAPPINGS
from .init_audio_nodes import NODE_CLASS_MAPPINGS as INIT_AUDIO_NODE_CLASS_MAPPINGS
from .init_audio_nodes import NODE_DISPLAY_NAME_MAPPINGS as INIT_AUDIO_NODE_DISPLAY_NAME_MAPPINGS
from .optional_image_nodes import NODE_CLASS_MAPPINGS as OPTIONAL_IMAGE_NODE_CLASS_MAPPINGS
from .optional_image_nodes import NODE_DISPLAY_NAME_MAPPINGS as OPTIONAL_IMAGE_NODE_DISPLAY_NAME_MAPPINGS
from .reference_gallery_nodes import NODE_CLASS_MAPPINGS as REFERENCE_GALLERY_NODE_CLASS_MAPPINGS
from .reference_gallery_nodes import NODE_DISPLAY_NAME_MAPPINGS as REFERENCE_GALLERY_NODE_DISPLAY_NAME_MAPPINGS
from .resolution_sync_nodes import NODE_CLASS_MAPPINGS as RESOLUTION_SYNC_NODE_CLASS_MAPPINGS
from .resolution_sync_nodes import NODE_DISPLAY_NAME_MAPPINGS as RESOLUTION_SYNC_NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS.update(FUZZPUPPY_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(INIT_AUDIO_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(OPTIONAL_IMAGE_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(REFERENCE_GALLERY_NODE_CLASS_MAPPINGS)
NODE_CLASS_MAPPINGS.update(RESOLUTION_SYNC_NODE_CLASS_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS = {
    **FUZZPUPPY_NODE_DISPLAY_NAME_MAPPINGS,
    **INIT_AUDIO_NODE_DISPLAY_NAME_MAPPINGS,
    **OPTIONAL_IMAGE_NODE_DISPLAY_NAME_MAPPINGS,
    **REFERENCE_GALLERY_NODE_DISPLAY_NAME_MAPPINGS,
    **RESOLUTION_SYNC_NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
