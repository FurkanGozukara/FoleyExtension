"""SECourses Reference Gallery: SwarmUI-style dynamic media references for ComfyUI.

Two nodes cooperate to replace banks of LoadImage / LoadVideo / LoadAudio nodes:

- ``SECoursesReferenceGallery`` is model-agnostic. Its frontend widget (see
  ``web/js/secourses_reference_gallery.js``) uploads images, videos, and audio
  through ComfyUI's native ``/upload/image`` endpoint and stores an ordered
  JSON manifest in the hidden ``references`` widget. The Python side loads the
  files (videos are resampled to ``video_fps`` and capped at ``max_seconds``
  with their soundtrack paired automatically) and emits one reference pack.
- ``SECoursesMiniMaxH3References`` adapts a reference pack to MiniMax H3: it
  rewrites ``@image1`` / ``@video1`` / ``@audio1`` prompt tokens into the
  ``<Picture i>`` / ``<Video k>`` / ``<Audio j>`` labels the model expects and
  then defers to ComfyUI's own ``MiniMaxH3ReferenceToVideo`` implementation.

Future reference-driven models only need another thin adapter node; the
gallery, its UI, and the ``@`` token grammar stay identical.
"""

from __future__ import annotations

import hashlib
import json
import math
import re


REF_PACK_TYPE = "SECOURSES_REF_PACK"

# Mirrors the MiniMax H3 reference pipeline's canvas area cap (768*1344). Video
# frames are decoded straight to at most this many pixels per frame so a phone
# 4K clip does not expand to tens of gigabytes of float tensors. Reference
# images are NOT capped here: the adapter's "max" mode wants full detail.
VIDEO_DECODE_AREA_CAP = 768 * 1344

# One entry per '@' alias, canonicalized: @img2 == @image2, @pic1 == @picture1 == @image1.
TOKEN_MATCHER = re.compile(
    r"(?<![0-9A-Za-z_@])@(?P<type>image|img|picture|pic|video|vid|audio|aud|sound)#?(?P<num>\d{1,2})(?![0-9A-Za-z])",
    re.IGNORECASE,
)

_CANONICAL_TYPE = {
    "image": "image", "img": "image", "picture": "image", "pic": "image",
    "video": "video", "vid": "video",
    "audio": "audio", "aud": "audio", "sound": "audio",
}


def translate_reference_tokens(prompt, image_count, video_count, audio_count, audio_label_offset):
    """Rewrites '@image1' style tokens into the '<Picture 1>' labels MiniMax H3 expects.

    Audio labels index video soundtracks first, so standalone audio tokens are
    shifted by ``audio_label_offset`` (the number of videos that carry sound).
    Legacy '<Picture 1>' labels typed directly in the prompt pass through
    unchanged. Raises ``ValueError`` on tokens that point at missing references.
    """
    if not prompt or "@" not in prompt:
        return prompt

    def replace(match):
        kind = _CANONICAL_TYPE[match.group("type").lower()]
        number = int(match.group("num"))
        label, count, offset = {
            "image": ("Picture", image_count, 0),
            "video": ("Video", video_count, 0),
            "audio": ("Audio", audio_count, audio_label_offset),
        }[kind]
        if number < 1:
            raise ValueError(
                f"The prompt reference '{match.group(0)}' is invalid: reference numbering starts at 1, eg '@{kind}1'."
            )
        if number > count:
            plural = " is" if count == 1 else "s are"
            raise ValueError(
                f"The prompt references '{match.group(0)}' but only {count} {kind} reference{plural} attached."
            )
        return f"<{label} {offset + number}>"

    return TOKEN_MATCHER.sub(replace, prompt)


def _parse_manifest(references):
    """Parses the gallery JSON manifest into {'images': [...], 'videos': [...], 'audios': [...]}."""
    if not references or not str(references).strip():
        return {"images": [], "videos": [], "audios": []}
    try:
        data = json.loads(references)
    except (TypeError, ValueError) as error:
        raise ValueError(f"The reference gallery manifest is not valid JSON: {error}")
    if not isinstance(data, dict):
        raise ValueError("The reference gallery manifest must be a JSON object.")
    manifest = {}
    for key in ("images", "videos", "audios"):
        entries = data.get(key) or []
        if not isinstance(entries, list):
            raise ValueError(f"Reference gallery manifest field '{key}' must be a list.")
        cleaned = []
        for entry in entries:
            if isinstance(entry, str):
                entry = {"file": entry}
            if not isinstance(entry, dict) or not entry.get("file"):
                raise ValueError(f"Reference gallery manifest field '{key}' has an entry without a file.")
            name = str(entry.get("name") or "")
            if not name:
                name = re.sub(r" \[(input|output|temp)\]$", "", str(entry["file"]))
                name = name.replace("\\", "/").rsplit("/", 1)[-1]
            cleaned.append({"file": str(entry["file"]), "name": name})
        manifest[key] = cleaned
    return manifest


def _resolve_reference_path(file):
    import folder_paths

    # get_annotated_filepath rejects path traversal outside the input directory.
    return folder_paths.get_annotated_filepath(file)


def _load_reference_image(path):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        if img.mode == "I":
            img = img.point(lambda value: value * (1 / 255))
        img = img.convert("RGB")
        array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None,]


def _decode_video_frames(path, fps_out, max_seconds):
    """Streams a video into a uint8 [T, H, W, 3] tensor resampled to ``fps_out``.

    Frames are downscaled (aspect preserved, never upscaled) during decode so
    each frame stays at or below VIDEO_DECODE_AREA_CAP pixels.
    """
    import av
    import torch
    import torch.nn.functional as functional

    fps_out = max(1.0, float(fps_out))
    max_frames = max(1, round(max_seconds * fps_out))
    frames = []
    target_size = None

    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in reference video '{path}'.")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate) if stream.average_rate else fps_out
        decoded_count = 0
        first_time = None
        next_output_time = 0.0
        output_interval = 1.0 / fps_out

        for frame in container.decode(stream):
            if frame.pts is not None:
                frame_time = float(frame.pts * frame.time_base)
            else:
                frame_time = decoded_count / source_fps
            decoded_count += 1
            if first_time is None:
                first_time = frame_time
            relative_time = max(0.0, frame_time - first_time)
            if relative_time + 1e-7 < next_output_time:
                continue

            rgb = torch.from_numpy(frame.to_ndarray(format="rgb24"))
            if target_size is None:
                height, width = rgb.shape[0], rgb.shape[1]
                scale = min(1.0, math.sqrt(VIDEO_DECODE_AREA_CAP / max(1, width * height)))
                target_size = (max(1, round(height * scale)), max(1, round(width * scale)))
            if (rgb.shape[0], rgb.shape[1]) != target_size:
                resized = functional.interpolate(
                    rgb.permute(2, 0, 1)[None].float(),
                    size=target_size, mode="bilinear", align_corners=False, antialias=True,
                )
                rgb = resized[0].permute(1, 2, 0).clamp_(0, 255).to(torch.uint8)

            while len(frames) < max_frames:
                frames.append(rgb)
                next_output_time += output_interval
                if relative_time + 1e-7 < next_output_time:
                    break
            if len(frames) >= max_frames:
                break

    if not frames:
        raise ValueError(f"No decodable frames were found in reference video '{path}'.")
    return torch.stack(frames, dim=0)


def _decode_video_audio(path, max_seconds):
    """Returns the video's soundtrack as a ComfyUI AUDIO dict, or None when it has no usable audio."""
    import av
    import torch

    try:
        with av.open(path, mode="r") as container:
            if not container.streams.audio:
                return None
            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            if not sample_rate:
                return None
            n_channels = stream.channels or 1
            max_samples = int(max_seconds * sample_rate)
            chunks = []
            collected = 0
            for frame in container.decode(streams=stream.index):
                buffer = torch.from_numpy(frame.to_ndarray())
                if buffer.shape[0] != n_channels:
                    buffer = buffer.view(-1, n_channels).t()
                chunks.append(buffer)
                collected += buffer.shape[1]
                if collected >= max_samples:
                    break
            if not chunks:
                return None
            waveform = torch.cat(chunks, dim=1)[:, :max_samples]
    except (av.error.FFmpegError, OSError):
        return None
    waveform = _f32_pcm(waveform)
    return {"waveform": waveform[None,], "sample_rate": sample_rate}


def _f32_pcm(waveform):
    import torch

    if waveform.dtype.is_floating_point:
        return waveform.float()
    if waveform.dtype == torch.int16:
        return waveform.float() / (2 ** 15)
    if waveform.dtype == torch.int32:
        return waveform.float() / (2 ** 31)
    raise ValueError(f"Unsupported reference audio dtype: {waveform.dtype}")


def _load_reference_audio(path, max_seconds):
    try:
        from comfy_extras.nodes_audio import load as load_audio_file
    except ImportError:
        load_audio_file = None
    if load_audio_file is not None:
        waveform, sample_rate = load_audio_file(path)
    else:  # Very old ComfyUI: decode with PyAV directly, mirroring nodes_audio.load.
        import av
        import torch

        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError(f"No audio stream found in reference audio '{path}'.")
            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            n_channels = stream.channels or 1
            chunks = []
            for frame in container.decode(streams=stream.index):
                buffer = torch.from_numpy(frame.to_ndarray())
                if buffer.shape[0] != n_channels:
                    buffer = buffer.view(-1, n_channels).t()
                chunks.append(buffer)
            if not chunks:
                raise ValueError(f"No audio frames decoded from reference audio '{path}'.")
            waveform = _f32_pcm(torch.cat(chunks, dim=1))
    max_samples = int(max_seconds * sample_rate)
    waveform = waveform[..., :max_samples]
    return {"waveform": waveform[None,] if waveform.dim() == 2 else waveform, "sample_rate": sample_rate}


class SECoursesReferenceGallery:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "dynamicPrompts": True,
                    "tooltip": "Prompt for the generation. Type '@' to reference gallery attachments, eg '@image1', '@video1', '@audio1' (aliases like '@img1', '@pic1', '@vid1', '@sound1' also work).",
                }),
                "references": ("STRING", {
                    "multiline": False,
                    "default": "{}",
                    "tooltip": "JSON manifest managed by the gallery widget. Add references with the 'Add references' button, drag-and-drop, or paste; do not edit by hand.",
                }),
                "video_fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Reference videos are resampled to this frame rate while decoding. MiniMax H3 expects 24.",
                }),
                "max_seconds": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Reference videos and audio are trimmed to at most this many seconds. MiniMax H3 supports 2-15 second references; shorter references encode faster.",
                }),
            }
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = (REF_PACK_TYPE, "STRING")
    RETURN_NAMES = ("references", "prompt")
    OUTPUT_TOOLTIPS = (
        "Every gallery reference bundled in upload order, ready for a model adapter node such as 'MiniMax H3 References (Gallery)'.",
        "The prompt exactly as typed, including '@' reference tokens.",
    )
    FUNCTION = "collect"
    DESCRIPTION = (
        "SwarmUI-style unified reference uploader: add images, videos, and audio next to the prompt and mention "
        "them as '@image1', '@video1', or '@audio1'. Video soundtracks are paired automatically. Feed the "
        "references output into a model adapter node (eg 'MiniMax H3 References (Gallery)')."
    )

    def collect(self, prompt, references, video_fps, max_seconds):
        manifest = _parse_manifest(references)
        max_seconds = max(1.0, float(max_seconds))

        images = []
        for entry in manifest["images"]:
            path = _resolve_reference_path(entry["file"])
            images.append({"pixels": _load_reference_image(path), "name": entry["name"]})

        videos = []
        for entry in manifest["videos"]:
            path = _resolve_reference_path(entry["file"])
            frames = _decode_video_frames(path, video_fps, max_seconds)
            videos.append({
                "frames": frames,
                "fps": float(video_fps),
                "audio": _decode_video_audio(path, max_seconds),
                "name": entry["name"],
            })

        audios = []
        for entry in manifest["audios"]:
            path = _resolve_reference_path(entry["file"])
            audios.append({"audio": _load_reference_audio(path, max_seconds), "name": entry["name"]})

        pack = {
            "version": 1,
            "prompt": prompt,
            "images": images,
            "videos": videos,
            "audios": audios,
        }
        summary = ", ".join(
            f"{len(values)} {kind}" for kind, values in
            (("image(s)", images), ("video(s)", videos), ("audio", audios))
        )
        print(f"[SECoursesReferenceGallery] collected {summary}", flush=True)
        return (pack, prompt)

    @classmethod
    def IS_CHANGED(cls, prompt, references, video_fps, max_seconds):
        digest = hashlib.sha256()
        digest.update(repr((prompt, references, float(video_fps), float(max_seconds))).encode("utf-8"))
        try:
            manifest = _parse_manifest(references)
        except ValueError:
            return float("nan")
        for kind in ("images", "videos", "audios"):
            for entry in manifest[kind]:
                try:
                    path = _resolve_reference_path(entry["file"])
                    with open(path, "rb") as handle:
                        digest.update(handle.read())
                except (OSError, ValueError):
                    digest.update(f"missing:{entry['file']}".encode("utf-8"))
        return digest.hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, references):
        import folder_paths

        try:
            manifest = _parse_manifest(references)
        except ValueError as error:
            return str(error)
        for kind in ("images", "videos", "audios"):
            for entry in manifest[kind]:
                if not folder_paths.exists_annotated_filepath(entry["file"]):
                    return f"Reference file not found: {entry['file']}. Re-add it in the gallery."
        return True


class SECoursesMiniMaxH3References:
    MAX_IMAGES = 9
    MAX_VIDEOS = 3
    MAX_AUDIOS = 3

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import nodes
            max_resolution = nodes.MAX_RESOLUTION
        except ImportError:
            max_resolution = 16384
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MiniMax H3 text/vision encoder (Qwen3-VL-32B)."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
                "references": (REF_PACK_TYPE, {"tooltip": "Reference pack from the SECourses Reference Gallery node."}),
                "width": ("INT", {"default": 1344, "min": 32, "max": max_resolution, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": max_resolution, "step": 32}),
                "length": ("INT", {
                    "default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362).",
                }),
                "ref_image_size": (["match", "max"], {
                    "default": "match",
                    "tooltip": "'match' scales each reference image to the generation's pixel area; 'max' keeps up to a 2048px short edge for best identity fidelity at a large speed and memory cost.",
                }),
            },
            "optional": {
                "prompt_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional external prompt. When connected and non-empty it replaces the gallery prompt; '@' reference tokens work here too.",
                }),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    DESCRIPTION = (
        "Feeds a SECourses Reference Gallery pack to ComfyUI's native MiniMax H3 Ref2VA conditioning. '@image1' / "
        "'@video1' / '@audio1' prompt tokens are translated to the '<Picture 1>' / '<Video 1>' / '<Audio 1>' labels "
        "the model expects, with audio labels offset past video soundtracks automatically."
    )

    def encode(self, clip, vae, audio_vae, references, width, height, length, ref_image_size, prompt_override=None):
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ReferenceToVideo
        except ImportError:
            raise RuntimeError(
                "MiniMax H3 nodes were not found. Update ComfyUI to a version that ships comfy_extras/nodes_minimax_h3.py."
            )

        if not isinstance(references, dict):
            raise ValueError("The references input must come from a SECourses Reference Gallery node.")
        images = references.get("images") or []
        videos = references.get("videos") or []
        audios = references.get("audios") or []
        if len(images) > self.MAX_IMAGES:
            raise ValueError(f"MiniMax H3 supports at most {self.MAX_IMAGES} image references; the gallery has {len(images)}.")
        if len(videos) > self.MAX_VIDEOS:
            raise ValueError(f"MiniMax H3 supports at most {self.MAX_VIDEOS} video references; the gallery has {len(videos)}.")
        if len(audios) > self.MAX_AUDIOS:
            raise ValueError(f"MiniMax H3 supports at most {self.MAX_AUDIOS} audio references; the gallery has {len(audios)}.")

        prompt = references.get("prompt") or ""
        if prompt_override is not None and str(prompt_override).strip():
            prompt = str(prompt_override)

        videos_with_audio = sum(1 for video in videos if video.get("audio") is not None)
        translated = translate_reference_tokens(prompt, len(images), len(videos), len(audios), videos_with_audio)

        ref_images = {}
        for index, image in enumerate(images):
            ref_images[f"ref_image_{index}"] = image["pixels"]

        import torch

        ref_videos = {}
        ref_video_audios = {}
        for index, video in enumerate(videos):
            frames = video["frames"]
            if frames.dtype == torch.uint8:
                frames = frames.float() / 255.0
            if frames.shape[0] < 5:
                raise ValueError(
                    f"Reference video {index + 1} ('{video.get('name', '?')}') has fewer than 5 usable frames (~0.2s at 24 fps)."
                )
            ref_videos[f"ref_video_{index}"] = frames
            if video.get("audio") is not None:
                ref_video_audios[f"ref_video_audio_{index}"] = video["audio"]

        ref_audios = {}
        for index, audio in enumerate(audios):
            ref_audios[f"ref_audio_{index}"] = audio["audio"]

        if translated != prompt:
            print(f"[SECoursesMiniMaxH3References] prompt for the model: {translated}", flush=True)
        output = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=vae, audio_vae=audio_vae, prompt=translated,
            width=width, height=height, length=length, ref_image_size=ref_image_size,
            ref_images=ref_images or None, ref_videos=ref_videos or None,
            ref_video_audios=ref_video_audios or None, ref_audios=ref_audios or None,
        )
        conditioning, latent = output.args[0], output.args[1]
        return (conditioning, latent)


NODE_CLASS_MAPPINGS = {
    "SECoursesReferenceGallery": SECoursesReferenceGallery,
    "SECoursesMiniMaxH3References": SECoursesMiniMaxH3References,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SECoursesReferenceGallery": "SECourses Reference Gallery (Images / Videos / Audio)",
    "SECoursesMiniMaxH3References": "MiniMax H3 References (Gallery)",
}
