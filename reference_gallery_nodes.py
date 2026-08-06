"""SECourses Reference Gallery: SwarmUI-style dynamic media references for ComfyUI.

Two nodes cooperate to replace banks of LoadImage / LoadVideo / LoadAudio nodes:

- ``SECoursesReferenceGallery`` is model-agnostic. Its frontend widget (see
  ``web/js/secourses_reference_gallery.js``) uploads images, videos, and audio
  through ComfyUI's native ``/upload/image`` endpoint and stores an ordered
  JSON manifest in the hidden ``references`` widget. The Python side emits a
  lightweight reference pack; model adapters inspect and decode the files only
  when their target canvas and duration are known.
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
import warnings


REF_PACK_TYPE = "SECOURSES_REF_PACK"

CANVAS_MULTIPLE = 32
RGB_FLOAT_BYTES_PER_PIXEL = 3 * 4
IMAGE_MAX_SOURCE_PIXELS = 100_000_000
VIDEO_MAX_SOURCE_PIXELS = 40_000_000
IMAGE_MAX_AREA = 8 * 1024 * 1024
IMAGE_MAX_SHORT_EDGE = 2048
IMAGE_FLOAT_BUDGET = 512 * 1024 * 1024
VIDEO_FLOAT_BUDGET = 512 * 1024 * 1024
MIN_MEDIA_BUDGET = 16 * 1024 * 1024

# Mirrors the MiniMax H3 reference pipeline's 768*1344 canvas cap. The actual
# per-video cap is lowered when necessary to keep the decoded float batch inside
# VIDEO_FLOAT_BUDGET.
VIDEO_DECODE_AREA_CAP = 768 * 1344
HASH_CHUNK_SIZE = 1024 * 1024

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
    unchanged. Tokens that point at a missing reference (eg '@image3' with two
    images attached) are silently omitted, together with one adjacent space, so
    a stale token left in the prompt never blocks execution.
    """
    if not prompt or "@" not in prompt:
        return prompt

    pieces = []
    last = 0
    omitted = []
    for match in TOKEN_MATCHER.finditer(prompt):
        pieces.append(prompt[last:match.start()])
        last = match.end()
        kind = _CANONICAL_TYPE[match.group("type").lower()]
        number = int(match.group("num"))
        label, count, offset = {
            "image": ("Picture", image_count, 0),
            "video": ("Video", video_count, 0),
            "audio": ("Audio", audio_count, audio_label_offset),
        }[kind]
        if 1 <= number <= count:
            pieces.append(f"<{label} {offset + number}>")
            continue
        omitted.append(match.group(0))
        if last < len(prompt) and prompt[last] == " ":
            last += 1
        elif pieces and pieces[-1].endswith(" "):
            pieces[-1] = pieces[-1][:-1]
    pieces.append(prompt[last:])
    if omitted:
        print(
            "[SECoursesMiniMaxH3References] ignoring prompt reference(s) with no matching attachment: "
            + ", ".join(omitted),
            flush=True,
        )
    return "".join(pieces)


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


def _available_media_budget(maximum):
    """Use at most one eighth of currently available RAM for one media class."""
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
    except (ImportError, AttributeError, OSError):
        return maximum
    return max(MIN_MEDIA_BUDGET, min(maximum, available // 8))


def _fit_dimensions(width, height, max_pixels, max_short_edge=None, multiple=CANVAS_MULTIPLE):
    """Aspect-preserving, down-only dimensions bounded by area and short edge."""
    width = max(1, int(width))
    height = max(1, int(height))
    max_pixels = max(1, int(max_pixels))
    scale = 1.0
    if max_short_edge:
        scale = min(scale, float(max_short_edge) / min(width, height))
    if width * height * scale * scale > max_pixels:
        scale = min(scale, math.sqrt(float(max_pixels) / (width * height)))

    if multiple <= 1:
        return max(1, round(width * scale)), max(1, round(height * scale))

    ideal_width = width * scale
    ideal_height = height * scale
    width_steps = max(1, math.ceil(ideal_width / multiple))
    height_steps = max(1, math.ceil(ideal_height / multiple))
    source_ratio = width / height
    candidates = []

    # Search the small 32px canvas grid instead of rounding each axis
    # independently. This avoids stretching 16:9 and portrait media when a
    # memory budget lands between two grid sizes.
    def add_candidate(width_step, height_step):
        if width_step < 1 or height_step < 1 or width_step > width_steps or height_step > height_steps:
            return
        target_width = width_step * multiple
        target_height = height_step * multiple
        if width >= multiple and target_width > width:
            return
        if height >= multiple and target_height > height:
            return
        area = target_width * target_height
        if area > max_pixels:
            return
        if max_short_edge and min(target_width, target_height) > max_short_edge:
            return
        ratio_error = abs((target_width / target_height) / source_ratio - 1.0)
        candidates.append((ratio_error, area, target_width, target_height))

    if width_steps <= height_steps:
        for width_step in range(1, width_steps + 1):
            nearest_height_step = (width_step * multiple / source_ratio) / multiple
            add_candidate(width_step, max(1, math.floor(nearest_height_step)))
            add_candidate(width_step, max(1, math.ceil(nearest_height_step)))
    else:
        for height_step in range(1, height_steps + 1):
            nearest_width_step = (height_step * multiple * source_ratio) / multiple
            add_candidate(max(1, math.floor(nearest_width_step)), height_step)
            add_candidate(max(1, math.ceil(nearest_width_step)), height_step)

    if not candidates:
        return multiple, multiple

    ratio_safe = [candidate for candidate in candidates if candidate[0] <= 0.02]
    if ratio_safe:
        _error, _area, target_width, target_height = max(
            ratio_safe, key=lambda candidate: (candidate[1], -candidate[0])
        )
    else:
        _error, _area, target_width, target_height = min(
            candidates, key=lambda candidate: (candidate[0], -candidate[1])
        )
    return target_width, target_height


def _apply_total_pixel_budget(sizes, byte_budget):
    """Scale a list of (width, height) pairs to a shared RGB float budget."""
    total_pixels = sum(width * height for width, height in sizes)
    pixel_budget = max(1, int(byte_budget) // RGB_FLOAT_BYTES_PER_PIXEL)
    if total_pixels <= pixel_budget:
        return list(sizes)
    factor = pixel_budget / total_pixels
    return [
        _fit_dimensions(width, height, max(1, math.floor(width * height * factor)))
        for width, height in sizes
    ]


def _aligned_frame_count(length):
    frame_count = max(5, int(length))
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def _usable_video_frame_count(length, fps, max_seconds):
    """Frames the native H3 node can use after duration and 17k+5 trimming."""
    frame_count = min(
        _aligned_frame_count(length),
        max(1, round(float(max_seconds) * max(1.0, float(fps)))),
    )
    if frame_count < 5:
        return frame_count
    while frame_count % 17 != 5:
        frame_count -= 1
    return frame_count


def _oriented_image_dimensions(path):
    from PIL import Image

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as img:
                width, height = img.size
                orientation = img.getexif().get(274, 1)
    except Image.DecompressionBombError as error:
        raise ValueError(f"Reference image '{path}' is too large to decode safely: {error}") from error
    pixels = width * height
    if pixels > IMAGE_MAX_SOURCE_PIXELS:
        raise ValueError(
            f"Reference image '{path}' is {pixels / 1_000_000:.1f} MP; "
            f"the safety limit is {IMAGE_MAX_SOURCE_PIXELS / 1_000_000:.0f} MP."
        )
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return width, height, orientation


def _video_metadata(path):
    import av

    try:
        with av.open(path, mode="r") as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in reference video '{path}'.")
            stream = container.streams.video[0]
            width = int(stream.codec_context.width or 0)
            height = int(stream.codec_context.height or 0)
            has_audio = bool(container.streams.audio)
    except (av.error.FFmpegError, OSError) as error:
        raise ValueError(f"Could not inspect reference video '{path}': {error}") from error

    pixels = width * height
    if pixels > VIDEO_MAX_SOURCE_PIXELS:
        raise ValueError(
            f"Reference video '{path}' is {pixels / 1_000_000:.1f} MP per frame; "
            f"the safety limit is {VIDEO_MAX_SOURCE_PIXELS / 1_000_000:.0f} MP."
        )
    return width, height, has_audio


def _load_reference_image(path, target_width=None, target_height=None):
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    width, height, orientation = _oriented_image_dimensions(path)
    target_width = min(width, int(target_width or width))
    target_height = min(height, int(target_height or height))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(path) as img:
            stored_target = (target_height, target_width) if orientation in (5, 6, 7, 8) else (target_width, target_height)
            img.draft("RGB", stored_target)
            img = ImageOps.exif_transpose(img)
            if img.mode == "I":
                img = img.point(lambda value: value * (1 / 255))
            img = img.convert("RGB")
            if img.size != (target_width, target_height):
                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
            array = np.array(img, dtype=np.float32, copy=True)
            array *= 1.0 / 255.0
    return torch.from_numpy(array)[None,]


def _decode_video_frames(path, fps_out, max_seconds=None, *, max_frames=None, area_cap=VIDEO_DECODE_AREA_CAP):
    """Streams a video into a uint8 [T, H, W, 3] tensor resampled to ``fps_out``.

    Frames are downscaled by FFmpeg before conversion to a NumPy/Torch RGB
    buffer, avoiding a full-resolution RGB float interpolation allocation.
    """
    import av
    import torch

    fps_out = max(1.0, float(fps_out))
    if max_frames is None:
        max_frames = max(1, round(float(max_seconds) * fps_out))
    else:
        max_frames = max(1, int(max_frames))
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

            if target_size is None:
                width, height = frame.width, frame.height
                pixels = width * height
                if pixels > VIDEO_MAX_SOURCE_PIXELS:
                    raise ValueError(
                        f"Reference video '{path}' is {pixels / 1_000_000:.1f} MP per frame; "
                        f"the safety limit is {VIDEO_MAX_SOURCE_PIXELS / 1_000_000:.0f} MP."
                    )
                target_width, target_height = _fit_dimensions(width, height, area_cap)
                target_size = (target_height, target_width)
            if (frame.height, frame.width) != target_size:
                frame = frame.reformat(
                    width=target_size[1], height=target_size[0], format="rgb24",
                    interpolation=av.video.reformatter.Interpolation.LANCZOS,
                )
            rgb = torch.from_numpy(frame.to_ndarray(format="rgb24"))

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
    audio = _decode_video_audio(path, max_seconds)
    if audio is None:
        raise ValueError(f"No usable audio stream found in reference audio '{path}'.")
    return audio


def _prepare_image_references(entries, width, height, ref_image_size, byte_budget=None):
    target_area = IMAGE_MAX_AREA
    target_short_edge = IMAGE_MAX_SHORT_EDGE
    if ref_image_size == "match":
        target_area = min(IMAGE_MAX_AREA, max(CANVAS_MULTIPLE ** 2, int(width) * int(height)))
        target_short_edge = None

    specs = []
    requested_sizes = []
    for entry in entries:
        path = _resolve_reference_path(entry["file"])
        source_width, source_height, _orientation = _oriented_image_dimensions(path)
        target_width, target_height = _fit_dimensions(
            source_width,
            source_height,
            target_area,
            max_short_edge=target_short_edge,
        )
        specs.append({
            "path": path,
            "name": entry.get("name") or "?",
            "source_size": (source_width, source_height),
        })
        requested_sizes.append((target_width, target_height))

    budget = _available_media_budget(IMAGE_FLOAT_BUDGET) if byte_budget is None else int(byte_budget)
    target_sizes = _apply_total_pixel_budget(requested_sizes, budget)
    for spec, target_size in zip(specs, target_sizes):
        spec["target_size"] = target_size
    return specs


def _prepare_video_references(entries, fps, max_seconds, length, byte_budget=None):
    max_frames = _usable_video_frame_count(length, fps, max_seconds)
    budget = _available_media_budget(VIDEO_FLOAT_BUDGET) if byte_budget is None else int(byte_budget)
    float_pixel_budget = max(1, budget // (max(1, max_frames) * RGB_FLOAT_BYTES_PER_PIXEL))
    area_cap = max(CANVAS_MULTIPLE ** 2, min(VIDEO_DECODE_AREA_CAP, float_pixel_budget))
    audio_seconds = min(float(max_seconds), max_frames / max(1.0, float(fps)))

    specs = []
    for entry in entries:
        path = _resolve_reference_path(entry["file"])
        source_width, source_height, has_audio = _video_metadata(path)
        target_size = None
        if source_width and source_height:
            target_size = _fit_dimensions(source_width, source_height, area_cap)
        specs.append({
            "path": path,
            "name": entry.get("name") or "?",
            "source_size": (source_width, source_height),
            "target_size": target_size,
            "area_cap": area_cap,
            "max_frames": max_frames,
            "audio_seconds": audio_seconds,
            "has_audio": has_audio,
        })
    return specs


def _log_media_resize(kind, name, source_size, target_size):
    if target_size and source_size != target_size:
        print(
            f"[SECoursesMiniMaxH3References] safely downscaling {kind} '{name}' "
            f"from {source_size[0]}x{source_size[1]} to {target_size[0]}x{target_size[1]}",
            flush=True,
        )


class _LazyImageReferences:
    def __init__(self, specs):
        self.specs = specs

    def __len__(self):
        return len(self.specs)

    def values(self):
        for spec in self.specs:
            target_width, target_height = spec["target_size"]
            _log_media_resize("image", spec["name"], spec["source_size"], spec["target_size"])
            yield _load_reference_image(spec["path"], target_width, target_height)


class _LazyVideoReferences:
    def __init__(self, specs, fps):
        self.specs = specs
        self.fps = fps

    def __len__(self):
        return len(self.specs)

    def items(self):
        import torch

        for index, spec in enumerate(self.specs):
            _log_media_resize("video", spec["name"], spec["source_size"], spec["target_size"])
            uint8_frames = _decode_video_frames(
                spec["path"],
                self.fps,
                max_frames=spec["max_frames"],
                area_cap=spec["area_cap"],
            )
            if uint8_frames.shape[0] < 5:
                raise ValueError(
                    f"Reference video {index + 1} ('{spec['name']}') has fewer than 5 usable frames "
                    "(~0.2s at 24 fps)."
                )
            frames = uint8_frames.to(dtype=torch.float32)
            del uint8_frames
            frames.mul_(1.0 / 255.0)
            yield f"ref_video_{index}", frames


class _LazyVideoAudioReferences:
    def __init__(self, specs):
        self.specs = specs
        self._cache = {}

    def __len__(self):
        return sum(
            1 for index, spec in enumerate(self.specs)
            if spec["has_audio"] and self.get(f"ref_video_audio_{index}") is not None
        )

    def get(self, name, default=None):
        try:
            index = int(str(name).rsplit("_", 1)[-1])
            spec = self.specs[index]
        except (IndexError, TypeError, ValueError):
            return default
        if not spec["has_audio"]:
            return default
        if index not in self._cache:
            self._cache[index] = _decode_video_audio(spec["path"], spec["audio_seconds"])
        return self._cache[index] or default


class _LazyAudioReferences:
    def __init__(self, entries, max_seconds):
        self.entries = entries
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.entries)

    def values(self):
        for entry in self.entries:
            path = _resolve_reference_path(entry["file"])
            yield _load_reference_audio(path, self.max_seconds)


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
                    "tooltip": "Reference videos are resampled to this frame rate during model-aware decoding. MiniMax H3 expects 24.",
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
        "references output into a model adapter node (eg 'MiniMax H3 References (Gallery)'). Media stays lazy "
        "until the adapter can apply its canvas, duration, and memory limits."
    )

    def collect(self, prompt, references, video_fps, max_seconds):
        manifest = _parse_manifest(references)
        max_seconds = max(1.0, float(max_seconds))
        pack = {
            "version": 2,
            "prompt": prompt,
            "video_fps": float(video_fps),
            "max_seconds": max_seconds,
            "images": [dict(entry) for entry in manifest["images"]],
            "videos": [dict(entry) for entry in manifest["videos"]],
            "audios": [dict(entry) for entry in manifest["audios"]],
        }
        summary = ", ".join(
            f"{len(values)} {kind}" for kind, values in
            (("image(s)", pack["images"]), ("video(s)", pack["videos"]), ("audio", pack["audios"]))
        )
        print(f"[SECoursesReferenceGallery] prepared {summary} for target-aware decoding", flush=True)
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
                        while chunk := handle.read(HASH_CHUNK_SIZE):
                            digest.update(chunk)
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
                    "tooltip": "'match' scales each reference image to the generation's pixel area; 'max' uses the largest memory-safe canvas up to a 2048px short edge for stronger identity fidelity.",
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
        "the model expects, with audio labels offset past video soundtracks automatically. High-resolution media "
        "is decoded lazily into an aspect-preserving, memory-bounded canvas."
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

        if int(references.get("version", 1)) >= 2:
            fps = max(1.0, float(references.get("video_fps", 24.0)))
            max_seconds = max(1.0, float(references.get("max_seconds", 15.0)))
            image_specs = _prepare_image_references(images, width, height, ref_image_size)
            video_specs = _prepare_video_references(videos, fps, max_seconds, length)
            ref_images = _LazyImageReferences(image_specs)
            ref_videos = _LazyVideoReferences(video_specs, fps)
            ref_video_audios = _LazyVideoAudioReferences(video_specs)
            ref_audios = _LazyAudioReferences(audios, max_seconds)
            videos_with_audio = len(ref_video_audios)
        else:
            # Compatibility with in-memory packs produced before descriptor packs v2.
            ref_images = {
                f"ref_image_{index}": image["pixels"]
                for index, image in enumerate(images)
            }

            import torch

            ref_videos = {}
            ref_video_audios = {}
            for index, video in enumerate(videos):
                frames = video["frames"]
                if frames.dtype == torch.uint8:
                    frames = frames.to(dtype=torch.float32)
                    frames.mul_(1.0 / 255.0)
                if frames.shape[0] < 5:
                    raise ValueError(
                        f"Reference video {index + 1} ('{video.get('name', '?')}') has fewer than 5 usable "
                        "frames (~0.2s at 24 fps)."
                    )
                ref_videos[f"ref_video_{index}"] = frames
                if video.get("audio") is not None:
                    ref_video_audios[f"ref_video_audio_{index}"] = video["audio"]

            ref_audios = {
                f"ref_audio_{index}": audio["audio"]
                for index, audio in enumerate(audios)
            }
            videos_with_audio = len(ref_video_audios)

        translated = translate_reference_tokens(prompt, len(images), len(videos), len(audios), videos_with_audio)

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
