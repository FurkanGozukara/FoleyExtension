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
- ``SECoursesBatchVideoSaveMerge`` saves each sequential folder-batch item as
  soon as its own queued execution finishes, then concatenates one MP4 per
  prompt directory after the final item and previews the last complete merge.
- ``SECoursesBatchVideoMerge`` remains available for older workflows that use
  ComfyUI's list-style folder batching.
- ``SECoursesBatchAudioMerge`` does the same for the audio-only preset,
  concatenating lossless waveform data and saving one FLAC per directory after
  the individual audio clips have been saved.
- ``SECoursesBatchAudioSaveMerge`` is the audio preset's single result node: it
  saves every individual FLAC, optionally creates the directory merges, and
  returns only the final complete merge to ComfyUI's player when enabled.

Future reference-driven models only need another thin adapter node; the
gallery, its UI, and the ``@`` token grammar stay identical.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import threading
import time
import warnings
from pathlib import Path

try:
    from .media_extensions import audio_extensions, image_extensions
except ImportError:  # direct test-module import
    from media_extensions import audio_extensions, image_extensions


REF_PACK_TYPE = "SECOURSES_REF_PACK"
OPTIONAL_IMAGE_TYPE = "SECOURSES_OPTIONAL_IMAGE"

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

BATCH_PROMPT_EXTENSIONS = {".txt"}
BATCH_IMAGE_EXTENSIONS = image_extensions()
BATCH_VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v", ".mkv", ".avi"}
BATCH_AUDIO_EXTENSIONS = audio_extensions()
BATCH_MEDIA_EXTENSIONS = BATCH_IMAGE_EXTENSIONS | BATCH_VIDEO_EXTENSIONS | BATCH_AUDIO_EXTENSIONS
BATCH_MAX_PROMPTS = 1000
BATCH_MAX_PROMPT_BYTES = 1024 * 1024
BATCH_SESSION_TTL_SECONDS = 6 * 60 * 60

_BATCH_SESSION_LOCK = threading.Lock()
_BATCH_OUTPUT_SESSIONS = {"video": {}, "audio": {}}
_BATCH_CONTINUATION_SESSIONS = {}

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


def translate_reference_tokens(prompt, image_count, video_count, audio_count, audio_label_offset,
                               audio_number_map=None, image_number_map=None, video_number_map=None):
    """Rewrites '@image1' style tokens into the '<Picture 1>' labels MiniMax H3 expects.

    Audio labels index video soundtracks first, so standalone audio tokens are
    shifted by ``audio_label_offset`` (the number of videos that carry sound).
    Legacy '<Picture 1>' labels typed directly in the prompt pass through
    unchanged. Tokens that point at a missing reference (eg '@image3' with two
    images attached) are silently omitted, together with one adjacent space, so
    a stale token left in the prompt never blocks execution.

    The ``*_number_map`` arguments renumber tokens when the gallery holds more
    files of that modality than the model cap and only the prompt-mentioned
    subset is attached: '@audio5' with an audio map of ``{5: 1}`` becomes the
    first standalone audio label, and tokens missing from their map are omitted
    like stale tokens.
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
        label, count, offset, number_map = {
            "image": ("Picture", image_count, 0, image_number_map),
            "video": ("Video", video_count, 0, video_number_map),
            "audio": ("Audio", audio_count, audio_label_offset, audio_number_map),
        }[kind]
        if number_map is not None:
            number = number_map.get(number, 0)
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


def translate_audio_only_reference_tokens(prompt, image_count, video_count, audio_count,
                                          audio_number_map=None, image_number_map=None,
                                          video_number_map=None):
    """Translate video aliases to their extracted soundtracks for audio-only H3 runs.

    The ``*_number_map`` arguments behave exactly as in
    ``translate_reference_tokens``: they renumber tokens onto the attached
    prompt-mentioned subset when the gallery holds more files of a modality
    than the model cap.
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
        label, count, offset, number_map = {
            "image": ("Picture", image_count, 0, image_number_map),
            "video": ("Audio", video_count, 0, video_number_map),
            "audio": ("Audio", audio_count, video_count, audio_number_map),
        }[kind]
        if number_map is not None:
            number = number_map.get(number, 0)
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


def select_prompt_media_references(prompt, entries, max_count, kind):
    """Choose which attachments of one modality accompany one prompt.

    ``kind`` is the canonical token type: ``"image"``, ``"video"``, or
    ``"audio"``. Galleries within the MiniMax H3 cap (9 images, 3 videos,
    3 standalone audios) pass straight through, preserving the historical
    attach-everything behavior. A larger gallery acts as a roster: only the
    files the prompt actually mentions are attached, in first-mention order,
    capped at ``max_count``, so folder-batch prompts can address eg
    ``@audio5`` or ``@image12`` from a large roster. Returns the attached
    subset plus a renumbering map ({original_number: attached_position}) for
    the prompt translators; unmentioned attachments are discarded for this run
    only.
    """
    if len(entries) <= max_count:
        return entries, None

    mentioned = []
    for match in TOKEN_MATCHER.finditer(prompt or ""):
        if _CANONICAL_TYPE[match.group("type").lower()] != kind:
            continue
        number = int(match.group("num"))
        if 1 <= number <= len(entries) and number not in mentioned:
            mentioned.append(number)
    selected = mentioned[:max_count]

    def _describe(numbers):
        return ", ".join(
            f"@{kind}{number} ({entries[number - 1].get('name') or '?'})" for number in numbers
        )

    if len(mentioned) > max_count:
        print(
            f"[SECoursesMiniMaxH3References] the prompt mentions {len(mentioned)} {kind} references but "
            f"MiniMax H3 accepts {max_count}; keeping the first mentioned: {_describe(selected)}; "
            f"dropping: {_describe(mentioned[max_count:])}",
            flush=True,
        )
    elif selected:
        print(
            f"[SECoursesMiniMaxH3References] gallery holds {len(entries)} {kind} files; attaching the "
            f"{len(selected)} mentioned in the prompt: {_describe(selected)}",
            flush=True,
        )
    else:
        print(
            f"[SECoursesMiniMaxH3References] gallery holds {len(entries)} {kind} files but the prompt "
            f"mentions none of them; no {kind} reference is attached for this run",
            flush=True,
        )
    return (
        [entries[number - 1] for number in selected],
        {number: index + 1 for index, number in enumerate(selected)},
    )


def select_prompt_audio_references(prompt, audios, max_audios):
    """Backward-compatible wrapper for the audio modality."""
    return select_prompt_media_references(prompt, audios, max_audios, "audio")


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
            cleaned_entry = {"file": str(entry["file"]), "name": name}
            trim = _entry_trim(entry)
            if trim is not None:
                cleaned_entry["trim_start"] = trim[0]
                if trim[1] is not None:
                    cleaned_entry["trim_end"] = trim[1]
            cleaned.append(cleaned_entry)
        manifest[key] = cleaned
    return manifest


def _entry_trim(entry):
    """Validated (start_seconds, end_seconds|None) for a manifest entry, or None when untrimmed.

    Trims come from the gallery's optional 'Load + trim' loader. Degenerate or
    malformed ranges fall back to the untrimmed behavior instead of erroring so
    a hand-edited manifest never blocks execution.
    """
    try:
        start = float(entry.get("trim_start") or 0.0)
        end = entry.get("trim_end")
        end = None if end is None else float(end)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(start) or (end is not None and not math.isfinite(end)):
        return None
    start = max(0.0, start)
    if end is not None and end - start < 0.05:
        return None
    if start <= 0.0 and end is None:
        return None
    return start, end


def _resolve_reference_path(file):
    import folder_paths

    # get_annotated_filepath rejects path traversal outside the input directory.
    return folder_paths.get_annotated_filepath(file)


def _normalize_batch_folder(batch_folder):
    raw = str(batch_folder or "").strip()
    # unwrap Explorer "Copy as path" / shell quoting, including nested whitespace
    while len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1].strip()
    if not raw:
        return None
    expanded = os.path.expandvars(os.path.expanduser(raw))
    folder = Path(expanded)
    if os.name != "nt" and "\\" in expanded and not folder.is_dir():
        # windows-style separators or shell-escaped spaces pasted on posix
        candidate = Path(expanded.replace("\\ ", " ").replace("\\", "/"))
        if candidate.is_dir():
            folder = candidate
    folder = folder.resolve()
    if not folder.is_dir():
        raise ValueError(f"Folder batch path is not an existing directory: {folder}")
    return folder


def _windows_natural_sort_key(value):
    """Cross-platform approximation of Explorer's numeric filename ordering."""
    text = str(value).replace("\\", "/")
    parts = []
    for part in re.split(r"(\d+)", text):
        if part.isdecimal():
            # Explorer places a zero-padded spelling first when numeric values
            # match ("02" before "2"). Negative length mirrors that tie-break.
            parts.append((1, int(part), -len(part)))
        else:
            parts.append((0, part.casefold()))
    return tuple(parts), text.casefold(), text


def _batch_relevant_files(root):
    relevant_extensions = BATCH_PROMPT_EXTENSIONS | BATCH_MEDIA_EXTENSIONS
    try:
        files = [
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in relevant_extensions
        ]
    except OSError as error:
        raise ValueError(f"Could not scan folder batch path '{root}': {error}") from error
    return sorted(
        files,
        key=lambda path: _windows_natural_sort_key(path.relative_to(root).as_posix()),
    )


def _batch_prompt_files(root):
    prompt_files = [
        path for path in _batch_relevant_files(root)
        if path.suffix.lower() in BATCH_PROMPT_EXTENSIONS
    ]
    if not prompt_files:
        raise ValueError(f"Folder batch path contains no .txt prompt files: {root}")
    if len(prompt_files) > BATCH_MAX_PROMPTS:
        raise ValueError(
            f"Folder batch path contains {len(prompt_files)} prompt files; "
            f"the safety limit is {BATCH_MAX_PROMPTS}."
        )
    return prompt_files


def _inspect_folder_batch(batch_folder):
    root = _normalize_batch_folder(batch_folder)
    if root is None:
        raise ValueError("Folder batch path is empty.")
    prompt_files = _batch_prompt_files(root)
    folders = {
        path.parent.relative_to(root).as_posix()
        for path in prompt_files
    }
    return {
        "count": len(prompt_files),
        "folders": len(folders),
    }


def _register_folder_batch_inspect_route():
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    marker = "_secourses_folder_batch_inspect_registered"
    if getattr(server, marker, False):
        return

    @server.routes.post("/secourses/folder_batch/inspect")
    async def inspect_folder_batch(request):
        try:
            payload = await request.json()
            result = _inspect_folder_batch(payload.get("batch_folder", ""))
        except (ValueError, OSError, json.JSONDecodeError) as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response(result)

    setattr(server, marker, True)


_MEDIA_INFO_CACHE = {}
_MEDIA_INFO_CACHE_LIMIT = 4096
_MEDIA_INFO_LOCK = threading.Lock()


def _media_stream_duration(container, stream):
    """Seconds of one stream (container duration as the fallback)."""
    import av

    if stream is not None and stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / av.time_base)
    return None


def _media_info(path):
    """Dimensions / duration of an input-directory file, with the decoders the adapters use.

    Returns {"kind": "image"|"video"|"audio", "width", "height", "duration", "has_audio", "fps"};
    unknown values are None. Used by the gallery's live token estimate, so image sizes follow
    the same EXIF-oriented reading as _prepare_image_references.
    """
    lower = path.lower()
    suffix = os.path.splitext(lower)[1]
    if suffix in BATCH_IMAGE_EXTENSIONS:
        width, height, _orientation = _oriented_image_dimensions(path)
        return {"kind": "image", "width": width, "height": height, "duration": None, "has_audio": False, "fps": None}

    import av

    with av.open(str(path), mode="r") as container:
        video_stream = container.streams.video[0] if container.streams.video else None
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        # Still images that PIL cannot open (eg HEIC via ffmpeg) also arrive here as one-frame videos.
        if video_stream is not None and (suffix in BATCH_VIDEO_EXTENSIONS or audio_stream is None):
            duration = _media_stream_duration(container, video_stream)
            fps = None
            try:
                rate = video_stream.average_rate or video_stream.guessed_rate
                fps = float(rate) if rate else None
            except (TypeError, ValueError, ZeroDivisionError):
                fps = None
            frames = int(video_stream.frames or 0)
            if (duration is None or duration <= 0) and frames and fps:
                duration = frames / fps
            if suffix not in BATCH_VIDEO_EXTENSIONS and (frames == 1 or (duration is not None and duration <= 0)):
                return {"kind": "image", "width": int(video_stream.codec_context.width or 0),
                        "height": int(video_stream.codec_context.height or 0), "duration": None, "has_audio": False, "fps": None}
            return {"kind": "video", "width": int(video_stream.codec_context.width or 0),
                    "height": int(video_stream.codec_context.height or 0), "duration": duration,
                    "has_audio": audio_stream is not None, "fps": fps}
        if audio_stream is None:
            raise ValueError(f"No image, video, or audio stream found in '{path}'.")
        duration = _media_stream_duration(container, audio_stream)
        if duration is None:
            duration = _audio_duration_seconds(path)
        return {"kind": "audio", "width": None, "height": None, "duration": duration, "has_audio": True, "fps": None}


def _cached_media_info(file):
    path = _resolve_reference_path(file)
    if not path or not os.path.isfile(path):
        raise ValueError(f"Reference file was not found: {file}")
    stat = os.stat(path)
    key = (os.path.normcase(os.path.abspath(path)), int(stat.st_mtime_ns), int(stat.st_size))
    with _MEDIA_INFO_LOCK:
        cached = _MEDIA_INFO_CACHE.get(key)
    if cached is not None:
        return cached
    info = _media_info(path)
    with _MEDIA_INFO_LOCK:
        if len(_MEDIA_INFO_CACHE) >= _MEDIA_INFO_CACHE_LIMIT:
            _MEDIA_INFO_CACHE.clear()
        _MEDIA_INFO_CACHE[key] = info
    return info


def _register_media_info_route():
    """POST /secourses/media_info {"files": [...]} -> {"<file>": {kind, width, height, duration, has_audio, fps} | {"error": ...}}."""
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return

    server = getattr(PromptServer, "instance", None)
    if server is None:
        return
    marker = "_secourses_media_info_registered"
    if getattr(server, marker, False):
        return

    @server.routes.post("/secourses/media_info")
    async def media_info(request):
        try:
            payload = await request.json()
        except (ValueError, json.JSONDecodeError):
            return web.json_response({"error": "invalid JSON body"}, status=400)
        files = payload.get("files") if isinstance(payload, dict) else None
        if not isinstance(files, list) or len(files) > 256:
            return web.json_response({"error": "expected {'files': [up to 256 names]}"}, status=400)
        result = {}
        for file in files:
            if not isinstance(file, str) or not file.strip():
                continue
            try:
                result[file] = _cached_media_info(file)
            except Exception as error:  # a broken upload must not hide the other files
                result[file] = {"error": str(error)}
        return web.json_response(result)

    setattr(server, marker, True)


def _batch_media_entries(root, folder):
    by_kind = {"images": [], "videos": [], "audios": []}
    kinds = (
        ("images", BATCH_IMAGE_EXTENSIONS),
        ("videos", BATCH_VIDEO_EXTENSIONS),
        ("audios", BATCH_AUDIO_EXTENSIONS),
    )
    try:
        direct_files = sorted(
            (path for path in folder.iterdir() if path.is_file()),
            key=lambda path: _windows_natural_sort_key(path.name),
        )
    except OSError as error:
        raise ValueError(f"Could not inspect folder batch directory '{folder}': {error}") from error

    root_string = str(root)
    for path in direct_files:
        extension = path.suffix.lower()
        for key, supported in kinds:
            if extension in supported:
                by_kind[key].append({
                    "path": str(path.resolve()),
                    "name": path.name,
                    "source": "batch_folder",
                    "batch_root": root_string,
                })
                break
    return by_kind


def _read_batch_prompt(path):
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"Could not inspect folder prompt '{path}': {error}") from error
    if size > BATCH_MAX_PROMPT_BYTES:
        raise ValueError(
            f"Folder prompt '{path}' is {size / 1024:.1f} KiB; "
            f"the safety limit is {BATCH_MAX_PROMPT_BYTES / 1024:.0f} KiB."
        )
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"Folder prompt '{path}' must be UTF-8 text: {error}") from error
    except OSError as error:
        raise ValueError(f"Could not read folder prompt '{path}': {error}") from error


def _batch_prompt_duration_seconds(path):
    """Return a positive integer suffix from ``name_<seconds>.txt``."""
    stem = Path(path).stem
    if "_" not in stem:
        return None
    suffix = stem.rsplit("_", 1)[1]
    if not re.fullmatch(r"[0-9]+", suffix):
        return None
    duration = int(suffix)
    return duration if duration > 0 else None


def _collect_folder_batch(batch_folder, fallback_manifest, video_fps, max_seconds, match_init_media=False):
    root = _normalize_batch_folder(batch_folder)
    if root is None:
        return None

    prompt_files = _batch_prompt_files(root)
    folder_counts = {}
    for prompt_path in prompt_files:
        folder_counts[prompt_path.parent] = folder_counts.get(prompt_path.parent, 0) + 1
    prompt_stems = {
        folder: {
            prompt_path.stem.casefold()
            for prompt_path in prompt_files
            if prompt_path.parent == folder
        }
        for folder in folder_counts
    }

    folder_media = {}
    folder_indexes = {}
    packs = []
    prompts = []
    for index, prompt_path in enumerate(prompt_files, start=1):
        folder = prompt_path.parent
        folder_indexes[folder] = folder_indexes.get(folder, 0) + 1
        if folder not in folder_media:
            folder_media[folder] = _batch_media_entries(root, folder)
        media = folder_media[folder]
        media_count = sum(len(media[key]) for key in ("images", "videos", "audios"))
        chosen = {key: list(values) for key, values in media.items()} if media_count else fallback_manifest
        init_image = None
        init_audio = None
        if match_init_media and media_count:
            stem = prompt_path.stem.casefold()
            init_images = [entry for entry in media["images"] if Path(entry["name"]).stem.casefold() == stem]
            init_audios = [entry for entry in media["audios"] if Path(entry["name"]).stem.casefold() == stem]
            if len(init_images) > 1 or len(init_audios) > 1:
                duplicates = init_images if len(init_images) > 1 else init_audios
                raise ValueError(
                    f"Folder prompt '{prompt_path.name}' has multiple same-basename init files: "
                    + ", ".join(entry["name"] for entry in duplicates)
                )
            init_image = init_images[0] if init_images else None
            init_audio = init_audios[0] if init_audios else None
            reserved = prompt_stems[folder]
            chosen["images"] = [entry for entry in media["images"] if Path(entry["name"]).stem.casefold() not in reserved]
            chosen["audios"] = [entry for entry in media["audios"] if Path(entry["name"]).stem.casefold() not in reserved]
        prompt = _read_batch_prompt(prompt_path)
        relative_folder = folder.relative_to(root).as_posix()
        if relative_folder == ".":
            relative_folder = "root"
        filename_duration = _batch_prompt_duration_seconds(prompt_path)
        packs.append({
            "version": 4,
            "prompt": prompt,
            "video_fps": float(video_fps),
            "max_seconds": float(max_seconds),
            "images": [dict(entry) for entry in chosen["images"]],
            "videos": [dict(entry) for entry in chosen["videos"]],
            "audios": [dict(entry) for entry in chosen["audios"]],
            "init_image": dict(init_image) if init_image else None,
            "init_audio": dict(init_audio) if init_audio else None,
            "batch": {
                "root": str(root),
                "folder": relative_folder,
                "prompt_file": prompt_path.name,
                "index": index,
                "count": len(prompt_files),
                "folder_index": folder_indexes[folder],
                "folder_count": folder_counts[folder],
                "uses_folder_media": bool(media_count),
                "duration_seconds": filename_duration,
            },
        })
        prompts.append(prompt)
    return packs, prompts


def _batch_output_merge_groups(values, reference_packs, value_key):
    """Group generated values by their folder-batch directory in prompt order."""
    if len(values) != len(reference_packs):
        raise ValueError(
            f"Folder batch {value_key[:-1]} merge received a different number of {value_key} "
            f"({len(values)}) and reference packs ({len(reference_packs)})."
        )

    grouped = {}
    for value, pack in zip(values, reference_packs):
        batch = pack.get("batch") if isinstance(pack, dict) else None
        if not isinstance(batch, dict):
            continue
        root = str(batch.get("root") or "").strip()
        folder = str(batch.get("folder") or "root").strip() or "root"
        if not root:
            raise ValueError(
                f"Folder batch {value_key[:-1]} merge is missing the validated batch root."
            )
        key = (root, folder)
        group = grouped.setdefault(key, {"root": root, "folder": folder, "items": []})
        try:
            index = int(batch.get("index", len(group["items"]) + 1))
        except (TypeError, ValueError):
            index = len(group["items"]) + 1
        group["items"].append((index, value))

    groups = []
    for group in grouped.values():
        group["items"].sort(key=lambda item: item[0])
        group[value_key] = [value for _, value in group.pop("items")]
        groups.append(group)
    return groups


def _batch_video_merge_groups(videos, reference_packs):
    return _batch_output_merge_groups(videos, reference_packs, "videos")


def _batch_audio_merge_groups(audios, reference_packs):
    return _batch_output_merge_groups(audios, reference_packs, "audios")


def _sequential_batch_item(pack):
    batch = pack.get("batch") if isinstance(pack, dict) else None
    if not isinstance(batch, dict) or not batch.get("sequential"):
        return None
    run_id = str(batch.get("run_id") or "")
    if not re.fullmatch(r"[0-9A-Za-z_-]{8,128}", run_id):
        raise ValueError("Folder batch sequential run ID is invalid.")
    try:
        index = int(batch["index"])
        count = int(batch["count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Folder batch sequential metadata is incomplete.") from error
    if count < 1 or index < 1 or index > count:
        raise ValueError(
            f"Folder batch sequential item {index} is outside the expected 1-{count} range."
        )
    return run_id, index, count


def _prune_batch_output_sessions(now):
    for sessions in _BATCH_OUTPUT_SESSIONS.values():
        expired = [
            run_id for run_id, session in sessions.items()
            if now - session["updated"] > BATCH_SESSION_TTL_SECONDS
        ]
        for run_id in expired:
            sessions.pop(run_id, None)

    expired = [
        run_id for run_id, session in _BATCH_CONTINUATION_SESSIONS.items()
        if now - session["updated"] > BATCH_SESSION_TTL_SECONDS
    ]
    for run_id in expired:
        _BATCH_CONTINUATION_SESSIONS.pop(run_id, None)


def _previous_batch_video(pack, enabled):
    if not enabled:
        return None
    item = _sequential_batch_item(pack)
    if item is None:
        return None

    run_id, index, count = item
    now = time.monotonic()
    with _BATCH_SESSION_LOCK:
        _prune_batch_output_sessions(now)
        if index == 1:
            _BATCH_CONTINUATION_SESSIONS.pop(run_id, None)
            return None
        session = _BATCH_CONTINUATION_SESSIONS.get(run_id)
        if session is None or session["count"] != count or session["index"] != index - 1:
            raise ValueError(
                "Folder batch last-frame continuation could not find the previous completed video. "
                "Run the folder batch again without changing or skipping queued items."
            )
        session["updated"] = now
        return session["path"]


def _record_batch_video_for_continuation(pack, path, enabled):
    if not enabled:
        return
    item = _sequential_batch_item(pack)
    if item is None:
        return

    run_id, index, count = item
    now = time.monotonic()
    with _BATCH_SESSION_LOCK:
        _prune_batch_output_sessions(now)
        if index >= count:
            _BATCH_CONTINUATION_SESSIONS.pop(run_id, None)
            return
        _BATCH_CONTINUATION_SESSIONS[run_id] = {
            "count": count,
            "index": index,
            "path": str(path),
            "updated": now,
        }


def _accumulate_sequential_output(kind, value, pack):
    """Collects saved file paths across separately queued prompt executions."""
    item = _sequential_batch_item(pack)
    if item is None:
        return None
    if kind not in _BATCH_OUTPUT_SESSIONS:
        raise ValueError(f"Unsupported folder batch output kind: {kind}")

    run_id, index, count = item
    now = time.monotonic()
    with _BATCH_SESSION_LOCK:
        _prune_batch_output_sessions(now)
        sessions = _BATCH_OUTPUT_SESSIONS[kind]
        session = sessions.setdefault(run_id, {
            "count": count,
            "items": {},
            "updated": now,
        })
        if session["count"] != count:
            sessions.pop(run_id, None)
            raise ValueError(
                "Folder batch changed after it was queued; run the folder batch again."
            )
        session["items"][index] = (value, pack)
        session["updated"] = now
        received = len(session["items"])
        if received < count:
            return {"complete": False, "received": received, "count": count}

        missing = [item_index for item_index in range(1, count + 1) if item_index not in session["items"]]
        if missing:
            return {
                "complete": False,
                "received": received,
                "count": count,
                "missing": missing,
            }
        ordered = [session["items"][item_index] for item_index in range(1, count + 1)]
        sessions.pop(run_id, None)

    values = [item_value for item_value, _ in ordered]
    packs = [item_pack for _, item_pack in ordered]
    value_key = "videos" if kind == "video" else "audios"
    groups = _batch_output_merge_groups(values, packs, value_key)
    return {"complete": True, "received": count, "count": count, "groups": groups}


def _safe_merge_output_component(value, fallback):
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "").strip())
    cleaned = cleaned.strip(" .")
    if cleaned in {"", ".", ".."}:
        return fallback
    return cleaned


def _merge_output_prefix(root, folder):
    root_name = _safe_merge_output_component(Path(root).name, "batch")
    raw_parts = str(folder or "root").replace("\\", "/").split("/")
    folder_parts = [
        _safe_merge_output_component(part, "folder")
        for part in raw_parts
        if part not in {"", ".", ".."}
    ]
    if not folder_parts:
        folder_parts = ["root"]
    folder_name = "_".join(folder_parts)
    return f"video/MiniMax_H3_Merged_{root_name}_{folder_name}"


def _merge_audio_output_prefix(root, folder):
    video_prefix = _merge_output_prefix(root, folder)
    suffix = video_prefix.removeprefix("video/MiniMax_H3_Merged_")
    return f"audio/MiniMax_H3_Audio_Merged_{suffix}"


def _write_concat_manifest(path, video_paths):
    lines = []
    for video_path in video_paths:
        # FFmpeg's concat demuxer accepts POSIX separators on Windows. A single
        # quote inside a path is escaped by ending and reopening the quoted run.
        escaped = Path(video_path).resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _run_ffmpeg_concat(ffmpeg, manifest_path, output_path):
    import subprocess

    base = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", str(manifest_path),
    ]
    result = subprocess.run(
        base + ["-c", "copy", "-movflags", "+faststart", str(output_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return

    # All temporary segments use H.264/AAC, so stream copy is normally exact.
    # Re-encoding is a compatibility fallback for unusual per-item stream data.
    fallback = subprocess.run(
        base + [
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if fallback.returncode != 0:
        detail = fallback.stderr.strip() or result.stderr.strip() or "unknown FFmpeg error"
        raise RuntimeError(f"Folder batch video merge failed: {detail}")


def _merge_batch_video_group(group):
    import tempfile

    import folder_paths
    from comfy_api.latest import Types
    from imageio_ffmpeg import get_ffmpeg_exe

    prefix = _merge_output_prefix(group["root"], group["folder"])
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory()
    )
    output_name = f"{filename}_{counter:05}_.mp4"
    output_path = Path(full_output_folder) / output_name

    temp_root = Path(folder_paths.get_temp_directory())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="secourses_h3_merge_", dir=temp_root) as temp_dir:
        temp_path = Path(temp_dir)
        segment_paths = []
        for index, video in enumerate(group["videos"], start=1):
            segment_path = temp_path / f"segment_{index:05}.mp4"
            video.save_to(
                str(segment_path),
                format=Types.VideoContainer.MP4,
                codec=Types.VideoCodec.H264,
            )
            segment_paths.append(segment_path)
        manifest_path = temp_path / "concat.txt"
        _write_concat_manifest(manifest_path, segment_paths)
        _run_ffmpeg_concat(get_ffmpeg_exe(), manifest_path, output_path)

    return {
        "filename": output_name,
        "subfolder": subfolder,
        "type": "output",
        "format": "video/mp4",
        "fullpath": str(output_path),
    }


def _save_video_output(video, filename_prefix, prompt=None, extra_pnginfo=None):
    import folder_paths
    from comfy_api.latest import Types

    width, height = video.get_dimensions()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        str(filename_prefix or "video/MiniMax_H3"),
        folder_paths.get_output_directory(),
        width,
        height,
    )
    output_name = f"{filename}_{counter:05}_.mp4"
    output_path = Path(full_output_folder) / output_name
    metadata = {}
    if isinstance(extra_pnginfo, dict):
        metadata.update(extra_pnginfo)
    if isinstance(prompt, dict):
        metadata["prompt"] = prompt
    video.save_to(
        str(output_path),
        format=Types.VideoContainer.MP4,
        codec=Types.VideoCodec.H264,
        metadata=metadata or None,
    )
    return {
        "filename": output_name,
        "subfolder": subfolder,
        "type": "output",
        "format": "video/mp4",
        "fullpath": str(output_path),
    }


def _merge_saved_video_group(group):
    import tempfile

    import folder_paths
    from imageio_ffmpeg import get_ffmpeg_exe

    paths = [Path(path) for path in group["videos"]]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Folder batch could not merge missing saved video(s): {', '.join(missing)}"
        )

    prefix = _merge_output_prefix(group["root"], group["folder"])
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, folder_paths.get_output_directory()
    )
    output_name = f"{filename}_{counter:05}_.mp4"
    output_path = Path(full_output_folder) / output_name
    temp_root = Path(folder_paths.get_temp_directory())
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="secourses_h3_merge_", dir=temp_root) as temp_dir:
        manifest_path = Path(temp_dir) / "concat.txt"
        _write_concat_manifest(manifest_path, paths)
        _run_ffmpeg_concat(get_ffmpeg_exe(), manifest_path, output_path)

    return {
        "filename": output_name,
        "subfolder": subfolder,
        "type": "output",
        "format": "video/mp4",
        "fullpath": str(output_path),
    }


def _video_from_saved_output(saved):
    from comfy_api.latest import InputImpl

    return InputImpl.VideoFromFile(saved["fullpath"])


def _decode_last_video_frame(path):
    import av
    import torch

    def decode(container, stream):
        last = None
        for frame in container.decode(streams=stream.index):
            last = frame
        return last

    path = str(path)
    with av.open(path, mode="r") as container:
        if not container.streams.video:
            raise ValueError(f"Previous folder-batch output has no video stream: {path}")
        stream = container.streams.video[0]
        if stream.duration is not None and stream.time_base is not None:
            two_seconds = max(1, round(2.0 / float(stream.time_base)))
            target = int(stream.start_time or 0) + max(0, int(stream.duration) - two_seconds)
            container.seek(target, stream=stream, any_frame=False, backward=True)
        frame = decode(container, stream)

    if frame is None:
        with av.open(path, mode="r") as container:
            if not container.streams.video:
                raise ValueError(f"Previous folder-batch output has no video stream: {path}")
            frame = decode(container, container.streams.video[0])
    if frame is None:
        raise ValueError(f"Previous folder-batch output has no decodable video frame: {path}")

    pixels = frame.to_ndarray(format="rgb24").copy()
    image = torch.from_numpy(pixels).to(dtype=torch.float32)
    image.mul_(1.0 / 255.0)
    return image.unsqueeze(0)


def _concatenate_batch_audio(audios):
    import torch

    if not audios:
        raise ValueError("Folder batch audio merge received no audio clips.")

    waveforms = []
    sample_rate = None
    leading_shape = None
    dtype = None
    device = None
    for index, audio in enumerate(audios, start=1):
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError(f"Folder batch audio merge received invalid AUDIO item {index}.")
        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise ValueError(
                f"Folder batch audio merge expected item {index} waveform as [batch, channels, samples]."
            )
        if waveform.shape[0] != 1 or waveform.shape[-1] < 1:
            raise ValueError(
                "Folder batch audio merge requires one non-empty waveform per prompt."
            )
        current_rate = int(audio["sample_rate"])
        current_shape = tuple(waveform.shape[:-1])
        if sample_rate is None:
            sample_rate = current_rate
            leading_shape = current_shape
            dtype = waveform.dtype
            device = waveform.device
        elif (
            current_rate != sample_rate
            or current_shape != leading_shape
            or waveform.dtype != dtype
            or waveform.device != device
        ):
            raise ValueError(
                "Folder batch audio clips must have matching sample rates, channels, dtype, and device."
            )
        waveforms.append(waveform)

    merged = dict(audios[0])
    merged["waveform"] = torch.cat(waveforms, dim=-1)
    merged["sample_rate"] = sample_rate
    return merged


def _save_audio_output(audio, filename_prefix):
    import folder_paths
    from comfy_api.latest import io as comfy_io
    from comfy_api.latest import ui

    saved = ui.AudioSaveHelper.save_audio(
        audio,
        filename_prefix=filename_prefix,
        folder_type=comfy_io.FolderType.output,
        cls=None,
        format="flac",
    )
    if not saved:
        raise RuntimeError("MiniMax H3 audio output did not save a FLAC file.")
    result = dict(saved[-1])
    result["format"] = "audio/flac"
    result["fullpath"] = str(
        Path(folder_paths.get_output_directory()) / result["subfolder"] / result["filename"]
    )
    return result


def _merge_batch_audio_group(group):
    merged = _concatenate_batch_audio(group["audios"])
    prefix = _merge_audio_output_prefix(group["root"], group["folder"])
    return _save_audio_output(merged, prefix)


def _load_saved_audio_output(path):
    import av
    import torch

    chunks = []
    sample_rate = None
    channels = None
    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"Saved folder batch audio has no audio stream: {path}")
        stream = container.streams.audio[0]
        sample_rate = int(stream.codec_context.sample_rate or 0)
        channels = int(stream.channels or 0)
        if sample_rate < 1 or channels < 1:
            raise ValueError(f"Saved folder batch audio has invalid stream metadata: {path}")
        for frame in container.decode(streams=stream.index):
            chunk = torch.from_numpy(frame.to_ndarray())
            if chunk.shape[0] != channels:
                chunk = chunk.view(-1, channels).t()
            chunks.append(chunk)
    if not chunks:
        raise ValueError(f"Saved folder batch audio is empty: {path}")
    waveform = _f32_pcm(torch.cat(chunks, dim=1))
    return {"waveform": waveform[None,], "sample_rate": sample_rate}


def _merge_saved_audio_group(group):
    audios = [_load_saved_audio_output(path) for path in group["audios"]]
    return _merge_batch_audio_group({**group, "audios": audios})


def _resolve_reference_entry(entry):
    if entry.get("source") != "batch_folder":
        return _resolve_reference_path(entry["file"])

    root_value = entry.get("batch_root")
    path_value = entry.get("path")
    if not root_value or not path_value:
        raise ValueError("Folder batch reference is missing its validated root or file path.")
    root = Path(root_value).resolve()
    path = Path(path_value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Folder batch reference escapes its selected root: {path}") from error
    if not path.is_file():
        raise ValueError(f"Folder batch reference file was not found: {path}")
    return str(path)


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
    except OSError:
        import av

        try:
            with av.open(str(path), mode="r") as container:
                if not container.streams.video:
                    raise ValueError(f"No image stream found in reference image '{path}'.")
                stream = container.streams.video[0]
                width = int(stream.codec_context.width or 0)
                height = int(stream.codec_context.height or 0)
                orientation = 1
        except (av.error.FFmpegError, OSError) as error:
            raise ValueError(f"Could not inspect reference image '{path}': {error}") from error
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
        try:
            img = Image.open(path)
        except OSError:
            import av

            try:
                with av.open(str(path), mode="r") as container:
                    frame = next(container.decode(video=0), None)
                    if frame is None:
                        raise ValueError(f"No decodable image frame found in reference image '{path}'.")
                    img = frame.to_image()
            except (av.error.FFmpegError, OSError) as error:
                raise ValueError(f"Could not decode reference image '{path}': {error}") from error
        with img:
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


def _stream_start_seconds(container, stream):
    """The stream's own start time in seconds; HTML players show this as 0:00."""
    import av

    if stream.start_time is not None and stream.time_base:
        return float(stream.start_time * stream.time_base)
    if container.start_time is not None:
        return max(0.0, container.start_time / av.time_base)
    return 0.0


def _seek_to_trim_start(container, stream, seconds):
    """Best-effort keyframe seek to at/before ``seconds``; callers still drop up to the exact target."""
    import av

    try:
        if stream.time_base:
            container.seek(max(0, int(seconds / stream.time_base)), stream=stream)
        else:
            container.seek(max(0, int(seconds * av.time_base)))
    except (av.error.FFmpegError, OSError, OverflowError, ValueError):
        pass  # decoding from the file start reaches the target too, just slower


def _decode_video_frames(path, fps_out, max_seconds=None, *, max_frames=None, area_cap=VIDEO_DECODE_AREA_CAP,
                         trim_start=0.0):
    """Streams a video into a uint8 [T, H, W, 3] tensor resampled to ``fps_out``.

    Frames are downscaled by FFmpeg before conversion to a NumPy/Torch RGB
    buffer, avoiding a full-resolution RGB float interpolation allocation.
    ``trim_start`` skips the first seconds of the source; the duration cap
    (``max_frames`` / ``max_seconds``) then applies from that point.
    """
    import av
    import torch

    fps_out = max(1.0, float(fps_out))
    if max_frames is None:
        max_frames = max(1, round(float(max_seconds) * fps_out))
    else:
        max_frames = max(1, int(max_frames))
    trim_start = max(0.0, float(trim_start or 0.0))
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
        start_target = None
        if trim_start > 0.0:
            start_target = _stream_start_seconds(container, stream) + trim_start
            _seek_to_trim_start(container, stream, start_target)

        for frame in container.decode(stream):
            has_pts = frame.pts is not None
            frame_time = float(frame.pts * frame.time_base) if has_pts else decoded_count / source_fps
            decoded_count += 1
            if start_target is not None and has_pts and first_time is None and frame_time + 1e-4 < start_target:
                continue  # decoded only to reach the trim start
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
        if trim_start > 0.0:
            raise ValueError(
                f"No decodable frames were found in reference video '{path}' "
                f"after its {trim_start:.2f}s trim start. Re-trim it in the gallery."
            )
        raise ValueError(f"No decodable frames were found in reference video '{path}'.")
    return torch.stack(frames, dim=0)


def _decode_video_audio(path, max_seconds, trim_start=0.0):
    """Returns the video's soundtrack as a ComfyUI AUDIO dict, or None when it has no usable audio.

    ``trim_start`` skips the first seconds of the soundtrack; ``max_seconds``
    then caps the duration kept from that point.
    """
    import av
    import torch

    trim_start = max(0.0, float(trim_start or 0.0))
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
            start_target = None
            if trim_start > 0.0:
                start_target = _stream_start_seconds(container, stream) + trim_start
                _seek_to_trim_start(container, stream, start_target)
            chunks = []
            collected = 0
            for frame in container.decode(streams=stream.index):
                buffer = torch.from_numpy(frame.to_ndarray())
                if buffer.shape[0] != n_channels:
                    buffer = buffer.view(-1, n_channels).t()
                if start_target is not None and frame.pts is not None:
                    frame_time = float(frame.pts * frame.time_base)
                    if frame_time + buffer.shape[1] / sample_rate <= start_target:
                        continue  # decoded only to reach the trim start
                    if frame_time < start_target:
                        buffer = buffer[:, round((start_target - frame_time) * sample_rate):]
                        if not buffer.shape[1]:
                            continue
                    start_target = None  # aligned with the trim start; keep every later frame
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


def _load_reference_audio(path, max_seconds, trim_start=0.0):
    audio = _decode_video_audio(path, max_seconds, trim_start=trim_start)
    if audio is None:
        if trim_start > 0.0:
            raise ValueError(
                f"No usable audio was found in reference audio '{path}' "
                f"after its {trim_start:.2f}s trim start. Re-trim it in the gallery."
            )
        raise ValueError(f"No usable audio stream found in reference audio '{path}'.")
    return audio


def _audio_duration_seconds(path):
    """Read an audio stream duration, decoding only when the container omits it."""
    import av

    with av.open(str(path), mode="r") as container:
        if not container.streams.audio:
            raise ValueError(f"No usable audio stream found in folder-batch init audio '{path}'.")
        stream = container.streams.audio[0]
        if stream.duration is not None and stream.time_base is not None:
            return float(stream.duration * stream.time_base)
        if container.duration is not None:
            return float(container.duration / av.time_base)
        samples = sum(frame.samples for frame in container.decode(streams=stream.index))
        sample_rate = int(stream.codec_context.sample_rate or 0)
    return samples / sample_rate if sample_rate > 0 else 0.0


def _entry_trim_window(entry, max_seconds):
    """Per-entry (trim_start, effective_seconds) honoring an optional trim window."""
    trim = _entry_trim(entry)
    if trim is None:
        return 0.0, float(max_seconds)
    start, end = trim
    if end is None:
        return start, float(max_seconds)
    return start, min(float(max_seconds), end - start)


def _prepare_image_references(entries, width, height, ref_image_size, byte_budget=None):
    target_area = IMAGE_MAX_AREA
    target_short_edge = IMAGE_MAX_SHORT_EDGE
    if ref_image_size == "match":
        target_area = min(IMAGE_MAX_AREA, max(CANVAS_MULTIPLE ** 2, int(width) * int(height)))
        target_short_edge = None

    specs = []
    requested_sizes = []
    for entry in entries:
        path = _resolve_reference_entry(entry)
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
        path = _resolve_reference_entry(entry)
        source_width, source_height, has_audio = _video_metadata(path)
        target_size = None
        if source_width and source_height:
            target_size = _fit_dimensions(source_width, source_height, area_cap)
        trim_start, trim_seconds = _entry_trim_window(entry, max_seconds)
        entry_frames = min(max_frames, max(1, round(trim_seconds * max(1.0, float(fps)))))
        if entry_frames >= 5:
            while entry_frames % 17 != 5:
                entry_frames -= 1
        specs.append({
            "path": path,
            "name": entry.get("name") or "?",
            "source_size": (source_width, source_height),
            "target_size": target_size,
            "area_cap": area_cap,
            "max_frames": entry_frames,
            "audio_seconds": min(audio_seconds, trim_seconds),
            "trim_start": trim_start,
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


class _ImageReferencesWithContinuation:
    def __init__(self, references, continuation_frame):
        self.references = references
        self.continuation_frame = continuation_frame

    def __len__(self):
        return len(self.references) + 1

    def values(self):
        yield from self.references.values()
        yield self.continuation_frame


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
                trim_start=spec.get("trim_start", 0.0),
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
            self._cache[index] = _decode_video_audio(
                spec["path"], spec["audio_seconds"], trim_start=spec.get("trim_start", 0.0)
            )
        return self._cache[index] or default


class _LazyAudioReferences:
    def __init__(self, entries, max_seconds):
        self.entries = entries
        self.max_seconds = max_seconds

    def __len__(self):
        return len(self.entries)

    def values(self):
        for entry in self.entries:
            path = _resolve_reference_entry(entry)
            trim_start, seconds = _entry_trim_window(entry, self.max_seconds)
            yield _load_reference_audio(path, seconds, trim_start=trim_start)


class _LazyAudioOnlyReferences:
    """Expose video soundtracks and audio files as one ordered audio reference set."""

    def __init__(self, video_specs, audio_entries, max_seconds):
        self.video_specs = video_specs
        self.audio_entries = audio_entries
        self.per_item_seconds = float(max_seconds)

    def __len__(self):
        return len(self.video_specs) + len(self.audio_entries)

    def values(self):
        for spec in self.video_specs:
            audio = _decode_video_audio(
                spec["path"],
                min(float(spec["audio_seconds"]), self.per_item_seconds),
                trim_start=spec.get("trim_start", 0.0),
            )
            if audio is None:
                raise ValueError(
                    f"Reference video '{spec['name']}' has no usable soundtrack. "
                    "Audio-only MiniMax H3 mode uses a video's audio stream, not its frames."
                )
            yield audio
        for entry in self.audio_entries:
            path = _resolve_reference_entry(entry)
            trim_start, seconds = _entry_trim_window(entry, self.per_item_seconds)
            yield _load_reference_audio(path, seconds, trim_start=trim_start)


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
                    "default": 15.0, "min": 1.0, "max": 3600.0, "step": 0.5,
                    "tooltip": "Maximum duration used from each reference. Clean 2-15 second clips are the quality-tested recommendation, but longer references are allowed; they use more memory/time and may be less reliable.",
                }),
                "batch_folder": ("STRING", {
                    "default": "",
                    "tooltip": "Optional local folder containing UTF-8 .txt prompts. With init matching enabled, prompt.txt + prompt.<image> is image-to-video, prompt.txt + prompt.<audio> is audio-to-video, and all three combine. Other media beside the prompts are references in natural order. Gallery attachments are the fallback when a directory has no media. Subfolders never share media.",
                }),
                "merge_batch_videos": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When Folder batch is active, save each queued prompt before starting the next, then concatenate each prompt directory after the final job. The complete last merge is returned.",
                }),
                "continue_batch_with_last_frame": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When Folder batch is active, use the last frame of each completed video as the next prompt's starting image. The first prompt uses no continuation frame.",
                }),
                "match_batch_init_media": ("BOOLEAN", {
                    "default": False,
                    "label_on": "SAME-NAME INIT MEDIA: ON",
                    "label_off": "SAME-NAME INIT MEDIA: OFF",
                    "tooltip": "When enabled, prompt.txt + prompt.<image> uses the image as the first frame, prompt.txt + prompt.<audio> uses the audio as the locked soundtrack, and all three combine. Other media remain references. Disabled by default for compatibility; MiniMax H3 video presets enable it.",
                }),
            },
            "optional": {
                "batch_run_id": ("STRING", {"default": ""}),
                "batch_item_index": ("INT", {"default": -1, "min": -1, "max": BATCH_MAX_PROMPTS - 1}),
                "batch_item_count": ("INT", {"default": 0, "min": 0, "max": BATCH_MAX_PROMPTS}),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = (REF_PACK_TYPE, "STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = (
        "references", "prompt", "folder_batch_active", "merge_batch_videos",
        "continue_batch_with_last_frame",
    )
    OUTPUT_IS_LIST = (True, True, True, True, True)
    OUTPUT_TOOLTIPS = (
        "Every gallery reference bundled in upload order, ready for a model adapter node such as 'MiniMax H3 References (Gallery)'.",
        "The prompt exactly as typed, or one output per naturally ordered .txt file when Folder batch is active.",
        "True for folder-batch items and false for the normal single prompt.",
        "True for every sequential folder job when the adjacent merge toggle is enabled.",
        "True for every sequential folder job when last-frame continuation is enabled.",
    )
    FUNCTION = "collect"
    DESCRIPTION = (
        "SwarmUI-style unified reference uploader: add images, videos, and audio next to the prompt and mention "
        "them as '@image1', '@video1', or '@audio1'. Video soundtracks are paired automatically. The optional "
        "'Load + trim' loader previews a video or audio file and selects a start/end window before adding it; "
        "only that window is decoded at generation time. Feed the references output into a model adapter node "
        "(eg 'MiniMax H3 References (Gallery)'). Media stays lazy until the adapter can apply its canvas, "
        "duration, and memory limits. An optional folder path queues one complete job per recursively discovered "
        ".txt prompt, saving each output before the next job starts. Compatible video presets can match a prompt's "
        "basename to an init image, init audio, or both; other media remains reference material. Media comes only "
        "from the prompt's own directory, with gallery attachments as fallback. Optional toggles merge after the final "
        "job or feed each completed video's final frame into the next prompt. A prompt filename ending in "
        "'_<integer>.txt' overrides that item's output duration in compatible presets."
    )

    def collect(
        self,
        prompt,
        references,
        video_fps,
        max_seconds,
        batch_folder="",
        merge_batch_videos=False,
        continue_batch_with_last_frame=False,
        batch_run_id="",
        batch_item_index=-1,
        batch_item_count=0,
        match_batch_init_media=False,
    ):
        manifest = _parse_manifest(references)
        max_seconds = max(1.0, float(max_seconds))
        folder_batch = _collect_folder_batch(
            batch_folder, manifest, video_fps, max_seconds, bool(match_batch_init_media)
        )
        if folder_batch is not None:
            packs, prompts = folder_batch
            run_id = str(batch_run_id or "").strip()
            if run_id:
                if not re.fullmatch(r"[0-9A-Za-z_-]{8,128}", run_id):
                    raise ValueError("Folder batch sequential run ID is invalid.")
                item_index = int(batch_item_index)
                expected_count = int(batch_item_count)
                if expected_count != len(packs):
                    raise ValueError(
                        "Folder batch changed after it was queued; run the folder batch again."
                    )
                if item_index < 0 or item_index >= len(packs):
                    raise ValueError(
                        f"Folder batch item index {item_index} is outside the expected "
                        f"0-{len(packs) - 1} range."
                    )
                pack = packs[item_index]
                pack["batch"]["run_id"] = run_id
                pack["batch"]["sequential"] = True
                packs = [pack]
                prompts = [prompts[item_index]]
                print(
                    f"[SECoursesReferenceGallery] prepared sequential folder prompt "
                    f"{item_index + 1}/{expected_count}: {pack['batch']['prompt_file']}",
                    flush=True,
                )
            elif int(batch_item_index) >= 0 or int(batch_item_count) > 0:
                raise ValueError("Folder batch sequential metadata is missing its run ID.")
            folders = len({pack["batch"]["folder"] for pack in packs})
            if not run_id:
                print(
                    f"[SECoursesReferenceGallery] prepared {len(packs)} folder prompt(s) "
                    f"across {folders} unique folder(s)",
                    flush=True,
                )
            merge_flags = [bool(merge_batch_videos)] * len(packs)
            continuation_flags = [bool(continue_batch_with_last_frame)] * len(packs)
            return (packs, prompts, [True] * len(packs), merge_flags, continuation_flags)

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
        return ([pack], [prompt], [False], [False], [False])

    @classmethod
    def IS_CHANGED(
        cls,
        prompt,
        references,
        video_fps,
        max_seconds,
        batch_folder="",
        merge_batch_videos=False,
        continue_batch_with_last_frame=False,
        batch_run_id="",
        batch_item_index=-1,
        batch_item_count=0,
        match_batch_init_media=False,
    ):
        digest = hashlib.sha256()
        digest.update(
            repr(
                (
                    prompt,
                    references,
                    float(video_fps),
                    float(max_seconds),
                    str(batch_folder),
                    bool(merge_batch_videos),
                    bool(continue_batch_with_last_frame),
                    bool(match_batch_init_media),
                    str(batch_run_id),
                    int(batch_item_index),
                    int(batch_item_count),
                )
            ).encode("utf-8")
        )
        try:
            root = _normalize_batch_folder(batch_folder)
        except ValueError:
            return float("nan")
        if root is not None:
            try:
                for path in _batch_relevant_files(root):
                    stat = path.stat()
                    digest.update(path.relative_to(root).as_posix().encode("utf-8"))
                    digest.update(repr((stat.st_size, stat.st_mtime_ns)).encode("ascii"))
            except (OSError, ValueError):
                return float("nan")
            return digest.hexdigest()

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
    def VALIDATE_INPUTS(
        cls,
        references,
        batch_folder="",
        merge_batch_videos=False,
        continue_batch_with_last_frame=False,
        batch_run_id="",
        batch_item_index=-1,
        batch_item_count=0,
        match_batch_init_media=False,
    ):
        try:
            manifest = _parse_manifest(references)
        except ValueError as error:
            return str(error)
        try:
            root = _normalize_batch_folder(batch_folder)
            if root is not None:
                prompt_files = _batch_prompt_files(root)
                run_id = str(batch_run_id or "").strip()
                if run_id:
                    if not re.fullmatch(r"[0-9A-Za-z_-]{8,128}", run_id):
                        return "Folder batch sequential run ID is invalid."
                    if int(batch_item_count) != len(prompt_files):
                        return "Folder batch changed after it was queued; run the folder batch again."
                    if int(batch_item_index) < 0 or int(batch_item_index) >= len(prompt_files):
                        return "Folder batch sequential item index is outside the prompt list."
                elif int(batch_item_index) >= 0 or int(batch_item_count) > 0:
                    return "Folder batch sequential metadata is missing its run ID."
        except ValueError as error:
            return str(error)
        folder_paths = None
        for kind in ("images", "videos", "audios"):
            for entry in manifest[kind]:
                if folder_paths is None:
                    import folder_paths
                if not folder_paths.exists_annotated_filepath(entry["file"]):
                    return f"Reference file not found: {entry['file']}. Re-add it in the gallery."
        return True


class SECoursesBatchDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": (REF_PACK_TYPE, {
                    "tooltip": "Prompt and filename metadata from SECourses Reference Gallery.",
                }),
                "default_duration_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.01, "max": 3600.0, "step": 0.01,
                    "tooltip": "Duration used unless the folder prompt filename ends in '_<integer>.txt'.",
                }),
            }
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("duration_seconds",)
    FUNCTION = "resolve"
    DESCRIPTION = (
        "Uses the positive integer suffix in a folder prompt filename such as 'scene_8.txt' as that prompt's "
        "duration. A matched folder-batch init audio uses its own duration; otherwise filenames without an "
        "underscore-integer suffix use the connected default duration."
    )

    def resolve(self, references, default_duration_seconds):
        default = float(default_duration_seconds)
        if not math.isfinite(default) or default <= 0:
            raise ValueError("The default MiniMax H3 duration must be a positive finite number.")
        batch = references.get("batch") if isinstance(references, dict) else None
        init_audio = references.get("init_audio") if isinstance(references, dict) else None
        if init_audio:
            duration = _audio_duration_seconds(_resolve_reference_entry(init_audio))
            if duration <= 0:
                raise ValueError(f"Folder-batch init audio '{init_audio['name']}' is empty.")
            return (duration,)
        override = batch.get("duration_seconds") if isinstance(batch, dict) else None
        return (float(override) if override is not None else default,)


class SECoursesBatchContinuationFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": (REF_PACK_TYPE, {
                    "tooltip": "Sequential folder-batch metadata from SECourses Reference Gallery.",
                }),
                "continue_batch_with_last_frame": ("BOOLEAN", {
                    "forceInput": True,
                    "tooltip": "Connect the gallery's last-frame continuation output.",
                }),
            }
        }

    CATEGORY = "SECourses/video"
    RETURN_TYPES = (OPTIONAL_IMAGE_TYPE,)
    RETURN_NAMES = ("first_frame",)
    FUNCTION = "load"
    DESCRIPTION = (
        "Returns no image for the first folder prompt, then decodes only the final frame of each immediately "
        "preceding saved video for use as the next MiniMax H3 starting image."
    )

    def load(self, references, continue_batch_with_last_frame):
        path = _previous_batch_video(references, bool(continue_batch_with_last_frame))
        if path is None:
            # A literal None output is treated as an unavailable dependency by
            # ComfyUI's graph executor. Keep the optional value concrete so the
            # first batch item can continue through the Auto adapter normally.
            return ({"image": None},)
        print(
            f"[SECoursesBatchContinuationFrame] using previous final frame from {path}",
            flush=True,
        )
        return ({"image": _decode_last_video_frame(path)},)


class SECoursesBatchVideoSaveMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": "The generated video. Folder batches are queued one prompt at a time so this file is written before the next prompt starts.",
                }),
                "references": (REF_PACK_TYPE, {
                    "tooltip": "Folder and prompt-order metadata from SECourses Reference Gallery.",
                }),
                "merge_batch_videos": ("BOOLEAN", {
                    "tooltip": "After the final queued prompt, merge the saved MP4 files once per prompt directory and return the last complete merge.",
                }),
                "filename_prefix": ("STRING", {"default": "video/MiniMax_H3"}),
            },
            "optional": {
                "continue_batch_with_last_frame": ("BOOLEAN", {
                    "forceInput": True,
                    "tooltip": "Connect the gallery's last-frame continuation output so each saved video becomes the next queued prompt's starting frame.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "SECourses/video"
    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "save_and_merge"
    DESCRIPTION = (
        "Saves each MiniMax H3 folder prompt as its own MP4 before the next queued prompt starts. After the "
        "last prompt, it concatenates the already-saved files once per prompt directory and returns only the "
        "last complete merged MP4. Normal non-folder generations are saved exactly once."
    )

    @staticmethod
    def _first(values, default=None):
        if isinstance(values, (list, tuple)):
            return values[0] if values else default
        return values if values is not None else default

    @staticmethod
    def _preview(saved):
        return {key: saved[key] for key in ("filename", "subfolder", "type")}

    def save_and_merge(
        self,
        video,
        references,
        merge_batch_videos,
        filename_prefix,
        continue_batch_with_last_frame=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        videos = list(video)
        packs = list(references)
        if len(videos) != len(packs):
            raise ValueError(
                "Folder batch video save received a different number of videos "
                f"({len(videos)}) and reference packs ({len(packs)})."
            )

        prefix = self._first(filename_prefix, "video/MiniMax_H3")
        prompt_value = self._first(prompt)
        extra_value = self._first(extra_pnginfo)
        if isinstance(continue_batch_with_last_frame, (list, tuple)):
            continuation_flags = list(continue_batch_with_last_frame)
        elif continue_batch_with_last_frame is None:
            continuation_flags = []
        else:
            continuation_flags = [continue_batch_with_last_frame]
        if len(continuation_flags) == 1 and len(packs) > 1:
            continuation_flags *= len(packs)
        if continuation_flags and len(continuation_flags) != len(packs):
            raise ValueError(
                "Folder batch video save received a different number of continuation flags "
                f"({len(continuation_flags)}) and reference packs ({len(packs)})."
            )
        if not continuation_flags:
            continuation_flags = [False] * len(packs)
        saved = []
        for clip, pack, continue_enabled in zip(videos, packs, continuation_flags):
            result = _save_video_output(clip, prefix, prompt_value, extra_value)
            saved.append(result)
            _record_batch_video_for_continuation(
                pack, result["fullpath"], bool(continue_enabled)
            )
            print(
                f"[SECoursesBatchVideoSaveMerge] saved individual video -> {result['fullpath']}",
                flush=True,
            )

        display_saved = saved
        display_video = _video_from_saved_output(saved[-1])
        merge_enabled = any(bool(value) for value in merge_batch_videos)
        if merge_enabled and any(_sequential_batch_item(pack) is not None for pack in packs):
            if len(saved) != 1:
                raise ValueError("Sequential folder batching expects exactly one saved video per queued job.")
            progress = _accumulate_sequential_output("video", saved[0]["fullpath"], packs[0])
            if progress["complete"]:
                merged_saved = []
                for group in progress["groups"]:
                    result = _merge_saved_video_group(group)
                    merged_saved.append(result)
                    print(
                        f"[SECoursesBatchVideoSaveMerge] merged {len(group['videos'])} saved video(s) "
                        f"for folder '{group['folder']}' -> {result['fullpath']}",
                        flush=True,
                    )
                display_saved = [merged_saved[-1]]
                display_video = _video_from_saved_output(merged_saved[-1])
            else:
                print(
                    f"[SECoursesBatchVideoSaveMerge] saved sequential item "
                    f"{progress['received']}/{progress['count']}; merge waits for the final item",
                    flush=True,
                )
        elif merge_enabled:
            groups = _batch_video_merge_groups(
                [result["fullpath"] for result in saved], packs
            )
            if groups:
                merged_saved = []
                for group in groups:
                    result = _merge_saved_video_group(group)
                    merged_saved.append(result)
                    print(
                        f"[SECoursesBatchVideoSaveMerge] merged {len(group['videos'])} saved video(s) "
                        f"for folder '{group['folder']}' -> {result['fullpath']}",
                        flush=True,
                    )
                display_saved = [merged_saved[-1]]
                display_video = _video_from_saved_output(merged_saved[-1])

        return {
            "ui": {
                "images": [self._preview(item) for item in display_saved],
                "animated": (True,),
            },
            "result": (display_video,),
        }


class SECoursesBatchVideoMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": "The generated folder-batch videos. Connect Save Video's VIDEO output so every individual clip is saved before the final merged preview.",
                }),
                "references": (REF_PACK_TYPE, {
                    "tooltip": "Folder and prompt-order metadata from SECourses Reference Gallery.",
                }),
                "merge_batch_videos": ("BOOLEAN", {
                    "tooltip": "Connect the gallery's Merge batch videos output. Disabled leaves every existing output unchanged and performs no additional save.",
                }),
            }
        }

    CATEGORY = "SECourses/video"
    RETURN_TYPES = ()
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "merge"
    DESCRIPTION = (
        "Optionally concatenates generated MiniMax H3 folder-batch videos without changing the normal per-prompt "
        "Save Video output. Videos are grouped by the directory containing their .txt prompts and saved under "
        "output/video alongside the individual clips. Every directory receives one merged MP4, while the node "
        "previews only the last merged MP4."
    )

    def merge(self, video, references, merge_batch_videos):
        if not any(bool(value) for value in merge_batch_videos):
            return {}

        groups = _batch_video_merge_groups(video, references)
        if not groups:
            print(
                "[SECoursesBatchVideoMerge] Merge videos is enabled, but Folder batch is not active; skipping.",
                flush=True,
            )
            return {}

        saved = []
        for group in groups:
            result = _merge_batch_video_group(group)
            saved.append(result)
            print(
                f"[SECoursesBatchVideoMerge] merged {len(group['videos'])} video(s) for "
                f"folder '{group['folder']}' -> {result['fullpath']}",
                flush=True,
            )

        preview = {
            key: saved[-1][key]
            for key in ("filename", "subfolder", "type")
        }
        return {"ui": {"images": [preview], "animated": (True,)}}


class SECoursesBatchAudioMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Connect Save Audio's AUDIO output so every individual FLAC is saved before the final merged preview.",
                }),
                "references": (REF_PACK_TYPE, {
                    "tooltip": "Folder and prompt-order metadata from SECourses Reference Gallery.",
                }),
                "merge_batch_audio": ("BOOLEAN", {
                    "tooltip": "Connect the gallery's merge output. Disabled leaves every existing audio output unchanged and performs no additional save.",
                }),
            }
        }

    CATEGORY = "SECourses/audio"
    RETURN_TYPES = ()
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "merge"
    DESCRIPTION = (
        "Optionally concatenates generated MiniMax H3 folder-batch audio after the normal per-prompt Save Audio "
        "node. Every prompt directory receives one lossless FLAC in output/audio beside the individual clips, "
        "and only the complete last merged FLAC is previewed."
    )

    def merge(self, audio, references, merge_batch_audio):
        if not any(bool(value) for value in merge_batch_audio):
            return {}

        groups = _batch_audio_merge_groups(audio, references)
        if not groups:
            print(
                "[SECoursesBatchAudioMerge] Merge audio is enabled, but Folder batch is not active; skipping.",
                flush=True,
            )
            return {}

        saved = []
        for group in groups:
            result = _merge_batch_audio_group(group)
            saved.append(result)
            print(
                f"[SECoursesBatchAudioMerge] merged {len(group['audios'])} audio clip(s) for "
                f"folder '{group['folder']}' -> {result['fullpath']}",
                flush=True,
            )

        preview = {
            key: saved[-1][key]
            for key in ("filename", "subfolder", "type")
        }
        return {"ui": {"audio": [preview]}}


class SECoursesBatchAudioSaveMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Generated audio clips. Every clip is saved individually before any optional folder merge.",
                }),
                "references": (REF_PACK_TYPE, {
                    "tooltip": "Folder and prompt-order metadata from SECourses Reference Gallery.",
                }),
                "merge_batch_audio": ("BOOLEAN", {
                    "tooltip": "Connect the gallery's merge output. Enabled returns only the complete last merged FLAC to ComfyUI while retaining every individual file.",
                }),
                "filename_prefix": ("STRING", {
                    "default": "audio/MiniMax_H3_Audio_Only",
                    "tooltip": "Output prefix for the individual lossless FLAC files.",
                }),
            }
        }

    CATEGORY = "SECourses/audio"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    INPUT_IS_LIST = True
    OUTPUT_NODE = True
    FUNCTION = "save_and_merge"
    DESCRIPTION = (
        "Saves each MiniMax H3 folder prompt as an individual lossless FLAC before the next queued prompt "
        "starts. After the final prompt, it saves one merged FLAC per prompt directory and returns only the "
        "complete last merge to ComfyUI's audio player."
    )

    def save_and_merge(self, audio, references, merge_batch_audio, filename_prefix):
        if not audio:
            raise ValueError("MiniMax H3 audio output received no audio clips to save.")

        prefix = str(filename_prefix[0] if filename_prefix else "").strip()
        if not prefix:
            prefix = "audio/MiniMax_H3_Audio_Only"

        individual_saved = []
        for clip in audio:
            result = _save_audio_output(clip, prefix)
            individual_saved.append(result)
            print(
                f"[SECoursesBatchAudioSaveMerge] saved individual audio -> {result['fullpath']}",
                flush=True,
            )

        merged_saved = []
        display_audio = audio[-1]
        merge_enabled = any(bool(value) for value in merge_batch_audio)
        sequential = merge_enabled and any(
            _sequential_batch_item(pack) is not None for pack in references
        )
        if sequential:
            if len(individual_saved) != 1 or len(references) != 1:
                raise ValueError("Sequential folder batching expects exactly one saved audio clip per queued job.")
            progress = _accumulate_sequential_output(
                "audio", individual_saved[0]["fullpath"], references[0]
            )
            if progress["complete"]:
                for group in progress["groups"]:
                    result = _merge_saved_audio_group(group)
                    merged_saved.append(result)
                    print(
                        f"[SECoursesBatchAudioSaveMerge] merged {len(group['audios'])} saved audio clip(s) "
                        f"for folder '{group['folder']}' -> {result['fullpath']}",
                        flush=True,
                    )
                display_audio = _load_saved_audio_output(merged_saved[-1]["fullpath"])
            else:
                print(
                    f"[SECoursesBatchAudioSaveMerge] saved sequential item "
                    f"{progress['received']}/{progress['count']}; merge waits for the final item",
                    flush=True,
                )
        elif merge_enabled:
            groups = _batch_audio_merge_groups(audio, references)
            if not groups:
                print(
                    "[SECoursesBatchAudioSaveMerge] Merge audio is enabled, but Folder batch is not active; "
                    "returning the individual audio output.",
                    flush=True,
                )
            for group in groups:
                result = _merge_batch_audio_group(group)
                merged_saved.append(result)
                print(
                    f"[SECoursesBatchAudioSaveMerge] merged {len(group['audios'])} audio clip(s) for "
                    f"folder '{group['folder']}' -> {result['fullpath']}",
                    flush=True,
                )

        if merged_saved:
            displayed = [merged_saved[-1]]
            if not sequential:
                display_audio = _concatenate_batch_audio(groups[-1]["audios"])
        else:
            displayed = individual_saved
        preview = [
            {key: result[key] for key in ("filename", "subfolder", "type")}
            for result in displayed
        ]
        return {"ui": {"audio": preview}, "result": (display_audio,)}


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
                    "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; ~4-15 seconds is quality-tested, and longer generation is allowed but experimental).",
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
                "audio_only_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "AUDIO ONLY: extract video soundtracks",
                    "label_off": "FULL REFERENCES: keep video frames",
                    "tooltip": "For audio-only output, omit reference-video frames and use only their soundtracks. @video1 becomes the corresponding <Audio 1> reference, preserving 32x32 speed.",
                }),
                "continuation_frame": ("IMAGE", {
                    "tooltip": "Optional final frame from the preceding folder-batch video. Ref2VA receives it as an additional starting-frame picture reference.",
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
        "is decoded lazily into an aspect-preserving, memory-bounded canvas. The gallery may hold more files than "
        "the model's per-run caps (9 images, 3 videos, 3 audios): each run then attaches only the files of that "
        "modality its prompt mentions (first-mention order, capped at the model limit), so folder-batch prompts "
        "can address eg '@image12', '@video4', or '@audio5' from a larger roster."
    )

    def encode(self, clip, vae, audio_vae, references, width, height, length, ref_image_size,
               prompt_override=None, audio_only_mode=False, continuation_frame=None):
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

        prompt = references.get("prompt") or ""
        if prompt_override is not None and str(prompt_override).strip():
            prompt = str(prompt_override)

        # A gallery larger than a model cap acts as a roster: the prompt
        # decides which attachments accompany this run (first-mention order,
        # capped at the model limit for each modality).
        images, image_number_map = select_prompt_media_references(prompt, images, self.MAX_IMAGES, "image")
        videos, video_number_map = select_prompt_media_references(prompt, videos, self.MAX_VIDEOS, "video")
        audios, audio_number_map = select_prompt_media_references(prompt, audios, self.MAX_AUDIOS, "audio")

        if int(references.get("version", 1)) >= 2:
            fps = max(1.0, float(references.get("video_fps", 24.0)))
            max_seconds = max(1.0, float(references.get("max_seconds", 15.0)))
            image_specs = _prepare_image_references(images, width, height, ref_image_size)
            video_specs = _prepare_video_references(videos, fps, max_seconds, length)
            ref_images = _LazyImageReferences(image_specs)
            if audio_only_mode:
                missing_audio = [spec["name"] for spec in video_specs if not spec["has_audio"]]
                if missing_audio:
                    raise ValueError(
                        "Audio-only MiniMax H3 mode received reference video(s) without a soundtrack: "
                        + ", ".join(missing_audio)
                    )
                ref_videos = None
                ref_video_audios = None
                ref_audios = _LazyAudioOnlyReferences(video_specs, audios, max_seconds)
                videos_with_audio = len(video_specs)
            else:
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

            if audio_only_mode:
                missing_audio = [
                    video.get("name", "?") for video in videos if video.get("audio") is None
                ]
                if missing_audio:
                    raise ValueError(
                        "Audio-only MiniMax H3 mode received reference video(s) without a soundtrack: "
                        + ", ".join(missing_audio)
                    )
                merged_audio = [video["audio"] for video in videos]
                merged_audio.extend(audio["audio"] for audio in audios)
                ref_videos = {}
                ref_video_audios = {}
                ref_audios = {
                    f"ref_audio_{index}": audio
                    for index, audio in enumerate(merged_audio)
                }
                videos_with_audio = len(videos)
            else:
                ref_audios = {
                    f"ref_audio_{index}": audio["audio"]
                    for index, audio in enumerate(audios)
                }
                videos_with_audio = len(ref_video_audios)

        if audio_only_mode:
            translated = translate_audio_only_reference_tokens(
                prompt, len(images), videos_with_audio, len(audios),
                audio_number_map=audio_number_map, image_number_map=image_number_map,
                video_number_map=video_number_map,
            )
        else:
            translated = translate_reference_tokens(
                prompt, len(images), len(videos), len(audios), videos_with_audio,
                audio_number_map=audio_number_map, image_number_map=image_number_map,
                video_number_map=video_number_map,
            )

        if continuation_frame is not None:
            if audio_only_mode:
                raise ValueError("Last-frame continuation is not available in MiniMax H3 audio-only mode.")
            if len(images) >= self.MAX_IMAGES:
                raise ValueError(
                    "Last-frame continuation needs one Ref2VA image slot. Use at most eight other image "
                    "references for this folder prompt."
                )
            continuation_number = len(images) + 1
            if int(references.get("version", 1)) >= 2:
                ref_images = _ImageReferencesWithContinuation(ref_images, continuation_frame)
            else:
                ref_images[f"ref_image_{len(ref_images)}"] = continuation_frame
            translated = (
                f"At 0.00 seconds, <Picture {continuation_number}> is the exact starting frame. "
                "Continue its action and camera motion seamlessly.\n\n"
                + translated
            )

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


class SECoursesMiniMaxH3ReferenceMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": (REF_PACK_TYPE, {
                    "tooltip": "Reference pack from SECourses Reference Gallery."
                }),
            }
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("has_references",)
    FUNCTION = "detect"
    DESCRIPTION = "Selects MiniMax H3 Ref2VA only when the current prompt actually has media references."

    def detect(self, references):
        if not isinstance(references, dict):
            raise ValueError("The references input must come from a SECourses Reference Gallery node.")
        return (any(references.get(kind) for kind in ("images", "videos", "audios")),)


class SECoursesMiniMaxH3TextOnly:
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
                "references": (REF_PACK_TYPE, {"tooltip": "Prompt pack from SECourses Reference Gallery."}),
                "width": ("INT", {"default": 32, "min": 32, "max": max_resolution, "step": 32}),
                "height": ("INT", {"default": 32, "min": 32, "max": max_resolution, "step": 32}),
                "length": ("INT", {
                    "default": 124, "min": 5, "max": 3600, "step": 17,
                    "tooltip": "Frame count at 24 fps, snapped up to the model's 17k+5 grid. Longer than 15 seconds is allowed but remains experimental.",
                }),
            },
            "optional": {
                "prompt_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional external prompt that replaces the gallery prompt.",
                }),
                "first_frame": ("IMAGE", {
                    "tooltip": "Optional final frame from the preceding folder-batch video, used as FL2VA's first-frame keyframe.",
                }),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    DESCRIPTION = "Uses MiniMax H3's FL2VA checkpoint for text-only or first-frame generation."

    def encode(self, clip, vae, references, width, height, length, prompt_override=None, first_frame=None):
        try:
            from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo
        except ImportError:
            raise RuntimeError(
                "MiniMax H3 nodes were not found. Update ComfyUI to a version that ships comfy_extras/nodes_minimax_h3.py."
            )
        if not isinstance(references, dict):
            raise ValueError("The references input must come from a SECourses Reference Gallery node.")
        prompt = references.get("prompt") or ""
        if prompt_override is not None and str(prompt_override).strip():
            prompt = str(prompt_override)
        prompt = translate_reference_tokens(prompt, 0, 0, 0, 0)
        output = MiniMaxH3ImageToVideo.execute(
            clip=clip,
            vae=vae,
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            first_frame=first_frame,
        )
        return (output.args[0], output.args[1])


class SECoursesMiniMaxH3Auto:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = SECoursesMiniMaxH3References.INPUT_TYPES()
        optional = dict(inputs.get("optional", {}))
        optional.pop("audio_only_mode", None)
        optional["continuation_frame"] = (OPTIONAL_IMAGE_TYPE, {
            "tooltip": "Optional value from MiniMax H3 Previous Batch Final Frame.",
        })
        return {**inputs, "optional": optional}

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "BOOLEAN")
    RETURN_NAMES = ("positive", "latent", "uses_ref2va")
    FUNCTION = "encode"
    DESCRIPTION = (
        "Automatically prepares Ref2VA conditioning when the current prompt pack has media references, otherwise "
        "prepares FL2VA text/first-frame conditioning. The uses_ref2va output is diagnostic; select the checkpoint "
        "with MiniMax H3 Reference Mode before any model-dependent VAE optimization."
    )

    def encode(
        self,
        clip,
        vae,
        audio_vae,
        references,
        width,
        height,
        length,
        ref_image_size,
        prompt_override=None,
        continuation_frame=None,
    ):
        if not isinstance(references, dict):
            raise ValueError("The references input must come from a SECourses Reference Gallery node.")
        if isinstance(continuation_frame, dict) and set(continuation_frame).issubset({"image"}):
            continuation_frame = continuation_frame.get("image")
        init_image = references.get("init_image")
        if init_image:
            continuation_frame = _load_reference_image(
                _resolve_reference_entry(init_image), int(width), int(height)
            )
            print(
                f"[SECoursesMiniMaxH3Auto] using folder-batch init image '{init_image['name']}'.",
                flush=True,
            )
        has_references = any(references.get(kind) for kind in ("images", "videos", "audios"))
        if has_references:
            positive, latent = SECoursesMiniMaxH3References().encode(
                clip=clip,
                vae=vae,
                audio_vae=audio_vae,
                references=references,
                width=width,
                height=height,
                length=length,
                ref_image_size=ref_image_size,
                prompt_override=prompt_override,
                continuation_frame=continuation_frame,
            )
        else:
            positive, latent = SECoursesMiniMaxH3TextOnly().encode(
                clip=clip,
                vae=vae,
                references=references,
                width=width,
                height=height,
                length=length,
                prompt_override=prompt_override,
                first_frame=continuation_frame,
            )
        return positive, latent, has_references


class SECoursesLoadVideoAudioB64:
    """Decode only a base64 video's soundtrack, without materializing its frames."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_base64": ("STRING", {"multiline": True}),
                "max_seconds": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 3600.0, "step": 0.5,
                    "tooltip": "Maximum soundtrack duration to decode. Clean 2-15 second references are recommended; longer references are allowed.",
                }),
            },
            "optional": {
                "start_seconds": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.05,
                    "tooltip": "Skip this many seconds from the start of the soundtrack before max_seconds is applied (used by trimmed references).",
                }),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    DESCRIPTION = "Extracts a video's soundtrack directly from base64 data without decoding any video frames."

    def load(self, video_base64, max_seconds, start_seconds=0.0):
        payload = str(video_base64).strip()
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        try:
            video_bytes = base64.b64decode(payload)
        except (ValueError, TypeError) as error:
            raise ValueError("Reference video contains invalid base64 data.") from error
        if not video_bytes:
            raise ValueError("Reference video is empty.")
        audio = _decode_video_audio(
            io.BytesIO(video_bytes), max(1.0, float(max_seconds)),
            trim_start=max(0.0, float(start_seconds)))
        if audio is None:
            raise ValueError(
                "Reference video has no usable soundtrack. Audio-only MiniMax H3 mode does not use its frames."
            )
        return (audio,)


class SECoursesTrimAudio:
    """Trim an AUDIO value to a user-selected duration without imposing a model policy cap."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "max_seconds": ("FLOAT", {
                    "default": 15.0, "min": 1.0, "max": 3600.0, "step": 0.5,
                    "tooltip": "Maximum duration used from this reference. Values above 15 seconds are allowed but experimental for MiniMax H3.",
                }),
            }
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "trim"
    DESCRIPTION = "Trims a ComfyUI AUDIO value to the selected maximum duration."

    def trim(self, audio, max_seconds):
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("SECourses Trim Audio requires a valid ComfyUI AUDIO input.")
        sample_rate = int(audio["sample_rate"])
        max_samples = max(1, round(max(1.0, float(max_seconds)) * sample_rate))
        trimmed = dict(audio)
        trimmed["waveform"] = audio["waveform"][..., :max_samples]
        return (trimmed,)


NODE_CLASS_MAPPINGS = {
    "SECoursesReferenceGallery": SECoursesReferenceGallery,
    "SECoursesBatchDuration": SECoursesBatchDuration,
    "SECoursesBatchContinuationFrame": SECoursesBatchContinuationFrame,
    "SECoursesBatchVideoSaveMerge": SECoursesBatchVideoSaveMerge,
    "SECoursesBatchVideoMerge": SECoursesBatchVideoMerge,
    "SECoursesBatchAudioMerge": SECoursesBatchAudioMerge,
    "SECoursesBatchAudioSaveMerge": SECoursesBatchAudioSaveMerge,
    "SECoursesMiniMaxH3References": SECoursesMiniMaxH3References,
    "SECoursesMiniMaxH3ReferenceMode": SECoursesMiniMaxH3ReferenceMode,
    "SECoursesMiniMaxH3TextOnly": SECoursesMiniMaxH3TextOnly,
    "SECoursesMiniMaxH3Auto": SECoursesMiniMaxH3Auto,
    "SECoursesLoadVideoAudioB64": SECoursesLoadVideoAudioB64,
    "SECoursesTrimAudio": SECoursesTrimAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SECoursesReferenceGallery": "SECourses Reference Gallery (Images / Videos / Audio)",
    "SECoursesBatchDuration": "MiniMax H3 Folder Batch Duration",
    "SECoursesBatchContinuationFrame": "MiniMax H3 Folder Batch Previous Last Frame",
    "SECoursesBatchVideoSaveMerge": "Save + Merge MiniMax H3 Folder Batch Videos",
    "SECoursesBatchVideoMerge": "Merge MiniMax H3 Folder Batch Videos",
    "SECoursesBatchAudioMerge": "Merge MiniMax H3 Folder Batch Audio",
    "SECoursesBatchAudioSaveMerge": "Save + Merge MiniMax H3 Folder Batch Audio",
    "SECoursesMiniMaxH3References": "MiniMax H3 References (Gallery)",
    "SECoursesMiniMaxH3ReferenceMode": "MiniMax H3 Reference Mode",
    "SECoursesMiniMaxH3TextOnly": "MiniMax H3 Text Only (Gallery Prompt)",
    "SECoursesMiniMaxH3Auto": "MiniMax H3 Auto FL2VA / Ref2VA (Gallery)",
    "SECoursesLoadVideoAudioB64": "Load Video Soundtrack (Base64, No Frames)",
    "SECoursesTrimAudio": "Trim Reference Audio",
}

_register_folder_batch_inspect_route()
_register_media_info_route()
