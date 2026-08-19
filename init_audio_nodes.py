"""Optional init audio: a soundtrack the generation must follow (MiniMax H3).

Works like the optional init image: nothing is emitted while no file is
selected, and every node here passes its inputs through untouched when no
init audio arrives, so one workflow serves text-only, first-frame, and
reference generation with or without a soundtrack.

The file may be an audio file or a video (its soundtrack is used), and an
optional trim window (``trim_start`` / ``trim_end`` seconds, set visually by
the trim panel under the player) keeps only one part of it. Decoding seeks to
the trim start and stops at the trim end, so a short window of a long video
stays fast and the file itself is never re-encoded.

``SECoursesMiniMaxH3InitAudio`` conditions MiniMax H3 the way the
``multimodalart/minimax-h3-audio-to-video`` Space does with diffusers, using
only mechanisms ComfyUI's native MiniMax H3 implementation already has:

- the encoded soundtrack replaces the audio half of the joint AV latent and a
  nested noise mask (video 1, audio 0) keeps those rows on the given audio at
  every sampling step, so the soundtrack is exact and the video is denoised
  alongside it (the same audio lock the face-refine pass uses);
- a t=1.0 audio guide (ComfyUI's ``MiniMaxH3AddGuide`` keyframe mechanism at
  frame 0) hands the transformer a clean copy of the whole soundtrack from the
  first step, the presentation the Space gives its locked audio rows.

The waveform is normalized to the audio VAE's 32 kHz stereo, silence-padded or
cut to the latent's exact duration before encoding (no centre-crop offset), and
the same normalized waveform, cut to the video duration, is returned so the
final MP4 carries the user's audio instead of a VAE round trip.
"""

from __future__ import annotations

import math
import os

import torch

try:
    from .media_extensions import audio_input_extensions, has_extension
except ImportError:  # direct test-module import
    from media_extensions import audio_input_extensions, has_extension

NO_AUDIO = "(none - disabled)"
TRIM_MAX_SECONDS = 360000.0
# Files up to this size are content-hashed like core LoadAudio; larger uploads
# (long videos used for their soundtrack) are fingerprinted by size and mtime.
HASH_FINGERPRINT_LIMIT = 64 * 1024 * 1024
DURATION_MATCH = "match init audio length"
DURATION_KEEP = "keep workflow duration"
CONDITIONING_MODES = ["lock soundtrack + guide", "lock soundtrack only", "guide only (model re-voices)"]
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_HOP = 800
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24


def _load_audio_node():
    try:
        from comfy_extras.nodes_audio import LoadAudio
    except ImportError as error:  # pragma: no cover - depends on the ComfyUI version
        raise RuntimeError("ComfyUI's LoadAudio node was not found. Update ComfyUI.") from error
    return LoadAudio


def audio_duration_seconds(audio):
    """Duration in seconds of a ComfyUI AUDIO value."""
    waveform = audio["waveform"]
    return float(waveform.shape[-1]) / float(audio["sample_rate"])


def align_frame_count(frames):
    """MiniMax H3's 17k+5 frame grid, rounded up (5, 22, 39, ...)."""
    frames = max(5, int(frames))
    while frames % 17 != 5:
        frames += 1
    return frames


def frames_for_seconds(seconds, fps=FPS):
    return align_frame_count(round(float(seconds) * fps))


def normalize_waveform(audio, sample_rate):
    """First batch item as float32 stereo ``[2, L]`` at ``sample_rate``."""
    waveform = audio["waveform"]
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"Init audio must be a [channels, samples] waveform, got {tuple(waveform.shape)}.")
    waveform = waveform.detach().to(dtype=torch.float32, device="cpu")
    if waveform.shape[0] > 2:
        # More than stereo is downmixed rather than refused; this is a user upload, not a curated reference.
        waveform = waveform.mean(dim=0, keepdim=True)
    source_rate = int(audio["sample_rate"])
    if source_rate != int(sample_rate):
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, source_rate, int(sample_rate))
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
    return waveform.contiguous()


def fit_waveform(waveform, samples):
    """Cut or silence-pad a ``[C, L]`` waveform to exactly ``samples``."""
    length = waveform.shape[-1]
    if length > samples:
        return waveform[..., :samples].contiguous()
    if length < samples:
        return torch.nn.functional.pad(waveform, (0, samples - length))
    return waveform


def video_frame_count(video_latent):
    """Pixel frames represented by a MiniMax H3 video latent ``[B, 24, T, H, W]``."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(video_latent.shape[2])))


def _av_parts(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None or not getattr(samples, "is_nested", False):
        raise ValueError(
            "MiniMax H3 Init Audio expects the joint video+audio LATENT of a MiniMax H3 conditioning node "
            "(Image to Video, Reference to Video, or the SECourses gallery adapters)."
        )
    parts = list(samples.unbind())
    if len(parts) < 2 or parts[0].ndim != 5 or parts[1].ndim != 4:
        raise ValueError("MiniMax H3 Init Audio could not find the video and audio streams of the AV latent.")
    return parts[0], parts[1]


def trim_window(trim_start=0.0, trim_end=0.0):
    """Validated ``(start_seconds, end_seconds | None)`` of the optional trim inputs.

    ``trim_end`` of 0 means "until the end of the file", so ``(0.0, None)`` is
    the untrimmed whole file. Raises ``ValueError`` for negative, non-finite,
    or inverted windows.
    """
    try:
        start = float(trim_start or 0.0)
        end = float(trim_end or 0.0)
    except (TypeError, ValueError) as error:
        raise ValueError("Init audio trim_start and trim_end must be seconds.") from error
    if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < 0.0:
        raise ValueError("Init audio trim_start and trim_end must be non-negative seconds (0 = no trim).")
    if end <= 0.0:
        return start, None
    if end <= start:
        raise ValueError(
            f"Init audio trim_end ({end:.2f}s) must be after trim_start ({start:.2f}s); "
            "set trim_end to 0 to use the file until its end."
        )
    return start, end


def describe_trim_window(start, end):
    """Human-readable suffix for log lines, empty when the whole file is used."""
    if end is None and start <= 0.0:
        return ""
    end_text = "end" if end is None else f"{end:.2f}s"
    return f" (trimmed {start:.2f}s-{end_text})"


def load_audio_window(audio, trim_start=0.0, trim_end=None):
    """ComfyUI AUDIO of an input-directory audio or video file, optionally only ``[trim_start, trim_end)`` seconds.

    Uses the reference gallery's seek-aware decoder: a video's first audio
    stream is decoded from the trim start and stops at the trim end, so a
    10 second window of a two hour recording stays fast.
    """
    import folder_paths

    try:
        from .reference_gallery_nodes import _decode_video_audio
    except ImportError:  # direct test-module import
        from reference_gallery_nodes import _decode_video_audio

    path = folder_paths.get_annotated_filepath(audio)
    start = max(0.0, float(trim_start or 0.0))
    window = None if trim_end is None else max(0.0, float(trim_end) - start)
    loaded = _decode_video_audio(path, window, trim_start=start)
    if loaded is None:
        where = f" after its {start:.2f}s trim start" if start > 0.0 else ""
        raise ValueError(
            f"Init audio '{audio}' has no decodable audio stream{where}. "
            "Pick an audio file or a video with a soundtrack, or move the trim start earlier."
        )
    return loaded


def input_file_fingerprint(audio):
    """Cache fingerprint of an input-directory file: content hash, or size+mtime for very large uploads."""
    import folder_paths

    path = folder_paths.get_annotated_filepath(audio)
    try:
        size = os.path.getsize(path)
    except OSError:
        return f"missing:{audio}"
    if size > HASH_FINGERPRINT_LIMIT:
        return f"{size}:{os.stat(path).st_mtime_ns}:{audio}"
    return _load_audio_node().fingerprint_inputs(audio)


class SECoursesInitAudio:
    """Optional init audio or video file (soundtrack) with an optional trim window, plus the duration it drives."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        files = [
            name for name in os.listdir(input_dir)
            if os.path.isfile(os.path.join(input_dir, name))
            and has_extension(name, audio_input_extensions())
        ]
        return {
            "required": {
                "audio": ([NO_AUDIO, *sorted(files)], {
                    "audio_upload": True,
                    "tooltip": "No audio is emitted until a file is selected. Upload or select an audio file, or a video whose soundtrack should be used: the generated video follows this audio (lipsync, action timing) and keeps it as its audio track. The trim panel under the player keeps only one part of the file.",
                }),
                "duration_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.01, "max": 3600.0, "step": 0.01,
                    "tooltip": "The workflow's own duration control (usually connected). It is passed through unchanged while no init audio is selected.",
                }),
                "duration_mode": ([DURATION_MATCH, DURATION_KEEP], {
                    "default": DURATION_MATCH,
                    "tooltip": "'match init audio length' makes the video as long as the selected audio (the trimmed part, rounded up to the model's frame grid). 'keep workflow duration' keeps the connected duration; longer audio is cut and shorter audio is padded with silence.",
                }),
            },
            "optional": {
                "trim_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": TRIM_MAX_SECONDS, "step": 0.01,
                    "tooltip": "Use the file from this many seconds in (0 = from the beginning). Set visually with the trim panel under the player; the part before it is skipped without re-encoding the file.",
                }),
                "trim_end": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": TRIM_MAX_SECONDS, "step": 0.01,
                    "tooltip": "Use the file up to this many seconds (0 = until the end of the file). Must be later than trim_start; set visually with the trim panel under the player.",
                }),
            },
        }

    CATEGORY = "SECourses/audio"
    RETURN_TYPES = ("AUDIO", "FLOAT")
    RETURN_NAMES = ("init_audio", "duration_seconds")
    FUNCTION = "load"
    DESCRIPTION = (
        "Optional init audio, enabled automatically when a file is selected: an audio file, or a video whose "
        "soundtrack is used. An optional trim window (trim_start / trim_end seconds, set visually in the trim panel "
        "under the player) keeps only one part of it. Outputs the loaded audio (or nothing) and the duration the "
        "workflow should use: the (trimmed) audio's own length by default, otherwise the connected value."
    )

    def load(self, audio, duration_seconds, duration_mode=DURATION_MATCH, trim_start=0.0, trim_end=0.0):
        duration = float(duration_seconds)
        if not audio or audio == NO_AUDIO:
            return (None, duration)
        start, end = trim_window(trim_start, trim_end)
        loaded = load_audio_window(audio, start, end)
        seconds = audio_duration_seconds(loaded)
        if seconds <= 0:
            raise ValueError(f"Init audio '{audio}' is empty{describe_trim_window(start, end)}.")
        window = describe_trim_window(start, end)
        if duration_mode == DURATION_MATCH:
            note = " (longer than the quality-tested 15 second range, allowed but experimental)" if seconds > 15 else ""
            print(
                f"[SECoursesInitAudio] '{audio}'{window} is {seconds:.2f}s; the video follows the audio length{note}.",
                flush=True,
            )
            return (loaded, seconds)
        print(
            f"[SECoursesInitAudio] '{audio}'{window} is {seconds:.2f}s; keeping the workflow duration of {duration:.2f}s.",
            flush=True,
        )
        return (loaded, duration)

    @classmethod
    def IS_CHANGED(cls, audio, duration_seconds=None, duration_mode=None, trim_start=0.0, trim_end=0.0):
        if not audio or audio == NO_AUDIO:
            return NO_AUDIO
        # The trim inputs are part of the cache key already; only the file content needs a fingerprint.
        return input_file_fingerprint(audio)

    @classmethod
    def VALIDATE_INPUTS(cls, audio, trim_start=0.0, trim_end=0.0):
        if not audio or audio == NO_AUDIO:
            return True
        try:
            trim_window(trim_start, trim_end)
        except ValueError as error:
            return str(error)
        return _load_audio_node().validate_inputs(audio)


class SECoursesMiniMaxH3InitAudio:
    """Make MiniMax H3 follow a given soundtrack; passthrough when none is connected."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "MiniMax H3 conditioning (text, first/last frame, or references)."}),
                "latent": ("LATENT", {"tooltip": "The joint AV latent produced by the same MiniMax H3 conditioning node."}),
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE."}),
            },
            "optional": {
                "init_audio": ("AUDIO", {
                    "tooltip": "Optional soundtrack the video must follow. Leave unconnected (or feed the Init Audio node's disabled output) for normal generation.",
                }),
                "audio_conditioning": (CONDITIONING_MODES, {
                    "default": CONDITIONING_MODES[0],
                    "tooltip": "'lock soundtrack + guide' (recommended) keeps the given audio exact and shows it to the model from the first step. 'lock soundtrack only' keeps it exact without the guide. 'guide only' lets the model generate its own audio while following the given one, so the output soundtrack is the model's.",
                }),
                "references": ("SECOURSES_REF_PACK", {
                    "tooltip": "Optional folder-batch pack. A same-basename audio file in an init-media-enabled batch overrides the single INIT AUDIO selection for this item.",
                }),
            },
        }

    CATEGORY = "SECourses/audio"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "AUDIO")
    RETURN_NAMES = ("positive", "latent", "init_audio")
    FUNCTION = "apply"
    DESCRIPTION = (
        "Optional init audio for MiniMax H3 (FL2VA text/first-frame and Ref2VA references alike). The soundtrack is "
        "encoded with the audio VAE, locked into the AV latent so it survives sampling exactly, and given to the "
        "transformer as a clean t=1.0 audio guide so the video is generated to match it. Without init audio "
        "everything passes through unchanged. The init_audio output is the normalized soundtrack cut to the video "
        "length (or nothing), ready to replace the decoded audio in the final video."
    )

    def apply(self, positive, latent, audio_vae, init_audio=None, audio_conditioning=CONDITIONING_MODES[0], references=None):
        batch_audio = references.get("init_audio") if isinstance(references, dict) else None
        if batch_audio:
            try:
                from .reference_gallery_nodes import _load_reference_audio, _resolve_reference_entry
            except ImportError:  # direct test-module import
                from reference_gallery_nodes import _load_reference_audio, _resolve_reference_entry
            seconds = video_frame_count(_av_parts(latent)[0]) / FPS
            init_audio = _load_reference_audio(_resolve_reference_entry(batch_audio), seconds)
            print(
                f"[SECoursesMiniMaxH3InitAudio] using folder-batch init audio '{batch_audio['name']}'.",
                flush=True,
            )
        if init_audio is None:
            return (positive, latent, None)
        if not isinstance(init_audio, dict) or "waveform" not in init_audio or "sample_rate" not in init_audio:
            raise ValueError("MiniMax H3 Init Audio needs a ComfyUI AUDIO value (waveform + sample_rate).")
        if audio_conditioning not in CONDITIONING_MODES:
            raise ValueError(f"Unknown audio_conditioning '{audio_conditioning}'.")
        lock = audio_conditioning != CONDITIONING_MODES[2]
        guide = audio_conditioning != CONDITIONING_MODES[1]

        video, audio_slot = _av_parts(latent)
        audio_t = int(audio_slot.shape[-1])
        sample_rate = int(getattr(audio_vae, "audio_sample_rate", DEFAULT_SAMPLE_RATE))
        hop = int(getattr(audio_vae, "downscale_ratio", DEFAULT_HOP))
        frame_count = video_frame_count(video)

        waveform = normalize_waveform(init_audio, sample_rate)
        source_seconds = waveform.shape[-1] / sample_rate
        latent_seconds = audio_t * hop / sample_rate
        # Exactly audio_t hops: the encoder then neither pads nor centre-crops, so
        # the latent starts at the audio's first sample and has the slot's length.
        encoded = audio_vae.encode(fit_waveform(waveform, audio_t * hop).unsqueeze(0).movedim(1, -1))
        if encoded.ndim != 4 or encoded.shape[1:3] != audio_slot.shape[1:3]:
            raise ValueError(
                f"The audio VAE produced a {tuple(encoded.shape)} latent, but the AV latent's audio slot is "
                f"{tuple(audio_slot.shape)}. Connect the MiniMax H3 audio VAE."
            )
        if encoded.shape[-1] != audio_t:  # defensive: the VAE contract is one latent frame per hop
            encoded = torch.nn.functional.pad(encoded[..., :audio_t], (0, max(0, audio_t - encoded.shape[-1])))
        encoded = encoded.to(device=audio_slot.device, dtype=audio_slot.dtype)

        import comfy.nested_tensor

        out = latent
        if lock:
            out = dict(latent)
            out["samples"] = comfy.nested_tensor.NestedTensor((video, encoded))
            previous = latent.get("noise_mask")
            video_mask = torch.ones_like(video)
            if previous is not None and getattr(previous, "is_nested", False):
                video_mask = previous.unbind()[0]
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, torch.zeros_like(encoded)))

        if guide:
            import node_helpers

            keyframes = list(positive[0][1].get("minimax_keyframes", []))
            keyframes.append({"resolved_frame_index": 0, "audio_latent": encoded})
            positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})

        fitted = None
        if lock:
            fitted = {
                "waveform": fit_waveform(waveform, int(round(frame_count / FPS * sample_rate))).unsqueeze(0),
                "sample_rate": sample_rate,
            }
        action = "cut to" if source_seconds > latent_seconds + 1e-6 else "silence-padded to" if source_seconds < latent_seconds - 1e-6 else "matching"
        print(
            f"[SECoursesMiniMaxH3InitAudio] {audio_conditioning}: {source_seconds:.2f}s init audio {action} the "
            f"{latent_seconds:.2f}s audio latent ({audio_t} frames) for {frame_count} video frames.",
            flush=True,
        )
        return (positive, out, fitted)


class SECoursesMiniMaxH3AudioFrames:
    """Frame count for a MiniMax H3 video that should last as long as an optional init audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fallback_frames": ("INT", {
                    "default": 124, "min": 1, "max": 3600, "step": 1,
                    "tooltip": "Frame count used when no init audio is connected (rounded up to the 17k+5 grid).",
                }),
            },
            "optional": {
                "init_audio": ("AUDIO", {"tooltip": "Optional init audio; when connected the frame count covers its duration at 24 FPS."}),
                "match_audio": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off: keep the fallback frame count even when init audio is connected.",
                }),
            },
        }

    CATEGORY = "SECourses/audio"
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "resolve"
    DESCRIPTION = "MiniMax H3 frame count (24 FPS, 17k+5 grid) covering an optional init audio, otherwise the fallback."

    def resolve(self, fallback_frames, init_audio=None, match_audio=True):
        if init_audio is None or not match_audio:
            return (align_frame_count(fallback_frames),)
        seconds = audio_duration_seconds(init_audio)
        if seconds <= 0:
            raise ValueError("Init audio is empty.")
        frames = frames_for_seconds(seconds)
        print(f"[SECoursesMiniMaxH3AudioFrames] {seconds:.2f}s init audio -> {frames} frames.", flush=True)
        return (frames,)


class SECoursesAudioFallback:
    """Prefer an optional override audio (the exact init audio) over the decoded audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"lazy": True, "tooltip": "Normal audio, eg the decoded generated soundtrack."}),
            },
            "optional": {
                "override": ("AUDIO", {"tooltip": "Optional replacement, eg the init_audio output of MiniMax H3 Init Audio. Used whenever it is present."}),
            },
        }

    CATEGORY = "SECourses/audio"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "pick"
    DESCRIPTION = "Returns the override audio when one is connected and present, otherwise the normal audio (which is only computed when needed)."

    def check_lazy_status(self, audio=None, override=None):
        return [] if override is not None else ["audio"]

    def pick(self, audio=None, override=None):
        if override is not None:
            return (override,)
        if audio is None:
            raise ValueError("Audio Fallback received neither an override nor a normal audio.")
        return (audio,)


NODE_CLASS_MAPPINGS = {
    "SECoursesInitAudio": SECoursesInitAudio,
    "SECoursesMiniMaxH3InitAudio": SECoursesMiniMaxH3InitAudio,
    "SECoursesMiniMaxH3AudioFrames": SECoursesMiniMaxH3AudioFrames,
    "SECoursesAudioFallback": SECoursesAudioFallback,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SECoursesInitAudio": "Init Audio (Optional, Auto Enable)",
    "SECoursesMiniMaxH3InitAudio": "MiniMax H3 Init Audio (Optional)",
    "SECoursesMiniMaxH3AudioFrames": "MiniMax H3 Frames From Init Audio",
    "SECoursesAudioFallback": "Audio Fallback (Optional Override)",
}
