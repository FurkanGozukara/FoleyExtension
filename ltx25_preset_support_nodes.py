"""Lightweight ComfyUI helpers used by the maintained LTX-2.5 presets."""

import math

import torch
import torch.nn.functional as torch_functional


_DISTILLED_SIGMAS = torch.tensor(
    [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    dtype=torch.float32,
)


def _next_multiple(value, multiple):
    return max(multiple, math.ceil(value / multiple) * multiple)


def _scalar(value, default=0):
    while isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[0]
    if torch.is_tensor(value):
        return value.item()
    return value


def _list_value(values, index, default):
    if not isinstance(values, (list, tuple)):
        return values
    if not values:
        return default
    return values[min(index, len(values) - 1)]


class LTX25UpscaleControls:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_latent_tokens": (
                    "INT",
                    {"default": 18000, "min": 1024, "max": 1048576, "step": 1024},
                ),
                "max_chunk_frames": (
                    "INT",
                    {"default": 121, "min": 1, "max": 4097, "step": 8},
                ),
                "overlap_frames": (
                    "INT",
                    {"default": 1, "min": 0, "max": 65, "step": 1},
                ),
                "sampling_steps": (
                    "INT",
                    {"default": 8, "min": 1, "max": 64, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = (
        "max_latent_tokens",
        "max_chunk_frames",
        "overlap_frames",
        "sampling_steps",
    )
    FUNCTION = "values"
    CATEGORY = "SECourses/LTX-2.5"
    DESCRIPTION = (
        "Exposes the chunking and sampling controls used by the LTX-2.5 "
        "2x pixel-upscale presets."
    )

    @staticmethod
    def values(max_latent_tokens, max_chunk_frames, overlap_frames, sampling_steps):
        return max_latent_tokens, max_chunk_frames, overlap_frames, sampling_steps


class LTX25DistilledSigmaSchedule:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampling_steps": (
                    "INT",
                    {"default": 8, "min": 1, "max": 64, "step": 1},
                )
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    FUNCTION = "build"
    CATEGORY = "SECourses/LTX-2.5"
    DESCRIPTION = (
        "Returns the official LTX-2.5 distilled sigma curve at eight steps. "
        "Other step counts interpolate that curve for experimentation."
    )

    @staticmethod
    def build(sampling_steps):
        sampling_steps = max(1, int(sampling_steps))
        if sampling_steps == len(_DISTILLED_SIGMAS) - 1:
            return (_DISTILLED_SIGMAS.clone(),)

        positions = torch.linspace(
            0,
            len(_DISTILLED_SIGMAS) - 1,
            sampling_steps + 1,
            dtype=torch.float32,
        )
        low = positions.floor().long()
        high = positions.ceil().long()
        fraction = positions - low
        sigmas = (
            _DISTILLED_SIGMAS[low] * (1.0 - fraction)
            + _DISTILLED_SIGMAS[high] * fraction
        )
        return (sigmas,)


class LTX25PrepareVideoChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "max_latent_tokens": (
                    "INT",
                    {"default": 18000, "min": 1024, "max": 1048576, "step": 1024},
                ),
                "max_chunk_frames": (
                    "INT",
                    {"default": 121, "min": 1, "max": 4097, "step": 8},
                ),
                "overlap_frames": (
                    "INT",
                    {"default": 1, "min": 0, "max": 65, "step": 1},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "INT",
        "STRING",
    )
    RETURN_NAMES = (
        "chunks",
        "generation_width",
        "generation_height",
        "chunk_length",
        "keep_length",
        "output_width",
        "output_height",
        "overlap_frames",
        "total_frames",
        "plan_summary",
    )
    OUTPUT_IS_LIST = (True,) * 10
    FUNCTION = "prepare"
    CATEGORY = "SECourses/LTX-2.5"
    DESCRIPTION = (
        "Pads source frames to valid LTX geometry, computes the exact 2x output "
        "size, and divides long videos into overlapping 8n+1-frame chunks."
    )

    @staticmethod
    def prepare(images, max_latent_tokens, max_chunk_frames, overlap_frames):
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError(
                "images must be a ComfyUI IMAGE tensor with shape "
                "[frames, height, width, channels]"
            )
        total_frames, source_height, source_width, _ = images.shape
        if total_frames < 1 or source_height < 1 or source_width < 1:
            raise ValueError("The source video must contain at least one non-empty frame")

        guide_width = _next_multiple(source_width, 32)
        guide_height = _next_multiple(source_height, 32)
        generation_width = guide_width * 2
        generation_height = guide_height * 2
        output_width = source_width * 2
        output_height = source_height * 2

        pad_right = guide_width - source_width
        pad_bottom = guide_height - source_height
        if pad_right or pad_bottom:
            channels_first = images.permute(0, 3, 1, 2)
            channels_first = torch_functional.pad(
                channels_first,
                (0, pad_right, 0, pad_bottom),
                mode="replicate",
            )
            images = channels_first.permute(0, 2, 3, 1)

        spatial_tokens = (generation_width // 32) * (generation_height // 32)
        temporal_tokens = max(1, int(max_latent_tokens) // max(1, spatial_tokens))
        budget_frames = 1 + 8 * (temporal_tokens - 1)
        frame_cap = min(int(max_chunk_frames), budget_frames)
        frame_cap = max(1, 1 + 8 * ((frame_cap - 1) // 8))

        overlap = max(0, min(int(overlap_frames), frame_cap - 1))
        chunks = []
        chunk_lengths = []
        keep_lengths = []
        chunk_overlaps = []
        start = 0

        while start < total_frames:
            stop = min(total_frames, start + frame_cap)
            chunk = images[start:stop]
            keep_length = stop - start
            padded_length = 1 + 8 * math.ceil(max(0, keep_length - 1) / 8)
            if padded_length > keep_length:
                padding = chunk[-1:].expand(padded_length - keep_length, -1, -1, -1)
                chunk = torch.cat((chunk, padding), dim=0)

            chunks.append(chunk)
            chunk_lengths.append(padded_length)
            keep_lengths.append(keep_length)
            chunk_overlaps.append(0 if not chunk_overlaps else overlap)

            if stop >= total_frames:
                break
            start = stop - overlap

        count = len(chunks)
        summary = (
            f"{source_width}x{source_height} source -> "
            f"{generation_width}x{generation_height} generation canvas -> "
            f"{output_width}x{output_height} exact crop; "
            f"{total_frames} frames in {count} chunk(s), cap {frame_cap}, "
            f"overlap {overlap}."
        )

        return (
            chunks,
            [generation_width] * count,
            [generation_height] * count,
            chunk_lengths,
            keep_lengths,
            [output_width] * count,
            [output_height] * count,
            chunk_overlaps,
            [total_frames] * count,
            [summary] * count,
        )


class LTX25MergeVideoChunks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "keep_length": ("INT", {"default": 9, "min": 1, "max": 4097}),
                "overlap_frames": ("INT", {"default": 1, "min": 0, "max": 65}),
                "total_frames": (
                    "INT",
                    {"default": 1, "min": 1, "max": 1048576},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    INPUT_IS_LIST = True
    FUNCTION = "merge"
    CATEGORY = "SECourses/LTX-2.5"
    DESCRIPTION = (
        "Trims padded chunk frames, blends overlaps, and restores the source "
        "video's frame count."
    )

    @staticmethod
    def merge(images, keep_length, overlap_frames, total_frames):
        chunks = list(images) if isinstance(images, (list, tuple)) else [images]
        if not chunks:
            raise ValueError("No decoded video chunks were provided")

        merged = None
        for index, chunk in enumerate(chunks):
            if not torch.is_tensor(chunk) or chunk.ndim != 4:
                raise ValueError("Each decoded chunk must be an IMAGE tensor")
            keep = max(
                1,
                int(
                    _scalar(
                        _list_value(keep_length, index, chunk.shape[0]),
                        chunk.shape[0],
                    )
                ),
            )
            chunk = chunk[:keep]
            if merged is None:
                merged = chunk
                continue

            overlap = max(
                0,
                int(_scalar(_list_value(overlap_frames, index, 0), 0)),
            )
            overlap = min(overlap, merged.shape[0], chunk.shape[0])
            if overlap:
                weights = torch.linspace(
                    0.0,
                    1.0,
                    overlap + 2,
                    device=chunk.device,
                    dtype=chunk.dtype,
                )[1:-1].view(overlap, 1, 1, 1)
                blended = merged[-overlap:] * (1.0 - weights) + chunk[:overlap] * weights
                merged = torch.cat(
                    (merged[:-overlap], blended, chunk[overlap:]),
                    dim=0,
                )
            else:
                merged = torch.cat((merged, chunk), dim=0)

        expected = max(1, int(_scalar(total_frames, merged.shape[0])))
        if merged.shape[0] < expected:
            padding = merged[-1:].expand(expected - merged.shape[0], -1, -1, -1)
            merged = torch.cat((merged, padding), dim=0)
        return (merged[:expected],)


NODE_CLASS_MAPPINGS = {
    "LTX25UpscaleControls": LTX25UpscaleControls,
    "LTX25DistilledSigmaSchedule": LTX25DistilledSigmaSchedule,
    "LTX25PrepareVideoChunks": LTX25PrepareVideoChunks,
    "LTX25MergeVideoChunks": LTX25MergeVideoChunks,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX25UpscaleControls": "LTX-2.5 Upscale Controls",
    "LTX25DistilledSigmaSchedule": "LTX-2.5 Distilled Sigma Schedule",
    "LTX25PrepareVideoChunks": "LTX-2.5 Prepare Video Chunks",
    "LTX25MergeVideoChunks": "LTX-2.5 Merge Video Chunks",
}
