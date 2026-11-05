"""Videos decoded at the output size: the gallery's downscale_videos switch, the Genjutsu clip and the SwarmUI
loader that replaces Video Slice -> Get Video Components -> SwarmVideoResampleFPS."""
import base64
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import genjutsu_nodes  # noqa: E402
import reference_gallery_nodes as gallery  # noqa: E402

HAS_COMFY = importlib.util.find_spec("comfy_api") is not None and importlib.util.find_spec("comfy") is not None


def write_video(path, width, height, fps, frame_count, with_audio=True):
    """A moving gradient (so frame blending is visible) with a 440 Hz tone."""
    import av
    import numpy as np
    from fractions import Fraction

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=Fraction(fps).limit_denominator(1001))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        audio = container.add_stream("aac", rate=48000) if with_audio else None
        xs = np.linspace(0, 255, width, dtype=np.float32)
        for index in range(frame_count):
            pixels = np.empty((height, width, 3), dtype=np.uint8)
            pixels[..., 0] = (xs + index * 7) % 256
            pixels[..., 1] = np.linspace(0, 255, height, dtype=np.float32)[:, None]
            pixels[..., 2] = (index * 11) % 256
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        if audio is not None:
            samples = int(48000 * frame_count / fps)
            tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(samples) / 48000)).astype(np.float32)
            for start in range(0, samples, 1024):
                chunk = np.stack([tone[start:start + 1024]] * 2)
                frame = av.AudioFrame.from_ndarray(chunk, format="fltp", layout="stereo")
                frame.sample_rate = 48000
                frame.pts = start
                for packet in audio.encode(frame):
                    container.mux(packet)
            for packet in audio.encode():
                container.mux(packet)


def swarm_resample_linear(images, fps_in, fps_out):
    """SwarmVideoResampleFPS ("linear") from SwarmUI's SwarmComfyCommon/SwarmVideo.py."""
    count = int(images.shape[0])
    if count <= 1 or math.isclose(fps_in, fps_out):
        return images
    output_count = max(1, round(count / fps_in * fps_out))
    positions = torch.arange(output_count, dtype=torch.float64) / fps_out * fps_in
    lower = torch.clamp(positions.floor().long(), 0, count - 1)
    upper = torch.clamp(lower + 1, 0, count - 1)
    weight = (positions - lower.to(torch.float64)).to(images.dtype).view(-1, 1, 1, 1)
    return ((1.0 - weight) * images[lower] + weight * images[upper]).contiguous()


@unittest.skipUnless(HAS_COMFY, "needs ComfyUI on the path")
class SwarmLoaderParityTests(unittest.TestCase):
    def old_chain(self, data, start, duration, size):
        import io

        import comfy.utils
        from comfy_api.latest._input_impl.video_types import VideoFromFile

        components = VideoFromFile(io.BytesIO(data)).as_trimmed(start, duration, strict_duration=False).get_components()
        frames = swarm_resample_linear(components.images, float(components.frame_rate), 24.0)
        frames = comfy.utils.common_upscale(frames.movedim(-1, 1), size[0], size[1], "area", "disabled").movedim(1, -1)
        return frames, components.audio

    def check(self, path, start, duration, width, height):
        data = Path(path).read_bytes()
        payload = base64.b64encode(data).decode()
        frames, audio = gallery.SECoursesLoadVideoB64AtSize().load(payload, start, duration, 24.0, width, height)
        again, _ = gallery.SECoursesLoadVideoB64AtSize().load(payload, start, duration, 24.0, width, height)
        old, old_audio = self.old_chain(data, start, duration, (frames.shape[2], frames.shape[1]))
        self.assertTrue(torch.equal(frames, again), "decoding must be deterministic")
        self.assertEqual(frames.shape, old.shape)
        self.assertLess(float((frames - old).abs().max()), 1e-4)
        if old_audio is None:
            self.assertIsNone(audio)
        else:
            self.assertEqual(audio["sample_rate"], old_audio["sample_rate"])
            self.assertTrue(torch.equal(audio["waveform"], old_audio["waveform"]))
        return frames

    def test_60_fps_clip_matches_the_old_chain_at_the_output_size(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clip.mp4"
            write_video(path, 640, 360, 60, 120)
            frames = self.check(path, 0.0, 60.0, 320, 192)
            trimmed = self.check(path, 0.55, 1.0, 320, 192)
        self.assertEqual(tuple(frames.shape), (48, 186, 330, 3))  # 640x360 fitted to 320*192 pixels, 16:9 kept
        self.assertEqual(trimmed.shape[0], 24)

    def test_width_not_a_multiple_of_32_and_no_frame_rate_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "odd.mp4"
            write_video(path, 600, 338, 24, 30, with_audio=False)
            frames = self.check(path, 0.0, 60.0, 256, 160)
        self.assertEqual(frames.shape[0], 30)

    def test_a_small_clip_is_never_enlarged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "small.mp4"
            write_video(path, 320, 192, 24, 10, with_audio=False)
            frames = self.check(path, 0.0, 60.0, 1344, 768)
        self.assertEqual(tuple(frames.shape[1:3]), (192, 320))


class GalleryVideoSizeTests(unittest.TestCase):
    def test_pack_carries_the_switch_and_it_defaults_on(self):
        node = gallery.SECoursesReferenceGallery()
        manifest = '{"images": [], "videos": [], "audios": []}'
        packs, *_ = node.collect("", manifest, 24.0, 15.0)
        self.assertTrue(packs[0]["downscale_videos"])
        packs, *_ = node.collect("", manifest, 24.0, 15.0, downscale_videos=False)
        self.assertFalse(packs[0]["downscale_videos"])
        self.assertIn("downscale_videos", gallery.SECoursesReferenceGallery.INPUT_TYPES()["optional"])
        self.assertNotEqual(
            gallery.SECoursesReferenceGallery.IS_CHANGED("", manifest, 24.0, 15.0, downscale_videos=True),
            gallery.SECoursesReferenceGallery.IS_CHANGED("", manifest, 24.0, 15.0, downscale_videos=False),
        )

    def test_reference_videos_follow_the_output_area(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clip.mp4"
            write_video(path, 1920, 1080, 24, 10, with_audio=False)
            entry = {"file": "reference_gallery/clip.mp4", "name": "clip.mp4"}
            with mock.patch.object(gallery, "_resolve_reference_entry", return_value=str(path)):
                full, = gallery._prepare_video_references([entry], 24, 15.0, 124, byte_budget=8 << 30)
                small, = gallery._prepare_video_references([entry], 24, 15.0, 124, byte_budget=8 << 30,
                                                           max_pixels=640 * 352)
                large, = gallery._prepare_video_references([entry], 24, 15.0, 124, byte_budget=8 << 30,
                                                           max_pixels=1920 * 1088)
        self.assertEqual(full["target_size"], (1344, 768))   # the previous canvas
        self.assertEqual(small["target_size"], (576, 320))  # the 32 px grid size closest to 16:9 within 640*352
        self.assertEqual(large["target_size"], (1344, 768))  # never above the previous canvas


class GenjutsuClipSizeTests(unittest.TestCase):
    def prepare(self, path, width, height, downscale, match_source_aspect=True):
        seen = {}

        def guide(frames, *_args):
            seen["frames"] = tuple(frames.shape)
            return frames[:, :8, :8], torch.zeros(frames.shape[:3], dtype=torch.bool)

        references = {
            "videos": [{"file": "reference_gallery/clip.mp4", "name": "clip.mp4"}],
            "images": [{"file": "reference_gallery/place.png", "name": "place.png"}],
            "video_fps": 24.0, "max_seconds": 15.0, "prompt": "", "downscale_videos": downscale,
        }
        with mock.patch.object(genjutsu_nodes, "_resolve_reference_entry", return_value=str(path)), \
                mock.patch.object(genjutsu_nodes, "method_guide", side_effect=guide), \
                mock.patch.object(genjutsu_nodes, "guide_previews", return_value=(None, None)):
            result = genjutsu_nodes.SECoursesH3Genjutsu().prepare(
                references, None, None, "person", genjutsu_nodes.BACKGROUNDS[0], 0.15, 0.5, 0, width, height,
                match_source_aspect, method=genjutsu_nodes.BACKGROUND_REPLACE)
        return seen["frames"], (result[2], result[3])

    def test_clip_is_decoded_at_the_output_area_and_the_output_size_does_not_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clip.mp4"
            write_video(path, 1920, 1080, 24, 22, with_audio=False)
            for width, height, match in ((1344, 768, True), (832, 480, True), (640, 640, True), (832, 480, False)):
                old_frames, old_size = self.prepare(path, width, height, False, match)
                new_frames, new_size = self.prepare(path, width, height, True, match)
                self.assertEqual(new_size, old_size)
                self.assertEqual(old_frames[1:3], (768, 1344))
                self.assertLessEqual(new_frames[1] * new_frames[2], new_size[0] * new_size[1])
                self.assertEqual(new_frames[0], old_frames[0])


if __name__ == "__main__":
    unittest.main()
