import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class ReferenceGalleryMediaSafetyTests(unittest.TestCase):
    def assertAspectRatio(self, source_size, target_size, tolerance=0.025):
        source_ratio = source_size[0] / source_size[1]
        target_ratio = target_size[0] / target_size[1]
        self.assertLessEqual(abs(target_ratio / source_ratio - 1.0), tolerance)

    def test_fit_dimensions_preserves_common_aspect_ratios(self):
        cases = [
            (7680, 4320),
            (4320, 7680),
            (6000, 6000),
            (12000, 1000),
            (1000, 12000),
            (4032, 3024),
        ]
        for source_size in cases:
            with self.subTest(source_size=source_size):
                target_size = gallery._fit_dimensions(
                    *source_size,
                    gallery.VIDEO_DECODE_AREA_CAP,
                )
                self.assertLessEqual(target_size[0] * target_size[1], gallery.VIDEO_DECODE_AREA_CAP)
                self.assertEqual(target_size[0] % gallery.CANVAS_MULTIPLE, 0)
                self.assertEqual(target_size[1] % gallery.CANVAS_MULTIPLE, 0)
                self.assertLessEqual(target_size[0], source_size[0])
                self.assertLessEqual(target_size[1], source_size[1])
                self.assertAspectRatio(source_size, target_size)

    def test_shared_image_budget_preserves_ratios_and_stays_bounded(self):
        sizes = [(4096, 3072), (3072, 4096), (7680, 4320)] * 3
        byte_budget = 64 * 1024 * 1024
        bounded = gallery._apply_total_pixel_budget(sizes, byte_budget)

        self.assertLessEqual(
            sum(width * height for width, height in bounded) * gallery.RGB_FLOAT_BYTES_PER_PIXEL,
            byte_budget,
        )
        for source_size, target_size in zip(sizes, bounded):
            self.assertAspectRatio(source_size, target_size)

    def test_24_megapixel_jpeg_is_downscaled_before_float_tensor(self):
        import torch
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "24mp.jpg"
            with Image.new("RGB", (6000, 4000), (27, 91, 173)) as image:
                image.save(path, quality=85)

            tensor = gallery._load_reference_image(str(path), 960, 640)

        self.assertEqual(tuple(tensor.shape), (1, 640, 960, 3))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertAspectRatio((6000, 4000), (tensor.shape[2], tensor.shape[1]), tolerance=0.001)

    def test_exif_rotation_is_applied_without_stretching(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rotated.jpg"
            with Image.new("RGB", (1200, 800), (120, 30, 80)) as image:
                exif = image.getexif()
                exif[274] = 6
                image.save(path, exif=exif)

            self.assertEqual(gallery._oriented_image_dimensions(str(path))[:2], (800, 1200))
            tensor = gallery._load_reference_image(str(path), 384, 576)

        self.assertEqual(tuple(tensor.shape), (1, 576, 384, 3))
        self.assertAspectRatio((800, 1200), (tensor.shape[2], tensor.shape[1]), tolerance=0.001)

    def test_image_above_hard_pixel_limit_fails_before_decode(self):
        class HeaderOnlyImage:
            size = (15000, 8000)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def getexif(self):
                return {}

        with mock.patch("PIL.Image.open", return_value=HeaderOnlyImage()):
            with self.assertRaisesRegex(ValueError, r"120\.0 MP.*100 MP"):
                gallery._oriented_image_dimensions("too-large.png")

    def test_video_above_hard_pixel_limit_fails_during_metadata_preflight(self):
        import av

        class FakeContainer:
            def __init__(self):
                codec = type("Codec", (), {"width": 10000, "height": 5000})()
                stream = type("Stream", (), {"codec_context": codec})()
                self.streams = type("Streams", (), {"video": [stream], "audio": []})()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with mock.patch.object(av, "open", return_value=FakeContainer()):
            with self.assertRaisesRegex(ValueError, r"50\.0 MP per frame.*40 MP"):
                gallery._video_metadata("too-large.mp4")

    def test_video_budget_is_based_on_final_usable_frame_count(self):
        entries = [{"file": "8k.mp4", "name": "8K"}]
        with (
            mock.patch.object(gallery, "_resolve_reference_path", return_value="8k.mp4"),
            mock.patch.object(gallery, "_video_metadata", return_value=(7680, 4320, True)),
        ):
            specs = gallery._prepare_video_references(
                entries,
                fps=24,
                max_seconds=15,
                length=124,
                byte_budget=64 * 1024 * 1024,
            )

        spec = specs[0]
        self.assertEqual(spec["max_frames"], 124)
        self.assertLessEqual(
            spec["target_size"][0]
            * spec["target_size"][1]
            * spec["max_frames"]
            * gallery.RGB_FLOAT_BYTES_PER_PIXEL,
            64 * 1024 * 1024,
        )
        self.assertAspectRatio(spec["source_size"], spec["target_size"])

    def test_video_decode_scales_before_stacking_and_preserves_ratio(self):
        import av
        import numpy as np
        import torch

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.mp4"
            with av.open(str(path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=24)
                stream.width = 1280
                stream.height = 720
                stream.pix_fmt = "yuv420p"
                for index in range(12):
                    pixels = np.empty((720, 1280, 3), dtype=np.uint8)
                    pixels[..., 0] = index * 10
                    pixels[..., 1] = 80
                    pixels[..., 2] = 160
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            frames = gallery._decode_video_frames(
                str(path),
                fps_out=24,
                max_frames=8,
                area_cap=128 * 1024,
            )

        self.assertEqual(frames.dtype, torch.uint8)
        self.assertEqual(frames.shape[0], 8)
        self.assertLessEqual(frames.shape[1] * frames.shape[2], 128 * 1024)
        self.assertAspectRatio((1280, 720), (frames.shape[2], frames.shape[1]))

    def test_standalone_audio_decode_stops_at_duration_limit(self):
        import wave

        import numpy as np
        import torch

        sample_rate = 8000
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.wav"
            samples = np.zeros((sample_rate * 2, 2), dtype=np.int16)
            with wave.open(str(path), "wb") as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())

            audio = gallery._load_reference_audio(str(path), max_seconds=1.0)

        self.assertEqual(audio["sample_rate"], sample_rate)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 2, sample_rate))
        self.assertEqual(audio["waveform"].dtype, torch.float32)

    def test_base64_video_soundtrack_loader_respects_longer_user_limit(self):
        import base64

        expected = {"waveform": object(), "sample_rate": 32000}
        with mock.patch.object(gallery, "_decode_video_audio", return_value=expected) as decode:
            output, = gallery.SECoursesLoadVideoAudioB64().load(
                base64.b64encode(b"fake-container").decode("ascii"),
                max_seconds=42.0,
            )

        self.assertIs(output, expected)
        stream, max_seconds = decode.call_args.args
        self.assertEqual(stream.getvalue(), b"fake-container")
        self.assertEqual(max_seconds, 42.0)

    def test_trim_audio_allows_limits_above_fifteen_seconds(self):
        import torch

        audio = {
            "waveform": torch.zeros((1, 2, 400), dtype=torch.float32),
            "sample_rate": 10,
        }
        trimmed, = gallery.SECoursesTrimAudio().trim(audio, max_seconds=30.0)

        self.assertEqual(trimmed["waveform"].shape[-1], 300)
        self.assertEqual(audio["waveform"].shape[-1], 400)

    def test_gallery_pack_keeps_descriptors_and_prompt_without_decoding(self):
        manifest = json.dumps({
            "images": [{"file": "reference_gallery/large.jpg", "name": "large.jpg"}],
            "videos": [{"file": "reference_gallery/large.mp4", "name": "large.mp4"}],
            "audios": [],
        })
        node = gallery.SECoursesReferenceGallery()
        with (
            mock.patch.object(gallery, "_load_reference_image", side_effect=AssertionError("eager image decode")),
            mock.patch.object(gallery, "_decode_video_frames", side_effect=AssertionError("eager video decode")),
        ):
            packs, prompts, active, merge = node.collect("keep @image1 and @video1", manifest, 24, 15)

        pack = packs[0]
        self.assertEqual(pack["version"], 2)
        self.assertEqual(pack["prompt"], "keep @image1 and @video1")
        self.assertEqual(prompts, ["keep @image1 and @video1"])
        self.assertEqual(active, [False])
        self.assertEqual(merge, [False])
        self.assertNotIn("pixels", pack["images"][0])
        self.assertNotIn("frames", pack["videos"][0])

    def test_h3_frame_limit_matches_native_alignment(self):
        self.assertEqual(gallery._usable_video_frame_count(124, 24, 15), 124)
        self.assertEqual(gallery._usable_video_frame_count(362, 24, 15), 345)
        self.assertEqual(gallery._usable_video_frame_count(362, 24, 2), 39)


if __name__ == "__main__":
    unittest.main()
