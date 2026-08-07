import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


@unittest.skipUnless((COMFY_ROOT / "comfy_extras" / "nodes_minimax_h3.py").exists(), "ComfyUI checkout not available")
class NativeMiniMaxH3IntegrationTests(unittest.TestCase):
    def test_descriptor_pack_runs_through_native_h3_reference_node(self):
        import av
        import numpy as np
        import torch
        from PIL import Image

        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

        class FakeVAE:
            def __init__(self):
                self.encoded_shapes = []

            def encode(self, pixels):
                self.encoded_shapes.append(tuple(pixels.shape))
                return torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32)

        class FakeClip:
            def __init__(self):
                self.prompt = None
                self.items = None

            def tokenize(self, prompt, minimax_ref_items):
                self.prompt = prompt
                self.items = [
                    {"type": item["type"], "shape": tuple(item["data"].shape) if "data" in item else None}
                    for item in minimax_ref_items
                ]
                return {"tokens": []}

            def encode_from_tokens_scheduled(self, _tokens):
                return [[torch.zeros((1, 1, 1)), {}]]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "wide.jpg"
            video_path = temp_path / "wide.mp4"
            with Image.new("RGB", (1800, 1200), (32, 96, 180)) as image:
                image.save(image_path, quality=85)

            with av.open(str(video_path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=24)
                stream.width = 640
                stream.height = 360
                stream.pix_fmt = "yuv420p"
                for index in range(24):
                    pixels = np.empty((360, 640, 3), dtype=np.uint8)
                    pixels[..., 0] = index * 5
                    pixels[..., 1] = 70
                    pixels[..., 2] = 150
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            paths = {
                "reference_gallery/wide.jpg": str(image_path),
                "reference_gallery/wide.mp4": str(video_path),
            }
            pack = {
                "version": 2,
                "prompt": "Match @image1, then move like @video1",
                "video_fps": 24.0,
                "max_seconds": 1.0,
                "images": [{"file": "reference_gallery/wide.jpg", "name": "wide.jpg"}],
                "videos": [{"file": "reference_gallery/wide.mp4", "name": "wide.mp4"}],
                "audios": [],
            }
            clip = FakeClip()
            vae = FakeVAE()
            node = gallery.SECoursesMiniMaxH3References()
            with (
                mock.patch.object(gallery, "_resolve_reference_path", side_effect=paths.__getitem__),
                mock.patch.object(
                    native_h3,
                    "_empty_av_latent",
                    side_effect=lambda _width, _height, length: ({"samples": "test"}, gallery._aligned_frame_count(length)),
                ),
                mock.patch.object(
                    native_h3.node_helpers,
                    "conditioning_set_values",
                    side_effect=lambda conditioning, _values: conditioning,
                ),
            ):
                conditioning, latent = node.encode(
                    clip=clip,
                    vae=vae,
                    audio_vae=object(),
                    references=pack,
                    width=640,
                    height=384,
                    length=22,
                    ref_image_size="match",
                )

        self.assertEqual(clip.prompt, "Match <Picture 1>, then move like <Video 1>")
        self.assertEqual([item["type"] for item in clip.items], ["image", "video"])
        self.assertEqual(len(vae.encoded_shapes), 2)
        self.assertEqual(latent, {"samples": "test"})
        self.assertEqual(len(conditioning), 1)

        for shape, source_ratio in zip(vae.encoded_shapes, (1.5, 16 / 9)):
            target_ratio = shape[2] / shape[1]
            self.assertLessEqual(abs(target_ratio / source_ratio - 1.0), 0.02)
            self.assertEqual(shape[1] % gallery.CANVAS_MULTIPLE, 0)
            self.assertEqual(shape[2] % gallery.CANVAS_MULTIPLE, 0)

    def test_mixed_pack_keeps_image_video_soundtrack_and_standalone_audio(self):
        import av
        import numpy as np
        import torch
        from PIL import Image

        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

        class FakeVideoVAE:
            def __init__(self):
                self.encoded_shapes = []

            def encode(self, pixels):
                self.encoded_shapes.append(tuple(pixels.shape))
                return torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32)

        class FakeAudioVAE:
            audio_sample_rate = 32000

            def __init__(self):
                self.encoded_shapes = []

            def encode(self, waveform):
                self.encoded_shapes.append(tuple(waveform.shape))
                return torch.zeros((1, 32, 2, 8), dtype=torch.float32)

        class FakeClip:
            def __init__(self):
                self.prompt = None
                self.item_types = None

            def tokenize(self, prompt, minimax_ref_items):
                self.prompt = prompt
                self.item_types = [item["type"] for item in minimax_ref_items]
                return {"tokens": []}

            def encode_from_tokens_scheduled(self, _tokens):
                return [[torch.zeros((1, 1, 1)), {}]]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "subject.jpg"
            video_path = temp_path / "motion.mp4"
            audio_path = temp_path / "voice.wav"
            with Image.new("RGB", (960, 640), (64, 110, 170)) as image:
                image.save(image_path, quality=85)

            with av.open(str(video_path), mode="w") as container:
                stream = container.add_stream("mpeg4", rate=24)
                stream.width = 640
                stream.height = 360
                stream.pix_fmt = "yuv420p"
                for index in range(24):
                    pixels = np.empty((360, 640, 3), dtype=np.uint8)
                    pixels[..., 0] = index * 5
                    pixels[..., 1] = 80
                    pixels[..., 2] = 145
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            paths = {
                "reference_gallery/subject.jpg": str(image_path),
                "reference_gallery/motion.mp4": str(video_path),
                "reference_gallery/voice.wav": str(audio_path),
            }
            pack = {
                "version": 2,
                "prompt": "Keep @image1, move like @video1, and use @audio1 for the voice",
                "video_fps": 24.0,
                "max_seconds": 1.0,
                "images": [{"file": "reference_gallery/subject.jpg", "name": "subject.jpg"}],
                "videos": [{"file": "reference_gallery/motion.mp4", "name": "motion.mp4"}],
                "audios": [{"file": "reference_gallery/voice.wav", "name": "voice.wav"}],
            }
            fake_audio = {
                "waveform": torch.zeros((1, 2, 32000), dtype=torch.float32),
                "sample_rate": 32000,
            }
            clip = FakeClip()
            video_vae = FakeVideoVAE()
            audio_vae = FakeAudioVAE()
            node = gallery.SECoursesMiniMaxH3References()
            with (
                mock.patch.object(gallery, "_resolve_reference_path", side_effect=paths.__getitem__),
                mock.patch.object(gallery, "_video_metadata", return_value=(640, 360, True)),
                mock.patch.object(gallery, "_decode_video_audio", return_value=fake_audio),
                mock.patch.object(
                    native_h3,
                    "_empty_av_latent",
                    side_effect=lambda _width, _height, length: ({"samples": "test"}, gallery._aligned_frame_count(length)),
                ),
                mock.patch.object(
                    native_h3.node_helpers,
                    "conditioning_set_values",
                    side_effect=lambda conditioning, _values: conditioning,
                ),
            ):
                conditioning, latent = node.encode(
                    clip=clip,
                    vae=video_vae,
                    audio_vae=audio_vae,
                    references=pack,
                    width=640,
                    height=384,
                    length=22,
                    ref_image_size="match",
                )

        self.assertEqual(
            clip.prompt,
            "Keep <Picture 1>, move like <Video 1>, and use <Audio 2> for the voice",
        )
        self.assertEqual(clip.item_types, ["image", "audio", "video", "audio"])
        self.assertEqual(len(video_vae.encoded_shapes), 2)
        self.assertEqual(len(audio_vae.encoded_shapes), 2)
        self.assertEqual(latent, {"samples": "test"})
        self.assertEqual(len(conditioning), 1)

    def test_audio_only_mode_uses_video_soundtrack_without_video_frames(self):
        import torch

        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

        class FakeVideoVAE:
            def __init__(self):
                self.encoded_shapes = []

            def encode(self, pixels):
                self.encoded_shapes.append(tuple(pixels.shape))
                return torch.zeros((1, 1, 2, 1, 1), dtype=torch.float32)

        class FakeAudioVAE:
            audio_sample_rate = 32000

            def __init__(self):
                self.encoded_shapes = []

            def encode(self, waveform):
                self.encoded_shapes.append(tuple(waveform.shape))
                return torch.zeros((1, 32, 2, 8), dtype=torch.float32)

        class FakeClip:
            def __init__(self):
                self.prompt = None
                self.item_types = None

            def tokenize(self, prompt, minimax_ref_items):
                self.prompt = prompt
                self.item_types = [item["type"] for item in minimax_ref_items]
                return {"tokens": []}

            def encode_from_tokens_scheduled(self, _tokens):
                return [[torch.zeros((1, 1, 1)), {}]]

        fake_audio = {
            "waveform": torch.zeros((1, 2, 32000), dtype=torch.float32),
            "sample_rate": 32000,
        }
        pack = {
            "version": 2,
            "prompt": "Use @video1 for delivery and @audio1 for tone",
            "video_fps": 24.0,
            "max_seconds": 30.0,
            "images": [],
            "videos": [{"file": "reference_gallery/source.mp4", "name": "source.mp4"}],
            "audios": [{"file": "reference_gallery/voice.wav", "name": "voice.wav"}],
        }
        clip = FakeClip()
        video_vae = FakeVideoVAE()
        audio_vae = FakeAudioVAE()
        with (
            mock.patch.object(gallery, "_resolve_reference_path", side_effect=lambda path: path),
            mock.patch.object(gallery, "_video_metadata", return_value=(1920, 1080, True)),
            mock.patch.object(gallery, "_decode_video_audio", return_value=fake_audio) as decode_video_audio,
            mock.patch.object(gallery, "_load_reference_audio", return_value=fake_audio) as load_reference_audio,
            mock.patch.object(
                native_h3,
                "_empty_av_latent",
                side_effect=lambda _width, _height, length: (
                    {"samples": "test"},
                    gallery._aligned_frame_count(length),
                ),
            ),
        ):
            conditioning, latent = gallery.SECoursesMiniMaxH3References().encode(
                clip=clip,
                vae=video_vae,
                audio_vae=audio_vae,
                references=pack,
                width=32,
                height=32,
                length=770,
                ref_image_size="max",
                audio_only_mode=True,
            )

        self.assertEqual(clip.prompt, "Use <Audio 1> for delivery and <Audio 2> for tone")
        self.assertEqual(clip.item_types, ["audio", "audio"])
        self.assertEqual(video_vae.encoded_shapes, [])
        self.assertEqual(len(audio_vae.encoded_shapes), 2)
        self.assertAlmostEqual(decode_video_audio.call_args.args[1], 30.0, delta=0.1)
        self.assertEqual(load_reference_audio.call_args.args[1], 30.0)
        self.assertEqual(latent, {"samples": "test"})
        self.assertEqual(len(conditioning), 1)


if __name__ == "__main__":
    unittest.main()
