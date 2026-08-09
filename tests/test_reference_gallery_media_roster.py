import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


def _entries(kind, count):
    return [
        {"file": f"reference_gallery/{kind}{number}.bin", "name": f"{kind}{number}.bin"}
        for number in range(1, count + 1)
    ]


class SelectPromptMediaReferencesTests(unittest.TestCase):
    def test_image_roster_selects_mentioned_in_first_mention_order(self):
        images = _entries("image", 12)
        selected, number_map = gallery.select_prompt_media_references(
            "start from @image10, then match @image2", images, 9, "image"
        )
        self.assertEqual([entry["name"] for entry in selected], ["image10.bin", "image2.bin"])
        self.assertEqual(number_map, {10: 1, 2: 2})

    def test_image_mentions_beyond_nine_keep_first_nine(self):
        images = _entries("image", 11)
        prompt = " ".join(f"@image{number}" for number in (4, 1, 11, 2, 5, 6, 7, 8, 9, 3))
        selected, number_map = gallery.select_prompt_media_references(prompt, images, 9, "image")
        self.assertEqual(
            [entry["name"] for entry in selected],
            [f"image{number}.bin" for number in (4, 1, 11, 2, 5, 6, 7, 8, 9)],
        )
        self.assertNotIn(3, number_map)
        self.assertEqual(number_map[4], 1)
        self.assertEqual(number_map[9], 9)

    def test_video_roster_selects_mentioned_with_aliases(self):
        videos = _entries("video", 5)
        selected, number_map = gallery.select_prompt_media_references(
            "move like @video4 and cut like @vid2", videos, 3, "video"
        )
        self.assertEqual([entry["name"] for entry in selected], ["video4.bin", "video2.bin"])
        self.assertEqual(number_map, {4: 1, 2: 2})

    def test_galleries_within_caps_pass_through_unchanged(self):
        images = _entries("image", 9)
        videos = _entries("video", 3)
        self.assertEqual(
            gallery.select_prompt_media_references("no tokens", images, 9, "image"),
            (images, None),
        )
        self.assertEqual(
            gallery.select_prompt_media_references("no tokens", videos, 3, "video"),
            (videos, None),
        )

    def test_no_mentions_attaches_nothing(self):
        videos = _entries("video", 4)
        selected, number_map = gallery.select_prompt_media_references(
            "a prompt that only uses @image1 and @audio1", videos, 3, "video"
        )
        self.assertEqual(selected, [])
        self.assertEqual(number_map, {})

    def test_tokens_of_other_modalities_are_ignored(self):
        images = _entries("image", 10)
        selected, number_map = gallery.select_prompt_media_references(
            "@video2 moves while @image3 stands and @audio2 speaks", images, 9, "image"
        )
        self.assertEqual([entry["name"] for entry in selected], ["image3.bin"])
        self.assertEqual(number_map, {3: 1})

    def test_audio_wrapper_still_delegates(self):
        audios = _entries("audio", 5)
        selected, number_map = gallery.select_prompt_audio_references(
            "use @audio5 now", audios, 3
        )
        self.assertEqual([entry["name"] for entry in selected], ["audio5.bin"])
        self.assertEqual(number_map, {5: 1})


class TranslateWithMediaMapsTests(unittest.TestCase):
    def test_image_map_renumbers_pictures(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "match @image10 beside @image2", 2, 0, 0, 0,
                image_number_map={10: 1, 2: 2},
            ),
            "match <Picture 1> beside <Picture 2>",
        )

    def test_video_map_renumbers_and_audio_offset_follows_selection(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "move like @video4 with the voice of @audio5", 0, 1, 1, 1,
                audio_number_map={5: 1}, video_number_map={4: 1},
            ),
            "move like <Video 1> with the voice of <Audio 2>",
        )

    def test_unmapped_tokens_are_omitted_like_stale_tokens(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "keep @image7 and @image2 here", 1, 0, 0, 0,
                image_number_map={2: 1},
            ),
            "keep and <Picture 1> here",
        )

    def test_audio_only_mode_maps_videos_onto_audio_labels(self):
        self.assertEqual(
            gallery.translate_audio_only_reference_tokens(
                "@video4 delivery with @audio5 tone", 0, 1, 1,
                audio_number_map={5: 1}, video_number_map={4: 1},
            ),
            "<Audio 1> delivery with <Audio 2> tone",
        )

    def test_without_maps_behavior_is_unchanged(self):
        self.assertEqual(
            gallery.translate_reference_tokens("a photo of @image1 and @image2", 2, 0, 0, 0),
            "a photo of <Picture 1> and <Picture 2>",
        )
        self.assertEqual(
            gallery.translate_reference_tokens("play @audio1 loud", 0, 2, 1, 2),
            "play <Audio 3> loud",
        )


@unittest.skipUnless((COMFY_ROOT / "comfy_extras" / "nodes_minimax_h3.py").exists(), "ComfyUI checkout not available")
class LargeMediaRosterIntegrationTests(unittest.TestCase):
    def _fakes(self):
        import torch

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

        return FakeVideoVAE(), FakeAudioVAE(), FakeClip()

    def test_twelve_image_roster_attaches_only_the_mentioned_two(self):
        import torch

        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

        video_vae, audio_vae, clip = self._fakes()
        pack = {
            "version": 2,
            "prompt": "Match @image10 and @image2 side by side",
            "video_fps": 24.0,
            "max_seconds": 15.0,
            "images": _entries("image", 12),
            "videos": [],
            "audios": [],
        }
        with (
            mock.patch.object(gallery, "_resolve_reference_path", side_effect=lambda path: path),
            mock.patch.object(gallery, "_oriented_image_dimensions", return_value=(640, 360, 1)),
            mock.patch.object(
                gallery, "_load_reference_image",
                return_value=torch.zeros((1, 96, 160, 3), dtype=torch.float32),
            ) as load_image,
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
                clip=clip, vae=video_vae, audio_vae=audio_vae, references=pack,
                width=640, height=384, length=22, ref_image_size="match",
            )

        self.assertEqual(clip.prompt, "Match <Picture 1> and <Picture 2> side by side")
        self.assertEqual(clip.item_types, ["image", "image"])
        self.assertEqual(len(video_vae.encoded_shapes), 2)
        self.assertEqual(
            [call.args[0] for call in load_image.call_args_list],
            ["reference_gallery/image10.bin", "reference_gallery/image2.bin"],
        )
        self.assertEqual(latent, {"samples": "test"})
        self.assertEqual(len(conditioning), 1)

    def test_combined_rosters_attach_only_mentions_across_modalities(self):
        import torch

        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

        video_vae, audio_vae, clip = self._fakes()
        fake_audio = {
            "waveform": torch.zeros((1, 2, 32000), dtype=torch.float32),
            "sample_rate": 32000,
        }
        pack = {
            "version": 2,
            "prompt": "Keep @image10, move like @video4, speak like @audio5",
            "video_fps": 24.0,
            "max_seconds": 15.0,
            "images": _entries("image", 10),
            "videos": _entries("video", 4),
            "audios": _entries("audio", 5),
        }
        with (
            mock.patch.object(gallery, "_resolve_reference_path", side_effect=lambda path: path),
            mock.patch.object(gallery, "_oriented_image_dimensions", return_value=(640, 360, 1)),
            mock.patch.object(
                gallery, "_load_reference_image",
                return_value=torch.zeros((1, 96, 160, 3), dtype=torch.float32),
            ),
            mock.patch.object(gallery, "_video_metadata", return_value=(640, 360, True)),
            mock.patch.object(
                gallery, "_decode_video_frames",
                return_value=torch.zeros((22, 96, 160, 3), dtype=torch.uint8),
            ) as decode_frames,
            mock.patch.object(
                gallery, "_decode_video_audio", return_value=fake_audio
            ) as decode_video_audio,
            mock.patch.object(
                gallery, "_load_reference_audio", return_value=fake_audio
            ) as load_audio,
            mock.patch.object(
                native_h3,
                "_empty_av_latent",
                side_effect=lambda _width, _height, length: (
                    {"samples": "test"},
                    gallery._aligned_frame_count(length),
                ),
            ),
        ):
            conditioning, _latent = gallery.SECoursesMiniMaxH3References().encode(
                clip=clip, vae=video_vae, audio_vae=audio_vae, references=pack,
                width=640, height=384, length=124, ref_image_size="match",
            )

        # The selected video carries audio, so the standalone audio label lands
        # one slot past the soundtrack: <Picture 1>, <Video 1>, <Audio 2>.
        self.assertEqual(
            clip.prompt,
            "Keep <Picture 1>, move like <Video 1>, speak like <Audio 2>",
        )
        self.assertEqual(clip.item_types, ["image", "audio", "video", "audio"])
        # Video VAE encodes the image and the video frames; audio VAE encodes
        # the soundtrack and the standalone audio.
        self.assertEqual(len(video_vae.encoded_shapes), 2)
        self.assertEqual(len(audio_vae.encoded_shapes), 2)
        self.assertEqual(decode_frames.call_args.args[0], "reference_gallery/video4.bin")
        self.assertEqual(decode_video_audio.call_args.args[0], "reference_gallery/video4.bin")
        self.assertEqual(load_audio.call_args.args[0], "reference_gallery/audio5.bin")
        self.assertEqual(len(conditioning), 1)

    def test_audio_only_mode_uses_selected_video_soundtrack(self):
        import torch

        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

        video_vae, audio_vae, clip = self._fakes()
        fake_audio = {
            "waveform": torch.zeros((1, 2, 32000), dtype=torch.float32),
            "sample_rate": 32000,
        }
        pack = {
            "version": 2,
            "prompt": "Narrate over @video4 with the tone of @audio5",
            "video_fps": 24.0,
            "max_seconds": 15.0,
            "images": [],
            "videos": _entries("video", 4),
            "audios": _entries("audio", 5),
        }
        with (
            mock.patch.object(gallery, "_resolve_reference_path", side_effect=lambda path: path),
            mock.patch.object(gallery, "_video_metadata", return_value=(640, 360, True)),
            mock.patch.object(gallery, "_decode_video_audio", return_value=fake_audio) as decode_video_audio,
            mock.patch.object(gallery, "_load_reference_audio", return_value=fake_audio) as load_audio,
            mock.patch.object(
                native_h3,
                "_empty_av_latent",
                side_effect=lambda _width, _height, length: (
                    {"samples": "test"},
                    gallery._aligned_frame_count(length),
                ),
            ),
        ):
            _conditioning, _latent = gallery.SECoursesMiniMaxH3References().encode(
                clip=clip, vae=video_vae, audio_vae=audio_vae, references=pack,
                width=32, height=32, length=124, ref_image_size="match",
                audio_only_mode=True,
            )

        self.assertEqual(clip.prompt, "Narrate over <Audio 1> with the tone of <Audio 2>")
        self.assertEqual(clip.item_types, ["audio", "audio"])
        self.assertEqual(video_vae.encoded_shapes, [])
        self.assertEqual(len(audio_vae.encoded_shapes), 2)
        self.assertEqual(decode_video_audio.call_args.args[0], "reference_gallery/video4.bin")
        self.assertEqual(load_audio.call_args.args[0], "reference_gallery/audio5.bin")


if __name__ == "__main__":
    unittest.main()
