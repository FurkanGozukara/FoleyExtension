import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


def _audios(count):
    return [
        {"file": f"reference_gallery/voice{number}.wav", "name": f"voice{number}.wav"}
        for number in range(1, count + 1)
    ]


class SelectPromptAudioReferencesTests(unittest.TestCase):
    def test_gallery_within_cap_passes_through_unchanged(self):
        audios = _audios(3)
        selected, number_map = gallery.select_prompt_audio_references(
            "no tokens at all", audios, 3
        )
        self.assertIs(selected, audios)
        self.assertIsNone(number_map)

    def test_large_gallery_attaches_mentioned_in_first_mention_order(self):
        audios = _audios(5)
        selected, number_map = gallery.select_prompt_audio_references(
            "voice of @audio5 first, then @audio2 replies", audios, 3
        )
        self.assertEqual([entry["name"] for entry in selected], ["voice5.wav", "voice2.wav"])
        self.assertEqual(number_map, {5: 1, 2: 2})

    def test_aliases_and_repeat_mentions_count_once(self):
        audios = _audios(4)
        selected, number_map = gallery.select_prompt_audio_references(
            "@sound3 speaks, @audio3 again, then @aud1", audios, 3
        )
        self.assertEqual([entry["name"] for entry in selected], ["voice3.wav", "voice1.wav"])
        self.assertEqual(number_map, {3: 1, 1: 2})

    def test_more_than_cap_mentions_keep_first_three(self):
        audios = _audios(5)
        selected, number_map = gallery.select_prompt_audio_references(
            "@audio4 then @audio1 then @audio5 then @audio2", audios, 3
        )
        self.assertEqual(
            [entry["name"] for entry in selected],
            ["voice4.wav", "voice1.wav", "voice5.wav"],
        )
        self.assertEqual(number_map, {4: 1, 1: 2, 5: 3})

    def test_out_of_range_mentions_are_ignored(self):
        audios = _audios(4)
        selected, number_map = gallery.select_prompt_audio_references(
            "@audio9 is missing but @audio2 exists", audios, 3
        )
        self.assertEqual([entry["name"] for entry in selected], ["voice2.wav"])
        self.assertEqual(number_map, {2: 1})

    def test_no_mentions_attaches_nothing(self):
        audios = _audios(5)
        selected, number_map = gallery.select_prompt_audio_references(
            "a prompt with only @image1 and @video1", audios, 3
        )
        self.assertEqual(selected, [])
        self.assertEqual(number_map, {})


class TranslateWithAudioNumberMapTests(unittest.TestCase):
    def test_mapped_audio_token_renumbers_past_video_soundtracks(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "use @audio5 now", 0, 2, 1, 2, audio_number_map={5: 1}
            ),
            "use <Audio 3> now",
        )

    def test_unmapped_audio_token_is_omitted_like_a_stale_token(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "use @audio5 and @audio4 now", 0, 0, 1, 0, audio_number_map={5: 1}
            ),
            "use <Audio 1> and now",
        )

    def test_map_does_not_affect_image_or_video_tokens(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "keep @image1 with @video1 and @audio7", 1, 1, 1, 1, audio_number_map={7: 1}
            ),
            "keep <Picture 1> with <Video 1> and <Audio 2>",
        )

    def test_audio_only_mode_maps_past_video_soundtracks(self):
        self.assertEqual(
            gallery.translate_audio_only_reference_tokens(
                "@video1 delivery with @audio5 tone", 0, 1, 1, audio_number_map={5: 1}
            ),
            "<Audio 1> delivery with <Audio 2> tone",
        )

    def test_without_map_behavior_is_unchanged(self):
        self.assertEqual(
            gallery.translate_reference_tokens("play @audio1 loud", 0, 2, 1, 2),
            "play <Audio 3> loud",
        )


@unittest.skipUnless((COMFY_ROOT / "comfy_extras" / "nodes_minimax_h3.py").exists(), "ComfyUI checkout not available")
class LargeAudioRosterIntegrationTests(unittest.TestCase):
    def _run_encode(self, prompt, audio_only_mode=False):
        import torch

        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

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
            "prompt": prompt,
            "video_fps": 24.0,
            "max_seconds": 15.0,
            "images": [],
            "videos": [],
            "audios": _audios(5),
        }
        clip = FakeClip()
        audio_vae = FakeAudioVAE()
        with (
            mock.patch.object(gallery, "_resolve_reference_path", side_effect=lambda path: path),
            mock.patch.object(
                gallery, "_load_reference_audio", return_value=fake_audio
            ) as load_reference_audio,
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
                vae=object(),
                audio_vae=audio_vae,
                references=pack,
                width=32,
                height=32,
                length=124,
                ref_image_size="match",
                audio_only_mode=audio_only_mode,
            )
        loaded_paths = [call.args[0] for call in load_reference_audio.call_args_list]
        return clip, audio_vae, loaded_paths, conditioning, latent

    def test_five_audio_roster_attaches_only_the_mentioned_two(self):
        clip, audio_vae, loaded_paths, conditioning, latent = self._run_encode(
            "Use @audio5 for the voice and @audio2 for the reply"
        )
        self.assertEqual(clip.prompt, "Use <Audio 1> for the voice and <Audio 2> for the reply")
        self.assertEqual(clip.item_types, ["audio", "audio"])
        self.assertEqual(len(audio_vae.encoded_shapes), 2)
        self.assertEqual(
            loaded_paths,
            ["reference_gallery/voice5.wav", "reference_gallery/voice2.wav"],
        )
        self.assertEqual(latent, {"samples": "test"})
        self.assertEqual(len(conditioning), 1)

    def test_five_audio_roster_in_audio_only_mode(self):
        clip, audio_vae, loaded_paths, _conditioning, _latent = self._run_encode(
            "Narrate like @audio4 with a hint of @audio1", audio_only_mode=True
        )
        self.assertEqual(clip.prompt, "Narrate like <Audio 1> with a hint of <Audio 2>")
        self.assertEqual(clip.item_types, ["audio", "audio"])
        self.assertEqual(len(audio_vae.encoded_shapes), 2)
        self.assertEqual(
            loaded_paths,
            ["reference_gallery/voice4.wav", "reference_gallery/voice1.wav"],
        )

    def test_five_audio_roster_with_no_mentions_attaches_none(self):
        clip, audio_vae, loaded_paths, _conditioning, _latent = self._run_encode(
            "A quiet scene with no narration"
        )
        self.assertEqual(clip.prompt, "A quiet scene with no narration")
        self.assertEqual(clip.item_types, [])
        self.assertEqual(audio_vae.encoded_shapes, [])
        self.assertEqual(loaded_paths, [])


if __name__ == "__main__":
    unittest.main()
