import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for candidate in (ROOT.parents[1], ROOT.parents[1] / "ComfyUI", ROOT.parents[2] / "ComfyUI"):
    if (candidate / "comfy" / "nested_tensor.py").exists():
        sys.path.insert(0, str(candidate))
        break

import init_audio_nodes as init_audio  # noqa: E402

try:
    import comfy.nested_tensor  # noqa: E402
    HAS_COMFY = True
except Exception:  # pragma: no cover - depends on the checkout layout
    HAS_COMFY = False


class FakeAudioVAE:
    audio_sample_rate = 32000
    downscale_ratio = 800

    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        # ComfyUI hands [B, samples, channels]; one latent frame per 800-sample hop, mean over the hop as content.
        self.encoded.append(tuple(pixels.shape))
        b, samples, channels = pixels.shape
        frames = samples // 800
        window = pixels[:, : frames * 800, :].reshape(b, frames, 800, channels).mean(dim=2)  # [B, T, C]
        latent = window.permute(0, 2, 1).unsqueeze(1).expand(b, 32, channels, frames).clone()  # [B, 32, C, T]
        return latent


def audio(seconds, sample_rate=32000, channels=1, value=0.5):
    return {"waveform": torch.full((1, channels, int(seconds * sample_rate)), value), "sample_rate": sample_rate}


class HelperTests(unittest.TestCase):
    def test_frame_grid_and_seconds(self):
        self.assertEqual([init_audio.align_frame_count(n) for n in (1, 5, 6, 22, 23, 124)], [5, 5, 22, 22, 39, 124])
        self.assertEqual(init_audio.frames_for_seconds(5.0), 124)
        self.assertEqual(init_audio.frames_for_seconds(7.3), 175)  # 175.2 -> 175, already on the 17k+5 grid
        self.assertEqual(init_audio.frames_for_seconds(7.6), 192)
        self.assertEqual(init_audio.frames_for_seconds(0.1), 5)

    def test_normalize_mono_resample_and_downmix(self):
        stereo = init_audio.normalize_waveform(audio(1.0, 16000, 1), 32000)
        self.assertEqual(tuple(stereo.shape), (2, 32000))
        six = init_audio.normalize_waveform(audio(0.5, 32000, 6), 32000)
        self.assertEqual(tuple(six.shape), (2, 16000))
        self.assertTrue(torch.allclose(six, torch.full_like(six, 0.5)))

    def test_fit_waveform_pads_with_silence_or_cuts(self):
        wave = torch.ones((2, 1000))
        padded = init_audio.fit_waveform(wave, 1600)
        self.assertEqual(tuple(padded.shape), (2, 1600))
        self.assertEqual(padded[:, 1000:].abs().sum().item(), 0.0)
        self.assertEqual(tuple(init_audio.fit_waveform(wave, 400).shape), (2, 400))
        self.assertIs(init_audio.fit_waveform(wave, 1000), wave)

    def test_video_frame_count_matches_h3_grid(self):
        self.assertEqual(init_audio.video_frame_count(torch.zeros((1, 24, 2, 4, 4))), 5)
        self.assertEqual(init_audio.video_frame_count(torch.zeros((1, 24, 37, 4, 4))), 124)


class InitAudioLoaderTests(unittest.TestCase):
    def test_disabled_passes_duration_through(self):
        node = init_audio.SECoursesInitAudio()
        self.assertEqual(node.load(init_audio.NO_AUDIO, 6.5), (None, 6.5))
        self.assertEqual(node.IS_CHANGED(init_audio.NO_AUDIO), init_audio.NO_AUDIO)
        self.assertIs(node.VALIDATE_INPUTS(init_audio.NO_AUDIO), True)

    def test_selected_file_drives_duration_by_mode(self):
        loaded = audio(7.3)

        class Output:
            args = (loaded,)

        class FakeLoadAudio:
            @classmethod
            def execute(cls, name):
                return Output()

            @classmethod
            def fingerprint_inputs(cls, name):
                return f"hash:{name}"

            @classmethod
            def validate_inputs(cls, name):
                return True

        node = init_audio.SECoursesInitAudio()
        with mock.patch.object(init_audio, "_load_audio_node", return_value=FakeLoadAudio):
            result, seconds = node.load("voice.wav", 5.0, init_audio.DURATION_MATCH)
            self.assertIs(result, loaded)
            self.assertAlmostEqual(seconds, 7.3, places=5)
            result, seconds = node.load("voice.wav", 5.0, init_audio.DURATION_KEEP)
            self.assertIs(result, loaded)
            self.assertEqual(seconds, 5.0)
            self.assertEqual(node.IS_CHANGED("voice.wav"), "hash:voice.wav")

    def test_picker_lists_extended_audio_and_video_containers(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("voice.ape", "music.m4b", "clip.wmv", "notes.txt"):
                Path(directory, name).touch()
            import folder_paths

            with mock.patch.object(folder_paths, "get_input_directory", return_value=directory):
                choices = init_audio.SECoursesInitAudio.INPUT_TYPES()["required"]["audio"][0]

        self.assertEqual(choices, [init_audio.NO_AUDIO, "clip.wmv", "music.m4b", "voice.ape"])


class FramesTests(unittest.TestCase):
    def test_frames_follow_audio_unless_disabled(self):
        node = init_audio.SECoursesMiniMaxH3AudioFrames()
        self.assertEqual(node.resolve(120), (124,))
        self.assertEqual(node.resolve(120, audio(7.6)), (192,))
        self.assertEqual(node.resolve(120, audio(7.3), match_audio=False), (124,))


class FallbackTests(unittest.TestCase):
    def test_override_wins_and_is_lazy(self):
        node = init_audio.SECoursesAudioFallback()
        decoded, override = audio(1.0), audio(2.0)
        self.assertEqual(node.check_lazy_status(override=override), [])
        self.assertEqual(node.check_lazy_status(), ["audio"])
        self.assertIs(node.pick(decoded, override)[0], override)
        self.assertIs(node.pick(decoded)[0], decoded)


@unittest.skipUnless(HAS_COMFY, "ComfyUI checkout not available")
class ConditioningTests(unittest.TestCase):
    def latent(self, frames=124, width=64, height=32):
        video_t = 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2
        audio_t = round(frames / 24 * 40)
        video = torch.zeros((1, 24, video_t, height // 16, width // 16))
        return {"samples": comfy.nested_tensor.NestedTensor((video, torch.zeros((1, 32, 2, audio_t))))}, audio_t

    def test_passthrough_without_audio(self):
        node = init_audio.SECoursesMiniMaxH3InitAudio()
        positive = [[torch.zeros((1, 4, 8)), {"minimax_keyframes": [{"resolved_frame_index": 0}]}]]
        latent, _ = self.latent()
        out = node.apply(positive, latent, FakeAudioVAE())
        self.assertIs(out[0], positive)
        self.assertIs(out[1], latent)
        self.assertIsNone(out[2])

    def test_lock_and_guide(self):
        node = init_audio.SECoursesMiniMaxH3InitAudio()
        positive = [[torch.zeros((1, 4, 8)), {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": torch.zeros(1)}]}]]
        latent, audio_t = self.latent()
        vae = FakeAudioVAE()
        cond, out, fitted = node.apply(positive, latent, vae, audio(3.0, 16000, 1, 0.25))

        # encoded exactly audio_t hops: no crop, silence padded (3s -> 5.175s)
        self.assertEqual(vae.encoded, [(1, audio_t * 800, 2)])
        video, encoded = out["samples"].unbind()
        self.assertEqual(tuple(encoded.shape), (1, 32, 2, audio_t))
        self.assertTrue(torch.allclose(encoded[..., 5:110], torch.full_like(encoded[..., 5:110], 0.25), atol=1e-3))
        self.assertEqual(encoded[..., 121:].abs().sum().item(), 0.0)  # 3s * 40 = 120 frames of audio, then silence
        video_mask, audio_mask = out["noise_mask"].unbind()
        self.assertTrue(torch.all(video_mask == 1))
        self.assertTrue(torch.all(audio_mask == 0))
        self.assertEqual(tuple(audio_mask.shape), tuple(encoded.shape))
        # original latent dict untouched
        self.assertNotIn("noise_mask", latent)

        keyframes = cond[0][1]["minimax_keyframes"]
        self.assertEqual(len(keyframes), 2)
        self.assertEqual(keyframes[1]["resolved_frame_index"], 0)
        self.assertIs(keyframes[1]["audio_latent"], encoded)
        self.assertEqual(len(positive[0][1]["minimax_keyframes"]), 1)  # input conditioning not mutated

        # muxed audio: 32 kHz stereo cut to the 124-frame video (5.1667 s)
        self.assertEqual(fitted["sample_rate"], 32000)
        self.assertEqual(tuple(fitted["waveform"].shape), (1, 2, round(124 / 24 * 32000)))

    def test_folder_batch_audio_overrides_single_init_audio(self):
        node = init_audio.SECoursesMiniMaxH3InitAudio()
        positive = [[torch.zeros((1, 4, 8)), {}]]
        latent, _ = self.latent(frames=22)
        batch_audio = audio(0.5, value=0.75)
        references = {"init_audio": {"name": "scene.ape", "path": "scene.ape"}}
        with (
            mock.patch("reference_gallery_nodes._resolve_reference_entry", return_value="scene.ape"),
            mock.patch("reference_gallery_nodes._load_reference_audio", return_value=batch_audio) as load,
        ):
            _cond, _latent, fitted = node.apply(
                positive, latent, FakeAudioVAE(), audio(0.5, value=0.1), references=references
            )

        load.assert_called_once()
        self.assertTrue(torch.allclose(fitted["waveform"][..., :100], torch.full_like(fitted["waveform"][..., :100], 0.75)))

    def test_locked_audio_mask_reaches_core_h3_timestep_conditioning(self):
        import comfy.model_base
        import comfy.utils

        positive = [[torch.zeros((1, 4, 8)), {}]]
        latent, _ = self.latent()
        _, out, _ = init_audio.SECoursesMiniMaxH3InitAudio().apply(
            positive, latent, FakeAudioVAE(), audio(3.0)
        )
        video_mask, audio_mask = out["noise_mask"].unbind()
        packed_mask, shapes = comfy.utils.pack_latents([video_mask, audio_mask])

        class MaskProbe:
            diffusion_model = SimpleNamespace(patch_size=(1, 2, 2))
            _pool_masks_to_token_grid = comfy.model_base.MiniMaxH3._pool_masks_to_token_grid
            _token_grid_masks = comfy.model_base.MiniMaxH3._token_grid_masks
            _denoise_mask_values = comfy.model_base.MiniMaxH3._denoise_mask_values

        values = MaskProbe()._denoise_mask_values(packed_mask, shapes)
        self.assertNotIn("denoise_mask", values)
        self.assertIn("audio_denoise_mask", values)
        self.assertEqual(tuple(values["audio_denoise_mask"].shape), (1, 1, 2, audio_mask.shape[-1]))
        self.assertTrue(torch.all(values["audio_denoise_mask"] == 0))

    def test_lock_only_and_guide_only(self):
        node = init_audio.SECoursesMiniMaxH3InitAudio()
        positive = [[torch.zeros((1, 4, 8)), {}]]
        latent, _ = self.latent(frames=22)
        cond, out, fitted = node.apply(positive, latent, FakeAudioVAE(), audio(9.0), init_audio.CONDITIONING_MODES[1])
        self.assertNotIn("minimax_keyframes", cond[0][1])
        self.assertIn("noise_mask", out)
        self.assertIsNotNone(fitted)
        self.assertEqual(tuple(fitted["waveform"].shape), (1, 2, round(22 / 24 * 32000)))  # 9s audio cut to the 22-frame video

        cond, out, fitted = node.apply(positive, latent, FakeAudioVAE(), audio(1.0), init_audio.CONDITIONING_MODES[2])
        self.assertIn("minimax_keyframes", cond[0][1])
        self.assertIs(out, latent)
        self.assertIsNone(fitted)

    def test_rejects_non_av_latent(self):
        node = init_audio.SECoursesMiniMaxH3InitAudio()
        with self.assertRaises(ValueError):
            node.apply([[torch.zeros((1, 4, 8)), {}]], {"samples": torch.zeros((1, 4, 8, 8))}, FakeAudioVAE(), audio(1.0))


if __name__ == "__main__":
    unittest.main()
