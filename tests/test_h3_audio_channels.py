"""Exercise native H3 row packing, not just an audio loader's output shape."""
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reference_gallery_nodes import _H3StereoAudioVAE
import init_audio_nodes
from comfy.ldm.minimax.model import PackedLayout, pack_audio
from comfy_extras.nodes_minimax_h3 import _encode_ref_audio
from comfy.nested_tensor import NestedTensor


class AudioVAE:
    audio_sample_rate = 32000
    downscale_ratio = 800

    def encode(self, samples):
        batch, length, channels = samples.shape
        return torch.zeros(batch, 32, channels, (length + 799) // 800)


class AudioChannelTests(unittest.TestCase):
    def test_unadapted_mono_reproduces_reported_shape_mismatch(self):
        audio = {'waveform': torch.zeros(1, 1, 356 * 800), 'sample_rate': 32000}
        encoded, length = _encode_ref_audio(AudioVAE(), audio)
        layout = PackedLayout(1, 2, 2, 2, 356,
                              refs=[{'kind': 'audio', 'ref_audio_t': length}])
        rows = torch.empty(len(layout.audio_update), 32)
        with self.assertRaisesRegex(RuntimeError, r'shape \[356, 32\].*shape \[712, 32\]'):
            rows[~layout.audio_update] = pack_audio(encoded)

    def test_mono_reference_and_all_init_audio_modes_pack_with_native_layout(self):
        audio = {'waveform': torch.zeros(1, 1, 356 * 800), 'sample_rate': 32000}
        encoded, length = _encode_ref_audio(_H3StereoAudioVAE(AudioVAE()), audio)
        reference = {'kind': 'audio', 'ref_audio_t': length, 'audio_latent': encoded}
        for mode in init_audio_nodes.CONDITIONING_MODES:
            with self.subTest(mode=mode):
                positive = [[torch.zeros(1, 1, 1), {'minimax_refs': [reference]}]]
                latent = {'samples': NestedTensor((torch.zeros(1, 24, 2, 2, 2), torch.zeros(1, 32, 2, 9)))}
                conditioning, out, fitted = init_audio_nodes.SECoursesMiniMaxH3InitAudio().apply(
                    positive, latent, AudioVAE(), audio, mode)
                keyframes = conditioning[0][1].get('minimax_keyframes', [])
                layout = PackedLayout(1, 2, 2, 2, 9, refs=[reference], keyframes=keyframes)
                guides = [k['audio_latent'] for k in keyframes] + [encoded]
                rows = torch.empty(len(layout.audio_update), 32)
                rows[~layout.audio_update] = torch.cat([pack_audio(z) for z in guides])
                self.assertEqual(encoded.shape[2], 2)
                self.assertEqual(fitted is None, mode == init_audio_nodes.CONDITIONING_MODES[2])

    def test_stereo_is_unchanged_and_surround_is_downmixed(self):
        class CaptureVAE(AudioVAE):
            def encode(self, samples):
                return samples

        vae = _H3StereoAudioVAE(CaptureVAE())
        stereo = torch.randn(1, 1600, 2)
        self.assertIs(vae.encode(stereo), stereo)
        surround = torch.randn(1, 1600, 6)
        self.assertTrue(torch.equal(vae.encode(surround), surround.mean(-1, keepdim=True).expand(-1, -1, 2)))


if __name__ == '__main__':
    unittest.main()
