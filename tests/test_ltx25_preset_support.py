import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ltx25_preset_support_nodes as support  # noqa: E402


class LTX25PresetSupportTests(unittest.TestCase):
    def test_distilled_default_matches_official_eight_step_curve(self):
        sigmas, = support.LTX25DistilledSigmaSchedule.build(8)
        expected = torch.tensor(
            [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
        )
        self.assertTrue(torch.equal(sigmas, expected))

    def test_non_default_step_count_preserves_schedule_endpoints(self):
        sigmas, = support.LTX25DistilledSigmaSchedule.build(12)
        self.assertEqual(sigmas.shape, (13,))
        self.assertEqual(sigmas[0].item(), 1.0)
        self.assertEqual(sigmas[-1].item(), 0.0)
        self.assertTrue(torch.all(sigmas[:-1] >= sigmas[1:]))

    def test_prepare_chunks_pads_geometry_and_retains_exact_output_size(self):
        images = torch.arange(35, dtype=torch.float32).view(35, 1, 1, 1)
        images = images.expand(35, 33, 65, 3).clone()

        result = support.LTX25PrepareVideoChunks.prepare(images, 18000, 17, 1)
        (
            chunks,
            widths,
            heights,
            chunk_lengths,
            keep_lengths,
            output_widths,
            output_heights,
            overlaps,
            totals,
            summaries,
        ) = result

        self.assertEqual(len(chunks), 3)
        self.assertEqual(widths, [192, 192, 192])
        self.assertEqual(heights, [128, 128, 128])
        self.assertEqual(chunk_lengths, [17, 17, 9])
        self.assertEqual(keep_lengths, [17, 17, 3])
        self.assertEqual(output_widths, [130, 130, 130])
        self.assertEqual(output_heights, [66, 66, 66])
        self.assertEqual(overlaps, [0, 1, 1])
        self.assertEqual(totals, [35, 35, 35])
        self.assertTrue(all(chunk.shape[1:3] == (64, 96) for chunk in chunks))
        self.assertIn("130x66 exact crop", summaries[0])

    def test_merge_removes_padding_and_preserves_all_source_frames(self):
        images = torch.arange(35, dtype=torch.float32).view(35, 1, 1, 1)
        images = images.expand(35, 33, 65, 3).clone()
        result = support.LTX25PrepareVideoChunks.prepare(images, 18000, 17, 1)
        chunks, keep_lengths, overlaps, totals = result[0], result[4], result[7], result[8]
        decoded = [chunk[:, :33, :65, :] for chunk in chunks]

        merged, = support.LTX25MergeVideoChunks.merge(
            decoded,
            keep_lengths,
            overlaps,
            totals,
        )

        self.assertEqual(merged.shape, (35, 33, 65, 3))
        self.assertTrue(
            torch.equal(merged[:, 0, 0, 0], torch.arange(35, dtype=torch.float32))
        )

    def test_invalid_image_tensor_has_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "ComfyUI IMAGE tensor"):
            support.LTX25PrepareVideoChunks.prepare(torch.zeros(3, 4, 5), 18000, 17, 1)


if __name__ == "__main__":
    unittest.main()
