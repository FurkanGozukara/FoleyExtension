import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class BatchAudioMergeTests(unittest.TestCase):
    def test_groups_audio_by_prompt_directory_in_prompt_order(self):
        groups = gallery._batch_audio_merge_groups(
            ["b2", "a1", "b1"],
            [
                {"batch": {"root": "C:/batch", "folder": "b", "index": 3}},
                {"batch": {"root": "C:/batch", "folder": "a", "index": 1}},
                {"batch": {"root": "C:/batch", "folder": "b", "index": 2}},
            ],
        )

        self.assertEqual([group["folder"] for group in groups], ["b", "a"])
        self.assertEqual(groups[0]["audios"], ["b1", "b2"])
        self.assertEqual(groups[1]["audios"], ["a1"])

    def test_concatenates_stereo_waveforms_without_reencoding(self):
        first = {
            "waveform": torch.tensor([[[1.0, 2.0], [10.0, 20.0]]]),
            "sample_rate": 32000,
        }
        second = {
            "waveform": torch.tensor([[[3.0], [30.0]]]),
            "sample_rate": 32000,
        }

        merged = gallery._concatenate_batch_audio([first, second])

        self.assertEqual(merged["sample_rate"], 32000)
        self.assertEqual(merged["waveform"].shape, (1, 2, 3))
        self.assertTrue(torch.equal(
            merged["waveform"],
            torch.tensor([[[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]]),
        ))

    def test_rejects_mismatched_sample_rates(self):
        clips = [
            {"waveform": torch.zeros((1, 2, 4)), "sample_rate": 32000},
            {"waveform": torch.zeros((1, 2, 4)), "sample_rate": 44100},
        ]

        with self.assertRaisesRegex(ValueError, "matching sample rates"):
            gallery._concatenate_batch_audio(clips)

    def test_audio_output_prefix_is_flat_beside_individual_audio(self):
        prefix = gallery._merge_audio_output_prefix(
            "C:/source/My Batch", "chapter 1/take:two"
        )

        self.assertEqual(
            prefix,
            "audio/MiniMax_H3_Audio_Merged_My Batch_chapter 1_take_two",
        )

    def test_enabled_node_previews_only_complete_last_group(self):
        packs = [
            {"batch": {"root": "C:/batch", "folder": "a", "index": 1}},
            {"batch": {"root": "C:/batch", "folder": "b", "index": 2}},
        ]

        def fake_merge(group):
            return {
                "filename": f"{group['folder']}.flac",
                "subfolder": "audio",
                "type": "output",
                "format": "audio/flac",
                "fullpath": f"C:/output/audio/{group['folder']}.flac",
            }

        with mock.patch.object(gallery, "_merge_batch_audio_group", side_effect=fake_merge) as merge:
            result = gallery.SECoursesBatchAudioMerge().merge(
                ["audio_a", "audio_b"], packs, [True, True]
            )

        self.assertEqual(merge.call_count, 2)
        self.assertEqual(result["ui"]["audio"], [{
            "filename": "b.flac",
            "subfolder": "audio",
            "type": "output",
        }])

    def test_disabled_node_does_not_save(self):
        with mock.patch.object(gallery, "_merge_batch_audio_group") as merge:
            result = gallery.SECoursesBatchAudioMerge().merge(
                ["audio"],
                [{"batch": {"root": "C:/batch", "folder": "root", "index": 1}}],
                [False],
            )

        self.assertEqual(result, {})
        merge.assert_not_called()

    def test_combined_output_saves_individuals_but_returns_only_last_merge(self):
        packs = [
            {"batch": {"root": "C:/batch", "folder": "a", "index": 1}},
            {"batch": {"root": "C:/batch", "folder": "b", "index": 2}},
        ]

        def fake_save(clip, prefix):
            return {
                "filename": f"{clip}.flac",
                "subfolder": "audio",
                "type": "output",
                "fullpath": f"C:/output/{prefix}/{clip}.flac",
            }

        def fake_merge(group):
            return {
                "filename": f"merged_{group['folder']}.flac",
                "subfolder": "audio",
                "type": "output",
                "fullpath": f"C:/output/audio/merged_{group['folder']}.flac",
            }

        with (
            mock.patch.object(gallery, "_save_audio_output", side_effect=fake_save) as save,
            mock.patch.object(gallery, "_merge_batch_audio_group", side_effect=fake_merge) as merge,
        ):
            result = gallery.SECoursesBatchAudioSaveMerge().save_and_merge(
                ["clip_a", "clip_b"], packs, [True, True], ["audio/individual"]
            )

        self.assertEqual(save.call_count, 2)
        self.assertEqual(merge.call_count, 2)
        self.assertEqual(result["ui"]["audio"], [{
            "filename": "merged_b.flac",
            "subfolder": "audio",
            "type": "output",
        }])

    def test_combined_output_returns_individuals_when_merge_is_disabled(self):
        def fake_save(clip, prefix):
            return {
                "filename": f"{clip}.flac",
                "subfolder": "audio",
                "type": "output",
                "fullpath": f"C:/output/{prefix}/{clip}.flac",
            }

        with (
            mock.patch.object(gallery, "_save_audio_output", side_effect=fake_save),
            mock.patch.object(gallery, "_merge_batch_audio_group") as merge,
        ):
            result = gallery.SECoursesBatchAudioSaveMerge().save_and_merge(
                ["clip_a", "clip_b"], [{}, {}], [False], ["audio/individual"]
            )

        merge.assert_not_called()
        self.assertEqual(
            [item["filename"] for item in result["ui"]["audio"]],
            ["clip_a.flac", "clip_b.flac"],
        )


if __name__ == "__main__":
    unittest.main()
