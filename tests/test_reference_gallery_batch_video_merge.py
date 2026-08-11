import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class BatchVideoMergeTests(unittest.TestCase):
    def setUp(self):
        gallery._BATCH_OUTPUT_SESSIONS["video"].clear()

    def tearDown(self):
        gallery._BATCH_OUTPUT_SESSIONS["video"].clear()

    def test_groups_by_prompt_directory_and_sorts_by_batch_index(self):
        videos = ["scene_b_2", "scene_a_1", "scene_b_1"]
        packs = [
            {"batch": {"root": "C:/batch", "folder": "scene_b", "index": 3}},
            {"batch": {"root": "C:/batch", "folder": "scene_a", "index": 1}},
            {"batch": {"root": "C:/batch", "folder": "scene_b", "index": 2}},
        ]

        groups = gallery._batch_video_merge_groups(videos, packs)

        self.assertEqual([group["folder"] for group in groups], ["scene_b", "scene_a"])
        self.assertEqual(groups[0]["videos"], ["scene_b_1", "scene_b_2"])
        self.assertEqual(groups[1]["videos"], ["scene_a_1"])

    def test_root_only_batch_is_one_group(self):
        groups = gallery._batch_video_merge_groups(
            ["one", "two"],
            [
                {"batch": {"root": "C:/batch", "folder": "root", "index": 1}},
                {"batch": {"root": "C:/batch", "folder": "root", "index": 2}},
            ],
        )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["videos"], ["one", "two"])

    def test_video_and_pack_count_must_match(self):
        with self.assertRaisesRegex(ValueError, "different number"):
            gallery._batch_video_merge_groups(["one"], [])

    def test_output_prefix_is_flat_beside_individual_videos(self):
        prefix = gallery._merge_output_prefix("C:/source/My Batch", "chapter 1/take:two")

        self.assertEqual(
            prefix,
            "video/MiniMax_H3_Merged_My Batch_chapter 1_take_two",
        )

    def test_enabled_node_saves_every_group_and_previews_only_last(self):
        packs = [
            {"batch": {"root": "C:/batch", "folder": "a", "index": 1}},
            {"batch": {"root": "C:/batch", "folder": "b", "index": 2}},
        ]

        def fake_merge(group):
            return {
                "filename": f"{group['folder']}.mp4",
                "subfolder": group["folder"],
                "type": "output",
                "format": "video/mp4",
                "fullpath": f"C:/output/{group['folder']}.mp4",
            }

        with mock.patch.object(gallery, "_merge_batch_video_group", side_effect=fake_merge) as merge:
            result = gallery.SECoursesBatchVideoMerge().merge(
                ["video_a", "video_b"], packs, [True, True]
            )

        self.assertEqual(merge.call_count, 2)
        self.assertEqual(result["ui"]["images"], [{
            "filename": "b.mp4",
            "subfolder": "b",
            "type": "output",
        }])
        self.assertEqual(result["ui"]["animated"], (True,))

    def test_disabled_node_leaves_outputs_untouched(self):
        with mock.patch.object(gallery, "_merge_batch_video_group") as merge:
            result = gallery.SECoursesBatchVideoMerge().merge(
                ["video"],
                [{"batch": {"root": "C:/batch", "folder": "root", "index": 1}}],
                [False],
            )

        self.assertEqual(result, {})
        merge.assert_not_called()

    def test_gallery_repeats_merge_flag_for_every_prompt(self):
        packs = ([{"batch": {"folder": "a"}}, {"batch": {"folder": "b"}}], ["one", "two"])
        with mock.patch.object(gallery, "_collect_folder_batch", return_value=packs):
            result = gallery.SECoursesReferenceGallery().collect(
                "fallback", "{}", 24, 15, "C:/batch", True
            )

        self.assertEqual(result[2], [True, True])
        self.assertEqual(result[3], [True, True])
        self.assertEqual(result[4], [False, False])

    def test_gallery_selects_one_sequential_prompt_per_queued_job(self):
        packs = (
            [
                {"batch": {"folder": "root", "prompt_file": "1.txt", "index": 1, "count": 2}},
                {"batch": {"folder": "root", "prompt_file": "2.txt", "index": 2, "count": 2}},
            ],
            ["one", "two"],
        )
        with mock.patch.object(gallery, "_collect_folder_batch", return_value=packs):
            result = gallery.SECoursesReferenceGallery().collect(
                "fallback",
                "{}",
                24,
                15,
                "C:/batch",
                True,
                False,
                "run_12345678",
                1,
                2,
            )

        self.assertEqual(result[1], ["two"])
        self.assertEqual(result[0][0]["batch"]["run_id"], "run_12345678")
        self.assertTrue(result[0][0]["batch"]["sequential"])

    def test_combined_video_output_saves_each_job_then_merges_on_final_job(self):
        events = []

        def fake_save(clip, prefix, prompt, extra):
            events.append(f"save:{clip}")
            return {
                "filename": f"{clip}.mp4",
                "subfolder": "video",
                "type": "output",
                "fullpath": f"C:/output/{clip}.mp4",
            }

        def fake_merge(group):
            events.append("merge:" + ",".join(group["videos"]))
            return {
                "filename": "merged.mp4",
                "subfolder": "video",
                "type": "output",
                "fullpath": "C:/output/merged.mp4",
            }

        pack_one = {"batch": {
            "root": "C:/batch", "folder": "root", "index": 1, "count": 2,
            "run_id": "run_12345678", "sequential": True,
        }}
        pack_two = {"batch": {
            "root": "C:/batch", "folder": "root", "index": 2, "count": 2,
            "run_id": "run_12345678", "sequential": True,
        }}

        with (
            mock.patch.object(gallery, "_save_video_output", side_effect=fake_save),
            mock.patch.object(gallery, "_merge_saved_video_group", side_effect=fake_merge),
            mock.patch.object(gallery, "_video_from_saved_output", side_effect=lambda item: item["filename"]),
        ):
            first = gallery.SECoursesBatchVideoSaveMerge().save_and_merge(
                ["clip_1"], [pack_one], [True], ["video/MiniMax_H3"]
            )
            second = gallery.SECoursesBatchVideoSaveMerge().save_and_merge(
                ["clip_2"], [pack_two], [True], ["video/MiniMax_H3"]
            )

        self.assertEqual(first["ui"]["images"][0]["filename"], "clip_1.mp4")
        self.assertEqual(second["ui"]["images"][0]["filename"], "merged.mp4")
        self.assertEqual(second["result"], ("merged.mp4",))
        self.assertEqual(events, [
            "save:clip_1",
            "save:clip_2",
            "merge:C:/output/clip_1.mp4,C:/output/clip_2.mp4",
        ])

    def test_combined_video_output_exposes_video_result(self):
        self.assertEqual(gallery.SECoursesBatchVideoSaveMerge.RETURN_TYPES, ("VIDEO",))
        self.assertEqual(gallery.SECoursesBatchVideoSaveMerge.RETURN_NAMES, ("video",))


if __name__ == "__main__":
    unittest.main()
