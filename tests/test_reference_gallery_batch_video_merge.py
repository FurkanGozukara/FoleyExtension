import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class BatchVideoMergeTests(unittest.TestCase):
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

    def test_output_prefix_mirrors_batch_folder_without_path_escape(self):
        prefix = gallery._merge_output_prefix("C:/source/My Batch", "chapter 1/take:two")

        self.assertEqual(
            prefix,
            "video/MiniMax_H3_Merged/My Batch/chapter 1/take_two/merged",
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
        self.assertEqual(result["ui"]["gifs"], [{
            "filename": "b.mp4",
            "subfolder": "b",
            "type": "output",
            "format": "video/mp4",
            "fullpath": "C:/output/b.mp4",
        }])

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


if __name__ == "__main__":
    unittest.main()
