import sys
import subprocess
import tempfile
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
        self.assertEqual(result[5], [1, 1])

    def test_gallery_exposes_init_video_toggles_for_normal_runs(self):
        result = gallery.SECoursesReferenceGallery().collect(
            "prompt", "{}", 24, 15,
            merge_batch_videos=True,
            continue_batch_with_last_frame=True,
            continuation_context_frames=22,
        )

        self.assertEqual(result[2], [False])
        self.assertEqual(result[3], [True])
        self.assertEqual(result[4], [True])
        self.assertEqual(result[5], [22])

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

    def test_combined_video_output_removes_multi_frame_replay_before_saving(self):
        pack = {"batch": {
            "root": "C:/batch", "folder": "root", "index": 2, "count": 3,
            "run_id": "run_context_1234", "sequential": True,
        }}
        saved = {
            "filename": "trimmed.mp4", "subfolder": "video", "type": "output",
            "fullpath": "C:/output/trimmed.mp4",
        }
        with (
            mock.patch.object(gallery, "_trim_video_start", return_value="trimmed") as trim,
            mock.patch.object(gallery, "_save_video_output", return_value=saved) as save,
            mock.patch.object(gallery, "_video_from_saved_output", return_value="preview"),
        ):
            gallery.SECoursesBatchVideoSaveMerge().save_and_merge(
                ["generated"], [pack], [False], ["video/MiniMax_H3"], [True], [22]
            )
        trim.assert_called_once_with("generated", 22, trim_audio=True)
        self.assertEqual(save.call_args.args[0], "trimmed")
        gallery._BATCH_CONTINUATION_SESSIONS.clear()

    def test_normal_init_video_is_trimmed_and_merged_when_enabled(self):
        saved = {
            "filename": "generated.mp4", "subfolder": "video", "type": "output",
            "fullpath": "C:/output/generated.mp4",
        }
        merged = {
            "filename": "merged.mp4", "subfolder": "video", "type": "output",
            "fullpath": "C:/output/merged.mp4",
        }
        with tempfile.NamedTemporaryFile(suffix=".mp4") as source:
            with (
                mock.patch.object(gallery, "_trim_video_start", return_value="trimmed") as trim,
                mock.patch.object(gallery, "_save_video_output", return_value=saved) as save,
                mock.patch.object(gallery, "_merge_init_video_with_generated", return_value=merged) as merge,
                mock.patch.object(gallery, "_video_from_saved_output", return_value="preview"),
            ):
                result = gallery.SECoursesBatchVideoSaveMerge().save_and_merge(
                    ["generated"], [{"prompt": "normal"}], [True], ["video/MiniMax_H3"],
                    [False], [22], [{"path": source.name, "name": "source.mp4"}],
                )

        trim.assert_called_once_with("generated", 1, trim_audio=False)
        self.assertEqual(save.call_args.args[0], "trimmed")
        merge.assert_called_once_with(source.name, "C:/output/generated.mp4", "video/MiniMax_H3")
        self.assertEqual(result["ui"]["images"][0]["filename"], "merged.mp4")
        self.assertEqual(result["result"], ("preview",))

    def test_init_video_ffmpeg_merge_preserves_order_and_audio(self):
        try:
            import av
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as error:
            self.skipTest(f"FFmpeg integration dependencies are unavailable: {error}")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "source.mp4")
            generated = Path(directory, "generated.mp4")
            ffmpeg = get_ffmpeg_exe()
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=red:s=64x48:r=12:d=0.5",
                "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.5",
                "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(source),
            ], check=True)
            subprocess.run([
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=blue:s=64x48:r=12:d=0.5",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", str(generated),
            ], check=True)

            with (
                mock.patch("folder_paths.get_output_directory", return_value=directory),
                mock.patch(
                    "folder_paths.get_save_image_path",
                    return_value=(directory, "merged", 1, "", ""),
                ),
            ):
                merged = gallery._merge_init_video_with_generated(source, generated, "video/test")

            info = gallery._media_info(merged["fullpath"])
            self.assertEqual((info["width"], info["height"]), (64, 48))
            self.assertTrue(info["has_audio"])
            self.assertGreater(info["duration"], 0.8)
            self.assertLess(info["duration"], 1.2)
            with av.open(merged["fullpath"]) as container:
                frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            self.assertGreater(frames[0][..., 0].mean(), frames[0][..., 2].mean())
            self.assertGreater(frames[-1][..., 2].mean(), frames[-1][..., 0].mean())


if __name__ == "__main__":
    unittest.main()
