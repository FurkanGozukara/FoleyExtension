import tempfile
import unittest
from pathlib import Path
from unittest import mock


import reference_gallery_nodes as gallery


def sequential_pack(index, count=3, run_id="run_12345678"):
    return {
        "version": 3,
        "prompt": f"prompt {index}",
        "images": [],
        "videos": [],
        "audios": [],
        "batch": {
            "root": "C:/batch",
            "folder": "root",
            "index": index,
            "count": count,
            "run_id": run_id,
            "sequential": True,
        },
    }


class BatchDurationTests(unittest.TestCase):
    def test_only_underscore_integer_suffix_sets_duration(self):
        cases = {
            "scene_8.txt": 8,
            "scene_take_012.txt": 12,
            "_4.txt": 4,
            "1.txt": None,
            "200.txt": None,
            "scene.txt": None,
            "scene_.txt": None,
            "scene_4.5.txt": None,
            "scene_-5.txt": None,
            "scene_0.txt": None,
        }
        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(gallery._batch_prompt_duration_seconds(filename), expected)

    def test_collector_stores_each_filename_duration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "1.txt").write_text("default", encoding="utf-8")
            (root / "scene_7.txt").write_text("seven", encoding="utf-8")
            packs, prompts = gallery._collect_folder_batch(root, {
                "images": [], "videos": [], "audios": [],
            }, 24, 15)

        self.assertEqual(prompts, ["default", "seven"])
        self.assertIsNone(packs[0]["batch"]["duration_seconds"])
        self.assertEqual(packs[1]["batch"]["duration_seconds"], 7)

    def test_duration_node_uses_override_or_user_default(self):
        node = gallery.SECoursesBatchDuration()
        self.assertEqual(node.resolve({"prompt": "normal"}, 5.5), (5.5,))
        self.assertEqual(
            node.resolve({"batch": {"duration_seconds": 9}}, 5.5),
            (9.0,),
        )
        with self.assertRaisesRegex(ValueError, "positive finite"):
            node.resolve({}, 0)


class BatchContinuationTests(unittest.TestCase):
    def setUp(self):
        gallery._BATCH_CONTINUATION_SESSIONS.clear()

    def tearDown(self):
        gallery._BATCH_CONTINUATION_SESSIONS.clear()

    def test_previous_video_advances_only_after_completed_item(self):
        first = sequential_pack(1)
        second = sequential_pack(2)
        third = sequential_pack(3)

        self.assertIsNone(gallery._previous_batch_video(first, True))
        gallery._record_batch_video_for_continuation(first, "first.mp4", True)
        self.assertEqual(gallery._previous_batch_video(second, True), "first.mp4")
        gallery._record_batch_video_for_continuation(second, "second.mp4", True)
        self.assertEqual(gallery._previous_batch_video(third, True), "second.mp4")
        gallery._record_batch_video_for_continuation(third, "third.mp4", True)
        self.assertNotIn("run_12345678", gallery._BATCH_CONTINUATION_SESSIONS)

    def test_disabled_and_non_batch_generation_have_no_frame(self):
        self.assertIsNone(gallery._previous_batch_video(sequential_pack(2), False))
        self.assertIsNone(gallery._previous_batch_video({"prompt": "normal"}, True))

    def test_missing_previous_item_is_an_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "previous completed video"):
            gallery._previous_batch_video(sequential_pack(2), True)

    def test_continuation_node_decodes_registered_previous_video(self):
        first = sequential_pack(1, count=2)
        second = sequential_pack(2, count=2)
        gallery._record_batch_video_for_continuation(first, "first.mp4", True)
        expected = object()
        with mock.patch.object(gallery, "_decode_last_video_frame", return_value=expected) as decode:
            result = gallery.SECoursesBatchContinuationFrame().load(second, True)
        self.assertEqual(result, ({"image": expected},))
        decode.assert_called_once_with("first.mp4")

    def test_first_item_returns_a_concrete_empty_optional_value(self):
        self.assertEqual(
            gallery.SECoursesBatchContinuationFrame().load(sequential_pack(1), True),
            ({"image": None},),
        )


class MiniMaxAutoRoutingTests(unittest.TestCase):
    def test_text_only_pack_uses_fl2va_and_passes_continuation_frame(self):
        frame = object()
        with mock.patch.object(
            gallery.SECoursesMiniMaxH3TextOnly,
            "encode",
            return_value=("positive", "latent"),
        ) as text_only:
            result = gallery.SECoursesMiniMaxH3Auto().encode(
                clip=object(), vae=object(), audio_vae=object(),
                references={"prompt": "go", "images": [], "videos": [], "audios": []},
                width=640, height=384, length=124, ref_image_size="match",
                continuation_frame={"image": frame},
            )
        self.assertEqual(result, ("positive", "latent", False))
        self.assertIs(text_only.call_args.kwargs["first_frame"], frame)

    def test_media_pack_uses_ref2va_and_passes_continuation_frame(self):
        frame = object()
        with mock.patch.object(
            gallery.SECoursesMiniMaxH3References,
            "encode",
            return_value=("positive", "latent"),
        ) as references:
            result = gallery.SECoursesMiniMaxH3Auto().encode(
                clip=object(), vae=object(), audio_vae=object(),
                references={
                    "prompt": "use @image1",
                    "images": [{"file": "image.png"}],
                    "videos": [], "audios": [],
                },
                width=640, height=384, length=124, ref_image_size="match",
                continuation_frame={"image": frame},
            )
        self.assertEqual(result, ("positive", "latent", True))
        self.assertIs(references.call_args.kwargs["continuation_frame"], frame)


if __name__ == "__main__":
    unittest.main()
