import tempfile
import sys
import types
import unittest
from fractions import Fraction
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

    def test_only_multi_frame_replay_is_trimmed_from_later_batch_items(self):
        self.assertEqual(gallery._continuation_frames_to_trim(sequential_pack(1), True, 22), 0)
        self.assertEqual(gallery._continuation_frames_to_trim(sequential_pack(2), True, 1), 0)
        self.assertEqual(gallery._continuation_frames_to_trim(sequential_pack(2), False, 22), 0)
        self.assertEqual(gallery._continuation_frames_to_trim(sequential_pack(2), True, 22), 22)
        with_init = sequential_pack(2)
        with_init["init_image"] = {"file": "shot.png"}
        self.assertEqual(gallery._continuation_frames_to_trim(with_init, True, 22), 0)

    def test_context_frame_choices_are_strict(self):
        for value in (1, "5", 22, "39", 56):
            self.assertEqual(gallery._continuation_context_frame_count(value), int(value))
        for value in (0, 2, 57, "bad"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "1, 5, 22, 39, or 56"):
                gallery._continuation_context_frame_count(value)

    def test_continuation_node_decodes_registered_previous_video(self):
        first = sequential_pack(1, count=2)
        second = sequential_pack(2, count=2)
        gallery._record_batch_video_for_continuation(first, "first.mp4", True)
        expected = object()
        with mock.patch.object(gallery, "_decode_last_video_frames", return_value=expected) as decode:
            result = gallery.SECoursesBatchContinuationFrame().load(second, True)
        self.assertEqual(result, ({"image": expected, "context_frames": 1},))
        decode.assert_called_once_with("first.mp4", 1)

    def test_continuation_node_decodes_selected_native_context_length(self):
        first = sequential_pack(1, count=2)
        second = sequential_pack(2, count=2)
        gallery._record_batch_video_for_continuation(first, "first.mp4", True)
        expected = object()
        with mock.patch.object(gallery, "_decode_last_video_frames", return_value=expected) as decode:
            result = gallery.SECoursesBatchContinuationFrame().load(second, True, 39)
        self.assertEqual(result, ({"image": expected, "context_frames": 39},))
        decode.assert_called_once_with("first.mp4", 39)

    def test_first_item_returns_a_concrete_empty_optional_value(self):
        self.assertEqual(
            gallery.SECoursesBatchContinuationFrame().load(sequential_pack(1), True),
            ({"image": None, "context_frames": 0},),
        )

    def test_normal_run_passes_the_init_image_through(self):
        init = object()
        node = gallery.SECoursesBatchContinuationFrame()
        self.assertEqual(
            node.load({"prompt": "normal"}, False, init_image=init),
            ({"image": init, "context_frames": 0},),
        )
        self.assertEqual(
            node.load({"prompt": "normal"}, True, init_image=init),
            ({"image": init, "context_frames": 0},),
        )
        self.assertEqual(
            node.load({"prompt": "normal"}, False),
            ({"image": None, "context_frames": 0},),
        )

    def test_batch_items_ignore_the_init_image(self):
        init = object()
        node = gallery.SECoursesBatchContinuationFrame()
        self.assertEqual(
            node.load(sequential_pack(1), True, init_image=init),
            ({"image": None, "context_frames": 0},),
        )
        self.assertEqual(
            node.load(sequential_pack(2), False, init_image=init),
            ({"image": None, "context_frames": 0},),
        )

    def test_batch_continuation_frame_wins_over_the_init_image(self):
        first = sequential_pack(1, count=2)
        second = sequential_pack(2, count=2)
        gallery._record_batch_video_for_continuation(first, "first.mp4", True)
        expected = object()
        with mock.patch.object(gallery, "_decode_last_video_frames", return_value=expected):
            result = gallery.SECoursesBatchContinuationFrame().load(
                second, True, init_image=object()
            )
        self.assertEqual(result, ({"image": expected, "context_frames": 1},))


class ReferenceModeRoutingTests(unittest.TestCase):
    def detect(self, pack):
        return gallery.SECoursesMiniMaxH3ReferenceMode().detect(pack)

    def test_normal_pack_without_media_routes_to_the_normal_path(self):
        self.assertEqual(
            self.detect({"prompt": "text", "images": [], "videos": [], "audios": []}),
            (False, False),
        )

    def test_normal_pack_with_media_routes_to_the_auto_path(self):
        self.assertEqual(
            self.detect({"prompt": "p", "images": [{"file": "a.png"}], "videos": [], "audios": []}),
            (True, True),
        )
        self.assertEqual(
            self.detect({"prompt": "p", "images": [], "videos": [], "audios": [{"file": "a.mp3"}]}),
            (True, True),
        )

    def test_batch_items_always_route_to_the_auto_path(self):
        self.assertEqual(self.detect(sequential_pack(1)), (False, True))
        with_media = sequential_pack(2)
        with_media["images"] = [{"file": "a.png"}]
        self.assertEqual(self.detect(with_media), (True, True))


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

    def test_multi_frame_context_uses_native_guide_without_single_frame_conditioning(self):
        frames = object()
        guide = mock.Mock()
        guide.execute.return_value = types.SimpleNamespace(args=("guided-positive",))
        fake_module = types.SimpleNamespace(MiniMaxH3AddGuide=guide)
        with mock.patch.object(
            gallery.SECoursesMiniMaxH3TextOnly,
            "encode",
            return_value=("positive", "latent"),
        ) as text_only, mock.patch.dict(
            "sys.modules", {"comfy_extras.nodes_minimax_h3": fake_module}
        ):
            result = gallery.SECoursesMiniMaxH3Auto().encode(
                clip=object(), vae="vae", audio_vae=object(),
                references={"prompt": "go", "images": [], "videos": [], "audios": []},
                width=640, height=384, length=124, ref_image_size="match",
                continuation_frame={"image": frames, "context_frames": 22},
            )
        self.assertEqual(result, ("guided-positive", "latent", False))
        self.assertIsNone(text_only.call_args.kwargs["first_frame"])
        guide.execute.assert_called_once_with(
            positive="positive", vae="vae", latent="latent", image=frames, frame_idx=0
        )

    def test_multi_frame_context_requires_new_frames(self):
        with self.assertRaisesRegex(ValueError, "longer than its 39-frame"):
            gallery.SECoursesMiniMaxH3Auto().encode(
                clip=object(), vae=object(), audio_vae=object(),
                references={"prompt": "go", "images": [], "videos": [], "audios": []},
                width=640, height=384, length=39, ref_image_size="match",
                continuation_frame={"image": object(), "context_frames": 39},
            )


class ContinuationVideoTrimTests(unittest.TestCase):
    def test_video_and_audio_are_trimmed_by_the_same_frame_duration(self):
        comfy_root = str(Path(__file__).resolve().parents[3])
        if comfy_root not in sys.path:
            sys.path.insert(0, comfy_root)
        try:
            import torch
            from comfy_api.latest import Input, InputImpl, Types
        except (ImportError, RuntimeError) as error:
            self.skipTest(f"ComfyUI video API unavailable: {error}")

        images = torch.arange(30 * 2 * 2 * 3, dtype=torch.float32).reshape(30, 2, 2, 3)
        waveform = torch.arange(2 * 3000, dtype=torch.float32).reshape(1, 2, 3000)
        video = InputImpl.VideoFromComponents(Types.VideoComponents(
            images=images,
            audio=Input.Audio({"waveform": waveform, "sample_rate": 2400}),
            frame_rate=Fraction(24),
        ))

        components = gallery._trim_video_start(video, 22).get_components()
        self.assertEqual(tuple(components.images.shape), (8, 2, 2, 3))
        self.assertTrue(torch.equal(components.images[0], images[22]))
        self.assertEqual(tuple(components.audio["waveform"].shape), (1, 2, 800))
        self.assertTrue(torch.equal(components.audio["waveform"][..., 0], waveform[..., 2200]))


if __name__ == "__main__":
    unittest.main()
