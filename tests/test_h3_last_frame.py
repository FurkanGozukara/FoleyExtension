import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class KeyframeAlignmentTests(unittest.TestCase):
    def test_first_and_last_frame_line_uses_the_real_duration(self):
        line = gallery._keyframe_alignment_line(True, 124)
        self.assertIn("<Picture 1> (from [Shot 1]) aligns with the 0.00-second mark", line)
        self.assertIn("<Picture 2> (from [Shot 1]) aligns with the 5.17-second mark", line)

    def test_last_frame_only_line(self):
        line = gallery._keyframe_alignment_line(False, 243)
        self.assertIn("<Picture 1> (from [Shot 1]) aligns with the 10.12-second mark", line)
        self.assertNotIn("<Picture 2>", line)

    def test_existing_alignment_line_is_detected(self):
        self.assertTrue(gallery._has_keyframe_alignment("  How the reference pictures align with the target video ..."))
        self.assertTrue(gallery._has_keyframe_alignment("For the target video, at 0.00 seconds ..."))
        self.assertFalse(gallery._has_keyframe_alignment("integrated_multimodal_description: [Shot 1] ..."))


@unittest.skipUnless((COMFY_ROOT / "comfy_extras" / "nodes_minimax_h3.py").exists(), "ComfyUI checkout not available")
class TextOnlyLastFrameTests(unittest.TestCase):
    def run_text_only(self, prompt, first_frame, last_frame):
        sys.path.insert(0, str(COMFY_ROOT))
        import comfy_extras.nodes_minimax_h3 as native_h3

        captured = {}

        def fake_execute(**kwargs):
            captured.update(kwargs)
            return mock.Mock(args=("cond", "latent"))

        with mock.patch.object(native_h3.MiniMaxH3ImageToVideo, "execute", side_effect=fake_execute):
            gallery.SECoursesMiniMaxH3TextOnly().encode(
                clip=None, vae=None, references={"prompt": prompt}, width=1344, height=768, length=124,
                first_frame=first_frame, last_frame=last_frame,
            )
        return captured

    def test_last_frame_is_passed_and_alignment_line_added(self):
        captured = self.run_text_only("integrated_multimodal_description: [Shot 1] A ship.", "first", "last")
        self.assertEqual(captured["first_frame"], "first")
        self.assertEqual(captured["last_frame"], "last")
        self.assertTrue(captured["prompt"].startswith("How the reference pictures align with the target video"))
        self.assertTrue(captured["prompt"].endswith("integrated_multimodal_description: [Shot 1] A ship."))

    def test_user_alignment_line_is_kept(self):
        prompt = "How the reference pictures align with the target video — custom.\n\nintegrated_multimodal_description: x"
        captured = self.run_text_only(prompt, "first", "last")
        self.assertEqual(captured["prompt"], prompt)

    def test_first_frame_only_prompt_is_unchanged(self):
        captured = self.run_text_only("integrated_multimodal_description: x", "first", None)
        self.assertEqual(captured["prompt"], "integrated_multimodal_description: x")
        self.assertIsNone(captured["last_frame"])


if __name__ == "__main__":
    unittest.main()
