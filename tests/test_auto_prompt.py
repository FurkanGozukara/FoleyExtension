"""SECoursesH3AutoPrompt: a typed prompt passes through without touching the model; an empty prompt is written from the answer."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))

PACKAGE = "foley_auto_prompt_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def _load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", ROOT / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


auto = _load("auto_prompt_nodes")


class ParseAnswer(unittest.TestCase):
    def test_labelled_lines(self):
        fields = auto.parse_answer("**SUBJECT:** red car\nSCENE: A coast road\nat sunset.\nSOUND: Engine.")
        self.assertEqual(fields["SUBJECT"], "red car")
        self.assertEqual(fields["SCENE"], "A coast road at sunset.")
        self.assertEqual(fields["SOUND"], "Engine.")

    def test_no_labels(self):
        self.assertEqual(auto.parse_answer("just some words"), {})


class PathDirection(unittest.TestCase):
    def test_directions(self):
        self.assertEqual(auto.path_direction('[{"x": 10, "y": 50}, {"x": 90, "y": 60}]'), "to the right")
        self.assertEqual(auto.path_direction('[{"x": 90, "y": 50}, {"x": 10, "y": 60}]'), "to the left")
        self.assertEqual(auto.path_direction('[{"x": 10, "y": 90}, {"x": 20, "y": 10}]'), "upward")
        self.assertEqual(auto.path_direction('[{"x": 10, "y": 10}, {"x": 20, "y": 90}]'), "downward")

    def test_bad_path(self):
        self.assertEqual(auto.path_direction("not json"), "along its path")
        self.assertEqual(auto.path_direction(None), "along its path")


class Prompts(unittest.TestCase):
    def test_cut_and_drag_uses_picture_and_direction(self):
        text = auto.cut_and_drag_prompt({"SCENE": "A road.", "SOUND": "Engine."}, "red car", "to the right")
        self.assertIn("<Picture 1>", text)
        self.assertIn("The red car in <Picture 1> starts and moves smoothly to the right", text)
        self.assertIn("overall_soundscape: Engine.", text)

    def test_first_last_ends_on_picture_two_only_with_a_last_frame(self):
        self.assertIn("End aligned with <Picture 2>.", auto.first_last_prompt({}))
        self.assertNotIn("<Picture 2>", auto.first_last_prompt({}, has_last=False))

    def test_references_names_every_image(self):
        text = auto.references_prompt({"ROLES": "a man; a street"}, 2)
        self.assertIn("@image1, @image2", text)
        self.assertIn("a man; a street", text)


class PassThrough(unittest.TestCase):
    def test_typed_prompt_never_asks_for_the_model(self):
        node = auto.SECoursesH3AutoPrompt()
        self.assertEqual(node.check_lazy_status("typed", "first last frame", ""), [])
        self.assertEqual(node.check_lazy_status("typed", "cut and drag", "red car"), [])

    def test_empty_prompt_asks_for_the_model(self):
        node = auto.SECoursesH3AutoPrompt()
        self.assertEqual(node.check_lazy_status("", "first last frame", ""), ["describe_clip"])
        self.assertEqual(node.check_lazy_status("  ", "references", ""), ["describe_clip"])

    def test_cut_and_drag_without_subject_asks_for_the_model(self):
        node = auto.SECoursesH3AutoPrompt()
        self.assertEqual(node.check_lazy_status("typed", "cut and drag", ""), ["describe_clip"])

    def test_typed_values_are_returned_unchanged(self):
        pack = {"prompt": "typed", "images": []}
        result = auto.SECoursesH3AutoPrompt().write("typed", "cut and drag", " red car ", references=pack)
        self.assertEqual(result, ("typed", "red car", pack))


class FakeClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return [[f"embedding of {tokens}", {"pooled": tokens}]]


class Story(unittest.TestCase):
    TEXT = "A red balloon flies.\n\noverall_soundscape: Wind.\n\nnon_diegetic_music: N/A\n---\nOver a meadow.\n---\nOver mountains."

    def test_no_scenes_means_no_prompts(self):
        self.assertEqual(auto.story_prompts("A plain prompt - with a dash"), [])
        self.assertEqual(auto.story_prompts(""), [])

    def test_scene_goes_before_the_sound(self):
        prompts = auto.story_prompts(self.TEXT)
        self.assertEqual(len(prompts), 2)
        self.assertTrue(prompts[0].startswith("integrated_multimodal_description: [Shot 1] A red balloon flies. Over a meadow."))
        self.assertTrue(prompts[1].endswith("non_diegetic_music: N/A"))
        self.assertIn("Over mountains.\n\noverall_soundscape: Wind.", prompts[1])

    def test_a_full_prompt_keeps_its_own_head(self):
        prompts = auto.story_prompts("integrated_multimodal_description: [Shot 1] A boat.\n---\nAt dawn.\n---\nAt noon.")
        self.assertEqual(prompts[0], "integrated_multimodal_description: [Shot 1] A boat. At dawn.")

    def test_conditioning_is_untouched_without_scenes(self):
        positive = [["text", {"minimax_keyframes": 1}]]
        self.assertIs(auto.SECoursesH3StoryWindows().combine(positive, FakeClip(), "plain", True)[0], positive)

    def test_conditioning_is_untouched_when_long_video_is_off(self):
        positive = [["text", {}]]
        self.assertIs(auto.SECoursesH3StoryWindows().combine(positive, FakeClip(), self.TEXT, False)[0], positive)
        self.assertIs(auto.SECoursesH3StoryWindows().combine(positive, FakeClip(), self.TEXT, True, True)[0], positive)

    def test_each_scene_keeps_the_extras(self):
        positive = [["text", {"minimax_keyframes": 7, "pooled": "old"}]]
        out = auto.SECoursesH3StoryWindows().combine(positive, FakeClip(), self.TEXT, True)[0]
        self.assertEqual(len(out), 2)
        self.assertIn("Over a meadow.", out[0][0])
        self.assertEqual(out[1][1]["minimax_keyframes"], 7)
        self.assertNotEqual(out[1][1]["pooled"], "old")


if __name__ == "__main__":
    unittest.main()
