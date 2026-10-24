import math
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import genjutsu_nodes  # noqa: E402
from genjutsu_nodes import BACKGROUNDS, genjutsu_prompt, source_aspect_size, usable_frame_count  # noqa: E402


# The MDMZ "H3 Character Swap" workflow's RegexReplace nodes, as ComfyUI runs them:
# (pattern, replacement, case_insensitive, multiline, dotall, count).
SOURCE_TIDY = [
    (r"^\s*```[a-zA-Z]*\s*$", "", True, True, False, 0),
    (r"^[\s*#_`]*subject[_ ]definitions[\s*_`]*:[ \t*]*", "subject_definitions:\\n", True, True, False, 0),
    (r"^[\s*#_`]*retention[_ ]analysis[\s*_`]*:[ \t*]*", "retention_analysis:\\n", True, True, False, 0),
]
SOURCE_VIDEO = [
    (r"^[ \t>*_`#]*subject[ _]definitions[ \t*_`]*:[*_`]*", genjutsu_nodes._VIDEO_BACKGROUND_RULES[0][1], True, True, False, 1),
    (r"(?:\[[ \t]*reference[ \t]+generation([^\]\n]*)\]([\s\S]*?))?^[ \t>*_`#]*retention[ _]analysis[ \t*_`]*:[*_`]*",
     genjutsu_nodes._VIDEO_BACKGROUND_RULES[1][1], True, True, False, 1),
]
SOURCE_IMAGE = [
    (r"^[ \t>*_`#]*subject[ _]definitions[ \t*_`]*:[*_`]*", genjutsu_nodes._IMAGE_BACKGROUND_RULES[0][1], True, True, False, 1),
    (r"^[ \t>*_`#]*retention[ _]analysis[ \t*_`]*:[*_`]*", genjutsu_nodes._IMAGE_BACKGROUND_RULES[1][1], True, True, False, 1),
    (r"Setting\s+in\s+<Video\s*1>\s*:[^\n]*\n?|(Scene\s+lighting|Framing)\s+on\s+the\s+performer\s+in\s+<Video\s*1>\s*:.*?"
     r"(?=\n\s*\n|(Scene\s+lighting|Framing)\s+on\s+the\s+performer|\Z)", "", True, False, True, 0),
]


def run_source(prompt, rules):
    for pattern, replacement, case_insensitive, multiline, dotall, count in rules:
        flags = (re.IGNORECASE if case_insensitive else 0) | (re.MULTILINE if multiline else 0) | (re.DOTALL if dotall else 0)
        prompt = re.sub(pattern, replacement, prompt, count=count, flags=flags)
    return prompt


TEMPLATE = """subject_definitions:
<Subject 1> is the adult person shown in <Picture 1>, with the exact face, skin tone, hairstyle, body proportions and complete outfit from that image.
<Video 1> is the source clip; use its performer's motion, timing and camera movement.
<Audio 1> is the source soundtrack.

summary:
[reference generation] From the very first frame, <Subject 1> fully replaces the performer in <Video 1> and performs the same actions with the same timing for the entire clip.

retention_analysis:
<Subject 1>: attribute_transfer - transfer the face, hairstyle, outfit and body proportions consistently to every frame.
<Video 1>: partially_preserved - preserve the performer's action, pose, timing and camera movement.
<Audio 1>: reference - keep the timing and ambience of the source audio.

detailed_description:
Photorealistic, continuous shot.
[Shot 1] <Subject 1>, already fully transformed in the opening frame, walks toward the camera.
Setting in <Video 1>: the target video takes place in <Subject 2>, a battlefield before a stone castle at sunset.
Scene lighting on the performer in <Video 1>: warm low sun from behind, soft fill on the face.
Framing on the performer in <Video 1>: full shot, then a medium shot as the camera pushes in.

overall_soundscape:
Wind and distant fires. No invented dialogue.

non_diegetic_music:
N/A"""

# An LLM answer pasted with a code fence and markdown headings.
PASTED = "```text\n**Subject Definitions:**\n" + TEMPLATE.split("\n", 1)[1].replace(
    "retention_analysis:", "## retention analysis:", 1) + "\n```"


class GenjutsuPromptTests(unittest.TestCase):
    def test_matches_the_source_workflow_chain(self):
        for prompt in (TEMPLATE, PASTED):
            tidy = run_source(prompt, SOURCE_TIDY)
            self.assertEqual(genjutsu_prompt(prompt, BACKGROUNDS[0]), run_source(tidy, SOURCE_VIDEO))
            self.assertEqual(genjutsu_prompt(prompt, BACKGROUNDS[1]), run_source(tidy, SOURCE_IMAGE))

    def test_video_background_turns_the_generation_into_an_edit(self):
        text = genjutsu_prompt(TEMPLATE, BACKGROUNDS[0])
        self.assertIn("[video editing] The target video is an edited version of <Video 1>.", text)
        self.assertNotIn("[reference generation]", text)
        self.assertIn("<Subject 2> is the environment of <Video 1>", text)
        self.assertEqual(text.count("subject_definitions:"), 1)
        self.assertEqual(text.count("retention_analysis:"), 1)

    def test_image_background_drops_the_clip_setting_lines(self):
        text = genjutsu_prompt(TEMPLATE, BACKGROUNDS[1])
        self.assertNotIn("Setting in <Video 1>", text)
        self.assertNotIn("Scene lighting on the performer", text)
        self.assertNotIn("Framing on the performer", text)
        self.assertIn("Environment: taken from <Picture 1>", text)
        self.assertIn("[reference generation]", text)

    def test_free_form_prompt_still_gets_the_rules(self):
        text = genjutsu_prompt("@image1 performs the dance from @video1.", BACKGROUNDS[0])
        self.assertTrue(text.startswith("subject_definitions:\n<Subject 2> is the environment of <Video 1>"))
        self.assertIn("@image1 performs the dance from @video1.", text)
        self.assertIn("retention_analysis:\n<Subject 2> (appears in every shot): fully_preserved", text)

    def test_unknown_background_is_rejected(self):
        with self.assertRaises(ValueError):
            genjutsu_prompt(TEMPLATE, "video")


class GenjutsuTemplateTests(unittest.TestCase):
    def test_full_prompts_are_recognised(self):
        self.assertTrue(genjutsu_nodes.is_full_prompt(TEMPLATE))
        self.assertTrue(genjutsu_nodes.is_full_prompt(PASTED))
        self.assertFalse(genjutsu_nodes.is_full_prompt(""))
        self.assertFalse(genjutsu_nodes.is_full_prompt("He wears a red scarf."))

    def test_a_full_prompt_is_used_as_written(self):
        self.assertEqual(genjutsu_nodes.build_prompt(TEMPLATE, BACKGROUNDS[0]), genjutsu_prompt(TEMPLATE, BACKGROUNDS[0]))

    def test_described_template(self):
        text = genjutsu_nodes.template_prompt(
            character="A man in a black suit and a white shirt, with round glasses",
            action="The performer walks slowly toward the camera while the camera pushes in",
            setting="A battlefield before a stone castle at sunset",
            lighting="Warm backlight from the low sun",
            framing="Full shot that becomes a medium close-up",
        )
        self.assertIn("complete outfit from that image: A man in a black suit and a white shirt, with round glasses.", text)
        self.assertIn("already fully transformed in the opening frame, walks slowly toward the camera while the camera pushes in.", text)
        self.assertIn("Setting in <Video 1>: the target video takes place in <Subject 2>, a battlefield before a stone castle at sunset.", text)
        self.assertIn("<Audio 1>: reference - keep the timing and ambience of the source audio.", text)
        self.assertIn("No invented dialogue.", text)
        final = genjutsu_prompt(text, BACKGROUNDS[0])
        self.assertIn("[video editing] The target video is an edited version of <Video 1>.", final)

    def test_generic_template_keeps_the_new_outfit(self):
        text = genjutsu_nodes.build_prompt("", BACKGROUNDS[0])
        self.assertIn("wears only the clothing and accessories shown in <Picture 1>", text)
        self.assertIn("performs the exact movement from <Video 1>.", text)
        self.assertNotIn("Setting in <Video 1>: the target video", text)

    def test_notes_speech_and_silence(self):
        text = genjutsu_nodes.build_prompt("He wears a red scarf", BACKGROUNDS[0], speech="Hello  there",
                                           language="English")
        self.assertIn("<Audio 1> is the source soundtrack; its voice (S1) belongs to <Subject 1>.", text)
        self.assertIn("<Subject 1> speaks (S1) <d>[English] Hello there</d>. He wears a red scarf.", text)
        self.assertIn("<Audio 1>: fully_copy - copy the source audio exactly;", text)
        silent = genjutsu_nodes.build_prompt("", BACKGROUNDS[0], speech="Hello", has_audio=False)
        self.assertNotIn("<Audio 1>", silent)
        self.assertNotIn("speaks (S1)", silent)
        self.assertIn("Natural ambience matching the scene. No dialogue.", silent)

    def test_clip_answer_parsing(self):
        answer = ("**ACTION:** The performer raises both arms\nand turns to the left.\n"
                  "SETTING: A small kitchen.\nLIGHTING: Soft daylight from a window.\n"
                  "FRAMING: Medium shot from the head to the waist.\nSPEAKING: yes")
        fields = genjutsu_nodes.parse_clip_answer(answer)
        self.assertEqual(fields["ACTION"], "The performer raises both arms and turns to the left.")
        self.assertEqual(fields["SPEAKING"], "yes")
        self.assertEqual(genjutsu_nodes._continuation(fields["ACTION"]), "raises both arms and turns to the left.")

    def test_unlabelled_clip_answer_is_read_in_order(self):
        # Qwen3-VL 8B answered the warrior clip this way (2026-10-24): five lines, no labels.
        answer = ("Advances steadily toward the camera, head held high.  \nA vast, grassy battlefield.  \n"
                  "Golden sunset light from the right.  \nThe camera slowly pushes in.  \nNo")
        fields = genjutsu_nodes.parse_clip_answer(answer)
        self.assertEqual(fields["ACTION"], "Advances steadily toward the camera, head held high.")
        self.assertEqual(fields["SPEAKING"], "No")
        self.assertEqual(genjutsu_nodes.parse_clip_answer("Just one line about the clip."), {})


class GenjutsuSizeTests(unittest.TestCase):
    def test_usable_frame_count_rounds_down_to_the_h3_grid(self):
        self.assertEqual(usable_frame_count(107), 107)
        self.assertEqual(usable_frame_count(124), 124)
        self.assertEqual(usable_frame_count(130), 124)
        self.assertEqual(usable_frame_count(360), 345)
        self.assertEqual(usable_frame_count(5), 5)
        with self.assertRaises(ValueError):
            usable_frame_count(4)

    def test_source_aspect_keeps_the_area(self):
        self.assertEqual(source_aspect_size(1344, 768, 1920, 1080), (1344, 768))
        width, height = source_aspect_size(1344, 768, 832, 1248)
        self.assertEqual((width % 32, height % 32), (0, 0))
        self.assertAlmostEqual(width / height, 832 / 1248, delta=0.03)
        self.assertLess(abs(width * height - 1344 * 768) / (1344 * 768), 0.05)


class GenjutsuGuideTests(unittest.TestCase):
    def test_missed_frames_take_the_nearest_mask(self):
        import torch

        masks = torch.zeros((20, 4, 4), dtype=torch.bool)
        masks[0, 1, 1] = True
        masks[10, 2, 2] = True
        filled, count = genjutsu_nodes._fill_missed_masks(masks.clone())
        self.assertEqual(count, 15)  # frames 1-9 and 11-16 are within 6 frames of a detection; 17-19 are not
        self.assertTrue(filled[3, 1, 1])
        self.assertTrue(filled[13, 2, 2])
        self.assertFalse(filled[17].any())

    def test_guide_size_matches_image_scale_to_total_pixels(self):
        width, height = genjutsu_nodes._guide_size(1344, 768, 0.15)
        scale = math.sqrt(0.15 * 1024 * 1024 / (1344 * 768))
        self.assertEqual((width, height), (round(1344 * scale), round(768 * scale)))


if __name__ == "__main__":
    unittest.main()
