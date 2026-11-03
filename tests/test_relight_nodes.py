import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import relight_nodes  # noqa: E402
from relight_nodes import (  # noqa: E402
    DIRECTIONS, EFFECTS, PRESETS, SECoursesH3RelightPrompt, assign_roles, colour_words, compile_relight,
    direction_index, height_phrase, render_preview, side_words, wrap_azimuth,
)

HEADINGS = ("subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:",
            "non_diegetic_music:")


def defaults(**overrides):
    """The node's default widget values, with overrides."""
    values = {name: spec[1].get("default") for name, spec in SECoursesH3RelightPrompt.INPUT_TYPES()["required"].items()}
    values.update(overrides)
    return values


class RelightPromptTests(unittest.TestCase):
    def setUp(self):
        self.node = SECoursesH3RelightPrompt()

    def test_data_tables_and_combos(self):
        self.assertEqual(len(PRESETS), 20)
        self.assertEqual(len(EFFECTS), 25)
        self.assertTrue(all(name in EFFECTS for name in PRESETS), "each preset has the atmosphere of the same name")
        required = SECoursesH3RelightPrompt.INPUT_TYPES()["required"]
        self.assertEqual(len(required["preset"][0]), 21)
        self.assertEqual(len(required["atmosphere"][0]), 26)
        self.assertEqual(required["background"][0], ["keep and relight the original", "studio black", "studio white"])
        self.assertEqual(relight_nodes.NODE_DISPLAY_NAME_MAPPINGS["SECoursesH3RelightPrompt"], "MiniMax H3 Relight Prompt")
        self.assertEqual(SECoursesH3RelightPrompt.CATEGORY, "SECourses/references")

    def test_every_preset_compiles(self):
        for name, lights in PRESETS.items():
            with self.subTest(preset=name):
                prompt, summary, preview = self.node.build(**defaults(preset=name))
                self.assertIn("Atmosphere: " + EFFECTS[name], prompt)
                self.assertTrue(summary.startswith(f"Preset {name}. "))
                self.assertEqual(sum(prompt.count(role + ":") for role in
                                     ("Key light", "Backlight key", "Second key light", "Fill light", "Rim light")), len(lights))
                self.assertEqual(tuple(preview.shape), (1, 512, 512, 3))
                self.assertNotIn("<Picture 2>", prompt)
                self.assertIsNone(re.search(r"\b(spheres?|ball)\b", prompt, re.IGNORECASE))
                if len(lights) == 2:
                    self.assertIn("Both lights belong to one coherent setup", prompt)

    def test_source_tag_matches_source(self):
        video, _, _ = self.node.build(**defaults(source="video"))
        self.assertIn("<Video 1> is the source video to relight.", video)
        self.assertIn("[video editing]", video)
        self.assertNotIn("<Picture", video)
        image, _, _ = self.node.build(**defaults(source="image"))
        self.assertIn("<Picture 1> is the source image to relight.", image)
        self.assertIn("[reference generation]", image)
        self.assertNotIn("<Video", image)
        self.assertIn("no subject motion and no camera motion", image)

    def test_all_six_sections_in_order(self):
        prompt, _, _ = self.node.build(**defaults())
        positions = [prompt.find("\n" + heading) if i else prompt.find(heading) for i, heading in enumerate(HEADINGS)]
        self.assertNotIn(-1, positions)
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(prompt.rstrip().endswith("non_diegetic_music:\nN/A"))

    def test_azimuth_wrap_and_direction_words(self):
        self.assertEqual(wrap_azimuth(-180), 180)
        self.assertEqual(wrap_azimuth(190), -170)
        self.assertEqual(wrap_azimuth(-190), 170)
        self.assertEqual(wrap_azimuth(540), 180)
        self.assertEqual(wrap_azimuth(405), 45)
        cases = {0: "the front", 22.4: "the front", 22.5: "the front-right", 90: "the right", 135: "the back-right",
                 180: "behind", -135: "the back-left", -90: "the left", -45: "the front-left", 337.6: "the front", 270: "the left"}
        for azimuth, word in cases.items():
            self.assertEqual(DIRECTIONS[direction_index(azimuth)], word, azimuth)
        self.assertIn("viewer's right side of the frame, which is the subject's own left", side_words(direction_index(90)))
        self.assertIn("viewer's left side of the frame, which is the subject's own right", side_words(direction_index(-90)))
        prompt, _, _ = self.node.build(**defaults(light1_azimuth=-270.0))
        self.assertIn("coming from the right (azimuth 90)", prompt)
        self.assertIn("cast shadows fall toward the viewer's left", prompt)

    def test_height_bands_and_shadows(self):
        self.assertTrue(height_phrase(-20).startswith("from below"))
        self.assertEqual(height_phrase(-5), "just below eye level")
        self.assertEqual(height_phrase(0), "at eye level")
        self.assertTrue(height_phrase(20).startswith("low, 20 degrees"))
        self.assertTrue(height_phrase(45).startswith("mid-height"))
        self.assertTrue(height_phrase(70).startswith("high"))
        self.assertTrue(height_phrase(85).startswith("overhead"))
        prompt, _ = compile_relight([("soft area light", 80, 0, 90, 3200)])
        self.assertIn("Cast shadows pool short and directly beneath the subject.", prompt)
        prompt, _ = compile_relight([("hard spotlight", 60, 45, 10, 5600)])
        self.assertIn("long cast shadows", prompt)

    def test_colour_words(self):
        self.assertTrue(colour_words(3200).startswith("warm tungsten"))
        self.assertTrue(colour_words(1800).startswith("deep amber"))
        self.assertTrue(colour_words(10000).startswith("deep cold blue"))
        self.assertTrue(colour_words("#ff00ee").startswith("saturated magenta"))
        self.assertTrue(colour_words("#8c0000").startswith("deep red"))
        self.assertTrue(colour_words("#fff").startswith("neutral white"))

    def test_key_fill_rim_roles(self):
        lights = [("soft area light", 40, -60, 30, 5600), ("soft area light", 80, 45, 30, 5600), ("hard spotlight", 70, 180, 30, 5600)]
        key, roles = assign_roles(lights)
        self.assertEqual(key, 1)
        self.assertEqual(roles, ["Fill light", "Key light", "Rim light"])
        prompt, _ = compile_relight(lights)
        self.assertIn("Fill light: moderate at about 50% of the key's strength", prompt)
        self.assertLess(prompt.index("Key light:"), prompt.index("Fill light:"))
        self.assertIn("the key light from the front-right is dominant", prompt)
        self.assertEqual(assign_roles([("hard spotlight", 57, 90, 0, 5600), ("hard spotlight", 57, -90, 0, 5600)])[1],
                         ["Key light", "Second key light"])
        self.assertEqual(assign_roles([("hard spotlight", 60, 180, 10, 2200)])[1], ["Backlight key"])

    def test_custom_lights_enable_and_notes(self):
        prompt, _, _ = self.node.build(**defaults(light2_enable=False, light3_enable=False))
        self.assertEqual(prompt.count("light: ") + prompt.count("key: "), 1)
        prompt, summary, _ = self.node.build(**defaults(light2_enable=True, light3_enable=True, background="studio black",
                                                        atmosphere="volumetric light shafts",
                                                        notes="Additional direction: make the lamp glow"))
        self.assertIn("All three lights belong to one coherent setup", prompt)
        self.assertIn("seamless studio black", prompt)
        self.assertIn("Atmosphere: visible volumetric light shafts", prompt)
        self.assertEqual(prompt.count("Additional direction:"), 1)
        self.assertIn("Additional direction: make the lamp glow", prompt)
        self.assertNotIn("\n", summary)

    def test_light_preview(self):
        image = render_preview([("hard spotlight", 90, 90, 0, 5600)])
        self.assertEqual(tuple(image.shape), (1, 512, 512, 3))
        self.assertGreaterEqual(float(image.min()), 0.0)
        self.assertLessEqual(float(image.max()), 1.0)
        row = image[0, 256]
        self.assertGreater(float(row[360].mean()), float(row[152].mean()) + 0.2, "a light from the viewer's right lights the right side")


if __name__ == "__main__":
    unittest.main()
