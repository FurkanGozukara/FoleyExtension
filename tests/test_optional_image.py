import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import optional_image_nodes


class FakeLoadImage:
    loaded = []

    def load_image(self, image):
        self.loaded.append(image)
        return f"pixels:{image}", "mask"

    @classmethod
    def IS_CHANGED(cls, image):
        return f"hash:{image}"

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return f"valid:{image}"


class OptionalImageTests(unittest.TestCase):
    def setUp(self):
        FakeLoadImage.loaded.clear()
        self.load_image_patch = mock.patch.object(optional_image_nodes.nodes, "LoadImage", FakeLoadImage)
        self.load_image_patch.start()

    def tearDown(self):
        self.load_image_patch.stop()

    def test_disabled_value_emits_none_without_loading(self):
        node = optional_image_nodes.SECoursesOptionalImage()

        self.assertEqual(node.load_image(node.NO_IMAGE), (None,))
        self.assertEqual(node.IS_CHANGED(node.NO_IMAGE), node.NO_IMAGE)
        self.assertIs(node.VALIDATE_INPUTS(node.NO_IMAGE), True)
        self.assertEqual(FakeLoadImage.loaded, [])

    def test_selected_image_loads_and_delegates_validation(self):
        node = optional_image_nodes.SECoursesOptionalImage()

        self.assertEqual(node.load_image("ending.png"), ("pixels:ending.png",))
        self.assertEqual(node.IS_CHANGED("ending.png"), "hash:ending.png")
        self.assertEqual(node.VALIDATE_INPUTS("ending.png"), "valid:ending.png")
        self.assertEqual(FakeLoadImage.loaded, ["ending.png"])

    def test_disabled_choice_precedes_sorted_input_images(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "z.png").touch()
            Path(directory, "a.jpg").touch()
            Path(directory, "design.psd").touch()
            Path(directory, "notes.txt").touch()
            with mock.patch.object(optional_image_nodes.folder_paths, "get_input_directory", return_value=directory):
                image_spec = optional_image_nodes.SECoursesOptionalImage.INPUT_TYPES()["required"]["image"]

        self.assertEqual(image_spec[0], ["(none - disabled)", "a.jpg", "design.psd", "z.png"])
        self.assertIs(image_spec[1]["image_upload"], True)

    def test_required_loader_delegates_to_core_image_node(self):
        node = optional_image_nodes.SECoursesLoadImage()

        self.assertEqual(node.load_image("start.psd"), ("pixels:start.psd", "mask"))
        self.assertEqual(node.IS_CHANGED("start.psd"), "hash:start.psd")
        self.assertEqual(node.VALIDATE_INPUTS("start.psd"), "valid:start.psd")


if __name__ == "__main__":
    unittest.main()
