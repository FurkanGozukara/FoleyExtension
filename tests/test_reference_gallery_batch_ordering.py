import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class BatchFilenameOrderingTests(unittest.TestCase):
    def test_windows_natural_key_orders_numeric_filename_chunks(self):
        names = ["image10.png", "Image3.png", "image1.png", "image02.png", "image2.png"]

        ordered = sorted(names, key=gallery._windows_natural_sort_key)

        self.assertEqual(
            ordered,
            ["image1.png", "image02.png", "image2.png", "Image3.png", "image10.png"],
        )

    def test_batch_images_receive_natural_filename_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("10.png", "2.png", "1.png", "ignore.txt"):
                (root / name).touch()

            entries = gallery._batch_media_entries(root, root)

        self.assertEqual(
            [entry["name"] for entry in entries["images"]],
            ["1.png", "2.png", "10.png"],
        )

    def test_recursive_prompt_scan_uses_natural_path_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder_name in ("scene10", "scene2", "scene1"):
                folder = root / folder_name
                folder.mkdir()
                (folder / "prompt.txt").touch()

            files = gallery._batch_relevant_files(root)

        self.assertEqual(
            [path.relative_to(root).as_posix() for path in files],
            ["scene1/prompt.txt", "scene2/prompt.txt", "scene10/prompt.txt"],
        )


if __name__ == "__main__":
    unittest.main()
