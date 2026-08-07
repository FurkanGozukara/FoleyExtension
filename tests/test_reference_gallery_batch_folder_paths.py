import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class BatchFolderPathTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.folder = Path(self._temp.name) / "batch prompts"
        self.folder.mkdir()
        self.resolved = self.folder.resolve()

    def assertNormalizes(self, raw):
        self.assertEqual(gallery._normalize_batch_folder(raw), self.resolved)

    def test_plain_path_with_spaces(self):
        self.assertNormalizes(str(self.folder))

    def test_surrounding_whitespace(self):
        self.assertNormalizes(f"  {self.folder}  \n")

    def test_double_quoted(self):
        self.assertNormalizes(f'"{self.folder}"')

    def test_single_quoted(self):
        self.assertNormalizes(f"'{self.folder}'")

    def test_quoted_with_inner_whitespace(self):
        self.assertNormalizes(f'" {self.folder} "')

    def test_forward_slashes(self):
        self.assertNormalizes(str(self.folder).replace(os.sep, "/"))

    def test_doubled_separators(self):
        raw = str(self.folder).replace(os.sep, os.sep + os.sep)
        self.assertNormalizes(raw)

    def test_trailing_separator(self):
        self.assertNormalizes(str(self.folder) + os.sep)

    def test_empty_and_blank_return_none(self):
        self.assertIsNone(gallery._normalize_batch_folder(""))
        self.assertIsNone(gallery._normalize_batch_folder("   "))
        self.assertIsNone(gallery._normalize_batch_folder('""'))
        self.assertIsNone(gallery._normalize_batch_folder(None))

    def test_missing_directory_raises(self):
        with self.assertRaises(ValueError):
            gallery._normalize_batch_folder(str(self.folder / "does not exist"))

    @unittest.skipIf(os.name == "nt", "posix-only fallback")
    def test_backslash_separators_on_posix(self):
        raw = str(self.folder).replace("/", "\\")
        self.assertNormalizes(raw)

    @unittest.skipIf(os.name == "nt", "posix-only fallback")
    def test_shell_escaped_spaces_on_posix(self):
        raw = str(self.folder).replace(" ", "\\ ")
        self.assertNormalizes(raw)

    @unittest.skipUnless(os.name == "nt", "windows-only mixed separators")
    def test_mixed_separators_on_windows(self):
        raw = str(self.folder).replace("\\", "/", 1)
        self.assertNormalizes(raw)


if __name__ == "__main__":
    unittest.main()
