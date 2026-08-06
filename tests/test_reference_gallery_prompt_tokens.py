import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class TranslateReferenceTokensTests(unittest.TestCase):
    def test_attached_tokens_translate_to_labels(self):
        self.assertEqual(
            gallery.translate_reference_tokens("a photo of @image1 and @image2", 2, 0, 0, 0),
            "a photo of <Picture 1> and <Picture 2>",
        )

    def test_audio_tokens_offset_past_video_soundtracks(self):
        self.assertEqual(
            gallery.translate_reference_tokens("play @audio1 loud", 0, 2, 1, 2),
            "play <Audio 3> loud",
        )

    def test_dangling_token_is_omitted_not_an_error(self):
        self.assertEqual(
            gallery.translate_reference_tokens("a photo of @image1 and @image3 walking", 2, 0, 0, 0),
            "a photo of <Picture 1> and walking",
        )

    def test_zero_and_out_of_range_tokens_are_omitted(self):
        self.assertEqual(gallery.translate_reference_tokens("bad @image0 token", 1, 0, 0, 0), "bad token")
        self.assertEqual(gallery.translate_reference_tokens("@image5 hello", 1, 0, 0, 0), "hello")

    def test_prompt_with_only_dangling_tokens_still_executes(self):
        self.assertEqual(
            gallery.translate_reference_tokens("just @image1 and @video1 and @audio1", 0, 0, 0, 0),
            "just and and",
        )

    def test_omission_preserves_newlines(self):
        self.assertEqual(
            gallery.translate_reference_tokens("line1\n@image9\nline2", 1, 0, 0, 0),
            "line1\n\nline2",
        )

    def test_legacy_labels_pass_through_even_when_dangling(self):
        self.assertEqual(
            gallery.translate_reference_tokens("keep <Picture 7> as typed @img2", 2, 0, 0, 0),
            "keep <Picture 7> as typed <Picture 2>",
        )


if __name__ == "__main__":
    unittest.main()
