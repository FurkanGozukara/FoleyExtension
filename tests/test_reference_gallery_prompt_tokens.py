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

    def test_at_tokens_ignore_case_and_harmless_spacing(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "Use @IMAGE1, @ Image #2, @vIdEo 1, and @SoUnD# 1",
                2, 1, 1, 1,
            ),
            "Use <Picture 1>, <Picture 2>, <Video 1>, and <Audio 2>",
        )

    def test_turkish_i_case_variants_are_safe_at_the_backend(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "Use @\u0130MAGE1, @v\u0131deo1, and <P\u0130CTURE 2>",
                1, 1, 0, 0,
            ),
            "Use <Picture 1>, <Video 1>, and <Picture 2>",
        )

    def test_native_labels_are_canonicalized_at_the_backend(self):
        self.assertEqual(
            gallery.translate_reference_tokens(
                "Use <picture 1>, <PICTURE1>, < image #02 >, <vId 1>, and <aUdIo 3>",
                0, 0, 0, 0,
            ),
            "Use <Picture 1>, <Picture 1>, <Picture 2>, <Video 1>, and <Audio 3>",
        )

    def test_non_reference_text_that_only_looks_similar_is_untouched(self):
        prompt = "Keep <pictures 1>, <Picture 123>, and user@image1.com unchanged"
        self.assertEqual(
            gallery.translate_reference_tokens(prompt, 9, 3, 3, 0),
            prompt,
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

    def test_native_labels_keep_their_index_even_when_dangling(self):
        self.assertEqual(
            gallery.translate_reference_tokens("keep <pIcTuRe 7> as typed @img2", 2, 0, 0, 0),
            "keep <Picture 7> as typed <Picture 2>",
        )


if __name__ == "__main__":
    unittest.main()
