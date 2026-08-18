import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import media_extensions
import reference_gallery_nodes as gallery


class FolderBatchInitMediaTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.folder = Path(self._temp.name)

    def touch(self, name, content=b""):
        path = self.folder / name
        path.write_bytes(content)
        return path

    def collect(self, enabled=True):
        fallback = {"images": [], "videos": [], "audios": []}
        return gallery._collect_folder_batch(self.folder, fallback, 24, 15, enabled)[0]

    def test_exact_basename_image_audio_and_combined_pairs(self):
        self.touch("1.txt", b"image prompt")
        self.touch("1.psd")
        self.touch("2.txt", b"audio prompt")
        self.touch("2.aiff")
        self.touch("3.txt", b"combined prompt")
        self.touch("3.webp")
        self.touch("3.ape")
        self.touch("shared.jpg")
        self.touch("shared.flac")
        self.touch("motion.mp4")

        image, audio, combined = self.collect()

        self.assertEqual(image["init_image"]["name"], "1.psd")
        self.assertIsNone(image["init_audio"])
        self.assertIsNone(audio["init_image"])
        self.assertEqual(audio["init_audio"]["name"], "2.aiff")
        self.assertEqual(combined["init_image"]["name"], "3.webp")
        self.assertEqual(combined["init_audio"]["name"], "3.ape")
        for pack in (image, audio, combined):
            self.assertEqual([entry["name"] for entry in pack["images"]], ["shared.jpg"])
            self.assertEqual([entry["name"] for entry in pack["audios"]], ["shared.flac"])
            self.assertEqual([entry["name"] for entry in pack["videos"]], ["motion.mp4"])

    def test_matching_is_opt_in_for_existing_reference_and_audio_only_workflows(self):
        self.touch("scene.txt", b"prompt")
        self.touch("scene.png")
        self.touch("scene.wav")

        pack = self.collect(enabled=False)[0]

        self.assertIsNone(pack["init_image"])
        self.assertIsNone(pack["init_audio"])
        self.assertEqual([entry["name"] for entry in pack["images"]], ["scene.png"])
        self.assertEqual([entry["name"] for entry in pack["audios"]], ["scene.wav"])

    def test_duplicate_same_kind_init_files_fail_clearly(self):
        self.touch("scene.txt", b"prompt")
        self.touch("scene.png")
        self.touch("scene.jpg")

        with self.assertRaisesRegex(ValueError, "multiple same-basename init files"):
            self.collect()

    def test_extension_rosters_cover_installed_images_and_broad_audio_formats(self):
        self.assertIn(".psd", media_extensions.image_extensions())
        self.assertIn(".exr", media_extensions.image_extensions())
        self.assertIn(".aiff", media_extensions.audio_extensions())
        self.assertIn(".ape", media_extensions.audio_extensions())
        self.assertIn(".ac4", media_extensions.audio_extensions())
        self.assertIn(".wma", media_extensions.audio_extensions())
        self.assertNotIn(".mp4", media_extensions.audio_extensions())
        self.assertIn(".mp4", media_extensions.audio_input_extensions())

    def test_matched_audio_duration_wins_over_filename_and_workflow_duration(self):
        self.touch("speech_8.txt", b"prompt")
        path = self.folder / "speech_8.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(8000)
            output.writeframes(b"\0\0" * 10000)
        pack = self.collect()[0]

        duration = gallery.SECoursesBatchDuration().resolve(pack, 5.0)[0]

        self.assertAlmostEqual(duration, 1.25, places=3)


class AutoInitImageTests(unittest.TestCase):
    def pack(self, with_reference=False):
        return {
            "version": 4,
            "prompt": "move",
            "images": [{"file": "ref.png"}] if with_reference else [],
            "videos": [],
            "audios": [],
            "init_image": {"name": "scene.png", "path": "scene.png", "source": "batch_folder"},
        }

    def test_init_image_becomes_fl2va_first_frame_without_references(self):
        marker = torch.ones((1, 16, 16, 3))
        with (
            mock.patch.object(gallery, "_resolve_reference_entry", return_value="scene.png"),
            mock.patch.object(gallery, "_load_reference_image", return_value=marker),
            mock.patch.object(gallery.SECoursesMiniMaxH3TextOnly, "encode", return_value=("cond", "latent")) as encode,
        ):
            result = gallery.SECoursesMiniMaxH3Auto().encode(
                object(), object(), object(), self.pack(), 640, 384, 124, "match"
            )

        self.assertEqual(result, ("cond", "latent", False))
        self.assertIs(encode.call_args.kwargs["first_frame"], marker)

    def test_init_image_uses_ref2va_start_picture_when_other_references_exist(self):
        marker = torch.ones((1, 16, 16, 3))
        with (
            mock.patch.object(gallery, "_resolve_reference_entry", return_value="scene.png"),
            mock.patch.object(gallery, "_load_reference_image", return_value=marker),
            mock.patch.object(gallery.SECoursesMiniMaxH3References, "encode", return_value=("cond", "latent")) as encode,
        ):
            result = gallery.SECoursesMiniMaxH3Auto().encode(
                object(), object(), object(), self.pack(True), 640, 384, 124, "match"
            )

        self.assertEqual(result, ("cond", "latent", True))
        self.assertIs(encode.call_args.kwargs["continuation_frame"], marker)


if __name__ == "__main__":
    unittest.main()
