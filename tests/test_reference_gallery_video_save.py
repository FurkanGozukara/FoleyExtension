import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class VideoSaveTests(unittest.TestCase):
    def setUp(self):
        types = SimpleNamespace(
            VideoContainer=SimpleNamespace(MP4="mp4"),
            VideoCodec=SimpleNamespace(H264="h264"),
        )
        patcher = mock.patch.dict(sys.modules, {
            "comfy_api.latest": SimpleNamespace(Types=types),
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def video(self, chroma):
        video = SimpleNamespace(
            get_dimensions=lambda: (16, 16),
            get_chroma_subsampling=lambda: chroma,
        )

        def save_to(path, **options):
            effective_chroma = options.get("chroma_subsampling", chroma)
            if effective_chroma not in (None, "444", "422", "420"):
                raise ValueError(f"Unsupported chroma subsampling: {effective_chroma}")
            Path(path).write_bytes(b"saved")

        video.save_to = mock.Mock(side_effect=save_to)
        return video

    def test_individual_save_recovers_legacy_none_and_keeps_metadata(self):
        video = self.video("none")
        prompt = {"1": {"class_type": "CreateVideo"}}
        extra = {"workflow": {"nodes": []}}
        with tempfile.TemporaryDirectory() as directory:
            paths = SimpleNamespace(
                get_output_directory=lambda: directory,
                get_save_image_path=lambda *args: (directory, "clip", 1, "", None),
            )
            with mock.patch.dict(sys.modules, {"folder_paths": paths}):
                saved = gallery._save_video_output(video, "clip", prompt, extra)
            self.assertEqual(Path(saved["fullpath"]).read_bytes(), b"saved")
        options = video.save_to.call_args.kwargs
        self.assertEqual(options["chroma_subsampling"], "420")
        self.assertEqual(options["metadata"], {**extra, "prompt": prompt})
        self.assertEqual(options["format"], "mp4")
        self.assertEqual(options["codec"], "h264")

    def test_valid_chroma_choices_keep_the_video_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            for chroma in (None, "444", "422", "420"):
                with self.subTest(chroma=chroma):
                    video = self.video(chroma)
                    gallery._save_h264_video(video, Path(directory) / "clip.mp4")
                    self.assertNotIn("chroma_subsampling", video.save_to.call_args.kwargs)

    def test_older_video_api_receives_no_new_keyword(self):
        save_to = mock.Mock()
        video = SimpleNamespace(save_to=save_to)
        gallery._save_h264_video(video, Path("clip.mp4"))
        save_to.assert_called_once_with(
            "clip.mp4", format="mp4", codec="h264", metadata=None,
        )

    def test_unrelated_invalid_chroma_still_reports_its_error(self):
        video = self.video("invalid")
        with self.assertRaisesRegex(ValueError, "Unsupported chroma subsampling: invalid"):
            gallery._save_h264_video(video, Path("unused.mp4"))

    def test_batch_merge_recovers_legacy_none_for_each_segment(self):
        videos = [self.video("none"), self.video("none")]
        with tempfile.TemporaryDirectory() as directory:
            paths = SimpleNamespace(
                get_temp_directory=lambda: directory,
                get_output_directory=lambda: directory,
                get_save_image_path=lambda *args: (directory, "merged", 1, "", None),
            )

            def concat(ffmpeg, manifest, output):
                self.assertEqual(len(Path(manifest).read_text().splitlines()), 2)
                output.write_bytes(b"merged")

            with (
                mock.patch.dict(sys.modules, {
                    "folder_paths": paths,
                    "imageio_ffmpeg": SimpleNamespace(get_ffmpeg_exe=lambda: "ffmpeg"),
                }),
                mock.patch.object(gallery, "_run_ffmpeg_concat", side_effect=concat),
            ):
                saved = gallery._merge_batch_video_group({
                    "root": "batch", "folder": "scene", "videos": videos,
                })
            self.assertEqual(Path(saved["fullpath"]).read_bytes(), b"merged")
        for video in videos:
            self.assertEqual(video.save_to.call_args.kwargs["chroma_subsampling"], "420")


if __name__ == "__main__":
    unittest.main()
