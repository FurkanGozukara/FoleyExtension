import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


class ReferenceGalleryTrimManifestTests(unittest.TestCase):
    def parse_single_video(self, entry):
        manifest = gallery._parse_manifest(json.dumps({"videos": [entry]}))
        return manifest["videos"][0]

    def test_trim_window_is_preserved(self):
        entry = self.parse_single_video(
            {"file": "reference_gallery/clip.mp4", "name": "clip.mp4", "trim_start": 1.5, "trim_end": 4.0}
        )
        self.assertEqual(entry["trim_start"], 1.5)
        self.assertEqual(entry["trim_end"], 4.0)

    def test_untrimmed_entry_stays_untouched(self):
        entry = self.parse_single_video({"file": "reference_gallery/clip.mp4", "name": "clip.mp4"})
        self.assertNotIn("trim_start", entry)
        self.assertNotIn("trim_end", entry)

    def test_degenerate_and_malformed_trims_fall_back_to_untrimmed(self):
        for trim in (
            {"trim_start": 4.0, "trim_end": 4.0},
            {"trim_start": 6.0, "trim_end": 2.0},
            {"trim_start": "junk", "trim_end": 4.0},
            {"trim_start": float("nan"), "trim_end": 4.0},
        ):
            with self.subTest(trim=trim):
                entry = self.parse_single_video({"file": "reference_gallery/clip.mp4", **trim})
                self.assertNotIn("trim_start", entry)
                self.assertNotIn("trim_end", entry)

    def test_start_only_trim_is_preserved(self):
        entry = self.parse_single_video({"file": "reference_gallery/clip.mp4", "trim_start": 2.0})
        self.assertEqual(entry["trim_start"], 2.0)
        self.assertNotIn("trim_end", entry)

    def test_negative_start_clamps_to_zero(self):
        entry = self.parse_single_video(
            {"file": "reference_gallery/clip.mp4", "trim_start": -3.0, "trim_end": 5.0}
        )
        self.assertEqual(entry["trim_start"], 0.0)
        self.assertEqual(entry["trim_end"], 5.0)

    def test_entry_trim_window_caps_duration(self):
        self.assertEqual(gallery._entry_trim_window({}, 15.0), (0.0, 15.0))
        self.assertEqual(
            gallery._entry_trim_window({"trim_start": 2.0, "trim_end": 6.0}, 15.0), (2.0, 4.0)
        )
        self.assertEqual(
            gallery._entry_trim_window({"trim_start": 2.0, "trim_end": 60.0}, 15.0), (2.0, 15.0)
        )
        self.assertEqual(gallery._entry_trim_window({"trim_start": 2.0}, 15.0), (2.0, 15.0))

    def test_gallery_pack_carries_trim_fields_without_decoding(self):
        manifest = json.dumps({
            "videos": [{
                "file": "reference_gallery/clip.mp4",
                "name": "clip.mp4",
                "trim_start": 1.0,
                "trim_end": 3.5,
            }],
        })
        node = gallery.SECoursesReferenceGallery()
        with mock.patch.object(
            gallery, "_decode_video_frames", side_effect=AssertionError("eager video decode")
        ):
            packs, _prompts, _active, _merge = node.collect("use @video1", manifest, 24, 15)
        self.assertEqual(packs[0]["videos"][0]["trim_start"], 1.0)
        self.assertEqual(packs[0]["videos"][0]["trim_end"], 3.5)


class ReferenceGalleryTrimDecodeTests(unittest.TestCase):
    def write_index_video(self, path, frame_count=72, fps=24, step=3):
        import av
        import numpy as np

        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width = 320
            stream.height = 192
            stream.pix_fmt = "yuv420p"
            for index in range(frame_count):
                pixels = np.empty((192, 320, 3), dtype=np.uint8)
                pixels[..., 0] = index * step
                pixels[..., 1] = 80
                pixels[..., 2] = 160
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def test_video_trim_start_skips_leading_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trim.mp4"
            self.write_index_video(path)

            frames = gallery._decode_video_frames(
                str(path), fps_out=24, max_frames=12, trim_start=1.0
            )

        self.assertEqual(frames.shape[0], 12)
        # Frame 24 (t=1.0s) encodes red = 24 * 3 = 72; allow for codec loss.
        first_red = float(frames[0, 96, 160, 0])
        self.assertAlmostEqual(first_red, 72.0, delta=16.0)
        last_red = float(frames[-1, 96, 160, 0])
        self.assertAlmostEqual(last_red, 105.0, delta=16.0)

    def test_video_without_trim_still_starts_at_first_frame(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notrim.mp4"
            self.write_index_video(path)

            frames = gallery._decode_video_frames(str(path), fps_out=24, max_frames=6)

        self.assertEqual(frames.shape[0], 6)
        self.assertAlmostEqual(float(frames[0, 96, 160, 0]), 0.0, delta=16.0)

    def test_trim_beyond_video_end_raises_actionable_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "short.mp4"
            self.write_index_video(path, frame_count=24)

            with self.assertRaisesRegex(ValueError, "trim start"):
                gallery._decode_video_frames(str(path), fps_out=24, max_frames=6, trim_start=30.0)

    def write_ramp_wav(self, path, sample_rate=8000, seconds=2):
        import wave

        import numpy as np

        samples = np.arange(sample_rate * seconds, dtype=np.int16)
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(samples.tobytes())

    def test_audio_trim_start_is_sample_accurate(self):
        sample_rate = 8000
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ramp.wav"
            self.write_ramp_wav(path, sample_rate=sample_rate)

            audio = gallery._load_reference_audio(str(path), max_seconds=0.5, trim_start=1.0)

        self.assertEqual(audio["sample_rate"], sample_rate)
        self.assertEqual(tuple(audio["waveform"].shape), (1, 1, sample_rate // 2))
        first = float(audio["waveform"][0, 0, 0]) * (2 ** 15)
        last = float(audio["waveform"][0, 0, -1]) * (2 ** 15)
        self.assertAlmostEqual(first, sample_rate * 1.0, delta=1.0)
        self.assertAlmostEqual(last, sample_rate * 1.5 - 1, delta=1.0)

    def test_audio_without_trim_matches_previous_behavior(self):
        sample_rate = 8000
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "ramp.wav"
            self.write_ramp_wav(path, sample_rate=sample_rate)

            audio = gallery._load_reference_audio(str(path), max_seconds=1.0)

        self.assertEqual(tuple(audio["waveform"].shape), (1, 1, sample_rate))
        self.assertAlmostEqual(float(audio["waveform"][0, 0, 0]) * (2 ** 15), 0.0, delta=1.0)

    def test_video_spec_honors_trim_window(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "spec.mp4"
            self.write_index_video(path)
            entry = {
                "file": "reference_gallery/spec.mp4",
                "name": "spec.mp4",
                "trim_start": 0.5,
                "trim_end": 1.5,
            }
            with mock.patch.object(gallery, "_resolve_reference_entry", return_value=str(path)):
                spec, = gallery._prepare_video_references(
                    [entry], fps=24, max_seconds=15.0, length=124,
                    byte_budget=512 * 1024 * 1024,
                )

        self.assertEqual(spec["trim_start"], 0.5)
        # 1.0s window at 24 fps is 24 frames, aligned down to the 17k+5 grid.
        self.assertEqual(spec["max_frames"], 22)
        self.assertEqual(spec["audio_seconds"], 1.0)

    def test_base64_soundtrack_loader_forwards_start_seconds(self):
        import base64

        expected = {"waveform": object(), "sample_rate": 32000}
        with mock.patch.object(gallery, "_decode_video_audio", return_value=expected) as decode:
            output, = gallery.SECoursesLoadVideoAudioB64().load(
                base64.b64encode(b"fake-container").decode("ascii"),
                max_seconds=15.0,
                start_seconds=12.5,
            )

        self.assertIs(output, expected)
        self.assertEqual(decode.call_args.kwargs["trim_start"], 12.5)

    def test_base64_soundtrack_loader_defaults_to_no_trim(self):
        import base64

        expected = {"waveform": object(), "sample_rate": 32000}
        with mock.patch.object(gallery, "_decode_video_audio", return_value=expected) as decode:
            gallery.SECoursesLoadVideoAudioB64().load(
                base64.b64encode(b"fake-container").decode("ascii"),
                max_seconds=15.0,
            )

        self.assertEqual(decode.call_args.kwargs["trim_start"], 0.0)

    def test_untrimmed_video_spec_is_unchanged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "plain.mp4"
            self.write_index_video(path)
            entry = {"file": "reference_gallery/plain.mp4", "name": "plain.mp4"}
            with mock.patch.object(gallery, "_resolve_reference_entry", return_value=str(path)):
                spec, = gallery._prepare_video_references(
                    [entry], fps=24, max_seconds=15.0, length=124,
                    byte_budget=512 * 1024 * 1024,
                )

        self.assertEqual(spec["trim_start"], 0.0)
        self.assertEqual(spec["max_frames"], gallery._usable_video_frame_count(124, 24, 15.0))


if __name__ == "__main__":
    unittest.main()
