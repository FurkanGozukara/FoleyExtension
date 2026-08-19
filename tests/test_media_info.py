"""The /secourses/media_info helper the gallery's live token meter relies on."""
import math
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import reference_gallery_nodes as gallery


def _write_png(path, width, height):
    from PIL import Image

    Image.new("RGB", (width, height), (10, 20, 30)).save(path)


def _write_rotated_jpeg(path, width, height):
    """A JPEG stored landscape whose EXIF orientation (6) displays it as portrait."""
    from PIL import Image

    image = Image.new("RGB", (width, height), (40, 50, 60))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, format="JPEG", exif=exif.tobytes())


def _write_wav(path, seconds, rate=16000):
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 20.0))) for i in range(frames)))


def _write_mp4(path, width, height, frames, fps=12):
    import av
    import numpy as np

    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            array = np.full((height, width, 3), (index * 9) % 255, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class MediaInfoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        patcher = mock.patch.object(gallery, "_resolve_reference_path", side_effect=lambda file: str(self.root / file.split(" [")[0]))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        gallery._MEDIA_INFO_CACHE.clear()

    def test_image_dimensions_follow_exif_orientation(self):
        _write_png(self.root / "plain.png", 640, 360)
        _write_rotated_jpeg(self.root / "rotated.jpg", 800, 400)
        self.assertEqual(gallery._cached_media_info("plain.png [input]"), {
            "kind": "image", "width": 640, "height": 360, "duration": None, "has_audio": False, "fps": None,
        })
        rotated = gallery._cached_media_info("rotated.jpg")
        self.assertEqual((rotated["kind"], rotated["width"], rotated["height"]), ("image", 400, 800))

    def test_audio_duration(self):
        _write_wav(self.root / "voice.wav", 3.37)
        info = gallery._cached_media_info("voice.wav")
        self.assertEqual(info["kind"], "audio")
        self.assertAlmostEqual(info["duration"], 3.37, places=2)
        self.assertTrue(info["has_audio"])

    def test_video_dimensions_and_duration(self):
        try:
            _write_mp4(self.root / "clip.mp4", 320, 176, 24)
        except Exception as error:  # pragma: no cover - depends on the local ffmpeg build
            self.skipTest(f"cannot encode a test video here: {error}")
        info = gallery._cached_media_info("clip.mp4 [input]")
        self.assertEqual((info["kind"], info["width"], info["height"], info["has_audio"]), ("video", 320, 176, False))
        self.assertAlmostEqual(info["duration"], 2.0, delta=0.2)
        self.assertAlmostEqual(info["fps"], 12.0, delta=0.5)

    def test_cache_invalidates_when_the_file_changes(self):
        _write_png(self.root / "img.png", 64, 64)
        self.assertEqual(gallery._cached_media_info("img.png")["width"], 64)
        _write_png(self.root / "img.png", 128, 96)
        # a same-second rewrite could share mtime; force a distinct size so the key changes either way
        self.assertEqual(gallery._cached_media_info("img.png")["width"], 128)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            gallery._cached_media_info("nope.png")


if __name__ == "__main__":
    unittest.main()
