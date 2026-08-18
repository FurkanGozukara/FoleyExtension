"""Media extensions accepted by the SECourses file and folder inputs."""

from functools import lru_cache
from pathlib import Path


VIDEO_EXTENSIONS = {
    ".3g2", ".3gp", ".avi", ".f4v", ".flv", ".m2ts", ".m4v", ".mkv",
    ".mov", ".mp4", ".mpeg", ".mpg", ".mts", ".ogv", ".rm", ".ts",
    ".vob", ".webm", ".wmv",
}

# Dedicated audio containers/codecs supported by FFmpeg/PyAV. Video containers
# stay separate so a same-name MP4 remains a reference video in folder batches.
_AUDIO_EXTENSIONS = {
    ".302", ".aa", ".aa3", ".aac", ".aax", ".ac3", ".ac4", ".adts", ".aea",
    ".afc", ".aif", ".aifc", ".aiff", ".al", ".alac", ".amr", ".ape",
    ".apl", ".aptx", ".aptxhd", ".ast", ".au", ".aud", ".avr", ".bcstm",
    ".bfstm", ".binka", ".caf", ".daud", ".dff", ".dsf", ".dts", ".dtshd",
    ".eac3", ".ec3", ".fap", ".flac", ".g722", ".gsm", ".iamf", ".it",
    ".laf", ".loas", ".m2a", ".m4a", ".m4b", ".mac", ".mca", ".mka",
    ".mlp", ".mod", ".mp1", ".mp2", ".mp3", ".mpa", ".mpc", ".oga",
    ".ogg", ".oma", ".omg", ".opus", ".paf", ".pvf", ".ra", ".ram", ".rka",
    ".s3m", ".sb", ".sbc", ".sds", ".shn", ".sln", ".snd", ".sox",
    ".spx", ".sw", ".tak", ".tta", ".ub", ".ul", ".uw", ".voc",
    ".w64", ".wa", ".wav", ".wave", ".wma", ".wv", ".xm", ".xwma",
}


@lru_cache(maxsize=1)
def image_extensions():
    """Every still-image extension supported by the installed Pillow build."""
    try:
        from PIL import Image

        Image.init()
        extensions = {
            extension.lower()
            for extension, image_format in Image.registered_extensions().items()
            if image_format in Image.OPEN
        }
        try:
            import av

            extensions.update(f".{extension.lower()}" for extension in av.ContainerFormat("image2").extensions)
            extensions.update({".heic", ".heif"})
        except (ImportError, ValueError):
            pass
    except ImportError:
        extensions = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    return frozenset(extensions - VIDEO_EXTENSIONS)


def audio_extensions():
    return frozenset(_AUDIO_EXTENSIONS)


def audio_input_extensions():
    """Single-file init audio also accepts soundtracks from video containers."""
    return audio_extensions() | VIDEO_EXTENSIONS


def has_extension(path, extensions):
    return Path(path).suffix.lower() in extensions
