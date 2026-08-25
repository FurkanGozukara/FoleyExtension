"""Optional image input that is inactive until a file is selected."""

import os

import folder_paths
import nodes

try:
    from .media_extensions import VIDEO_EXTENSIONS, has_extension, image_extensions
except ImportError:  # direct test-module import
    from media_extensions import VIDEO_EXTENSIONS, has_extension, image_extensions


INIT_VIDEO_TYPE = "SECOURSES_INIT_VIDEO"


def _input_images():
    input_dir = folder_paths.get_input_directory()
    return sorted(
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
        and has_extension(name, image_extensions())
    )


def _input_images_and_videos():
    input_dir = folder_paths.get_input_directory()
    extensions = image_extensions() | VIDEO_EXTENSIONS
    return sorted(
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
        and has_extension(name, extensions)
    )


def _init_video_value(image):
    path = folder_paths.get_annotated_filepath(image)
    if not has_extension(path, VIDEO_EXTENSIONS):
        return {"path": None, "name": None}
    return {"path": path, "name": image}


class SECoursesLoadImage:
    """Image loader that also uses an uploaded video's final frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_input_images_and_videos(), {
                    "image_upload": True,
                    "tooltip": "Select or upload an image or video. A video outputs only its final frame and takes priority as the preset's init media.",
                }),
            }
        }

    CATEGORY = "SECourses/image"
    RETURN_TYPES = ("IMAGE", "MASK", INIT_VIDEO_TYPE)
    RETURN_NAMES = ("image", "mask", "init_video")
    FUNCTION = "load_image"
    DESCRIPTION = (
        "Loads any supported still image normally. When a video is selected, decodes only its final frame as "
        "the image and also exposes the source video to compatible continuation and merge nodes."
    )

    def load_image(self, image):
        init_video = _init_video_value(image)
        if init_video["path"] is None:
            loaded, mask = nodes.LoadImage().load_image(image)
            return loaded, mask, init_video

        import torch

        try:
            from .reference_gallery_nodes import _decode_last_video_frames
        except ImportError:  # direct test-module import
            from reference_gallery_nodes import _decode_last_video_frames

        frame = _decode_last_video_frames(init_video["path"], 1)
        mask = torch.zeros(frame.shape[:3], dtype=frame.dtype, device=frame.device)
        print(
            f"[SECoursesLoadImage] using the final frame of init video '{image}'.",
            flush=True,
        )
        return frame, mask, init_video

    @classmethod
    def IS_CHANGED(cls, image):
        return nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        return nodes.LoadImage.VALIDATE_INPUTS(image)


class SECoursesOptionalImage:
    NO_IMAGE = "(none - disabled)"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ([cls.NO_IMAGE, *_input_images()], {
                    "image_upload": True,
                    "tooltip": "No image is emitted until a file is selected. Uploading or selecting one enables the connected optional image input automatically.",
                }),
            }
        }

    CATEGORY = "SECourses/image"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_image"
    DESCRIPTION = "Returns no image when disabled, or loads the selected image automatically."

    def load_image(self, image):
        if not image or image == self.NO_IMAGE:
            return (None,)
        loaded, _mask = nodes.LoadImage().load_image(image)
        return (loaded,)

    @classmethod
    def IS_CHANGED(cls, image):
        if not image or image == cls.NO_IMAGE:
            return cls.NO_IMAGE
        return nodes.LoadImage.IS_CHANGED(image)

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not image or image == cls.NO_IMAGE:
            return True
        return nodes.LoadImage.VALIDATE_INPUTS(image)


NODE_CLASS_MAPPINGS = {
    "SECoursesLoadImage": SECoursesLoadImage,
    "SECoursesOptionalImage": SECoursesOptionalImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SECoursesLoadImage": "Load Init Image or Video (Final Frame)",
    "SECoursesOptionalImage": "Optional Image (Auto Enable)",
}
