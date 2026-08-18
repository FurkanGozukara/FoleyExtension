"""Optional image input that is inactive until a file is selected."""

import os

import folder_paths
import nodes

try:
    from .media_extensions import has_extension, image_extensions
except ImportError:  # direct test-module import
    from media_extensions import has_extension, image_extensions


def _input_images():
    input_dir = folder_paths.get_input_directory()
    return sorted(
        name
        for name in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, name))
        and has_extension(name, image_extensions())
    )


class SECoursesLoadImage:
    """Core-compatible image loader whose picker exposes every supported format."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_input_images(), {
                    "image_upload": True,
                    "tooltip": "Select or upload any still-image format supported by this ComfyUI installation.",
                }),
            }
        }

    CATEGORY = "SECourses/image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    DESCRIPTION = "Loads any still-image format supported by the installed Pillow/ComfyUI decoders."

    def load_image(self, image):
        return nodes.LoadImage().load_image(image)

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
    "SECoursesLoadImage": "Load Image (All Supported Formats)",
    "SECoursesOptionalImage": "Optional Image (Auto Enable)",
}
