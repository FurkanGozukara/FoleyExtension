"""Optional image input that is inactive until a file is selected."""

import os

import folder_paths
import nodes


class SECoursesOptionalImage:
    NO_IMAGE = "(none - disabled)"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": ([cls.NO_IMAGE, *sorted(files)], {
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
    "SECoursesOptionalImage": SECoursesOptionalImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SECoursesOptionalImage": "Optional Image (Auto Enable)",
}
