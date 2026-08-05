"""Resolution controls whose persisted width and height are authoritative."""


ASPECT_RATIOS = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
    "Custom",
]


class SECoursesResolutionSync:
    """Return the visible dimensions while the frontend keeps all controls in sync."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS, {"default": "16:9 (Widescreen)"}),
                "megapixels": (
                    "FLOAT",
                    {
                        "default": 0.4,
                        "min": 0.01,
                        "max": 64.0,
                        "step": 0.01,
                        "round": 0.01,
                        "tooltip": "Changing this recalculates width and height in the browser.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 864,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Authoritative output width. Manual edits update megapixels and aspect ratio.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 480,
                        "min": 1,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Authoritative output height. Manual edits update megapixels and aspect ratio.",
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 128,
                        "step": 1,
                        "advanced": True,
                        "tooltip": "Width and height are snapped to this multiple when a control is edited.",
                    },
                ),
            }
        }

    CATEGORY = "SECourses/utilities"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "resolve"
    DESCRIPTION = (
        "Keeps aspect ratio, megapixels, width, and height synchronized in ComfyUI. "
        "Width and height are the authoritative values returned to the workflow."
    )

    def resolve(self, aspect_ratio, megapixels, width, height, multiple):
        del aspect_ratio, megapixels, multiple
        return (int(width), int(height))


NODE_CLASS_MAPPINGS = {
    "SECoursesResolutionSync": SECoursesResolutionSync,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SECoursesResolutionSync": "SECourses Resolution Sync",
}
