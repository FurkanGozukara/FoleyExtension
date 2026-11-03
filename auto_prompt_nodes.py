"""Write the prompt for you: when the prompt box is empty, a local vision model looks at the inputs and writes a MiniMax H3 prompt.

A typed prompt passes through unchanged and the model is never loaded, so presets behave as before for anyone who writes prompts."""
import json
import re

from .genjutsu_nodes import _scaled
from .reference_gallery_nodes import REF_PACK_TYPE, _load_reference_image, _resolve_reference_entry

MODES = ("cut and drag", "first last frame", "references")
_PHOTO_MEGAPIXELS = 0.5
_MAX_REFERENCES = 3
_QUESTIONS = {
    MODES[0]: ("Look at this photo. Answer with exactly three labelled lines and nothing else.\n"
               "SUBJECT: the one main object, animal or person in the photo that could move, in at most four plain words, for example: red car\n"
               "SCENE: one sentence about the place, the light and the mood.\n"
               "SOUND: one sentence about what would be heard as this subject moves here."),
    MODES[1]: ("The first image is the first frame of one continuous video shot and the second image is its last frame. "
               "Answer with exactly two labelled lines and nothing else.\n"
               "ACTION: two sentences about how the action, the camera and the light go from the first image to the last image, in playback order.\n"
               "SOUND: one sentence about the ambience and the sounds of this action."),
    MODES[2]: ("These images are references for a short video: people or characters, objects and places. Answer with exactly three labelled lines and nothing else.\n"
               "ROLES: for each image in order, a few words saying what it shows.\n"
               "ACTION: two sentences of a natural short scene with these references, with the camera move, in playback order.\n"
               "SOUND: one sentence about the ambience and the sounds of this scene."),
}
_FIRST_FRAME_QUESTION = ("This image is the first frame of one continuous video shot. Answer with exactly two labelled lines and nothing else.\n"
                         "ACTION: two sentences about what naturally happens next and how the camera moves, in playback order.\n"
                         "SOUND: one sentence about the ambience and the sounds of this action.")
_LABEL = re.compile(r"^[\s*#_`-]*(SUBJECT|SCENE|SOUND|ACTION|ROLES)[\s*_`]*:[\s*_`]*(.*)$", re.IGNORECASE)


def parse_answer(text):
    """The labelled lines of an answer as {LABEL: text}; a field may continue on the following lines."""
    fields, current = {}, None
    for line in str(text or "").splitlines():
        match = _LABEL.match(line)
        if match:
            current = match.group(1).upper()
            fields[current] = match.group(2).strip()
        elif current and line.strip():
            fields[current] = (fields[current] + " " + line.strip()).strip()
    return fields


def path_direction(path):
    """Words for where a drawn path goes, from its first to its last point (screen coordinates: y grows downward)."""
    try:
        points = json.loads(path) if isinstance(path, str) else path
        if points and isinstance(points[0], list):
            points = points[0]
        first, last = points[0], points[-1]
        dx, dy = float(last["x"]) - float(first["x"]), float(last["y"]) - float(first["y"])
    except (TypeError, ValueError, KeyError, IndexError):
        return "along its path"
    if abs(dx) >= abs(dy):
        return "to the right" if dx > 0 else "to the left"
    return "downward" if dy > 0 else "upward"


def cut_and_drag_prompt(fields, subject, direction):
    scene = fields.get("SCENE") or "A natural, photographic scene."
    sound = fields.get("SOUND") or "Natural ambience and the sounds of the moving subject."
    return ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
            f"integrated_multimodal_description: [Shot 1] {scene} The {subject} in <Picture 1> starts and moves smoothly {direction}, "
            "naturally for what it is, while the rest of the scene keeps living quietly. The camera stays still.\n\n"
            f"overall_soundscape: {sound}\n\nnon_diegetic_music: N/A")


def first_last_prompt(fields, has_last=True):
    action = fields.get("ACTION") or "One continuous, natural move."
    sound = fields.get("SOUND") or "Natural ambience of the scene."
    end = " End aligned with <Picture 2>." if has_last else ""
    return (f"integrated_multimodal_description: [Shot 1] Begin aligned with <Picture 1>. {action}{end} "
            f"Keep the same subject, place and lighting throughout.\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: N/A")


def references_prompt(fields, count):
    roles = fields.get("ROLES") or "the subject and the place of the images"
    action = fields.get("ACTION") or "A natural short scene with a slow camera move."
    sound = fields.get("SOUND") or "Natural ambience of the scene."
    names = ", ".join(f"@image{i + 1}" for i in range(count))
    return (f"subject_definitions:\n<Subject 1> comes from the references {names}: {roles}. Reference role: identity and place anchor.\n\n"
            "summary:\n[reference generation] <Subject 1> appears in a short live-action scene built from the references.\n\n"
            "retention_analysis:\n<Subject 1> (appears in [Shot 1]): fully_preserved - retain what the references show.\n"
            "No audio reference is used in this generation.\n\n"
            f"detailed_description:\nThe video is live-action, cinematic, natural light.\n[Shot 1] {action}\n\n"
            f"overall_soundscape:\n{sound}\n\nnon_diegetic_music:\nN/A")


class SECoursesH3AutoPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True, "tooltip": "The gallery's prompt. Typed text passes through unchanged."}),
                "mode": (MODES, {"default": MODES[0]}),
                "subject": ("STRING", {"default": "", "tooltip": "Cut and drag: what to move, e.g. red car. Empty: found automatically."}),
            },
            "optional": {
                "describe_clip": ("CLIP", {"lazy": True, "tooltip": "Qwen3-VL 8B. Loaded only when something must be written."}),
                "image": ("IMAGE", {"tooltip": "The photo or first frame."}),
                "last_image": ("IMAGE", {"tooltip": "The last frame, for first last frame."}),
                "references": (REF_PACK_TYPE, {"tooltip": "The gallery's references: returned with the written prompt; references mode also reads its images."}),
                "path": ("STRING", {"forceInput": True, "tooltip": "The path of the Spline Editor, so the prompt names its direction."}),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("STRING", "STRING", REF_PACK_TYPE)
    RETURN_NAMES = ("prompt", "subject", "references")
    FUNCTION = "write"
    DESCRIPTION = ("When the prompt is empty, a local vision model looks at the inputs and writes a MiniMax H3 prompt (and, for cut and drag, "
                   "names the subject to move). A typed prompt and a typed subject pass through and the model is not loaded.")

    @staticmethod
    def _needed(prompt, mode, subject):
        return not str(prompt or "").strip() or (mode == MODES[0] and not str(subject or "").strip())

    def check_lazy_status(self, prompt, mode, subject, describe_clip=None, **_):
        return ["describe_clip"] if self._needed(prompt, mode, subject) and describe_clip is None else []

    def write(self, prompt, mode, subject, describe_clip=None, image=None, last_image=None, references=None, path=None):
        import torch
        from comfy_extras.nodes_textgen import TextGenerate

        prompt, subject = str(prompt or ""), str(subject or "").strip()
        if not self._needed(prompt, mode, subject):
            return (prompt, subject, references)
        count = 0
        if mode == MODES[2]:
            entries = (references or {}).get("images") or [] if isinstance(references, dict) else []
            count = min(len(entries), _MAX_REFERENCES)
            if count == 0 and not prompt.strip():
                raise ValueError("Add at least one reference image to the gallery, or write a prompt.")
            shots = [_scaled(_load_reference_image(_resolve_reference_entry(e)), _PHOTO_MEGAPIXELS) for e in entries[:count]]
        else:
            if image is None and not prompt.strip():
                raise ValueError("Add your photo first, or write a prompt.")
            shots = [_scaled(image[:1], _PHOTO_MEGAPIXELS)]
            if mode == MODES[1] and last_image is not None:
                shots.append(_scaled(last_image[:1], _PHOTO_MEGAPIXELS))
        size = shots[0].shape[1:3]
        same = [torch.nn.functional.interpolate(s.movedim(-1, 1), size=tuple(size), mode="area").movedim(1, -1) for s in shots]
        question = _FIRST_FRAME_QUESTION if mode == MODES[1] and last_image is None else _QUESTIONS[mode]
        answer = TextGenerate.execute(clip=describe_clip, prompt=question, max_length=300, sampling_mode={"sampling_mode": "off"},
                                      image=torch.cat(same)).args[0].strip()
        fields = parse_answer(answer)
        if mode == MODES[0] and not subject:
            subject = re.sub(r"[.\"']", "", fields.get("SUBJECT", "")).strip()[:60] or "main subject"
        if not prompt.strip():
            if mode == MODES[0]:
                prompt = cut_and_drag_prompt(fields, subject, path_direction(path))
            elif mode == MODES[1]:
                prompt = first_last_prompt(fields, last_image is not None)
            else:
                prompt = references_prompt(fields, count)
        print(f"[SECoursesH3AutoPrompt] {mode}: subject '{subject}'; prompt: {prompt[:300]!r}", flush=True)
        pack = dict(references, prompt=prompt) if isinstance(references, dict) else references
        return (prompt, subject, pack)


_STORY_SPLIT = re.compile(r"^[ \t]*-{3,}[ \t]*$", re.MULTILINE)
_SOUND_START = re.compile(r"\n\s*overall_soundscape\s*:", re.IGNORECASE)
_DESCRIPTION = "integrated_multimodal_description: [Shot 1] "


def story_prompts(prompt):
    """A prompt with --- lines as one full H3 prompt per scene: the part before the first --- (description, sound, music) is
    shared, each later part is one scene. Without scenes the list is empty and nothing changes."""
    parts = [part.strip() for part in _STORY_SPLIT.split(str(prompt or "")) if part.strip()]
    if len(parts) < 2:
        return []
    shared, scenes = parts[0], parts[1:]
    if "integrated_multimodal_description" not in shared and "detailed_description" not in shared:
        shared = _DESCRIPTION + shared
    match = _SOUND_START.search(shared)
    head, tail = (shared[:match.start()].rstrip(), shared[match.start():]) if match else (shared, "")
    return [f"{head} {scene}{tail}" for scene in scenes]


class SECoursesH3StoryWindows:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip": "The conditioning of the one-prompt video."}),
                "clip": ("CLIP", {"tooltip": "The MiniMax H3 text encoder."}),
                "prompt": ("STRING", {"forceInput": True, "tooltip": "Shared description, then --- lines, each followed by one scene."}),
                "enabled": ("BOOLEAN", {"forceInput": True, "tooltip": "9. LONG VIDEO. Scenes need context windows."}),
            },
            "optional": {
                "uses_ref2va": ("BOOLEAN", {"forceInput": True, "tooltip": "Reference runs keep their one prompt: their text is encoded with the reference images."}),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "combine"
    DESCRIPTION = ("A long video that develops instead of repeating: write the shared description, then --- lines with one scene each. Every context "
                   "window follows the scene for its place in the video. A prompt without --- is not touched.")

    def combine(self, positive, clip, prompt, enabled, uses_ref2va=False):
        prompts = story_prompts(prompt)
        if not prompts:
            return (positive,)
        if not enabled or uses_ref2va:
            print("[SECoursesH3Story] the prompt has --- scenes, but they need 9. LONG VIDEO on and no reference media: using the whole text as one prompt.", flush=True)
            return (positive,)
        extras = dict(positive[0][1]) if positive else {}
        out = []
        for text in prompts:
            for embedding, options in clip.encode_from_tokens_scheduled(clip.tokenize(text)):
                out.append([embedding, {**extras, **options}])
        print(f"[SECoursesH3Story] {len(prompts)} scenes over the context windows.", flush=True)
        return (out,)


NODE_CLASS_MAPPINGS = {"SECoursesH3AutoPrompt": SECoursesH3AutoPrompt, "SECoursesH3StoryWindows": SECoursesH3StoryWindows}
NODE_DISPLAY_NAME_MAPPINGS = {"SECoursesH3AutoPrompt": "MiniMax H3 Auto Prompt (empty prompt = written for you)",
                              "SECoursesH3StoryWindows": "MiniMax H3 Story (--- scenes over a long video)"}
