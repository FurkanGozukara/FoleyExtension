"""MiniMax H3 Genjutsu: swap the performer of a video for the character of a reference image.

SAM 3.1 finds the performer in every frame and only the performer is colour-inverted, so MiniMax H3 Ref2VA
copies the clip's motion, timing and camera without copying the performer's identity (the inverted motion
guide of MDMZ's "H3 Character Swap" workflow). The guide is shrunk to a small reference video, the character
comes from a reference image and the clip's soundtrack stays the soundtrack. Prompt rules decide whether the
clip's environment is kept (character replacement) or the image's (motion transfer).

- ``SECoursesH3GenjutsuMotionGuide`` builds the guide from decoded video frames (SwarmUI's MiniMax H3
  extension uses it, and so does the gallery node below).
- ``SECoursesH3GenjutsuPrompt`` uses a complete H3 prompt (eg an LLM answer) as written, or builds the Genjutsu
  template, optionally described by a local vision LLM (Qwen3-VL 8B), and adds the background rules.
- ``SECoursesH3Genjutsu`` does both for the first video of a SECourses Reference Gallery pack and hands the
  pack on, so the ComfyUI presets keep their gallery, folder batch and MiniMax H3 adapter nodes.
"""

from __future__ import annotations

import math
import re

try:
    from .reference_gallery_nodes import (
        REF_PACK_TYPE,
        VIDEO_DECODE_AREA_CAP,
        _decode_video_audio,
        _decode_video_frames,
        _entry_trim_window,
        _fit_dimensions,
        _load_reference_image,
        _oriented_image_dimensions,
        _resolve_reference_entry,
        _video_metadata,
    )
except ImportError:  # direct test-module import
    from reference_gallery_nodes import (
        REF_PACK_TYPE,
        VIDEO_DECODE_AREA_CAP,
        _decode_video_audio,
        _decode_video_frames,
        _entry_trim_window,
        _fit_dimensions,
        _load_reference_image,
        _oriented_image_dimensions,
        _resolve_reference_entry,
        _video_metadata,
    )


FPS = 24
BACKGROUNDS = ("video (character replacement)", "image (motion transfer)")
METHODS = ("original Genjutsu", "Genjutsu + task LoRA", "aligned guide + task LoRA", "masked edit", "face swap LoRA", "background replace")
MASKED_EDIT, FACE_SWAP, BACKGROUND_REPLACE = METHODS[3], METHODS[4], METHODS[5]
# Methods that hold pixels of the clip through the Masked Video Edit Latent node.
HOLD_METHODS = (MASKED_EDIT, BACKGROUND_REPLACE)
CAMERA_STILL_THRESHOLD = 0.012
MAIN_PEOPLE_RATIO = 0.12
_METHOD_INPUT = (METHODS, {
    "default": METHODS[0],
    "tooltip": "Original keeps the inverted native video reference. Task LoRA improves character transfer with the same guide. "
               "Aligned guide uses the original frames on H3's output timeline and the task LoRA; "
               "it uses more tokens and dense INT8 attention. Use Genjutsu Apply with either task method. "
               "Masked edit regenerates only the target (shown to H3 as a grey figure) and keeps every other pixel of "
               "the clip; it needs the Masked Video Edit Latent node of the presets. Face swap LoRA keeps the clip as it "
               "is and changes only the face to the one in @image1 (FaceSwap LoRA, trigger word Faceswap). Background replace "
               "keeps the target person exactly as filmed and regenerates everything around them from the location in "
               "@image1; the prompt is written for you (it also needs the Masked Video Edit Latent node).",
})
# Masked edit (grey mannequin guide and edit mask, as in openjutsu): the person's mask gets this much of its height as
# headroom above the head for hair and hats; the grey figure keeps soft shading so the motion stays readable.
MASK_HEADROOM = 0.1
MANNEQUIN_BLUR = 0.04
FACE_SWAP_TRIGGER = "Faceswap"
_MASKED_VIDEO_LINE = (
    "<Video 1> is the source footage being edited. In it the person to replace is shown as a smooth grey figure; the "
    "grey figure supplies movement only - the same gestures, steps, gaze, expressions and mouth shapes - and none of "
    "its appearance. The backgrounds of the pictures do not enter the scene."
)
# EmptyImage colour 8421504 (#808080) of the source workflow: the canvas behind the inverted performer when the
# background comes from the character image.
NEUTRAL_GREY = 128.0 / 255.0
CANVAS_MULTIPLE = 32
# Frames per SAM 3.1 detection call; bounds the float copy of the source frames.
DETECT_CHUNK = 16
# A frame where SAM 3.1 finds nobody borrows the nearest detected mask within this many frames, so a missed
# detection does not show the original performer uninverted for one frame.
MASK_FILL_FRAMES = 6
PREVIEW_FRAMES = 6

# The source workflow's RegexReplace chain, unchanged: (pattern, replacement, flags, count).
_SUBJECT_HEADING = r"^[ \t>*_`#]*subject[ _]definitions[ \t*_`]*:[*_`]*"
_RETENTION_HEADING = r"^[ \t>*_`#]*retention[ _]analysis[ \t*_`]*:[*_`]*"
_TIDY_RULES = (
    (r"^\s*```[a-zA-Z]*\s*$", "", re.IGNORECASE | re.MULTILINE, 0),
    (r"^[\s*#_`]*subject[_ ]definitions[\s*_`]*:[ \t*]*", r"subject_definitions:\n", re.IGNORECASE | re.MULTILINE, 0),
    (r"^[\s*#_`]*retention[_ ]analysis[\s*_`]*:[ \t*]*", r"retention_analysis:\n", re.IGNORECASE | re.MULTILINE, 0),
)
_VIDEO_BACKGROUND_RULES = (
    (
        _SUBJECT_HEADING,
        "subject_definitions:\n"
        "<Subject 2> is the environment of <Video 1>: its location, walls, floor, background, set, furniture, objects "
        "and light sources, exactly as they appear in <Video 1> (described in the 'Setting in <Video 1>' line if "
        "given).\n"
        "<Video 1> is the source video for the target video edit.\n"
        "Background rule: <Picture 1> provides only <Subject 1> - the character's identity, face, hair and outfit "
        "design. Everything around <Subject 1> in every frame is <Subject 2>: the location, background, set, "
        "objects and lighting of <Video 1>.\n"
        "Lighting rule: relight <Subject 1> to match <Video 1> exactly - same light direction, intensity, colour "
        "temperature, coloured light and reflections, shadows, contrast and colour grade as the original performer "
        "- so the character looks filmed in that scene, not composited. If a 'Scene lighting' line is given, match "
        "it exactly.\n"
        "Framing rule: keep exactly the shot size, crop, lens and subject scale of <Video 1> in every frame, as "
        "described in the 'Framing on the performer' line if given; <Subject 1> occupies the same area of the frame, "
        "cut by the frame edges at the same places on the body, as the performer.",
        re.IGNORECASE | re.MULTILINE,
        1,
    ),
    (
        r"(?:\[[ \t]*reference[ \t]+generation([^\]\n]*)\]([\s\S]*?))?" + _RETENTION_HEADING,
        "[video editing\\g<1>] The target video is an edited version of <Video 1>. <Subject 2> stays exactly as it is "
        "in <Video 1>, and the only edit is the performer, who becomes <Subject 1>.\\g<2>\n"
        "retention_analysis:\n"
        "<Subject 2> (appears in every shot): fully_preserved - the location, walls, floor, background, set, "
        "furniture, objects and light sources of <Video 1> are retained exactly, in the same positions and from the "
        "same camera angle.\n"
        "Lighting: preserved from <Video 1> - <Subject 1> takes the lighting and colour grade of <Subject 2>.",
        re.IGNORECASE | re.MULTILINE,
        1,
    ),
)
_IMAGE_BACKGROUND_RULES = (
    (
        _SUBJECT_HEADING,
        "subject_definitions:\n"
        "Background rule: <Subject 1> appears in the environment shown in <Picture 1>; keep that background, setting "
        "and lighting. Use <Video 1> only for motion, timing and camera movement; ignore its background and setting. "
        "Light <Subject 1> exactly as in <Picture 1>. Do not bring any coloured lights, haze, smoke or effects from "
        "<Video 1> into the scene.\n"
        "Framing rule: start from the framing, shot size and lens of <Picture 1>, and apply the camera movement of "
        "<Video 1> relative to that framing. Keep <Subject 1>'s full body in frame whenever the action needs it.",
        re.IGNORECASE | re.MULTILINE,
        1,
    ),
    (
        _RETENTION_HEADING,
        "retention_analysis:\n"
        "Environment: taken from <Picture 1> - keep the background, setting and lighting of <Picture 1>; do not "
        "reproduce the setting of <Video 1>.",
        re.IGNORECASE | re.MULTILINE,
        1,
    ),
    (
        r"Setting\s+in\s+<Video\s*1>\s*:[^\n]*\n?|(Scene\s+lighting|Framing)\s+on\s+the\s+performer\s+in\s+<Video\s*1>"
        r"\s*:.*?(?=\n\s*\n|(Scene\s+lighting|Framing)\s+on\s+the\s+performer|\Z)",
        "",
        re.IGNORECASE | re.DOTALL,
        0,
    ),
)


def genjutsu_prompt(prompt, background):
    """Tidy a pasted H3 prompt and add the background rules of ``background`` (one of BACKGROUNDS)."""
    if background not in BACKGROUNDS:
        raise ValueError(f"Unknown Genjutsu background '{background}'. Use one of: {', '.join(BACKGROUNDS)}.")
    text = str(prompt or "")
    for pattern, replacement, flags, count in _TIDY_RULES:
        text = re.sub(pattern, replacement, text, count=count, flags=flags)
    # A free-form prompt has no H3 section headings for the rules to attach to.
    if not re.search(_SUBJECT_HEADING, text, re.IGNORECASE | re.MULTILINE):
        text = "subject_definitions:\n" + text
    if not re.search(_RETENTION_HEADING, text, re.IGNORECASE | re.MULTILINE):
        text = text.rstrip() + "\n\nretention_analysis:\n"
    rules = _VIDEO_BACKGROUND_RULES if background == BACKGROUNDS[0] else _IMAGE_BACKGROUND_RULES
    for pattern, replacement, flags, count in rules:
        text = re.sub(pattern, replacement, text, count=count, flags=flags)
    return text


def is_full_prompt(text):
    """True for a complete MiniMax H3 prompt (an LLM answer or a hand-written one), which is used as written."""
    return re.search(r"subject[ _]definitions[\s*_`]*:", str(text or ""), re.IGNORECASE) is not None


def method_prompt(prompt, method, background, has_audio=True, target="", autogenerated=False, still_camera=False):
    if method not in METHODS:
        raise ValueError(f"Unknown Genjutsu method '{method}'.")
    if method == METHODS[0]:
        return prompt
    if method == FACE_SWAP:
        return prompt
    if method == BACKGROUND_REPLACE:
        return prompt if is_full_prompt(prompt) else background_replace_prompt(prompt, has_audio, still_camera)
    text = str(prompt)
    if method == MASKED_EDIT:
        # No task LoRA and no trigger: the grey figure is the motion source and the clip's own pixels are held.
        text = text.replace("<Video 1> is the source video for the target video edit.", _MASKED_VIDEO_LINE, 1)
        if _MASKED_VIDEO_LINE not in text:
            text = re.sub(r"^\s*subject_definitions:\s*\n", "subject_definitions:\n" + _MASKED_VIDEO_LINE + "\n", text,
                          count=1, flags=re.IGNORECASE | re.MULTILINE)
        if autogenerated and target.strip() and target.strip().lower() != "person":
            text = text.replace("subject_definitions:\n", "subject_definitions:\n"
                                f"Only the grey figure matching '{target.strip()}' is replaced.\n", 1)
        return text
    if method == METHODS[2]:
        # The aligned guide has no native Video/Audio reference slot. Preserve the other slots' identities.
        def video(match):
            number = int(match.group(1) or match.group(2))
            if number == 1:
                return "the guide video"
            return f"<Video {number - 1}>" if match.group(1) else f"@video{number - 1}"

        text = re.sub(r"<\s*Video\s*(\d+)\s*>|(?<![0-9A-Za-z_@])@(?:video|vid)#?(\d+)(?![0-9A-Za-z])", video, text, flags=re.IGNORECASE)
        if has_audio:
            text = re.sub(r"<\s*Audio\s*(\d+)\s*>",
                          lambda m: "the source soundtrack" if int(m.group(1)) == 1 else f"<Audio {int(m.group(1)) - 1}>",
                          text, flags=re.IGNORECASE)
    if autogenerated:
        text = text.replace("\nPhotorealistic, continuous shot.\n",
                            "\nKeep the character's visual style and body proportions from <Picture 1>. Continuous shot.\n")
        if target.strip() and target.strip().lower() != "person":
            text = text.replace("subject_definitions:\n", "subject_definitions:\n"
                                f"Only replace the guide performer matching '{target.strip()}'. Other people keep their own appearance.\n", 1)
    trigger = "body_swap" if background == BACKGROUNDS[0] else "motion_transfer"
    text = re.sub(r"^\s*(?:body_swap|motion_transfer)\s*:\s*", "", text, count=1, flags=re.IGNORECASE)
    return trigger + ": " + text


def camera_is_still(frames, threshold=CAMERA_STILL_THRESHOLD):
    """Whether the clip's camera stays put. Frames half a second apart are compared on a small copy; the median pixel ignores
    a moving person, and a plain wall counts as still, which is right: nothing there asks for a camera move."""
    import torch

    count = int(frames.shape[0])
    if count < 2:
        return True
    gap = max(1, min(FPS // 2, count - 1))
    starts = sorted({round(i * (count - gap - 1) / 11) for i in range(12)}) if count - gap > 1 else [0]
    changes = []
    for start in starts:
        pair = frames[[start, start + gap]].float().movedim(-1, 1)
        if pair.max() > 1.5:
            pair = pair / 255.0
        small = torch.nn.functional.interpolate(pair, size=(54, 96), mode="area")
        changes.append(float((small[0] - small[1]).abs().mean(dim=0).median()))
    changes.sort()
    return changes[len(changes) // 2] < threshold


def background_replace_prompt(notes="", has_audio=True, still_camera=False):
    """The whole prompt of the background replace method. The person's pixels are held in the latent and pasted back, so
    the prompt only describes the new place from @image1; no reference video is used."""
    notes = str(notes or "").strip()
    sound = ("The existing soundtrack continues unchanged; no new dialogue, no music." if has_audio
             else "Natural ambience of the location; no dialogue, no music.")
    return (
        "subject_definitions:\n"
        "<Picture 1> is the new location.\n\n"
        "summary:\n"
        "[video editing] A person stays exactly as they are; everything around them is the location of <Picture 1>, in "
        "every frame.\n\n"
        "retention_analysis:\n"
        "<Picture 1>: partially_preserved - preserve the place: its architecture or landscape, colours, light and mood.\n\n"
        "detailed_description:\n"
        "Continuous, photorealistic shot in the look of a real camera.\n"
        "[Shot 1] The whole environment around the person is the location of <Picture 1>, extended naturally to fill the "
        "frame, with natural perspective and depth of field, "
        + ("and the camera is locked off on a tripod and never moves: the framing of the location stays exactly the same from "
           "the first frame to the last. " if still_camera else "moving with the camera. ")
        + "The person stands on the ground of "
        "this location, with soft natural contact shadows. The light, colours and weather stay constant from the first "
        "frame to the last. No new people, no text, no logos, no watermarks.\n"
        + (notes + "\n" if notes else "")
        + "\noverall_soundscape:\n" + sound + "\n\nnon_diegetic_music:\nN/A"
    )


def face_swap_prompt(prompt):
    """The FaceSwap LoRA was trained on its trigger word alone; short notes, if any, follow it."""
    notes = str(prompt or "").strip()
    if notes.lower().startswith(FACE_SWAP_TRIGGER.lower()):
        return notes
    return FACE_SWAP_TRIGGER + (f". {notes}" if notes else "")


def _unit_images(images):
    import torch
    return images.float().mul(1.0 / 255.0) if images.dtype == torch.uint8 else images


def _sentence(text):
    text = " ".join(str(text or "").split()).strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def _continuation(action):
    """Turn 'The performer walks ...' into 'walks ...' so it continues '<Subject 1>, already fully transformed ...,'."""
    action = _sentence(action)
    action = re.sub(r"^(?:the\s+(?:main\s+)?(?:performer|character|person|subject|man|woman|figure)|he|she|they)\s+", "",
                    action, flags=re.IGNORECASE)
    return action[:1].lower() + action[1:]


def template_prompt(character="", action="", setting="", lighting="", framing="", speaking=False, speech="",
                    language="English", has_audio=True, notes=""):
    """A complete Genjutsu H3 prompt in the structure of the Gemini instructions; empty fields get generic wording."""
    speech = " ".join(str(speech or "").split())
    speaking = has_audio and (speaking or bool(speech))
    if character:
        outfit = f": {_sentence(character)}"
    else:
        outfit = (". <Subject 1> wears only the clothing and accessories shown in <Picture 1>; nothing of the "
                  "performer's own outfit or props is kept.")
    lines = [
        "subject_definitions:",
        "<Subject 1> is the adult person in <Picture 1>, with the exact face, skin tone, hairstyle, body proportions "
        "and complete outfit from that image" + outfit,
        "<Video 1> is the source clip; use its performer's motion, timing and camera movement.",
    ]
    if has_audio:
        lines.append("<Audio 1> is the source soundtrack; its voice (S1) belongs to <Subject 1>." if speaking
                     else "<Audio 1> is the source soundtrack.")
    lines += [
        "",
        "summary:",
        "[reference generation] From the very first frame, <Subject 1> fully replaces the performer in <Video 1> and "
        "performs the same actions with the same timing for the entire clip.",
        "",
        "retention_analysis:",
        "<Subject 1>: attribute_transfer - transfer the face, hairstyle, outfit and body proportions consistently to "
        "every frame.",
        "<Video 1>: partially_preserved - preserve the performer's action, pose, timing and camera movement; do not "
        "preserve the original performer's face, makeup or face paint, hair, hair accessories, clothing or props.",
    ]
    if has_audio:
        lines.append("<Audio 1>: fully_copy - copy the source audio exactly; <Subject 1>'s lip movements are "
                     "synchronized word for word to the speech (S1)." if speaking
                     else "<Audio 1>: reference - keep the timing and ambience of the source audio.")
    shot = "[Shot 1] <Subject 1>, already fully transformed in the opening frame, " + (
        _continuation(action) if action else "performs the exact movement from <Video 1>.")
    if speaking:
        shot += (f" <Subject 1> speaks (S1) <d>[{language or 'English'}] {speech}</d>." if speech
                 else " <Subject 1> speaks (S1) whenever the performer talks in <Video 1>.")
    if notes and str(notes).strip():
        shot += " " + _sentence(notes)
    shot += (" The camera follows the same path and framing as <Video 1>. Stable face, stable outfit, natural body "
             "motion, no identity drift, no morphing, no duplicated limbs.")
    lines += ["", "detailed_description:", "Photorealistic, continuous shot.", shot]
    if setting:
        setting = _sentence(setting)
        lines.append("Setting in <Video 1>: the target video takes place in <Subject 2>, " + setting[:1].lower() + setting[1:])
    if lighting:
        lines.append("Scene lighting on the performer in <Video 1>: " + _sentence(lighting))
    if framing:
        lines.append("Framing on the performer in <Video 1>: " + _sentence(framing))
    if not has_audio:
        soundscape = "Natural ambience matching the scene. No dialogue."
    elif speaking:
        soundscape = "Natural ambience and physical sounds matching the source clip."
    else:
        soundscape = "Natural ambience and physical sounds matching the source clip. No invented dialogue."
    lines += ["", "overall_soundscape:", soundscape, "", "non_diegetic_music:", "N/A"]
    return "\n".join(lines)


# Questions for the local describer (any ComfyUI text encoder that can generate text with images, eg Qwen3-VL 8B).
_CHARACTER_QUESTION = (
    "Describe the person in this image for a video generation prompt, in one or two sentences: apparent age, build, "
    "face, skin tone, hair, facial hair, glasses, and every piece of clothing and every accessory they wear. Visual "
    "facts only. Do not describe the background, the pose, the lighting or the image style. Answer with the "
    "description only."
)
_CLIP_QUESTION = (
    "These {count} images are frames of one video clip in time order; the first image is the start of the clip. "
    "Answer with exactly these five lines, each beginning with its label and a colon:\n"
    "ACTION: <what the main performer does from the first frame to the last - movements, direction, gestures and "
    "facial expression - and how the camera moves; two to four sentences that start with a verb, eg 'walks slowly "
    "toward the camera'; describe only body movement and never mention the performer's appearance, clothing, costume, "
    "armour, hair or any object they hold>\n"
    "SETTING: <the place where the clip was filmed, as it would look with nobody in it: the type of location, the "
    "background, furniture and objects, colours and materials; one or two sentences>\n"
    "LIGHTING: <the colour, direction and brightness of the light on the performer, the shadows and the contrast; "
    "one sentence>\n"
    "FRAMING: <the shot size at the first frame and at the last frame, eg full shot from head to feet, medium shot "
    "from head to waist, close-up of the face; one sentence>\n"
    "SPEAKING: <yes if the performer's mouth moves as in talking or singing in these frames, otherwise no>"
)
_CLIP_FIELDS = ("ACTION", "SETTING", "LIGHTING", "FRAMING", "SPEAKING")
DESCRIBE_FRAMES = 6
DESCRIBE_FRAME_MEGAPIXELS = 0.3
DESCRIBE_CHARACTER_MEGAPIXELS = 0.5


def parse_clip_answer(text):
    """The labelled lines of the clip answer as {field: text}; a field may continue on following lines. An answer of
    exactly one unlabelled line per field (small models sometimes drop the labels) is read in the asked order."""
    fields = {}
    current = None
    label = re.compile(r"^[\s*#_`-]*(" + "|".join(_CLIP_FIELDS) + r")[\s*_`]*:[\s*_`]*(.*)$", re.IGNORECASE)
    lines = [line for line in str(text or "").splitlines() if line.strip()]
    for line in lines:
        match = label.match(line)
        if match:
            current = match.group(1).upper()
            fields[current] = match.group(2).strip()
        elif current:
            fields[current] = (fields[current] + " " + line.strip()).strip()
    if not fields and len(lines) == len(_CLIP_FIELDS):
        fields = {name: line.strip() for name, line in zip(_CLIP_FIELDS, lines)}
    return fields


def _scaled(images, megapixels):
    import comfy.utils
    import torch

    images = images[..., :3]
    images = images.to(torch.float32).mul_(1.0 / 255.0) if images.dtype == torch.uint8 else images.to(torch.float32)
    width, height = _guide_size(int(images.shape[2]), int(images.shape[1]), megapixels)
    width, height = max(32, width // 16 * 16), max(32, height // 16 * 16)
    return comfy.utils.common_upscale(images.movedim(-1, 1), width, height, "area", "disabled").movedim(1, -1)


def describe(describe_clip, frames, character):
    """Character, action, setting, lighting, framing and speaking from a local vision LLM (greedy, so repeatable)."""
    from comfy_extras.nodes_textgen import TextGenerate

    def ask(question, images, max_length):
        return TextGenerate.execute(clip=describe_clip, prompt=question, max_length=max_length,
                                    sampling_mode={"sampling_mode": "off"}, image=images).args[0].strip()

    details = {}
    if character is not None:
        details["character"] = ask(_CHARACTER_QUESTION, _scaled(character[:1], DESCRIBE_CHARACTER_MEGAPIXELS), 200)
    if frames is not None:
        count = int(frames.shape[0])
        picks = sorted({round(i * (count - 1) / max(1, DESCRIBE_FRAMES - 1)) for i in range(DESCRIBE_FRAMES)})
        answer = ask(_CLIP_QUESTION.format(count=len(picks)), _scaled(frames[picks], DESCRIBE_FRAME_MEGAPIXELS), 450)
        fields = parse_clip_answer(answer)
        if "ACTION" not in fields:
            print(f"[SECoursesH3Genjutsu] the clip description has no labelled lines, using generic wording: {answer!r}", flush=True)
        for name in ("action", "setting", "lighting", "framing"):
            if fields.get(name.upper()):
                details[name] = fields[name.upper()]
        details["speaking"] = fields.get("SPEAKING", "").strip().lower().startswith("yes")
    print("[SECoursesH3Genjutsu] described: " + "; ".join(f"{key}: {value}" for key, value in details.items()), flush=True)
    return details


def build_prompt(prompt, background, describe_clip=None, frames=None, character=None, speech="", language="English",
                 has_audio=True):
    """The final Genjutsu prompt: a complete H3 prompt is used as written, anything else (empty, or short notes
    such as spoken words or outfit details) becomes the Genjutsu template, described by ``describe_clip`` when given."""
    if is_full_prompt(prompt):
        text = str(prompt)
    else:
        details = describe(describe_clip, frames, character) if describe_clip is not None else {}
        text = template_prompt(**details, speech=speech, language=language, has_audio=has_audio, notes=prompt)
    return genjutsu_prompt(text, background)


def usable_frame_count(frames):
    """Largest MiniMax H3 length (17k + 5) that fits ``frames`` source frames."""
    frames = int(frames)
    if frames < 5:
        raise ValueError(f"MiniMax H3 Genjutsu needs a source clip of at least 5 frames ({5 / FPS:.2f}s); got {frames}.")
    return frames - (frames - 5) % 17


def source_aspect_size(width, height, source_width, source_height):
    """Output size with the pixel area of ``width`` x ``height`` and the aspect ratio of the source clip."""
    area = int(width) * int(height)
    ratio = float(source_width) / float(source_height)
    out_width = max(CANVAS_MULTIPLE, round(math.sqrt(area * ratio) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    out_height = max(CANVAS_MULTIPLE, round(math.sqrt(area / ratio) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    return out_width, out_height


def _guide_size(width, height, megapixels):
    # ImageScaleToTotalPixels with resolution_steps 1, as in the source workflow.
    scale = math.sqrt(megapixels * 1024 * 1024 / (width * height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _fill_missed_masks(masks):
    """Give frames without a detection the nearest detected mask within MASK_FILL_FRAMES frames."""
    found = masks.flatten(1).any(dim=1).tolist()
    detected = [index for index, hit in enumerate(found) if hit]
    if not detected:
        return masks, 0
    filled = 0
    for index, hit in enumerate(found):
        if hit:
            continue
        nearest = min(detected, key=lambda other: abs(other - index))
        if abs(nearest - index) <= MASK_FILL_FRAMES:
            masks[index] = masks[nearest]
            filled += 1
    return masks, filled


def _grow(mask, pixels):
    import torch

    if pixels <= 0:
        return mask
    kernel = 2 * int(pixels) + 1
    return torch.nn.functional.max_pool2d(mask.unsqueeze(1), kernel, stride=1, padding=int(pixels)).squeeze(1)


def add_headroom(masks, headroom=MASK_HEADROOM):
    """Extend each frame's mask upward by ``headroom`` x its height, in the columns near its top (head, hair, raised
    hands - not outstretched arms), so a masked edit has room for hair and hats. ``masks``: bool [T, H, W]."""
    import torch

    out = masks.clone()
    rows = torch.arange(masks.shape[1]).unsqueeze(1)
    for index in range(masks.shape[0]):
        mask = masks[index]
        has = mask.any(dim=0)
        if not bool(has.any()):
            continue
        hit_rows = mask.any(dim=1).nonzero().flatten()
        top, bottom = int(hit_rows[0]), int(hit_rows[-1])
        room = round((bottom - top + 1) * float(headroom))
        if room <= 0:
            continue
        col_top = torch.where(has, mask.float().argmax(dim=0), torch.full_like(has, masks.shape[1], dtype=torch.long))
        near = has & (col_top <= top + room)
        out[index] |= (rows >= col_top - room) & (rows < col_top) & near.unsqueeze(0)
    return out


def _blur(image, sigma):
    """Separable Gaussian blur of [N, 1, H, W] with ``sigma`` in pixels."""
    import torch

    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=image.dtype)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = (kernel / kernel.sum()).view(1, 1, 1, -1)
    image = torch.nn.functional.conv2d(torch.nn.functional.pad(image, (radius, radius, 0, 0), mode="replicate"), kernel)
    return torch.nn.functional.conv2d(torch.nn.functional.pad(image, (0, 0, radius, radius), mode="replicate"),
                                      kernel.transpose(-1, -2))


def grey_mannequins(frames, masks, blur=MANNEQUIN_BLUR):
    """Draw every masked person as a smooth grey figure (shaded by the frame's own luma, softer for taller people) so
    H3 reads the motion without the identity. ``frames`` float [T, H, W, 3], ``masks`` float [T, H, W]."""
    import torch

    luma = (frames[..., 0] * 0.299 + frames[..., 1] * 0.587 + frames[..., 2] * 0.114).unsqueeze(1)
    heights, tops = [], []
    for rows in masks.amax(dim=2) > 0.5:  # [H] rows that hold the person, per frame
        hit = rows.nonzero().flatten()
        heights.append(int(hit[-1]) - int(hit[0]) + 1 if hit.numel() else 0)
        tops.append(int(hit[0]) if hit.numel() else 0)
    soft = ((torch.tensor(heights, dtype=frames.dtype) * float(blur) - 1.0) / 7.0).clamp(0.0, 1.0).view(-1, 1, 1, 1)
    luma = _blur(luma, 1.0) * (1.0 - soft) + _blur(luma, 8.0) * soft
    # The head band (top fifth of the figure) is blurred harder, so glasses, hats and hair lines of the original
    # performer do not show through the grey figure (the 2026-11-03 knight test kept the walker's glasses).
    rows = torch.arange(luma.shape[-2], dtype=frames.dtype).view(1, -1, 1)
    for index, (top, height) in enumerate(zip(tops, heights)):
        if height < 8:
            continue
        band = ((top + 0.22 * height - rows) / max(1.0, 0.05 * height)).clamp(0.0, 1.0)  # 1 on the head, ramps to 0
        head = _blur(luma[index:index + 1], max(2.0, 0.03 * height))
        luma[index] = luma[index] * (1.0 - band) + head[0] * band
    grey = (0.45 + 0.45 * luma).squeeze(1).unsqueeze(-1).expand_as(frames)
    mask = masks.unsqueeze(-1)
    return frames * (1.0 - mask) + grey * mask


def build_motion_guide(frames, sam3_model, sam3_clip, target, background, guide_megapixels, threshold=0.5,
                       refine_iterations=2, mask_grow=0, guide_style="inverted"):
    """Motion guide for ``frames`` ([T, H, W, 3] float 0-1 or uint8).

    ``guide_style`` "inverted" colour-inverts the performer (Genjutsu), "mannequin" draws a grey figure on the clip
    (masked edit), "person on grey" shows the person alone on neutral grey (background replace). Returns the guide frames (float, about ``guide_megapixels``), the performer masks (bool, frame size)
    and the number of frames that had no detection of their own."""
    import comfy.utils
    import torch
    from comfy_extras.nodes_sam3 import SAM3_Detect

    if background not in BACKGROUNDS:
        raise ValueError(f"Unknown Genjutsu background '{background}'. Use one of: {', '.join(BACKGROUNDS)}.")
    target = str(target or "").strip()
    if not target:
        raise ValueError("Write who to replace in 'target', eg 'person', 'man' or 'man in red:1'.")
    count, height, width = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    conditioning = sam3_clip.encode_from_tokens_scheduled(sam3_clip.tokenize(target))

    def as_float(chunk):
        chunk = chunk[..., :3]
        if chunk.dtype == torch.uint8:
            return chunk.to(torch.float32).mul_(1.0 / 255.0)
        return chunk.to(torch.float32)

    masks = torch.zeros((count, height, width), dtype=torch.bool)
    for start in range(0, count, DETECT_CHUNK):
        chunk = as_float(frames[start:start + DETECT_CHUNK])
        detected = SAM3_Detect.execute(model=sam3_model, image=chunk, conditioning=conditioning, threshold=threshold,
                                       refine_iterations=refine_iterations, individual_masks=False).args[0]
        masks[start:start + chunk.shape[0]] = detected.to("cpu") > 0.5
    masks, filled = _fill_missed_masks(masks)
    missed = int((~masks.flatten(1).any(dim=1)).sum())

    guide_width, guide_height = _guide_size(width, height, guide_megapixels)
    guide = torch.empty((count, guide_height, guide_width, 3), dtype=torch.float32)
    for start in range(0, count, DETECT_CHUNK):
        chunk = as_float(frames[start:start + DETECT_CHUNK])
        if guide_style == "mannequin":
            # Shade at the guide size, where openjutsu's blur constants were tuned (about 512 px).
            small = comfy.utils.common_upscale(chunk.movedim(-1, 1), guide_width, guide_height, "area", "disabled").movedim(1, -1)
            mask = masks[start:start + chunk.shape[0]].to(torch.float32).unsqueeze(1)
            mask = comfy.utils.common_upscale(mask, guide_width, guide_height, "bilinear", "disabled").squeeze(1)
            guide[start:start + chunk.shape[0]] = grey_mannequins(small, mask.clamp(0.0, 1.0))
            continue
        if guide_style == "person on grey":
            # Background replace: only for the preview; the generation has no reference video.
            mask = masks[start:start + chunk.shape[0]].to(torch.float32).unsqueeze(-1)
            composite = chunk * mask + torch.full_like(chunk, NEUTRAL_GREY) * (1.0 - mask)
            scaled = comfy.utils.common_upscale(composite.movedim(-1, 1), guide_width, guide_height, "area", "disabled")
            guide[start:start + chunk.shape[0]] = scaled.movedim(1, -1)
            continue
        mask = _grow(masks[start:start + chunk.shape[0]].to(torch.float32), mask_grow).unsqueeze(-1)
        base = chunk if background == BACKGROUNDS[0] else torch.full_like(chunk, NEUTRAL_GREY)
        composite = base * (1.0 - mask) + (1.0 - chunk) * mask
        scaled = comfy.utils.common_upscale(composite.movedim(-1, 1), guide_width, guide_height, "nearest-exact", "disabled")
        guide[start:start + chunk.shape[0]] = scaled.movedim(1, -1)
    print(
        f"[SECoursesH3Genjutsu] '{target}' found in {count - missed - filled}/{count} frames"
        + (f", {filled} missed frame(s) took the nearest mask" if filled else "")
        + (f", {missed} frame(s) without anyone stay uninverted" if missed else "")
        + f"; {background.split(' ')[0]} background guide {guide_width}x{guide_height}.",
        flush=True,
    )
    return guide, masks, missed


def keep_main_people(masks, ratio=MAIN_PEOPLE_RATIO):
    """Distant passers-by would float in the new place: in every frame keep the biggest person and anyone near their size.
    A small dilation joins the parts of one person (a hand, a gap at a strap), so they are never dropped. The decision is
    made on a copy about 192 lines high, so this stays fast at any clip size."""
    import numpy as np
    import torch
    from scipy import ndimage

    height, width = int(masks.shape[1]), int(masks.shape[2])
    step = max(1, height // 192)
    reach = max(1, round(height * 0.015 / step))
    kept = masks.clone()
    for index in range(int(masks.shape[0])):
        frame = masks[index:index + 1].float().unsqueeze(1)
        small = torch.nn.functional.max_pool2d(frame, step, stride=step) if step > 1 else frame
        joined = torch.nn.functional.max_pool2d(small, 2 * reach + 1, stride=1, padding=reach)[0, 0].numpy() > 0.5
        labels, count = ndimage.label(joined)
        if count < 2:
            continue
        areas = ndimage.sum_labels(small[0, 0].numpy(), labels, index=np.arange(1, count + 1))
        chosen = np.isin(labels, np.flatnonzero(areas >= ratio * areas.max()) + 1)
        whole = torch.nn.functional.interpolate(torch.from_numpy(chosen).float()[None, None], size=(height, width), mode="nearest")[0, 0] > 0.5
        kept[index] = masks[index] & whole
    return kept


def method_guide(frames, sam3_model, sam3_clip, target, background, guide_megapixels, threshold, refine_iterations,
                 mask_grow, method):
    """The @video1 guide and the target masks for ``method``. Masked edit: grey figures on the clip and masks with
    headroom (the region to regenerate). Face swap: the clip itself, unchanged, and no detection. Background replace:
    the person alone on grey (the preview) and the inverted person mask."""
    import torch

    if method == FACE_SWAP:
        return _unit_images(frames), torch.zeros(frames.shape[:3], dtype=torch.bool)
    if method == MASKED_EDIT and background != BACKGROUNDS[0]:
        raise ValueError("Masked edit keeps every pixel of the clip around the target: set background to 'video'.")
    if method == BACKGROUND_REPLACE:
        # The mask names what is regenerated (red in the preview): everything except the person.
        guide, masks, _missed = build_motion_guide(frames, sam3_model, sam3_clip, target, background, guide_megapixels,
                                                   threshold, refine_iterations, 0, "person on grey")
        return guide, ~keep_main_people(masks)
    guide, masks, _missed = build_motion_guide(
        frames, sam3_model, sam3_clip, target, background, guide_megapixels, threshold, refine_iterations,
        0 if method == MASKED_EDIT else mask_grow, "mannequin" if method == MASKED_EDIT else "inverted",
    )
    if method == MASKED_EDIT:
        masks = add_headroom(masks)
    return guide, masks


def guide_previews(frames, masks, guide):
    """A few evenly spaced frames: the source with the mask in red, and the guide."""
    import comfy.utils
    import torch

    count = int(frames.shape[0])
    picks = sorted({round(i * (count - 1) / max(1, PREVIEW_FRAMES - 1)) for i in range(PREVIEW_FRAMES)})
    source = frames[picks][..., :3]
    source = source.to(torch.float32).mul_(1.0 / 255.0) if source.dtype == torch.uint8 else source.to(torch.float32)
    mask = masks[picks].to(torch.float32).unsqueeze(-1)
    red = torch.tensor([1.0, 0.15, 0.15]).view(1, 1, 1, 3)
    tinted = source * (1.0 - 0.55 * mask) + red * (0.55 * mask)
    width, height = _guide_size(int(frames.shape[2]), int(frames.shape[1]), 0.25)
    tinted = comfy.utils.common_upscale(tinted.movedim(-1, 1), width, height, "bilinear", "disabled").movedim(1, -1)
    return tinted, guide[picks]


def cut_audio(audio, seconds):
    """ComfyUI AUDIO cut to ``seconds`` (None stays None)."""
    if audio is None:
        return None
    samples = round(float(seconds) * int(audio["sample_rate"]))
    return {**audio, "waveform": audio["waveform"][..., :samples]}


_SIZE_INPUTS = {
    "width": ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32, "tooltip": "Output width (see match_source_aspect)."}),
    "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32, "tooltip": "Output height (see match_source_aspect)."}),
    "match_source_aspect": ("BOOLEAN", {
        "default": True,
        "tooltip": "On: keep the pixel area of width x height but use the source clip's aspect ratio, so the result is framed like the clip. Off: use width and height as they are.",
    }),
}


class SECoursesH3GenjutsuMotionGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Source clip frames at 24 FPS (the performer to replace)."}),
                "sam3_model": ("MODEL", {"tooltip": "SAM 3.1 from Load Checkpoint (sam3.1_multiplex_fp16.safetensors)."}),
                "sam3_clip": ("CLIP", {"tooltip": "The CLIP output of the same SAM 3.1 checkpoint."}),
                "target": ("STRING", {
                    "default": "person",
                    "tooltip": "Who to replace, as SAM 3.1 text: 'person' for one person, 'man in red:1' or 'man:1' to pick one of several. One phrase only; a comma starts a second prompt.",
                }),
                "background": (BACKGROUNDS, {
                    "default": BACKGROUNDS[0],
                    "tooltip": "'video' keeps the clip's background around the inverted performer (character replacement). 'image' puts the inverted performer on neutral grey, so only the motion is taken (motion transfer); pair it with the same choice in MiniMax H3 Genjutsu Prompt.",
                }),
                "guide_megapixels": ("FLOAT", {
                    "default": 0.15, "min": 0.05, "max": 1.0, "step": 0.01,
                    "tooltip": "Size of the motion guide. 0.15 is fast and keeps lip sync; 0.3-0.4 shows hands, face and background in more detail and costs more reference tokens.",
                }),
                "threshold": ("FLOAT", {
                    "default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01,
                    "tooltip": "SAM 3.1 detection score threshold. Lower it when the performer is missed in some frames.",
                }),
                "mask_grow": ("INT", {
                    "default": 0, "min": 0, "max": 64, "step": 1,
                    "tooltip": "Grow the performer mask by this many pixels (at the clip's decoded size) so hair and loose clothing edges are inverted too.",
                }),
                **_SIZE_INPUTS,
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Optional soundtrack of the clip; returned cut to the output length."}),
                "refine_iterations": ("INT", {
                    "default": 2, "min": 0, "max": 5, "step": 1, "advanced": True,
                    "tooltip": "SAM decoder refinement passes per detection (as SAM3 Detect).",
                }),
                "method": _METHOD_INPUT,
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "AUDIO", "IMAGE")
    RETURN_NAMES = ("guide", "mask", "frames", "width", "height", "audio", "mask_preview")
    OUTPUT_TOOLTIPS = (
        "Native methods: reference video 1. Aligned method: the guide input of Genjutsu Apply.",
        "The performer mask of every input frame.",
        "Largest MiniMax H3 length (17k + 5 frames) the clip covers; connect it to the generation length.",
        "Output width.",
        "Output height.",
        "The soundtrack cut to the output length (nothing without an audio input).",
        "A few frames with the performer mask in red. Check it before a long run.",
    )
    FUNCTION = "build"
    DESCRIPTION = (
        "MiniMax H3 Genjutsu motion guide: SAM 3.1 detects the target for the mask preview. Native methods invert "
        "the performer and return a small reference video; aligned mode returns the original trimmed frames for "
        "Genjutsu Apply. The output size and length follow the source clip."
    )

    def build(self, images, sam3_model, sam3_clip, target, background, guide_megapixels, threshold, mask_grow,
              width, height, match_source_aspect, audio=None, refine_iterations=2, method=METHODS[0]):
        length = usable_frame_count(images.shape[0])
        guide, masks = method_guide(images[:length], sam3_model, sam3_clip, target, background, guide_megapixels,
                                    threshold, refine_iterations, mask_grow, method)
        preview, _guide_frames = guide_previews(images[:length], masks, guide)
        if method == METHODS[2]:
            guide = _unit_images(images[:length])
        if match_source_aspect:
            width, height = source_aspect_size(width, height, int(images.shape[2]), int(images.shape[1]))
        return (guide, masks.float(), length, int(width), int(height), cut_audio(audio, length / FPS), preview)


_DESCRIBE_INPUTS = {
    "describe_clip": ("CLIP", {
        "tooltip": "Optional local vision LLM for Auto Describe, eg Qwen3-VL 8B (qwen3vl_8b_int8_convrot.safetensors, CLIP type stable_diffusion). When the prompt is not a complete MiniMax H3 prompt, it describes the character image and the clip so the Genjutsu template names the new outfit, the action, the setting, the lighting and the framing. Leave it unconnected for the generic template.",
    }),
    "speech": ("STRING", {
        "multiline": True, "default": "",
        "tooltip": "Optional: the words spoken in the clip, for word-by-word lip sync in the Genjutsu template. Ignored when the prompt is a complete MiniMax H3 prompt.",
    }),
    "speech_language": ("STRING", {"default": "English", "tooltip": "Language of the spoken words, eg English, Turkish."}),
}

_PROMPT_TOOLTIP = (
    "A complete MiniMax H3 prompt (with subject_definitions:, eg pasted from an LLM given the Genjutsu instructions) is "
    "used as written. Anything else - empty, or short notes such as outfit details - becomes the Genjutsu template, "
    "with the notes added to the shot description."
)


class SECoursesH3GenjutsuPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "", "tooltip": _PROMPT_TOOLTIP}),
                "background": (BACKGROUNDS, {"default": BACKGROUNDS[0], "tooltip": "Use the same choice as the motion guide."}),
            },
            "optional": {
                **_DESCRIBE_INPUTS,
                "images": ("IMAGE", {"tooltip": "The clip's frames, for Auto Describe."}),
                "character": ("IMAGE", {"tooltip": "The new character's image, for Auto Describe."}),
                "audio": ("AUDIO", {"tooltip": "The clip's soundtrack. Without it the template has no audio lines."}),
                "method": _METHOD_INPUT,
                "target": ("STRING", {"default": "person"}),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "apply"
    DESCRIPTION = (
        "MiniMax H3 Genjutsu prompt. A complete H3 prompt is tidied (code fences, markdown headings) and used as written; "
        "anything else becomes the Genjutsu template, optionally described by a local vision LLM. Then the background "
        "rules are added: 'video' keeps the clip's location, lighting and framing as <Subject 2> and turns the "
        "generation into an edit of <Video 1>; 'image' keeps the character image's environment and removes the clip's "
        "setting, lighting and framing lines."
    )

    def apply(self, prompt, background, describe_clip=None, speech="", speech_language="English", images=None,
              character=None, audio=None, method=METHODS[0], target="person"):
        if method == FACE_SWAP:
            return (face_swap_prompt(prompt),)
        has_audio = audio is not None and audio["waveform"].numel() > 0
        if method == BACKGROUND_REPLACE:
            still = images is not None and camera_is_still(images)
            return (method_prompt(prompt, method, background, has_audio, target, False, still),)
        if images is not None:
            images = images[:usable_frame_count(images.shape[0])]
        text = build_prompt(prompt, background, describe_clip, images, character, speech, speech_language, has_audio)
        return (method_prompt(text, method, background, has_audio, target, not is_full_prompt(prompt)),)


class SECoursesH3Genjutsu:
    @classmethod
    def INPUT_TYPES(cls):
        guide_inputs = SECoursesH3GenjutsuMotionGuide.INPUT_TYPES()
        required = {"references": (REF_PACK_TYPE, {
            "tooltip": "Gallery pack. @video1 is the clip whose performer is replaced; @image1 is the new character.",
        })}
        required.update({name: spec for name, spec in guide_inputs["required"].items() if name != "images"})
        return {
            "required": required,
            "optional": {
                **_DESCRIBE_INPUTS,
                "refine_iterations": guide_inputs["optional"]["refine_iterations"],
                "prompt_override": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional prompt that replaces the gallery prompt when it is not empty. " + _PROMPT_TOOLTIP,
                }),
                "method": _METHOD_INPUT,
                "native_speedup": ("BOOLEAN", {
                    "default": True, "advanced": True,
                    "tooltip": "Allow the linked speed optimizer in the two native-reference methods. Aligned guides always disable sparse attention.",
                }),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = (REF_PACK_TYPE, "STRING", "INT", "INT", "INT", "AUDIO", "IMAGE", "IMAGE", "IMAGE", "STRING", "STRING",
                    "BOOLEAN", "MASK")
    RETURN_NAMES = ("references", "prompt", "width", "height", "length", "source_audio", "mask_preview", "guide_preview",
                    "aligned_guide", "method", "background", "native_speedup", "edit_mask")
    OUTPUT_TOOLTIPS = (
        "The reference pack: native methods replace @video1 with the inverted guide; aligned mode removes that slot.",
        "The final prompt; give it to the MiniMax H3 adapter and the face pass.",
        "Output width.",
        "Output height.",
        "Output length: the clip's frames at 24 FPS on MiniMax H3's 17k + 5 grid.",
        "The clip's own soundtrack cut to the output length (nothing when the clip is silent).",
        "A few source frames with the performer mask in red.",
        "The same frames of the motion guide.",
        "Original source frames: for Genjutsu Apply's aligned method, or the images of the Masked Video Edit Latent in "
        "masked edit (nothing in the other methods).",
        "Selected method for Genjutsu Apply.",
        "Selected background for Genjutsu Apply's task LoRA.",
        "Connect to the speed optimizer's master switch: sparse attention is disabled for aligned guides and masked edits.",
        "Masked edit only: the region to regenerate (the target with headroom above the head); connect it to the mask "
        "of the Masked Video Edit Latent. Nothing in the other methods.",
    )
    FUNCTION = "prepare"
    DESCRIPTION = (
        "MiniMax H3 Genjutsu for the SECourses Reference Gallery: @video1 becomes the SAM 3.1 inverted motion guide, "
        "the output takes the clip's length (its gallery trim window included) and aspect ratio, the prompt becomes the "
        "Genjutsu prompt with its background rules (Auto Describe with a local vision LLM when connected), and the "
        "clip's soundtrack is returned for the final video. Every other reference, RefMod and folder-batch field "
        "passes through unchanged."
    )

    def prepare(self, references, sam3_model, sam3_clip, target, background, guide_megapixels, threshold, mask_grow,
                width, height, match_source_aspect, describe_clip=None, speech="", speech_language="English",
                refine_iterations=2, prompt_override=None, method=METHODS[0], native_speedup=True):
        if not isinstance(references, dict):
            raise ValueError("The references input must come from a SECourses Reference Gallery node.")
        videos = references.get("videos") or []
        if not videos:
            raise ValueError(
                "MiniMax H3 Genjutsu needs the clip to swap as @video1: add the video to the gallery "
                "(Load + trim selects the part to use)."
            )
        entry = videos[0]
        path = _resolve_reference_entry(entry)
        fps = max(1.0, float(references.get("video_fps", FPS)))
        if abs(fps - FPS) > 1e-6:
            raise ValueError(f"MiniMax H3 Genjutsu needs the gallery's video_fps at {FPS}; it is {fps:g}.")
        trim_start, seconds = _entry_trim_window(entry, float(references.get("max_seconds", 15.0)))
        area_cap = VIDEO_DECODE_AREA_CAP
        canvas = None
        if references.get("downscale_videos", True):
            # Decode straight to the output's pixel area. The output size still follows the clip's usual canvas,
            # so it stays the same as without downscaling.
            source_width, source_height, _has_audio = _video_metadata(path)
            if source_width and source_height:
                canvas = _fit_dimensions(source_width, source_height, VIDEO_DECODE_AREA_CAP)
                out_width, out_height = source_aspect_size(width, height, *canvas) if match_source_aspect else (width, height)
                area_cap = min(area_cap, out_width * out_height)
        frames = _decode_video_frames(path, fps, max_frames=max(1, round(seconds * fps)),
                                      area_cap=area_cap, trim_start=trim_start)
        length = usable_frame_count(frames.shape[0])
        frames = frames[:length]
        guide, masks = method_guide(frames, sam3_model, sam3_clip, target, background, guide_megapixels, threshold,
                                    refine_iterations, mask_grow, method)
        mask_preview, guide_preview = guide_previews(frames, masks, guide)
        source_audio = _decode_video_audio(path, length / fps, trim_start=trim_start)

        prompt = references.get("prompt") or ""
        if prompt_override is not None and str(prompt_override).strip():
            prompt = str(prompt_override)
        if method == FACE_SWAP:
            prompt = face_swap_prompt(prompt)
        elif method == BACKGROUND_REPLACE:
            if not references.get("images"):
                raise ValueError("Background replace needs the new background: add an image to the gallery (it becomes @image1).")
            prompt = method_prompt(prompt, method, background, source_audio is not None, target, False, method == BACKGROUND_REPLACE and camera_is_still(frames))
        else:
            character = None
            images = references.get("images") or []
            if describe_clip is not None and images and not is_full_prompt(prompt):
                character_path = _resolve_reference_entry(images[0])
                source_width, source_height, _orientation = _oriented_image_dimensions(character_path)
                character = _load_reference_image(
                    character_path, *_guide_size(source_width, source_height, DESCRIBE_CHARACTER_MEGAPIXELS))
            autogenerated = not is_full_prompt(prompt)
            prompt = build_prompt(prompt, background, describe_clip, frames, character, speech, speech_language,
                                  has_audio=source_audio is not None)
            prompt = method_prompt(prompt, method, background, source_audio is not None, target, autogenerated)
        if match_source_aspect:
            width, height = source_aspect_size(width, height, *(canvas or (int(frames.shape[2]), int(frames.shape[1]))))
        # Aligned guides go to Genjutsu Apply; a masked edit hands the clip's own frames to the Masked Video Edit Latent.
        aligned = _unit_images(frames) if method in (METHODS[2], *HOLD_METHODS) else None
        edit_mask = masks.float() if method in HOLD_METHODS else None
        if method == METHODS[2]:
            mask_preview, guide_preview = guide_previews(frames, masks, aligned)
        del frames, masks
        pack = dict(references)
        pack["videos"] = videos[1:] if method in (METHODS[2], BACKGROUND_REPLACE) else [dict(entry, frames=guide), *videos[1:]]
        pack["prompt"] = prompt
        print(
            f"[SECoursesH3Genjutsu] @video1 '{entry.get('name') or '?'}': {length} frames "
            f"({length / fps:.2f}s{', from ' + format(trim_start, '.2f') + 's' if trim_start else ''}), "
            f"output {width}x{height}, soundtrack {'kept' if source_audio is not None else 'none (silent clip)'}.",
            flush=True,
        )
        return (pack, prompt, int(width), int(height), length, source_audio, mask_preview, guide_preview,
                aligned, method, background, bool(native_speedup and method not in (METHODS[2], *HOLD_METHODS)),
                edit_mask)


class SECoursesH3LoadClip:
    """The gallery's first video as frames at 24 FPS, on H3's frame grid, with its soundtrack (none for a silent clip)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "references": (REF_PACK_TYPE, {"tooltip": "From a SECourses Reference Gallery node; its first video is used (Load + trim picks the part)."}),
                "max_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01,
                    "tooltip": "A bigger clip is reduced to about this size for the generation; a smaller clip keeps its size."}),
            },
        }

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "INT", "INT")
    RETURN_NAMES = ("images", "audio", "frames", "width", "height")
    FUNCTION = "load"
    DESCRIPTION = "Decode the first video of the gallery for H3: 24 FPS, 17n+5 frames, a size divisible by 32 that keeps its aspect ratio."

    def load(self, references, max_megapixels):
        import comfy.utils

        videos = references.get("videos") or [] if isinstance(references, dict) else []
        if not videos:
            raise ValueError("Add your video to the gallery first.")
        entry = videos[0]
        path = _resolve_reference_entry(entry)
        trim_start, seconds = _entry_trim_window(entry, float(references.get("max_seconds", 15.0)))
        frames = _decode_video_frames(path, FPS, max_frames=max(1, round(seconds * FPS)), area_cap=round(max_megapixels * 1_000_000),
                                      trim_start=trim_start)
        length = usable_frame_count(frames.shape[0])
        frames = _unit_images(frames[:length])
        height, width = int(frames.shape[1]), int(frames.shape[2])
        fit_w = max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        fit_h = max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        if (fit_w, fit_h) != (width, height):
            frames = comfy.utils.common_upscale(frames.movedim(-1, 1), fit_w, fit_h, "lanczos", "center").clamp(0.0, 1.0).movedim(1, -1)
        audio = _decode_video_audio(path, length / FPS, trim_start=trim_start)
        return (frames, audio, length, fit_w, fit_h)


NODE_CLASS_MAPPINGS = {
    "SECoursesH3LoadClip": SECoursesH3LoadClip,
    "SECoursesH3GenjutsuMotionGuide": SECoursesH3GenjutsuMotionGuide,
    "SECoursesH3GenjutsuPrompt": SECoursesH3GenjutsuPrompt,
    "SECoursesH3Genjutsu": SECoursesH3Genjutsu,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SECoursesH3LoadClip": "MiniMax H3 Load Clip From Gallery",
    "SECoursesH3GenjutsuMotionGuide": "MiniMax H3 Genjutsu Motion Guide (SAM 3.1)",
    "SECoursesH3GenjutsuPrompt": "MiniMax H3 Genjutsu Prompt",
    "SECoursesH3Genjutsu": "MiniMax H3 Genjutsu (Gallery)",
}
