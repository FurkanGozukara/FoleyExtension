"""MiniMax H3 Relight Prompt: compiles 1-3 designed lights into a MiniMax H3 sections prompt.

Our own light-plan compiler, written from a description of the approach of the H3-Relight community project. It turns
light geometry into unambiguous words (viewer side versus subject side, shadow fall and length, colour from Kelvin,
key / fill / rim roles) and names the source in our tag convention: <Video 1> for a video (@video1 in the gallery) and
<Picture 1> for an image (@image1). The optional preview is a lit sphere for the user only; the prompt never mentions it.
"""

from __future__ import annotations

import colorsys
import math

import numpy as np

# Light type: (softness wording, then preview-only shading: factor, wrap, shininess, specular, rim)
LIGHT_TYPES = {
    "soft area light": ("a soft, large diffused source: the light wraps gently around the forms and shadow edges are feathered", 0.95, 0.35, 10, 0.14, 1.25),
    "hard spotlight": ("a small, hard, direct source: crisp sharp-edged shadows and bright specular highlights", 1.2, 0.0, 48, 0.45, 1.6),
    "sky / sun": ("broad sky light: very soft, even and directional, with faint diffuse shadows", 1.15, 0.5, 4, 0.0, 0.95),
}
# Background: (prompt sentence, then preview-only ambient, hemisphere, ground RGB, plate RGB)
BACKGROUNDS = {
    "keep and relight the original": ("Keep the original background of {src} and relight it with the same lights, so the subject and the background agree.", 0.2, 0.13, (220, 224, 234), (108, 108, 108)),
    "studio black": ("Replace the background with a seamless studio black backdrop; only the planned light separates the subject from it.", 0.06, 0.05, (58, 63, 74), (22, 22, 24)),
    "studio white": ("Replace the background with a seamless high-key white studio backdrop, with a soft contact shadow under the subject.", 0.28, 0.18, (236, 238, 243), (232, 232, 234)),
}
EFFECTS = {
    "volumetric light shafts": "visible volumetric light shafts cutting through a faint haze along the key light direction; the rays pass around the subject and never veil the face or the product",
    "venetian blind stripes": "hard parallel stripes of light and shadow, as if the key light passes through venetian blinds, projected across the subject and the background along the key direction",
    "water caustics": "rippling caustic light patterns, as if the key light is refracted through moving water or glass, playing across the surfaces it reaches",
    "chiaroscuro drama": "strong chiaroscuro: one dominant key, fast falloff into deep near-black shadow, high contrast and very little fill",
    "realistic diffuse": "natural, realistic diffuse light with gradual shadow transitions, true-to-life color and no stylization",
    "desert sun": "sun-baked desert warmth: amber-orange light, dusty warm haze in the air, sun-bleached highlights and warm bounce light from the ground",
    "dreamy teal mood": "dreamy, melancholic film mood: saturated teal-blue light against warm practical accents, soft bloom on highlights, light smoky haze and rich shadows",
    "neon noir": "neon-noir: saturated magenta and electric-blue light from low side angles, deep blacks, light haze and wet, reflective highlights",
    "old money window light": "quiet old-money interior: warm tungsten window light from high front-right, restrained contrast, rich muted wood and leather tones",
    "pastel storybook": "flat, symmetrical frontal soft light, a pastel pink palette, even exposure and minimal shadows, a storybook look",
    "blue hour": "cool blue-hour light from the side, soft and lonely city-night mood, gentle contrast with a faint neon spill",
    "80s supernatural": "1980s supernatural mood: deep red practical light from one side against a cold blue beam from high behind, light haze and strong color contrast",
    "golden sunset backlight": "warm low sunset backlight, a golden rim around the silhouette, long shadows reaching toward the camera and a classic Hollywood glow",
    "top light low key": "top light from directly above: warm amber tungsten, eye sockets and the lower face falling into shadow, low-key with deep blacks",
    "monochrome": "black-and-white monochrome with a rich tonal range and silver-gelatin contrast; every color is rendered as grayscale",
    "liquid gold": "warm backlight skimming glossy surfaces, molten golden reflections and bright specular streaks along the edges",
    "metallic sharp": "cold hard backlight, crisp specular edge highlights on metal and glass, clean black reflections and high contrast",
    "velvet skin": "soft, flattering diffuse key with velvety smooth tonal gradients and gentle wrap-around, no harsh shadow edges",
    "rim light halo": "a bright backlight rim that outlines the full silhouette with a glowing halo, clearly separating the subject from the background",
    "grazing texture": "hard light grazing across surfaces at a low angle to reveal texture and relief with strong micro-shadows",
    "butterfly product": "high frontal soft source, a small symmetrical shadow directly beneath the subject, a clean polished commercial look",
    "product spotlight": "a hard spotlight from above that pools light on the subject, with falloff into darker surroundings",
    "champagne glow": "warm champagne-gold backlight glow with elegant soft highlights and a luxurious warm sheen",
    "symmetry studio": "two equal lights from exactly left and right at eye level, mirrored highlights on both sides and a darker central line, clean studio look",
    "optical caustics": "cool light passing through glass or crystal, projecting sharp caustic light patterns onto nearby surfaces",
}
# Preset: lights as (type, intensity, azimuth, elevation, Kelvin or "#hex"); its atmosphere is the effect of the same name.
_SOFT, _HARD, _SKY = "soft area light", "hard spotlight", "sky / sun"
PRESETS = {
    "desert sun": [(_HARD, 73, -135, 45, 1800)],
    "dreamy teal mood": [(_SKY, 50, 90, 0, "#52bdff")],
    "neon noir": [(_HARD, 70, -55, -10, "#ff00ee"), (_HARD, 70, 55, -10, "#1a33ff")],
    "old money window light": [(_SKY, 47, 45, 45, 3200)],
    "pastel storybook": [(_SOFT, 43, 0, 0, 3000)],
    "blue hour": [(_SKY, 40, 90, 15, 8500)],
    "80s supernatural": [(_SKY, 70, 90, 0, "#8c0000"), (_HARD, 70, -125, 50, "#0084ff")],
    "golden sunset backlight": [(_HARD, 60, 180, 10, 2200)],
    "top light low key": [(_SOFT, 80, 0, 90, 3200)],
    "monochrome": [(_SOFT, 58, 90, 45, 5000)],
    "liquid gold": [(_HARD, 70, 180, 0, 2000)],
    "metallic sharp": [(_HARD, 77, 180, 0, 10000)],
    "velvet skin": [(_SOFT, 50, 45, 45, 4500)],
    "rim light halo": [(_HARD, 80, 180, 30, 4500)],
    "grazing texture": [(_HARD, 67, 90, 0, 5000)],
    "butterfly product": [(_SOFT, 57, 0, 58, 5500)],
    "product spotlight": [(_HARD, 73, 45, 90, 5600)],
    "champagne glow": [(_SOFT, 60, 135, 45, 3000)],
    "symmetry studio": [(_HARD, 57, 90, 0, 5600), (_HARD, 57, -90, 0, 5600)],
    "optical caustics": [(_SOFT, 63, 120, 45, 7500)],
}
STRENGTHS = {
    "subtle": "Relight gently: keep as much of the original image as possible and move its light only part of the way toward this plan.",
    "balanced": "Replace the original lighting with this plan.",
    "strong": "Relight completely: remove every trace of the original light direction, colour and shadows, and light the scene only as planned.",
}
# Subject: (who <Subject 1> is, preservation add-on)
SUBJECTS = {
    "auto": ("the main subject of {src}", ""),
    "portrait": ("the person in {src}", " Keep natural skin texture and true skin tone under the new light, with catchlights in the eyes that match the key direction."),
    "product": ("the product in {src}", " Keep the product's exact silhouette, labels, text and materials; its reflections and specular highlights follow the new lights."),
    "scene / exterior": ("the scene in {src}: its location, buildings, landscape and sky", " The key light is the sun: the sky, the cast shadows and the bounce light agree with its direction and colour."),
}
CONVENTION = ("Directions are seen from the camera: azimuth 0 is the camera side, +90 the viewer's right, -90 the viewer's left "
              "and 180 directly behind the subject.")
DIRECTIONS = ("the front", "the front-right", "the right", "the back-right", "behind", "the back-left", "the left", "the front-left")
_SHADOW_FALL = ("straight back, away from the camera", "back and toward the viewer's left", "toward the viewer's left",
                "forward and toward the viewer's left", "forward, toward the camera", "forward and toward the viewer's right",
                "toward the viewer's right", "back and toward the viewer's right")
_HIGHLIGHTS = ("the surfaces that face the camera", "the front and the viewer's right side", "the viewer's right side",
               "the viewer's right edges and contour", "the outline, the hair and the top edges",
               "the viewer's left edges and contour", "the viewer's left side", "the front and the viewer's left side")
_KELVIN_WORDS = ((2000, "deep amber, candle and firelight warmth"), (2800, "warm orange tungsten"), (3600, "warm tungsten"),
                 (4800, "slightly warm neutral"), (6000, "neutral daylight white"), (7500, "cool daylight"), (9000, "cool blue"),
                 (math.inf, "deep cold blue"))
_KELVIN_RGB = ((1000, 255, 140, 20), (1800, 255, 179, 71), (3200, 255, 210, 161), (5600, 255, 255, 255),
               (7000, 232, 241, 255), (10000, 150, 157, 245))
_HUES = ((12, "red"), (35, "orange"), (50, "amber"), (68, "yellow"), (95, "yellow-green"), (150, "green"), (175, "teal-green"),
         (195, "cyan"), (215, "sky blue"), (245, "blue"), (270, "indigo"), (292, "violet"), (330, "magenta"), (350, "rose pink"),
         (361, "red"))


def wrap_azimuth(azimuth):
    value = ((float(azimuth) + 180.0) % 360.0) - 180.0
    return 180.0 if value == -180.0 else value


def direction_index(azimuth):
    return int(math.floor(((float(azimuth) % 360.0) + 22.5) / 45.0)) % 8


def height_phrase(elevation):
    if elevation <= -15:
        return f"from below, {-elevation:g} degrees under the horizon"
    if elevation < -3:
        return "just below eye level"
    if elevation < 8:
        return "at eye level"
    band = "low" if elevation < 33 else "mid-height" if elevation < 58 else "high" if elevation < 80 else "overhead"
    return f"{band}, {elevation:g} degrees above the horizon"


def side_words(index):
    if 1 <= index <= 3:
        return "It lights the viewer's right side of the frame, which is the subject's own left when the subject faces the camera."
    if 5 <= index <= 7:
        return "It lights the viewer's left side of the frame, which is the subject's own right when the subject faces the camera."
    return ("It comes from the camera side and lights the surfaces that face the viewer." if index == 0
            else "It comes from behind the subject and outlines the silhouette.")


def shadow_words(index, elevation):
    if elevation >= 80:
        return "Cast shadows pool short and directly beneath the subject."
    if elevation <= -15:
        return "As an uplight it throws shadows upward, above the features it lights, for an unnatural, eerie look."
    length = "long" if elevation < 20 else "medium-length" if elevation < 50 else "short"
    return f"Highlights fall on {_HIGHLIGHTS[index]}; {length} cast shadows fall {_SHADOW_FALL[index]}."


def hex_rgb(colour):
    text = colour.strip().lstrip("#")
    text = "".join(c * 2 for c in text) if len(text) == 3 else text
    return tuple(int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def colour_words(colour):
    if isinstance(colour, str):
        r, g, b = hex_rgb(colour)
        hue, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
        if max(r, g, b) - min(r, g, b) < 0.08:
            return "neutral white" if lightness > 0.6 else "neutral grey-white"
        tone = "deep " if lightness < 0.32 else "pale " if lightness > 0.78 else "saturated " if saturation > 0.7 else ""
        return tone + next(name for limit, name in _HUES if hue * 360 < limit) + f" ({colour.lower()})"
    return next(words for limit, words in _KELVIN_WORDS if colour < limit) + f" ({int(colour)} K)"


def light_rgb(colour):
    if isinstance(colour, str):
        return hex_rgb(colour)
    kelvin = min(max(float(colour), 1000.0), 10000.0)
    for (k0, *c0), (k1, *c1) in zip(_KELVIN_RGB, _KELVIN_RGB[1:]):
        if kelvin <= k1:
            t = (kelvin - k0) / (k1 - k0)
            return tuple(math.floor(a + (b - a) * t + 0.5) / 255.0 for a, b in zip(c0, c1))


def intensity_word(intensity):
    level = intensity / 10.0
    return ("faint" if level <= 2.5 else "moderate" if level <= 4.5 else "bright" if level <= 6.5
            else "strong" if level <= 8.5 else "very strong")


def normalise(light):
    kind, intensity, azimuth, elevation, colour = light
    colour = colour if isinstance(colour, str) else int(min(max(colour, 1000), 10000))
    return (kind if kind in LIGHT_TYPES else _SOFT, int(min(max(intensity, 10), 100)), wrap_azimuth(azimuth),
            float(min(max(elevation, -90), 90)), colour)


def assign_roles(lights):
    """The strongest light is the key (ties: lowest index); other back lights are rims; >= 80 % of the key is a second key."""
    key = max(range(len(lights)), key=lambda i: (lights[i][1], -i))
    roles = []
    for i, (_, intensity, azimuth, _, _) in enumerate(lights):
        back = 3 <= direction_index(azimuth) <= 5
        if i == key:
            roles.append("Backlight key" if back else "Key light")
        else:
            roles.append("Rim light" if back else "Second key light" if intensity >= 0.8 * lights[key][1] else "Fill light")
    return key, roles


def light_sentence(light, role, key_intensity):
    kind, intensity, azimuth, elevation, colour = light
    index = direction_index(azimuth)
    share = f" at about {round(100 * intensity / key_intensity)}% of the key's strength" if role == "Fill light" else ""
    return (f"{role}: {intensity_word(intensity)}{share}, {colour_words(colour)}, coming from {DIRECTIONS[index]} "
            f"(azimuth {azimuth:g}), {height_phrase(elevation)}. It is {LIGHT_TYPES[kind][0]}. "
            f"{side_words(index)} {shadow_words(index, elevation)}")


def compile_relight(lights, background="keep and relight the original", atmosphere="none", subject="auto",
                    strength="balanced", source="video", notes=""):
    """Return (H3 sections prompt, one-paragraph plain summary) for 1-3 lights."""
    lights = [normalise(light) for light in lights][:3]
    video = source == "video"
    src = "<Video 1>" if video else "<Picture 1>"
    key, roles = assign_roles(lights)
    order = [key] + [i for i in range(len(lights)) if i != key]
    where = lambda i: DIRECTIONS[direction_index(lights[i][2])]
    look = (f"a {intensity_word(lights[key][1])} {colour_words(lights[key][4]).split(' (')[0].split(',')[0]} "
            f"{lights[key][0]} from {where(key)}" + "".join(f", plus a {roles[i].lower()} from {where(i)}" for i in order[1:]))
    plan = ["[Shot 1] Lighting plan. " + STRENGTHS.get(strength, STRENGTHS["balanced"]) + " " + CONVENTION]
    plan += [light_sentence(lights[i], roles[i], lights[key][1]) for i in order]
    plan.append(f"{'Both' if len(lights) == 2 else 'All three'} lights belong to one coherent setup; the {roles[key].lower()} from "
                f"{where(key)} is dominant and sets the main shadow direction."
                if len(lights) > 1 else "This is the only new light source; the rest of the scene gets only its soft natural bounce.")
    background = background if background in BACKGROUNDS else "keep and relight the original"
    plan.append(BACKGROUNDS[background][0].format(src=src))
    if atmosphere in EFFECTS:
        plan.append(f"Atmosphere: {EFFECTS[atmosphere]}. Use it only as far as it supports the mood; it must never hide "
                    "the subject's face, eyes or the product.")
    plan.append("Neutral colours stay neutral except where a planned light tints them. No new objects, people, text, "
                "watermarks or geometry, no identity change, " + ("no change to the motion or camera movement of <Video 1>."
                if video else "no subject motion and no camera motion.") + " The planned lights stay fixed for the whole "
                "clip: they never move, sweep, fade, pulse or flicker.")
    notes = str(notes or "").strip()
    notes = notes[len("additional direction:"):].strip() if notes.lower().startswith("additional direction:") else notes
    if notes:
        plan.append("Additional direction: " + notes)
    who, add_on = SUBJECTS.get(subject, SUBJECTS["auto"])
    motion = "; keep its motion, timing and camera movement exactly" if video else ""
    summary_line = (f"[video editing] The target video is {src} relit by {look}." if video else
                    f"[reference generation] The target video shows exactly the scene of {src}, relit by {look}.")
    lines = ["subject_definitions:", f"{src} is the source {'video' if video else 'image'} to relight.",
             f"<Subject 1> is {who.format(src=src)}.", "", "summary:",
             summary_line + f" Everything except the lighting stays exactly as in {src}.", "", "retention_analysis:",
             f"{src}: partially_preserved - preserve the identity, face, expression, pose, wardrobe, hair, text and logos, "
             f"geometry, framing, composition, lens and perspective{motion}; only the light changes.{add_on}",
             "", "detailed_description:", f"Continuous shot in the look and style of {src}.", *plan,
             "", "overall_soundscape:", "Natural ambience matching the scene; no new dialogue.", "", "non_diegetic_music:", "N/A"]
    summary = f"Relight of {src} ({source} source, {strength}). " + " ".join(
        f"{roles[i]}: {intensity_word(lights[i][1])} {colour_words(lights[i][4])} {lights[i][0]} from {where(i)} "
        f"(azimuth {lights[i][2]:g}), {height_phrase(lights[i][3])}." for i in order)
    summary += f" Background: {background}." + (f" Atmosphere: {atmosphere}." if atmosphere in EFFECTS else "")
    return "\n".join(lines), summary


def render_preview(lights, background="keep and relight the original", size=512):
    """A lit white sphere showing the plan at a glance (a preview only; never send it to the model)."""
    import torch

    _, ambient, hemisphere, ground, plate = BACKGROUNDS.get(background, BACKGROUNDS["keep and relight the original"])
    linear = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    scale = 2.0 / (0.46 * size)
    coords = (np.arange(size) + 0.5 - size / 2) * scale
    u, v = np.meshgrid(coords, -coords)  # x to the viewer's right, y up, z toward the camera
    radius = np.sqrt(u * u + v * v)
    nz = np.sqrt(np.clip(1 - radius * radius, 0, 1))
    normal = np.stack([u, v, nz], -1)
    ground_lin = linear(np.array(ground) / 255.0)
    total = ambient + hemisphere * (ground_lin + (1 - ground_lin) * ((np.clip(v, -1, 1) + 1) / 2)[..., None])
    specular_sum, halo = np.zeros_like(total), np.zeros_like(total)
    for kind, intensity, azimuth, elevation, colour in (normalise(light) for light in lights):
        _, factor, wrap, shininess, spec, rim = LIGHT_TYPES[kind]
        a, e = math.radians(azimuth), math.radians(elevation)
        direction = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        rgb = linear(np.array(light_rgb(colour)))
        power = (intensity / 100 * 3.2 + 0.3) * factor * 0.55
        n_dot_l = normal @ direction
        half = direction + np.array([0.0, 0.0, 1.0])
        half = half / max(np.linalg.norm(half), 1e-6)
        specular = np.where(n_dot_l > 0, spec * np.maximum(normal @ half, 0) ** shininess, 0)
        back = max(0.0, -direction[2])
        side = np.clip(0.5 + 0.5 * (u * direction[0] + v * direction[1])
                       / (np.maximum(radius, 1e-6) * (math.hypot(direction[0], direction[1]) or 1.0)), 0, 1)
        lit = np.maximum(0, (n_dot_l + wrap) / (1 + wrap)) + back * (1 - nz) ** 3 * side * rim
        total += power * lit[..., None] * rgb
        specular_sum += power * specular[..., None] * rgb
        halo += (back * np.exp(-9 * np.maximum(radius - 1, 0)) * side * rim * 0.22 * power)[..., None] * rgb
    ball = 1 - np.exp(-(0.85 * total + specular_sum))
    ball = np.where(ball <= 0.0031308, ball * 12.92, 1.055 * np.power(np.maximum(ball, 0.0031308), 1 / 2.4) - 0.055)
    cover = np.clip((1 - radius) / scale + 0.5, 0, 1)[..., None]
    image = cover * ball + (1 - cover) * np.clip(np.array(plate) / 255.0 + halo, 0, 1)
    return torch.from_numpy(np.clip(image, 0, 1).astype(np.float32))[None]


def _light_inputs(n, kind, intensity, azimuth, elevation, kelvin):
    fields = {f"light{n}_enable": ("BOOLEAN", {"default": False, "tooltip": f"Add light {n} to the plan (custom preset only)."})} if n > 1 else {}
    fields[f"light{n}_type"] = (list(LIGHT_TYPES), {"default": kind, "tooltip": "hard spotlight: crisp shadows and bright highlights. soft area light: soft wrap and feathered shadow edges. sky / sun: broad, very soft directional light."})
    fields[f"light{n}_intensity"] = ("INT", {"default": intensity, "min": 10, "max": 100, "step": 1, "tooltip": "Strength from 10 (faint) to 100 (very strong). The strongest light becomes the key light."})
    fields[f"light{n}_azimuth"] = ("FLOAT", {"default": azimuth, "min": -360.0, "max": 360.0, "step": 5.0, "tooltip": "Direction around the subject, seen from the camera: 0 = from the camera side, 90 = from the viewer's right, -90 = from the viewer's left, 180 = from behind."})
    fields[f"light{n}_elevation"] = ("FLOAT", {"default": elevation, "min": -90.0, "max": 90.0, "step": 5.0, "tooltip": "Height: 0 = eye level, 90 = straight overhead, negative = from below."})
    fields[f"light{n}_kelvin"] = ("INT", {"default": kelvin, "min": 1000, "max": 10000, "step": 100, "tooltip": "Colour: 1800 candle, 3200 tungsten, 5600 daylight, 8000 and above cool blue."})
    return fields


class SECoursesH3RelightPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "preset": (["custom", *PRESETS], {"default": "custom", "tooltip": "A ready lighting look. A preset sets the lights, the background (kept and relit) and the atmosphere; 'custom' uses the fields below."}),
            "source": (["video", "image"], {"default": "video", "tooltip": "What is relit. video: the clip (@video1 in the gallery), named <Video 1> in the prompt. image: the picture (@image1), named <Picture 1>."}),
            "subject": (list(SUBJECTS), {"default": "auto", "tooltip": "What to protect: portrait keeps skin and eye catchlights, product keeps labels and reflections, scene / exterior treats the key as the sun."}),
            "strength": (list(STRENGTHS), {"default": "balanced", "tooltip": "subtle if identity or shapes drift, strong if the result still looks like the original light."}),
            "background": (list(BACKGROUNDS), {"default": "keep and relight the original", "tooltip": "Keep the original background and relight it, or replace it with a seamless studio backdrop. Ignored with a preset."}),
            "atmosphere": (["none", *EFFECTS], {"default": "none", "tooltip": "An optional mood or light effect added to the plan. Ignored with a preset."}),
            **_light_inputs(1, _SOFT, 60, 45.0, 35.0, 5600),
            **_light_inputs(2, _SOFT, 30, -60.0, 30.0, 5600),
            **_light_inputs(3, _HARD, 45, 160.0, 35.0, 6500),
            "notes": ("STRING", {"multiline": True, "default": "", "tooltip": "What the lights cannot say: which subject, which material should shine, the mood. Do not repeat direction, colour or strength here."}),
        }}

    CATEGORY = "SECourses/references"
    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("prompt", "summary", "light_preview")
    OUTPUT_TOOLTIPS = ("MiniMax H3 prompt in sections layout with the lighting plan.", "One plain paragraph describing the plan.",
                       "Lit sphere showing the light directions. Preview only: do not feed it to the model.")
    FUNCTION = "build"
    DESCRIPTION = ("Builds a MiniMax H3 relight prompt from 1-3 lights (direction, height, softness, colour, strength), a "
                   "background and an atmosphere: key, fill and rim roles, shadow direction and length, and what must stay unchanged.")

    def build(self, preset, source, subject, strength, background, atmosphere, notes="", **fields):
        if preset in PRESETS:
            lights, background, atmosphere = PRESETS[preset], "keep and relight the original", preset
        else:
            lights = [tuple(fields[f"light{n}_{k}"] for k in ("type", "intensity", "azimuth", "elevation", "kelvin"))
                      for n in (1, 2, 3) if n == 1 or fields.get(f"light{n}_enable")]
        prompt, summary = compile_relight(lights, background, atmosphere, subject, strength, source, notes)
        return (prompt, (f"Preset {preset}. " if preset in PRESETS else "") + summary, render_preview(lights, background))


NODE_CLASS_MAPPINGS = {"SECoursesH3RelightPrompt": SECoursesH3RelightPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"SECoursesH3RelightPrompt": "MiniMax H3 Relight Prompt"}
