import assert from "node:assert/strict";
import test from "node:test";

import "../web/js/minimax_h3_tokens.js";
import {
    PromptResolver,
    estimateFromPrompt,
    evaluateMathExpression,
    initAudioTrimmedDuration,
    selectMentioned,
    setMediaInfoProvider,
} from "../web/js/secourses_token_graph.mjs";

const H3 = globalThis.SECoursesMiniMaxH3Tokens;
const LENGTH_GRID = "max(5, round(a * 24)) + (5 - (max(5, round(a * 24)) % 17)) % 17";

// Output nodes are recognised through LiteGraph's registry in the browser; emulate it here.
globalThis.LiteGraph = {
    registered_node_types: {
        SECoursesBatchVideoSaveMerge: { nodeData: { output_node: true } },
        SaveVideo: { nodeData: { output_node: true } },
        PreviewImage: { nodeData: { output_node: true } },
    },
};

const MEDIA = {
    "reference_gallery/a.png [input]": { kind: "image", width: 1920, height: 1080 },
    "reference_gallery/b.jpg [input]": { kind: "image", width: 1080, height: 1920 },
    "reference_gallery/clip.mp4 [input]": { kind: "video", width: 1920, height: 1080, duration: 8, has_audio: true },
    "reference_gallery/voice.wav [input]": { kind: "audio", duration: 3.37, has_audio: true },
    "start.jpg": { kind: "image", width: 640, height: 640 },
    "song.mp3": { kind: "audio", duration: 9.0, has_audio: true },
};
setMediaInfoProvider(async (file) => MEDIA[file] ?? null);

const gallery = (overrides = {}) => ({
    inputs: {
        prompt: "A cat.",
        references: "{}",
        video_fps: 24,
        max_seconds: 15,
        batch_folder: "",
        merge_batch_videos: false,
        continue_batch_with_last_frame: false,
        match_batch_init_media: true,
        ...overrides,
    },
    class_type: "SECoursesReferenceGallery",
});

/**
 * The Text/Image To Video preset shape: normal route through the I2V subgraph, folder route through the
 * auto adapter. `router: "auto"` is the current preset shape (the switch follows Reference Mode's
 * auto_route output, so single runs with references use the auto adapter too); the default `"legacy"`
 * shape switches on the gallery's folder_batch_active output only.
 */
function textToVideoPrompt({ duration = 5, refs = "{}", batchFolder = "", initAudio = "(none - disabled)", initAudioTrim = {}, durationMode = "match init audio length", firstFrame = null, faceOn = false, router = "legacy", initImage = null } = {}) {
    const output = {
        "119": gallery({ references: refs, batch_folder: batchFolder }),
        "115": { inputs: { aspect_ratio: "16:9 (Widescreen)", megapixels: 0.4, width: 864, height: 480, multiple: 32 }, class_type: "SECoursesResolutionSync" },
        "120": { inputs: { value: duration }, class_type: "PrimitiveFloat" },
        "187": { inputs: { audio: initAudio, duration_seconds: ["120", 0], duration_mode: durationMode, ...initAudioTrim }, class_type: "SECoursesInitAudio" },
        "143": { inputs: { references: ["119", 0], default_duration_seconds: ["187", 1] }, class_type: "SECoursesBatchDuration" },
        // "Image to Video (MiniMax H3)" subgraph, flattened
        "105:107": { inputs: { expression: LENGTH_GRID, "values.a": ["143", 0] }, class_type: "ComfyMathExpression" },
        "105:104": { inputs: { clip: ["105:13", 0], vae: ["105:11", 0], prompt: ["119", 1], width: ["115", 0], height: ["115", 1], length: ["105:107", 1], ...(firstFrame ? { first_frame: ["114", 0] } : {}) }, class_type: "MiniMaxH3ImageToVideo" },
        "105:183": { inputs: { positive: ["105:104", 0], latent: ["105:104", 1], audio_vae: ["105:24", 0], init_audio: ["187", 0], audio_conditioning: "lock soundtrack + guide" }, class_type: "SECoursesMiniMaxH3InitAudio" },
        "105:16": { inputs: { model: ["105:6", 0], conditioning: ["105:183", 0] }, class_type: "BasicGuider" },
        "105:14": { inputs: { noise: ["105:15", 0], guider: ["105:16", 0], sampler: ["105:17", 0], sigmas: ["105:9", 0], latent_image: ["105:183", 1] }, class_type: "SamplerCustomAdvanced" },
        "105:10": { inputs: { samples: ["105:14", 0], vae: ["105:11", 0] }, class_type: "VAEDecode" },
        "105:91": { inputs: { images: ["105:10", 0], fps: 24 }, class_type: "CreateVideo" },
        // "Folder Batch Auto" subgraph, flattened
        "121:136": { inputs: { expression: LENGTH_GRID, "values.a": ["143", 0] }, class_type: "ComfyMathExpression" },
        "121:135": { inputs: { clip: ["121:132", 0], vae: ["121:123", 0], audio_vae: ["121:124", 0], references: ["119", 0], continuation_frame: ["144", 0], width: ["115", 0], height: ["115", 1], length: ["121:136", 1], ref_image_size: "match" }, class_type: "SECoursesMiniMaxH3Auto" },
        "144": { inputs: { references: ["119", 0], continue_batch_with_last_frame: ["119", 4], ...(initImage ? { init_image: ["114", 0] } : {}) }, class_type: "SECoursesBatchContinuationFrame" },
        "121:185": { inputs: { positive: ["121:135", 0], latent: ["121:135", 1], audio_vae: ["121:124", 0], init_audio: ["187", 0], audio_conditioning: "lock soundtrack + guide", references: ["119", 0] }, class_type: "SECoursesMiniMaxH3InitAudio" },
        "121:130": { inputs: { model: ["121:131", 0], conditioning: ["121:185", 0] }, class_type: "BasicGuider" },
        "121:129": { inputs: { noise: ["121:133", 0], guider: ["121:130", 0], sampler: ["121:127", 0], sigmas: ["121:128", 0], latent_image: ["121:185", 1] }, class_type: "SamplerCustomAdvanced" },
        "121:126": { inputs: { samples: ["121:129", 0], vae: ["121:123", 0] }, class_type: "VAEDecode" },
        "121:134": { inputs: { images: ["121:126", 0], fps: 24 }, class_type: "CreateVideo" },
        // router + face pass + save
        "122": { inputs: { switch: router === "auto" ? ["148", 1] : ["119", 2], on_false: ["105:91", 0], on_true: ["121:134", 0] }, class_type: "ComfySwitchNode" },
        "153": { inputs: { value: faceOn }, class_type: "PrimitiveBoolean" },
        "154:3": { inputs: { clip: ["105:13", 0], vae: ["105:11", 0], audio_vae: ["105:24", 0], references: ["119", 0], prompt_override: ["119", 1], width: ["154:1", 4], height: ["154:1", 5], length: ["105:107", 1], ref_image_size: "match" }, class_type: "SECoursesMiniMaxH3Auto" },
        "154:1": { inputs: { images: ["105:10", 0], detector: "yolo.pt" }, class_type: "MiniMaxH3FaceTrackCrop" },
        "154:181": { inputs: { images: ["154:1", 0], guider_cond: ["154:3", 0] }, class_type: "SamplerCustomAdvanced" },
        "154:184": { inputs: { images: ["154:181", 0] }, class_type: "CreateVideo" },
        "163": { inputs: { switch: ["153", 0], on_false: ["122", 0], on_true: ["154:184", 0] }, class_type: "ComfySwitchNode" },
        "92": { inputs: { video: ["163", 0], references: ["119", 0], continue_batch_with_last_frame: ["119", 4], merge_batch_videos: ["119", 3], filename_prefix: "video/MiniMax_H3" }, class_type: "SECoursesBatchVideoSaveMerge" },
    };
    if (firstFrame) output["114"] = { inputs: { image: firstFrame }, class_type: "SECoursesLoadImage" };
    if (initImage) output["114"] = { inputs: { image: initImage }, class_type: "SECoursesLoadImage" };
    if (router === "auto") output["148"] = { inputs: { references: ["119", 0] }, class_type: "SECoursesMiniMaxH3ReferenceMode" };
    return output;
}

test("math expressions follow the ComfyUI node (Python round, int output)", () => {
    assert.equal(evaluateMathExpression(LENGTH_GRID, { a: 5 }), 124);
    assert.equal(evaluateMathExpression(LENGTH_GRID, { a: 15 }), 362);
    assert.equal(evaluateMathExpression(LENGTH_GRID, { a: 0.1 }), 5);
    assert.equal(evaluateMathExpression("ceil(a / 32) * 32", { a: 1080 }), 1088);
    assert.equal(evaluateMathExpression("round(a)", { a: 2.5 }), 2);
    assert.equal(evaluateMathExpression("(5 - 12) % 17", {}), 10);
    assert.equal(evaluateMathExpression("-7 // 2", {}), -4);
    assert.equal(evaluateMathExpression("2 ** 3 ** 2", {}), 512);
    assert.equal(evaluateMathExpression("a if a else b", { a: 1, b: 2 }), null);
    assert.equal(evaluateMathExpression("max(values) + min(a, b)", { a: 3, b: 9 }), 12);
    assert.equal(evaluateMathExpression("(a > 2) and (b < 2)", { a: 3, b: 1 }), 1);
    assert.equal(evaluateMathExpression("a + b", { a: 1 }), null);
    assert.equal(evaluateMathExpression("process.exit(1)", { a: 1 }), null);
    assert.equal(evaluateMathExpression("this.constructor", {}), null);
});

test("roster rule: everything within the cap, only mentions above it", () => {
    const entries = Array.from({ length: 12 }, (_, i) => ({ name: `img${i + 1}` }));
    assert.equal(selectMentioned("no mentions", entries.slice(0, 9), 9, "image").length, 9);
    assert.deepEqual(selectMentioned("@image12 and @img3 then @image12 @image1", entries, 9, "image").map((e) => e.name), ["img12", "img3", "img1"]);
    assert.equal(selectMentioned("nothing", entries, 9, "image").length, 0);
});

test("legacy preset shape: normal route ignores references, prompt from the gallery", async () => {
    const refs = JSON.stringify({ images: [{ file: "reference_gallery/a.png [input]", name: "a.png" }], videos: [], audios: [] });
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, refs }), 119);
    assert.ok(result.estimate, result.reason);
    const est = result.estimate;
    assert.equal(est.width, 864);
    assert.equal(est.height, 480);
    assert.equal(est.frames, 124);
    assert.equal(est.parts.video, 37 * 405);
    assert.equal(est.parts.audio, 207 * 2);
    assert.equal(est.parts.refImages, 0, "gallery references are only used on the folder route");
    assert.equal(est.parts.keyframes, 0);
    assert.equal(est.parts.text, H3.textTokens("A cat."));
    assert.equal(est.approximate, false);
    assert.equal(result.label, "text to video");
});

test("folder route: the auto adapter attaches the gallery references", async () => {
    const refs = JSON.stringify({ images: [{ file: "reference_gallery/a.png [input]", name: "a.png" }], videos: [], audios: [] });
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, refs, batchFolder: "D:/prompts" }), 119);
    assert.ok(result.estimate, result.reason);
    const expectedRef = H3.estimate({ width: 864, height: 480, frames: 124, refImages: [{ width: 1920, height: 1080 }], pipeline: "secourses" }).parts.refImages;
    assert.equal(result.estimate.parts.refImages, expectedRef);
    assert.equal(result.label, "reference to video");
});

test("auto-routed preset shape: single runs with references take the auto adapter", async () => {
    const refs = JSON.stringify({ images: [{ file: "reference_gallery/a.png [input]", name: "a.png" }], videos: [], audios: [] });
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, refs, router: "auto" }), 119);
    assert.ok(result.estimate, result.reason);
    const expectedRef = H3.estimate({ width: 864, height: 480, frames: 124, refImages: [{ width: 1920, height: 1080 }], pipeline: "secourses" }).parts.refImages;
    assert.equal(result.estimate.parts.refImages, expectedRef);
    assert.equal(result.estimate.approximate, false);
    assert.equal(result.label, "reference to video");
});

test("auto-routed preset shape: no references and no batch keeps the normal route", async () => {
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, router: "auto" }), 119);
    assert.ok(result.estimate, result.reason);
    assert.equal(result.estimate.parts.refImages, 0);
    assert.equal(result.estimate.approximate, false);
    assert.equal(result.label, "text to video");
});

test("auto-routed preset shape: a folder batch without references still takes the auto adapter", async () => {
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, batchFolder: "D:/prompts", router: "auto" }), 119);
    assert.ok(result.estimate, result.reason);
    assert.equal(result.label, "text to video");
    assert.equal(result.estimate.parts.refImages, 0);
});

test("init image passes through the continuation node on the auto route", async () => {
    const refs = JSON.stringify({ images: [{ file: "reference_gallery/a.png [input]", name: "a.png" }], videos: [], audios: [] });
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, refs, router: "auto", initImage: "start.jpg" }), 119);
    assert.ok(result.estimate, result.reason);
    const expectedRef = H3.estimate({
        width: 864, height: 480, frames: 124,
        refImages: [{ width: 1920, height: 1080 }, { width: 864, height: 480 }],
        pipeline: "secourses",
    }).parts.refImages;
    assert.equal(result.estimate.parts.refImages, expectedRef, "the init image is counted as the starting-frame picture reference");
    assert.equal(result.label, "reference to video");
});

test("init image without references stays a keyframe on the auto route", async () => {
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, router: "auto", initImage: "start.jpg", refs: JSON.stringify({ images: [{ file: "reference_gallery/a.png [input]", name: "a.png" }] }) }), 119);
    assert.ok(result.estimate, result.reason);
    // With a reference attached the auto route runs; remove it and the normal route runs instead:
    const plain = await estimateFromPrompt(textToVideoPrompt({ duration: 5, router: "auto", initImage: "start.jpg" }), 119);
    assert.ok(plain.estimate, plain.reason);
    assert.equal(plain.label, "text to video");
    assert.equal(plain.estimate.parts.keyframes, 0, "the normal route does not read the continuation node");
});

test("duration follows the primitive through the batch-duration helper and the length grid", async () => {
    const eight = await estimateFromPrompt(textToVideoPrompt({ duration: 8 }), 119);
    assert.equal(eight.estimate.frames, H3.alignFrames(192));
    const fifteen = await estimateFromPrompt(textToVideoPrompt({ duration: 15 }), 119);
    assert.equal(fifteen.estimate.frames, 362);
    assert.equal(fifteen.estimate.parts.video, 107 * 405);
});

test("init audio: video length follows the audio and the whole soundtrack is a t=1 guide", async () => {
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, initAudio: "song.mp3" }), 119);
    assert.ok(result.estimate, result.reason);
    const est = result.estimate;
    assert.equal(est.frames, H3.framesForSeconds(9.0));
    assert.equal(est.parts.keyframes, est.audioT * 2);
});

test("init audio trim window drives the video length in match mode", () => {
    assert.equal(initAudioTrimmedDuration(9.0, null, null), 9.0);
    assert.equal(initAudioTrimmedDuration(9.0, 2.0, 5.5), 3.5);
    assert.equal(initAudioTrimmedDuration(9.0, 3.0, 0), 6.0); // 0 end = until the end of the file
    assert.equal(initAudioTrimmedDuration(9.0, 2.0, 30.0), 7.0); // end beyond EOF is capped like the decoder
    assert.equal(initAudioTrimmedDuration(9.0, 12.0, null), 0); // start past EOF decodes nothing
    assert.equal(initAudioTrimmedDuration(null, 2.0, 5.5), 3.5); // unknown file length, exact window
    assert.equal(initAudioTrimmedDuration(null, 2.0, null), null); // unknown length, open end
    assert.equal(initAudioTrimmedDuration(9.0, 5.0, 2.0), null); // inverted window is rejected by the backend
});

test("init audio trim: frames follow the trimmed window, keep-duration mode ignores it", async () => {
    const trimmed = await estimateFromPrompt(
        textToVideoPrompt({ duration: 5, initAudio: "song.mp3", initAudioTrim: { trim_start: 2.0, trim_end: 5.5 } }), 119);
    assert.ok(trimmed.estimate, trimmed.reason);
    assert.equal(trimmed.estimate.frames, H3.framesForSeconds(3.5));
    assert.equal(trimmed.estimate.parts.keyframes, trimmed.estimate.audioT * 2);
    assert.equal(trimmed.estimate.approximate, false);

    const openEnd = await estimateFromPrompt(
        textToVideoPrompt({ duration: 5, initAudio: "song.mp3", initAudioTrim: { trim_start: 3.0, trim_end: 0 } }), 119);
    assert.equal(openEnd.estimate.frames, H3.framesForSeconds(6.0));

    const keep = await estimateFromPrompt(
        textToVideoPrompt({ duration: 5, initAudio: "song.mp3", initAudioTrim: { trim_start: 2.0, trim_end: 5.5 }, durationMode: "keep workflow duration" }), 119);
    assert.equal(keep.estimate.frames, H3.framesForSeconds(5.0));
    assert.equal(keep.estimate.parts.keyframes, keep.estimate.audioT * 2); // the guide still covers the target length

    const inverted = await estimateFromPrompt(
        textToVideoPrompt({ duration: 5, initAudio: "song.mp3", initAudioTrim: { trim_start: 5.0, trim_end: 2.0 } }), 119);
    assert.ok(inverted.estimate); // unresolvable duration falls back to the defaults, flagged approximate
    assert.equal(inverted.estimate.approximate, true);
});

test("first frame adds one keyframe block", async () => {
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, firstFrame: "start.jpg" }), 119);
    assert.equal(result.estimate.parts.keyframes, 405);
    assert.equal(result.label, "image to video");
});

test("the face pass canvas is unknown, so it never masks the main generation", async () => {
    const result = await estimateFromPrompt(textToVideoPrompt({ duration: 5, faceOn: true }), 119);
    assert.ok(result.estimate);
    assert.equal(result.estimate.width, 864);
    assert.equal(result.estimate.approximate, false);
});

test("reference preset: gallery adapter with images, a trimmed video and audio; roster and audio-only rules", async () => {
    const refs = JSON.stringify({
        images: [{ file: "reference_gallery/a.png [input]", name: "a.png" }, { file: "reference_gallery/b.jpg [input]", name: "b.jpg" }],
        videos: [{ file: "reference_gallery/clip.mp4 [input]", name: "clip.mp4", trim_start: 2, trim_end: 7.5 }],
        audios: [{ file: "reference_gallery/voice.wav [input]", name: "voice.wav" }],
    });
    const output = {
        "400": gallery({ references: refs, prompt: "@image1 @video1 @audio1" }),
        "115": { inputs: { aspect_ratio: "16:9 (Widescreen)", megapixels: 0.4, width: 864, height: 480, multiple: 32 }, class_type: "SECoursesResolutionSync" },
        "132": { inputs: { value: 5 }, class_type: "PrimitiveFloat" },
        "131": { inputs: { expression: LENGTH_GRID, "values.a": ["132", 0] }, class_type: "ComfyMathExpression" },
        "136": { inputs: { clip: ["1", 0], vae: ["2", 0], audio_vae: ["3", 0], references: ["400", 0], width: ["115", 0], height: ["115", 1], length: ["131", 1], ref_image_size: "match" }, class_type: "SECoursesMiniMaxH3Auto" },
        "50": { inputs: { model: ["4", 0], conditioning: ["136", 0] }, class_type: "BasicGuider" },
        "51": { inputs: { guider: ["50", 0], latent_image: ["136", 1] }, class_type: "SamplerCustomAdvanced" },
        "52": { inputs: { samples: ["51", 0], vae: ["2", 0] }, class_type: "VAEDecode" },
        "53": { inputs: { images: ["52", 0], fps: 24 }, class_type: "SaveVideo" },
    };
    const result = await estimateFromPrompt(output, 400);
    assert.ok(result.estimate, result.reason);
    const expected = H3.estimate({
        width: 864, height: 480, frames: 124, prompt: "@image1 @video1 @audio1", pipeline: "secourses", refImageSize: "match", maxSeconds: 15,
        refImages: [{ width: 1920, height: 1080 }, { width: 1080, height: 1920 }],
        refVideos: [{ width: 1920, height: 1080, duration: 8, hasAudio: true, trimStart: 2, trimEnd: 7.5 }],
        refAudios: [{ duration: 3.37 }],
    });
    assert.equal(result.estimate.total, expected.total);
    assert.equal(result.estimate.approximate, false);

    // audio-only variant: 32x32 canvas, video contributes only its soundtrack
    output["136"] = { inputs: { ...output["136"].inputs, width: 32, height: 32, audio_only_mode: true }, class_type: "SECoursesMiniMaxH3References" };
    const audioOnly = await estimateFromPrompt(output, 400);
    assert.equal(audioOnly.label, "audio only with references");
    assert.equal(audioOnly.estimate.parts.refVideos, 0);
    assert.ok(audioOnly.estimate.parts.refAudios > 0);
});

test("core reference node (SwarmUI-style workflow): image loaders and a LoadVideo clip", async () => {
    const output = {
        "1": { inputs: { image: "start.jpg" }, class_type: "LoadImage" },
        "2": { inputs: { file: "reference_gallery/clip.mp4 [input]" }, class_type: "LoadVideo" },
        "3": { inputs: { video: ["2", 0] }, class_type: "GetVideoComponents" },
        "6": { inputs: { clip: ["9", 0], vae: ["8", 0], audio_vae: ["7", 0], prompt: "hello <Picture 1>", width: 1344, height: 768, length: 124, ref_image_size: "max", "ref_images.ref_image_0": ["1", 0], "ref_videos.ref_video_0": ["3", 0], "ref_video_audios.ref_video_audio_0": ["3", 1] }, class_type: "MiniMaxH3ReferenceToVideo" },
        "10": { inputs: { conditioning: ["6", 0], latent_image: ["6", 1] }, class_type: "KSampler" },
        "11": { inputs: { samples: ["10", 0] }, class_type: "VAEDecode" },
        "12": { inputs: { images: ["11", 0] }, class_type: "SaveVideo" },
        "400": gallery(),
    };
    const result = await estimateFromPrompt(output, 400);
    assert.ok(result.estimate, result.reason);
    const expected = H3.estimate({
        width: 1344, height: 768, frames: 124, prompt: "hello <Picture 1>", pipeline: "core", refImageSize: "max", maxSeconds: 15,
        refImages: [{ width: 640, height: 640 }], refVideos: [{ width: 1920, height: 1080, duration: 8, hasAudio: true }],
    });
    assert.equal(result.estimate.total, expected.total);
});

test("no active conditioning node -> unavailable with a reason", async () => {
    const result = await estimateFromPrompt({ "400": gallery() }, 400);
    assert.equal(result.estimate, null);
    assert.match(result.reason, /no MiniMax H3 conditioning node/);
});

test("PromptResolver follows only the taken switch branch", async () => {
    const output = {
        "1": { inputs: { value: true }, class_type: "PrimitiveBoolean" },
        "2": { inputs: { value: 3 }, class_type: "PrimitiveInt" },
        "3": { inputs: { value: 4 }, class_type: "PrimitiveInt" },
        "4": { inputs: { switch: ["1", 0], on_false: ["2", 0], on_true: ["3", 0] }, class_type: "ComfySwitchNode" },
        "5": { inputs: { images: ["4", 0] }, class_type: "PreviewImage" },
    };
    const resolver = new PromptResolver(output);
    assert.equal(await resolver.resolve("4", 0), 4);
    const active = await resolver.activeNodes();
    assert.ok(active.has("3") && active.has("1") && !active.has("2"));
});
