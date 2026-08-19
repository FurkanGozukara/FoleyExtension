/**
 * MiniMax H3 packed-sequence token model (pure math, no DOM, no ComfyUI/SwarmUI dependencies).
 *
 * The H3 DiT runs full attention over ONE packed sequence:
 *   [text | keyframe / reference blocks | target audio | target video]
 * so VRAM and time scale with its length. This module reproduces ComfyUI's
 * `comfy/ldm/minimax/model.py` PackedLayout and `comfy_extras/nodes_minimax_h3.py`
 * (plus the SECourses gallery adapter's reference sizing) so a UI can show a live
 * "current / budget" token estimate before anything is queued.
 *
 * MiniMax documents no hard token limit (only 9 images / 3 videos / 3 audios, 15 s of
 * reference video and audio, 4-15 s output at a 768p canvas). `BUDGET_TOKENS` is the packed
 * length of that documented maximum single output (15 s at the 768x1344 canvas cap), which
 * is the envelope the released checkpoints were tested for; longer sequences run but cost
 * more, and SageAttention kernels overflow int32 above `SAGE_ATTENTION_MAX_TOKENS`.
 *
 * This file is loaded as a classic <script> by SwarmUI and as an ES module by ComfyUI, so it
 * only publishes a global (`globalThis.SECoursesMiniMaxH3Tokens`) and uses no import/export.
 * FoleyExtension/web/js/minimax_h3_tokens.js and
 * SwarmUI_Premium_Extensions/MiniMaxH3References/Assets/minimax_h3_tokens.js must stay identical.
 */
(function (root) {
    "use strict";

    const FPS = 24;
    const AUDIO_LATENT_FPS = 40;
    const CANVAS_MULTIPLE = 32;
    const BASE_SHORT_EDGE = 768;
    const MAX_PIXELS = 768 * 1344;
    const REF_IMAGE_SHORT_EDGE = 2048;
    const QWEN_MIN_PIXELS = 3136;
    const QWEN_MAX_PIXELS = 12845056;
    // SECourses gallery adapter (FoleyExtension reference_gallery_nodes.py) memory-safety caps.
    const IMAGE_MAX_AREA = 8 * 1024 * 1024;
    const IMAGE_FLOAT_BUDGET = 512 * 1024 * 1024;
    const VIDEO_FLOAT_BUDGET = 512 * 1024 * 1024;
    const RGB_FLOAT_BYTES_PER_PIXEL = 12;
    const MAX_IMAGES = 9;
    const MAX_VIDEOS = 3;
    const MAX_AUDIOS = 3;
    const DOCUMENTED_MAX_SECONDS = 15;
    // KJNodes MiniMaxH3TokenCounter: sageattention offsets overflow int32 at seq_len * 7168 >= 2^31.
    const SAGE_ATTENTION_MAX_TOKENS = Math.floor((2 ** 31 - 1) / 7168);

    /** Python round(): half to even. */
    function pyRound(value) {
        const floor = Math.floor(value);
        const diff = value - floor;
        if (diff > 0.5) return floor + 1;
        if (diff < 0.5) return floor;
        return floor % 2 === 0 ? floor : floor + 1;
    }

    /** MiniMax H3's 17k+5 frame grid, rounded up (5, 22, 39, ...). */
    function alignFrames(frames) {
        let n = Math.max(5, Math.round(Number(frames) || 0));
        while (n % 17 !== 5) n += 1;
        return n;
    }

    /** Same grid rounded down (reference clips are cropped, never padded). */
    function alignFramesDown(frames) {
        let n = Math.floor(Number(frames) || 0);
        if (n < 5) return n;
        while (n % 17 !== 5) n -= 1;
        return n;
    }

    function framesForSeconds(seconds) {
        return alignFrames(pyRound((Number(seconds) || 0) * FPS));
    }

    /** Video latent frames for an aligned pixel frame count. */
    function videoLatentT(frames) {
        return frames <= 5 ? 2 : Math.floor((frames - 5) / 17) * 5 + 2;
    }

    /** Target audio latent frames: round(duration * 40) like _empty_av_latent. */
    function audioLatentT(frames) {
        return pyRound((frames / FPS) * AUDIO_LATENT_FPS);
    }

    /** Encoded reference audio latent frames: the audio VAE right-pads to whole 800-sample hops. */
    function audioLatentForSeconds(seconds) {
        return Math.max(0, Math.ceil((Number(seconds) || 0) * AUDIO_LATENT_FPS - 1e-9));
    }

    /** DiT rows per latent frame: the empty latent is (H // 16, W // 16), padded up to the 2x2 patch. */
    function patchRows(width, height) {
        const latentH = Math.max(1, Math.floor((Number(height) || 0) / 16));
        const latentW = Math.max(1, Math.floor((Number(width) || 0) / 16));
        return Math.ceil(latentH / 2) * Math.ceil(latentW / 2);
    }

    /** Qwen3-VL vision tokens for one image / one 2-frame video block (patch 16, merge 2). */
    function qwenVisionTokens(width, height) {
        const factor = 32;
        let hBar = Math.round(height / factor) * factor;
        let wBar = Math.round(width / factor) * factor;
        if (hBar * wBar > QWEN_MAX_PIXELS) {
            const beta = Math.sqrt((height * width) / QWEN_MAX_PIXELS);
            hBar = Math.max(factor, Math.floor(height / beta / factor) * factor);
            wBar = Math.max(factor, Math.floor(width / beta / factor) * factor);
        } else if (hBar * wBar < QWEN_MIN_PIXELS) {
            const beta = Math.sqrt(QWEN_MIN_PIXELS / (height * width));
            hBar = Math.ceil((height * beta) / factor) * factor;
            wBar = Math.ceil((width * beta) / factor) * factor;
        }
        return (hBar / 16 / 2) * (wBar / 16 / 2);
    }

    /** nodes_minimax_h3.adapt_canvas: 768 short edge, 768*1344 area cap, per-axis round to 32. */
    function adaptCanvas(width, height) {
        const ratio = width / height;
        let nomW, nomH;
        if (ratio >= 1.0) {
            nomW = BASE_SHORT_EDGE * ratio;
            nomH = BASE_SHORT_EDGE;
        } else {
            nomW = BASE_SHORT_EDGE;
            nomH = BASE_SHORT_EDGE / ratio;
        }
        if (nomW * nomH > MAX_PIXELS) {
            const s = Math.sqrt(MAX_PIXELS / (nomW * nomH));
            nomW *= s;
            nomH *= s;
        }
        return [
            Math.max(CANVAS_MULTIPLE, pyRound(nomW / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            Math.max(CANVAS_MULTIPLE, pyRound(nomH / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        ];
    }

    /** reference_gallery_nodes._fit_dimensions: aspect-preserving, down-only, on the 32 px grid. */
    function fitDimensions(width, height, maxPixels, maxShortEdge = null, multiple = CANVAS_MULTIPLE) {
        width = Math.max(1, Math.trunc(width));
        height = Math.max(1, Math.trunc(height));
        maxPixels = Math.max(1, Math.trunc(maxPixels));
        let scale = 1.0;
        if (maxShortEdge) scale = Math.min(scale, maxShortEdge / Math.min(width, height));
        if (width * height * scale * scale > maxPixels) {
            scale = Math.min(scale, Math.sqrt(maxPixels / (width * height)));
        }
        if (multiple <= 1) {
            return [Math.max(1, pyRound(width * scale)), Math.max(1, pyRound(height * scale))];
        }
        const idealWidth = width * scale;
        const idealHeight = height * scale;
        const widthSteps = Math.max(1, Math.ceil(idealWidth / multiple));
        const heightSteps = Math.max(1, Math.ceil(idealHeight / multiple));
        const sourceRatio = width / height;
        const candidates = [];
        const addCandidate = (ws, hs) => {
            if (ws < 1 || hs < 1 || ws > widthSteps || hs > heightSteps) return;
            const tw = ws * multiple;
            const th = hs * multiple;
            if (width >= multiple && tw > width) return;
            if (height >= multiple && th > height) return;
            const area = tw * th;
            if (area > maxPixels) return;
            if (maxShortEdge && Math.min(tw, th) > maxShortEdge) return;
            candidates.push({ error: Math.abs(tw / th / sourceRatio - 1.0), area, tw, th });
        };
        if (widthSteps <= heightSteps) {
            for (let ws = 1; ws <= widthSteps; ws++) {
                const nearest = (ws * multiple / sourceRatio) / multiple;
                addCandidate(ws, Math.max(1, Math.floor(nearest)));
                addCandidate(ws, Math.max(1, Math.ceil(nearest)));
            }
        } else {
            for (let hs = 1; hs <= heightSteps; hs++) {
                const nearest = (hs * multiple * sourceRatio) / multiple;
                addCandidate(Math.max(1, Math.floor(nearest)), hs);
                addCandidate(Math.max(1, Math.ceil(nearest)), hs);
            }
        }
        if (!candidates.length) return [multiple, multiple];
        const ratioSafe = candidates.filter((c) => c.error <= 0.02);
        let pick;
        if (ratioSafe.length) {
            pick = ratioSafe.reduce((best, c) => (c.area > best.area || (c.area === best.area && c.error < best.error) ? c : best));
        } else {
            pick = candidates.reduce((best, c) => (c.error < best.error || (c.error === best.error && c.area > best.area) ? c : best));
        }
        return [pick.tw, pick.th];
    }

    /** Qwen text tokens for a reference label / timestamp tag (measured with the Qwen2 tokenizer). */
    function labelTokens(kind, index) {
        const digits = String(index).length;
        if (kind === "audio") return 4 + digits;      // "<Audio j>: "
        return 5 + digits;                            // "<Picture i>: " / "<Video k>: "
    }
    function timestampTokens(seconds) {
        return 3 + seconds.toFixed(1).length;         // "<T.T seconds>"
    }

    /** Rough Qwen tokenizer estimate: a letter run, a digit, or any other non-space char is one token. */
    function textTokens(text) {
        const s = String(text || "");
        if (!s.trim()) return 0;
        return (s.match(/[A-Za-z]+|\d|[^\sA-Za-z\d]/g) || []).length;
    }

    /** Encoded size of one reference image, per pipeline ('secourses' gallery adapter or 'core' node). */
    function referenceImageSize(image, canvasWidth, canvasHeight, refImageSize, pipeline) {
        const w = Math.max(1, Math.trunc(image.width || 0));
        const h = Math.max(1, Math.trunc(image.height || 0));
        if (pipeline === "secourses") {
            // _prepare_image_references pre-fits every image on the 32 px grid; the core node then
            // finds it already within its area / short-edge rule and keeps it as-is.
            if (refImageSize === "match") {
                const area = Math.min(IMAGE_MAX_AREA, Math.max(CANVAS_MULTIPLE ** 2, canvasWidth * canvasHeight));
                return fitDimensions(w, h, area, null);
            }
            return fitDimensions(w, h, IMAGE_MAX_AREA, REF_IMAGE_SHORT_EDGE);
        }
        let scale;
        if (refImageSize === "match") {
            scale = Math.min(1.0, Math.sqrt((canvasWidth * canvasHeight) / (w * h)));
        } else {
            scale = Math.min(1.0, REF_IMAGE_SHORT_EDGE / Math.min(w, h));
        }
        return [
            Math.max(CANVAS_MULTIPLE, pyRound((w * scale) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            Math.max(CANVAS_MULTIPLE, pyRound((h * scale) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        ];
    }

    /** Effective (start, seconds) window of a trimmed reference against the max-seconds cap. */
    function trimWindow(entry, maxSeconds) {
        const start = Math.max(0, Number(entry.trimStart) || 0);
        const end = entry.trimEnd == null ? null : Number(entry.trimEnd);
        let seconds = maxSeconds;
        if (end != null && Number.isFinite(end)) seconds = Math.min(maxSeconds, Math.max(0, end - start));
        if (entry.duration != null && Number.isFinite(entry.duration)) {
            seconds = Math.min(seconds, Math.max(0, entry.duration - start));
        }
        return { start, seconds: Math.max(0, seconds) };
    }

    /**
     * Estimate the packed sequence length of one MiniMax H3 generation.
     *
     * spec = {
     *   width, height,               // generation canvas in pixels
     *   frames | seconds,            // output length (snapped up to the 17k+5 grid at 24 fps)
     *   prompt,                      // prompt text (Qwen tokens are estimated)
     *   pipeline: 'secourses'|'core',// which reference sizing rules apply
     *   refImageSize: 'match'|'max',
     *   maxSeconds,                  // per-reference duration cap (video/audio)
     *   audioOnly,                   // reference videos contribute only their soundtrack
     *   refImages: [{width, height}], refVideos: [{width, height, duration, hasAudio, trimStart, trimEnd}],
     *   refAudios: [{duration, trimStart, trimEnd}],   // unknown metadata (null) is assumed at the caps
     *   keyframeImages,              // first/last-frame style image keyframes (fl2va)
     *   audioGuide,                  // init audio guide keyframe (whole soundtrack, t=1.0)
     *   textTokens                   // optional exact Qwen token count override
     * }
     */
    function estimate(spec) {
        const s = spec || {};
        const width = Math.max(CANVAS_MULTIPLE, Math.trunc(Number(s.width) || 0));
        const height = Math.max(CANVAS_MULTIPLE, Math.trunc(Number(s.height) || 0));
        const frames = alignFrames(s.frames != null ? s.frames : pyRound((Number(s.seconds) || 0) * FPS));
        const latentT = videoLatentT(frames);
        const audioT = audioLatentT(frames);
        const rows = patchRows(width, height);
        const pipeline = s.pipeline === "core" ? "core" : "secourses";
        const refImageSize = s.refImageSize === "max" ? "max" : "match";
        const maxSeconds = Math.max(0.05, Number(s.maxSeconds) || DOCUMENTED_MAX_SECONDS);
        const notes = [];
        let approximate = false;

        const parts = { text: 0, keyframes: 0, refImages: 0, refVideos: 0, refAudios: 0, audio: audioT * 2, video: latentT * rows };
        parts.text += s.textTokens != null ? Math.max(0, Math.trunc(s.textTokens)) : textTokens(s.prompt);

        const keyframeImages = Math.max(0, Math.trunc(Number(s.keyframeImages) || 0));
        if (keyframeImages) {
            parts.keyframes += keyframeImages * rows;                 // one latent frame each
            parts.text += keyframeImages * (labelTokens("image", 1) + 2) + keyframeImages * qwenVisionTokens(width, height);
        }
        if (s.audioGuide) parts.keyframes += audioT * 2;

        const images = (s.refImages || []).slice(0, MAX_IMAGES);
        images.forEach((image, index) => {
            let w = image && image.width, h = image && image.height;
            if (!(w > 0 && h > 0)) {
                approximate = true;
                [w, h] = refImageSize === "max" ? [Math.round(REF_IMAGE_SHORT_EDGE * 16 / 9), REF_IMAGE_SHORT_EDGE] : [width, height];
            }
            const [tw, th] = referenceImageSize({ width: w, height: h }, width, height, refImageSize, pipeline);
            parts.refImages += patchRows(tw, th);
            parts.text += labelTokens("image", index + 1) + 2 + qwenVisionTokens(tw, th);
        });

        // Video references: canvas + usable frames follow the pipeline that decodes them.
        const videos = (s.refVideos || []).slice(0, MAX_VIDEOS);
        let maxRefFrames = Math.min(frames, Math.max(1, pyRound(maxSeconds * FPS)));
        maxRefFrames = maxRefFrames < 5 ? maxRefFrames : alignFramesDown(maxRefFrames);
        let videoAreaCap = MAX_PIXELS;
        if (pipeline === "secourses") {
            const floatPixelBudget = Math.max(1, Math.floor(VIDEO_FLOAT_BUDGET / (Math.max(1, maxRefFrames) * RGB_FLOAT_BYTES_PER_PIXEL)));
            videoAreaCap = Math.max(CANVAS_MULTIPLE ** 2, Math.min(MAX_PIXELS, floatPixelBudget));
        }
        let audioLabel = 0;
        videos.forEach((video, index) => {
            const window = trimWindow(video || {}, maxSeconds);
            let vw = video && video.width, vh = video && video.height;
            if (!(vw > 0 && vh > 0)) {
                approximate = true;
                [vw, vh] = [1344, 768];
            }
            if (video && video.duration == null && video.trimEnd == null) approximate = true;
            const hasAudio = video ? video.hasAudio !== false : true;
            let n = Math.min(maxRefFrames, Math.max(1, pyRound(window.seconds * FPS)));
            n = n < 5 ? n : alignFramesDown(n);
            // The gallery adapter cuts a soundtrack to the generated duration; the core node
            // (SwarmUI's Video Slice path) encodes the whole sliced soundtrack.
            const audioSeconds = pipeline === "secourses"
                ? Math.min(window.seconds, maxRefFrames / FPS, maxSeconds)
                : Math.min(window.seconds, maxSeconds);
            if (s.audioOnly) {
                if (hasAudio) {
                    audioLabel += 1;
                    parts.refAudios += audioLatentForSeconds(audioSeconds) * 2;
                    parts.text += labelTokens("audio", audioLabel);
                }
                return;
            }
            let cw, ch;
            if (pipeline === "secourses") {
                [cw, ch] = fitDimensions(vw, vh, videoAreaCap);
            } else {
                [cw, ch] = adaptCanvas(vw, vh);
                if (vw * vh < cw * ch) {
                    cw = Math.max(CANVAS_MULTIPLE, pyRound(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE);
                    ch = Math.max(CANVAS_MULTIPLE, pyRound(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE);
                }
            }
            if (n < 5) {
                notes.push(`reference video ${index + 1} is shorter than 5 frames`);
                n = 5;
            }
            const clipLatentT = videoLatentT(n);
            const clipRows = patchRows(cw, ch);
            parts.refVideos += clipLatentT * clipRows;
            if (hasAudio) {
                audioLabel += 1;
                parts.refVideos += audioLatentForSeconds(audioSeconds) * 2;
                parts.text += labelTokens("audio", audioLabel);
            }
            // Qwen sees the clip at 2 fps as 2-frame blocks with a timestamp tag each.
            const sampled = Math.ceil(n / (FPS / 2));
            const blocks = Math.ceil(sampled / 2);
            parts.text += labelTokens("video", index + 1);
            for (let b = 0; b < blocks; b++) {
                parts.text += timestampTokens(b + 0.25) + 2 + qwenVisionTokens(cw, ch);
            }
        });

        const audios = (s.refAudios || []).slice(0, MAX_AUDIOS);
        audios.forEach((audio) => {
            if (!audio || audio.duration == null) approximate = true;
            const window = trimWindow(audio || {}, maxSeconds);
            audioLabel += 1;
            parts.refAudios += audioLatentForSeconds(window.seconds) * 2;
            parts.text += labelTokens("audio", audioLabel);
        });

        const total = parts.text + parts.keyframes + parts.refImages + parts.refVideos + parts.refAudios + parts.audio + parts.video;
        return {
            total,
            parts,
            width,
            height,
            frames,
            seconds: frames / FPS,
            latentT,
            audioT,
            rows,
            approximate,
            notes,
            budget: BUDGET_TOKENS,
            sageLimit: SAGE_ATTENTION_MAX_TOKENS,
        };
    }

    // The documented maximum single output: 15 s (362 frames) on the 768x1344 canvas cap, no text.
    const BUDGET_TOKENS = (() => {
        const frames = alignFrames(DOCUMENTED_MAX_SECONDS * FPS);
        return videoLatentT(frames) * patchRows(1344, 768) + audioLatentT(frames) * 2;
    })();

    function formatTokens(n) {
        n = Math.max(0, Math.round(Number(n) || 0));
        if (n < 1000) return String(n);
        if (n < 10000) return (n / 1000).toFixed(2).replace(/\.?0+$/, "") + "k";
        if (n < 100000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
        return Math.round(n / 1000) + "k";
    }

    /** Human readable breakdown lines for a tooltip. */
    function describe(est) {
        if (!est) return [];
        const p = est.parts;
        const lines = [
            `${est.width}x${est.height}, ${est.frames} frames (${est.seconds.toFixed(2)} s at ${FPS} fps): ${est.latentT} x ${est.rows} video patches`,
            `target video ${p.video.toLocaleString()} + target audio ${p.audio.toLocaleString()} + text ${p.text.toLocaleString()}`,
        ];
        if (p.keyframes) lines.push(`keyframes / init audio guide ${p.keyframes.toLocaleString()}`);
        if (p.refImages) lines.push(`reference images ${p.refImages.toLocaleString()}`);
        if (p.refVideos) lines.push(`reference videos ${p.refVideos.toLocaleString()}`);
        if (p.refAudios) lines.push(`reference audio ${p.refAudios.toLocaleString()}`);
        lines.push(`budget ${BUDGET_TOKENS.toLocaleString()} = MiniMax H3's documented maximum output (15 s at the 768x1344 canvas cap)`);
        if (est.total > SAGE_ATTENTION_MAX_TOKENS) {
            lines.push(`over ${SAGE_ATTENTION_MAX_TOKENS.toLocaleString()}: SageAttention kernels overflow int32, use another attention backend`);
        } else if (est.total > BUDGET_TOKENS) {
            lines.push("above the documented envelope: it still generates, but slower, with more VRAM and outside the quality-tested range");
        }
        for (const note of est.notes || []) lines.push(note);
        if (est.approximate) lines.push("some reference metadata is unknown; assumed at the size / duration caps");
        return lines;
    }

    root.SECoursesMiniMaxH3Tokens = {
        FPS, AUDIO_LATENT_FPS, CANVAS_MULTIPLE, MAX_PIXELS, REF_IMAGE_SHORT_EDGE,
        MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS, DOCUMENTED_MAX_SECONDS,
        BUDGET_TOKENS, SAGE_ATTENTION_MAX_TOKENS,
        pyRound, alignFrames, alignFramesDown, framesForSeconds, videoLatentT, audioLatentT, audioLatentForSeconds,
        patchRows, qwenVisionTokens, adaptCanvas, fitDimensions, referenceImageSize, labelTokens, timestampTokens,
        textTokens, trimWindow, estimate, formatTokens, describe,
    };
})(typeof globalThis !== "undefined" ? globalThis : this);
