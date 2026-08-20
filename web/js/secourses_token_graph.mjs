/**
 * Graph-side token estimation (pure: no DOM, no ComfyUI imports; unit-testable in Node).
 *
 * Works on the flat API prompt (`app.graphToPrompt().output`): the workflow with subgraphs
 * flattened and bypassed nodes resolved. Only nodes on the active route are considered (walked
 * back from the output nodes, lazy If/Else switches follow only the branch their `switch`
 * selects), the conditioning nodes the gallery feeds are found, their inputs are resolved
 * through a small evaluator that understands the pure helper nodes the presets use
 * (primitives, Math Expression, Resolution Sync, duration / init-audio helpers, image /
 * audio / video loaders, the gallery itself), and a registered per-model estimator turns the
 * resolved inputs into a token estimate (MiniMax H3: `minimax_h3_tokens.js`, the exact
 * ComfyUI PackedLayout math).
 */

import "./minimax_h3_tokens.js";

const H3 = globalThis.SECoursesMiniMaxH3Tokens;
export const GALLERY_CLASS = "SECoursesReferenceGallery";
const NO_IMAGE = "(none - disabled)";
const NO_AUDIO = "(none - disabled)";

/** Injected media probe: file -> Promise<{kind, width, height, duration, has_audio} | null>. */
let mediaInfo = async () => null;
export function setMediaInfoProvider(provider) {
    mediaInfo = typeof provider === "function" ? provider : async () => null;
}

async function imageDescriptor(file) {
    if (!file || file === NO_IMAGE) return null;
    const info = await mediaInfo(file);
    return { kind: "image", file, width: info?.width ?? null, height: info?.height ?? null };
}

async function audioDescriptor(file) {
    if (!file || file === NO_AUDIO) return null;
    const info = await mediaInfo(file);
    return { kind: "audio", file, duration: info?.duration ?? null };
}

/**
 * Mirrors init_audio_nodes.trim_window + the windowed decode: the seconds actually kept of an
 * init audio file trimmed to [trim_start, trim_end) (0 / null end = until the end of the file).
 * Returns null when the length is unknown or the window is invalid (the backend rejects it).
 */
export function initAudioTrimmedDuration(duration, trimStart, trimEnd) {
    const start = Math.max(0, trimStart ?? 0);
    const end = trimEnd != null && trimEnd > 0 ? trimEnd : null;
    if (end != null && end <= start) return null; // VALIDATE_INPUTS refuses this window
    if (duration == null) return end != null ? end - start : null;
    const stop = end == null ? duration : Math.min(end, duration);
    return Math.max(0, stop - start);
}

async function videoDescriptor(file) {
    if (!file) return null;
    const info = await mediaInfo(file);
    return {
        kind: "video", file,
        width: info?.width ?? null, height: info?.height ?? null,
        duration: info?.duration ?? null, hasAudio: info ? info.has_audio !== false : true,
    };
}

// ==================== Safe Math Expression evaluation ====================

const MATH_FUNCTIONS = {
    sum: (...args) => (args.length === 1 && Array.isArray(args[0]) ? args[0] : args).reduce((a, b) => a + b, 0),
    min: (...a) => Math.min(...(a.length === 1 && Array.isArray(a[0]) ? a[0] : a)),
    max: (...a) => Math.max(...(a.length === 1 && Array.isArray(a[0]) ? a[0] : a)),
    abs: Math.abs, round: (v) => H3.pyRound(v), pow: Math.pow, sqrt: Math.sqrt,
    ceil: Math.ceil, floor: Math.floor, log: Math.log, log2: Math.log2, log10: Math.log10,
    sin: Math.sin, cos: Math.cos, tan: Math.tan, int: Math.trunc, float: Number,
};

/**
 * Evaluates a ComfyMathExpression formula (Python / simpleeval semantics: floored %, //, **,
 * banker's round, and/or/not) over resolved numeric inputs. Returns null when the formula uses
 * anything outside numbers, the node's functions, and its variables. No code generation.
 */
export function evaluateMathExpression(expression, values) {
    const source = String(expression || "").trim();
    if (!source) return null;
    const tokens = [];
    const pattern = /\s*(?:(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+(?:e[+-]?\d+)?)|([A-Za-z_]\w*)|(\*\*|\/\/|<=|>=|==|!=|[-+*/%(),<>])|(.))/gy;
    let match;
    while ((match = pattern.exec(source)) !== null) {
        if (match[1] != null) tokens.push({ type: "num", value: Number(match[1]) });
        else if (match[2] != null) tokens.push({ type: "id", value: match[2] });
        else if (match[3] != null) tokens.push({ type: "op", value: match[3] });
        else return null; // unsupported character
    }
    let pos = 0;
    const peek = () => tokens[pos];
    const take = (type, value) => {
        const token = tokens[pos];
        if (!token || token.type !== type || (value !== undefined && token.value !== value)) throw new Error("syntax");
        pos += 1;
        return token;
    };
    const isOp = (value) => peek()?.type === "op" && peek().value === value;
    const isKeyword = (value) => peek()?.type === "id" && peek().value === value;
    const truthy = (v) => (Array.isArray(v) ? v.length > 0 : Boolean(v));

    function parseOr() {
        let left = parseAnd();
        while (isKeyword("or")) { pos += 1; const right = parseAnd(); left = truthy(left) ? left : right; }
        return left;
    }
    function parseAnd() {
        let left = parseNot();
        while (isKeyword("and")) { pos += 1; const right = parseNot(); left = truthy(left) ? right : left; }
        return left;
    }
    function parseNot() {
        if (isKeyword("not")) { pos += 1; return !truthy(parseNot()); }
        return parseComparison();
    }
    function parseComparison() {
        let left = parseArith();
        while (peek()?.type === "op" && ["<", ">", "<=", ">=", "==", "!="].includes(peek().value)) {
            const op = take("op").value;
            const right = parseArith();
            left = op === "<" ? left < right : op === ">" ? left > right : op === "<=" ? left <= right
                : op === ">=" ? left >= right : op === "==" ? left === right : left !== right;
        }
        return left;
    }
    function parseArith() {
        let left = parseTerm();
        while (isOp("+") || isOp("-")) {
            const op = take("op").value;
            const right = parseTerm();
            left = op === "+" ? Number(left) + Number(right) : Number(left) - Number(right);
        }
        return left;
    }
    function parseTerm() {
        let left = parseUnary();
        while (isOp("*") || isOp("/") || isOp("//") || isOp("%")) {
            const op = take("op").value;
            const right = Number(parseUnary());
            const l = Number(left);
            if (op === "*") left = l * right;
            else if (op === "/") { if (right === 0) throw new Error("division by zero"); left = l / right; }
            else if (op === "//") { if (right === 0) throw new Error("division by zero"); left = Math.floor(l / right); }
            else { if (right === 0) throw new Error("modulo by zero"); left = ((l % right) + right) % right; }
        }
        return left;
    }
    function parseUnary() {
        if (isOp("-")) { pos += 1; return -Number(parseUnary()); }
        if (isOp("+")) { pos += 1; return Number(parseUnary()); }
        return parsePower();
    }
    function parsePower() {
        const base = parsePrimary();
        if (isOp("**")) { pos += 1; return Math.pow(Number(base), Number(parseUnary())); }
        return base;
    }
    function parsePrimary() {
        const token = peek();
        if (!token) throw new Error("syntax");
        if (token.type === "num") { pos += 1; return token.value; }
        if (token.type === "op" && token.value === "(") {
            pos += 1;
            const value = parseOr();
            take("op", ")");
            return value;
        }
        if (token.type === "id") {
            pos += 1;
            if (token.value === "True") return true;
            if (token.value === "False") return false;
            if (token.value === "None") return null;
            if (Object.prototype.hasOwnProperty.call(MATH_FUNCTIONS, token.value)) {
                take("op", "(");
                const args = [];
                if (!isOp(")")) {
                    args.push(parseOr());
                    while (isOp(",")) { pos += 1; args.push(parseOr()); }
                }
                take("op", ")");
                return MATH_FUNCTIONS[token.value](...args);
            }
            if (token.value === "values") return Object.values(values);
            if (Object.prototype.hasOwnProperty.call(values, token.value)) return values[token.value];
            throw new Error(`unknown name ${token.value}`);
        }
        throw new Error("syntax");
    }
    try {
        const result = parseOr();
        if (pos !== tokens.length) return null;
        const number = typeof result === "boolean" ? Number(result) : result;
        return typeof number === "number" && Number.isFinite(number) ? number : null;
    } catch (error) {
        return null;
    }
}

// ==================== Flat-prompt evaluator ====================

const isLink = (value) => Array.isArray(value) && value.length === 2 && typeof value[0] === "string";
const asNumber = (value) => (typeof value === "number" && Number.isFinite(value) ? value
    : typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value)) ? Number(value) : null);
const asBool = (value) => (typeof value === "boolean" ? value
    : typeof value === "number" ? value !== 0
    : typeof value === "string" ? ["true", "1", "yes", "on"].includes(value.trim().toLowerCase()) : null);

/** Parses the gallery manifest widget value into per-type entry lists. */
function parseManifest(value) {
    let parsed = {};
    try { parsed = JSON.parse(value || "{}"); } catch (error) { parsed = {}; }
    const list = (key) => (Array.isArray(parsed[key]) ? parsed[key].filter((e) => e && e.file) : []);
    return { images: list("images"), videos: list("videos"), audios: list("audios") };
}

/**
 * Node evaluators: class -> async (ctx, node, slot) => value. `ctx.input(node, name)` resolves an
 * input (widget value or upstream link). Unknown classes resolve to `undefined`.
 */
export const NODE_EVALUATORS = {
    PrimitiveInt: (ctx, node) => ctx.input(node, "value"),
    PrimitiveFloat: (ctx, node) => ctx.input(node, "value"),
    PrimitiveBoolean: (ctx, node) => ctx.input(node, "value"),
    PrimitiveString: (ctx, node) => ctx.input(node, "value"),
    PrimitiveStringMultiline: (ctx, node) => ctx.input(node, "value"),
    SECoursesResolutionSync: async (ctx, node, slot) => asNumber(await ctx.input(node, slot === 0 ? "width" : "height")),
    SECoursesBatchDuration: async (ctx, node) => asNumber(await ctx.input(node, "default_duration_seconds")),
    SECoursesInitAudio: async (ctx, node, slot) => {
        let audio = await audioDescriptor(await ctx.input(node, "audio"));
        if (audio) {
            const duration = initAudioTrimmedDuration(
                audio.duration,
                asNumber(await ctx.input(node, "trim_start")),
                asNumber(await ctx.input(node, "trim_end")),
            );
            audio = { ...audio, duration };
        }
        if (slot === 0) return audio;
        const mode = await ctx.input(node, "duration_mode");
        if (audio && (mode == null || mode === "match init audio length")) return audio.duration; // null = unknown
        return asNumber(await ctx.input(node, "duration_seconds"));
    },
    SECoursesMiniMaxH3AudioFrames: async (ctx, node) => {
        const audio = await ctx.input(node, "init_audio");
        const match = asBool(await ctx.input(node, "match_audio"));
        if (audio && audio.duration != null && match !== false) return H3.framesForSeconds(audio.duration);
        if (audio && match !== false) return null;
        return H3.alignFrames(asNumber(await ctx.input(node, "fallback_frames")) ?? 124);
    },
    ComfyMathExpression: async (ctx, node, slot) => {
        const values = {};
        for (const name of Object.keys(node.inputs)) {
            if (!name.startsWith("values.")) continue;
            const value = asNumber(await ctx.input(node, name));
            if (value == null) return null;
            values[name.slice("values.".length)] = value;
        }
        const result = evaluateMathExpression(await ctx.input(node, "expression"), values);
        if (result == null) return null;
        return slot === 1 ? Math.trunc(result) : slot === 2 ? result !== 0 : result;
    },
    ComfySwitchNode: async (ctx, node) => {
        const on = asBool(await ctx.input(node, "switch"));
        if (on == null) return undefined;
        return ctx.input(node, on ? "on_true" : "on_false");
    },
    ComfySoftSwitchNode: async (ctx, node) => {
        const on = asBool(await ctx.input(node, "switch"));
        if (on == null) return undefined;
        return ctx.input(node, on ? "on_true" : "on_false");
    },
    LoadImage: async (ctx, node, slot) => (slot === 0 ? imageDescriptor(await ctx.input(node, "image")) : undefined),
    SECoursesLoadImage: async (ctx, node, slot) => (slot === 0 ? imageDescriptor(await ctx.input(node, "image")) : undefined),
    SECoursesOptionalImage: async (ctx, node) => imageDescriptor(await ctx.input(node, "image")),
    SECoursesBatchContinuationFrame: async (ctx, node) => {
        // Folder-batch continuation frames are unknowable ahead of time; normal runs
        // pass the optional init image through as the Auto adapter's starting frame.
        const pack = await ctx.input(node, "references");
        if (pack?.batchActive) return null;
        return (await ctx.input(node, "init_image")) ?? null;
    },
    LoadAudio: async (ctx, node) => audioDescriptor(await ctx.input(node, "audio")),
    SECoursesTrimAudio: async (ctx, node) => {
        const audio = await ctx.input(node, "audio");
        const max = asNumber(await ctx.input(node, "max_seconds"));
        if (!audio) return audio;
        return { ...audio, duration: audio.duration == null || max == null ? audio.duration : Math.min(audio.duration, max) };
    },
    LoadVideo: async (ctx, node) => videoDescriptor(await ctx.input(node, "file")),
    GetVideoComponents: async (ctx, node, slot) => {
        const video = await ctx.input(node, "video");
        if (!video) return null;
        if (slot === 0) return video;
        if (slot === 1) return video.hasAudio ? { kind: "audio", file: video.file, duration: video.duration } : null;
        return null;
    },
    GetImageSize: async (ctx, node, slot) => {
        const image = await ctx.input(node, "image");
        if (!image) return null;
        return slot === 0 ? image.width : slot === 1 ? image.height : 1;
    },
    SECoursesMiniMaxH3ReferenceMode: async (ctx, node, slot) => {
        const pack = await ctx.input(node, "references");
        if (!pack) return undefined;
        const hasRefs = pack.images.length + pack.videos.length + pack.audios.length > 0;
        // slot 0 = has_references, slot 1 = auto_route (references or folder-batch item)
        return slot === 1 ? hasRefs || pack.batchActive === true : hasRefs;
    },
    [GALLERY_CLASS]: async (ctx, node, slot) => {
        const manifest = parseManifest(await ctx.input(node, "references"));
        const prompt = String((await ctx.input(node, "prompt")) ?? "");
        const batchFolder = String((await ctx.input(node, "batch_folder")) ?? "").trim();
        if (slot === 1) return prompt;
        if (slot === 2) return batchFolder !== "";
        if (slot === 3) return asBool(await ctx.input(node, "merge_batch_videos")) === true && batchFolder !== "";
        if (slot === 4) return asBool(await ctx.input(node, "continue_batch_with_last_frame")) === true && batchFolder !== "";
        const describe = async (entries, kind) => Promise.all(entries.map(async (entry) => {
            const info = await mediaInfo(entry.file);
            const base = { name: entry.name || entry.file, trimStart: entry.trim_start ?? null, trimEnd: entry.trim_end ?? null };
            if (kind === "image") return { ...base, width: info?.width ?? null, height: info?.height ?? null };
            if (kind === "video") {
                return { ...base, width: info?.width ?? null, height: info?.height ?? null, duration: info?.duration ?? null,
                    hasAudio: info ? info.has_audio !== false : true };
            }
            return { ...base, duration: info?.duration ?? null };
        }));
        return {
            kind: "refpack",
            prompt,
            batchActive: batchFolder !== "",
            videoFps: asNumber(await ctx.input(node, "video_fps")) ?? 24,
            maxSeconds: asNumber(await ctx.input(node, "max_seconds")) ?? 15,
            images: await describe(manifest.images, "image"),
            videos: await describe(manifest.videos, "video"),
            audios: await describe(manifest.audios, "audio"),
        };
    },
};

const SWITCH_CLASSES = new Set(["ComfySwitchNode", "ComfySoftSwitchNode"]);

/** Memoized evaluator over one flat prompt. */
export class PromptResolver {
    constructor(output) {
        this.output = output;
        this.memo = new Map();
        this.consumers = new Map();
        for (const [id, node] of Object.entries(output)) {
            for (const value of Object.values(node.inputs || {})) {
                if (!isLink(value)) continue;
                if (!this.consumers.has(value[0])) this.consumers.set(value[0], []);
                this.consumers.get(value[0]).push(id);
            }
        }
    }

    /** Resolves an input of a node: literal widget value or the upstream output it links to. */
    input(node, name) {
        const value = node.inputs?.[name];
        if (isLink(value)) return this.resolve(value[0], value[1]);
        if (value && typeof value === "object" && "__value__" in value) return value.__value__;
        return value;
    }

    /** Value of output `slot` of node `id`; `undefined` when the node is not understood. */
    resolve(id, slot) {
        const key = `${id}#${slot}`;
        if (this.memo.has(key)) return this.memo.get(key);
        const node = this.output[id];
        const evaluator = node && NODE_EVALUATORS[node.class_type];
        const promise = (async () => {
            if (!evaluator) return undefined;
            try {
                return await evaluator(this, node, Number(slot));
            } catch (error) {
                return undefined;
            }
        })();
        this.memo.set(key, promise);
        return promise;
    }

    /** Ids of nodes that will execute: walked back from the output nodes, following only the taken switch branch. */
    async activeNodes() {
        const isOutput = (id) => {
            const type = this.output[id]?.class_type;
            const registered = globalThis.LiteGraph?.registered_node_types?.[type];
            return Boolean(registered?.nodeData?.output_node);
        };
        let roots = Object.keys(this.output).filter(isOutput);
        if (!roots.length) roots = Object.keys(this.output).filter((id) => !this.consumers.has(id));
        const active = new Set();
        const stack = [...roots];
        while (stack.length) {
            const id = stack.pop();
            if (active.has(id)) continue;
            const node = this.output[id];
            if (!node) continue;
            active.add(id);
            const inputs = node.inputs || {};
            let names = Object.keys(inputs);
            if (SWITCH_CLASSES.has(node.class_type)) {
                const on = asBool(await this.input(node, "switch"));
                if (on != null) names = ["switch", on ? "on_true" : "on_false"];
            }
            for (const name of names) {
                const value = inputs[name];
                if (isLink(value)) stack.push(value[0]);
            }
        }
        return active;
    }

    /** Ids reachable downstream from `id` (through any links). */
    downstream(id) {
        const seen = new Set();
        const stack = [id];
        while (stack.length) {
            const current = stack.pop();
            for (const consumer of this.consumers.get(current) || []) {
                if (!seen.has(consumer)) {
                    seen.add(consumer);
                    stack.push(consumer);
                }
            }
        }
        return seen;
    }
}

// ==================== Model estimators (per conditioning node class) ====================

/** Applies the gallery adapter's roster rule: within the cap everything attaches, above it only mentions. */
export function selectMentioned(prompt, entries, cap, kind) {
    if (entries.length <= cap) return entries;
    const aliases = { image: ["image", "img", "picture", "pic"], video: ["video", "vid"], audio: ["audio", "aud", "sound"] }[kind];
    const regex = /(?<![0-9A-Za-z_@])@(image|img|picture|pic|video|vid|audio|aud|sound)#?(\d{1,2})(?![0-9A-Za-z])/gi;
    const mentioned = [];
    let match;
    while ((match = regex.exec(prompt || "")) !== null) {
        if (!aliases.includes(match[1].toLowerCase())) continue;
        const n = parseInt(match[2], 10);
        if (n >= 1 && n <= entries.length && !mentioned.includes(n)) mentioned.push(n);
    }
    return mentioned.slice(0, cap).map((n) => entries[n - 1]);
}

/** Text prompt / keyframes / init audio guides added by nodes downstream of a conditioning output. */
async function downstreamConditioningExtras(resolver, id, active) {
    const extras = { audioGuide: false, keyframeImages: 0, audioKeyframes: 0 };
    const seen = new Set();
    let frontier = [id];
    while (frontier.length) {
        const next = [];
        for (const current of frontier) {
            for (const consumer of resolver.consumers.get(current) || []) {
                if (seen.has(consumer) || !active.has(consumer)) continue;
                seen.add(consumer);
                const node = resolver.output[consumer];
                const feeds = Object.values(node.inputs || {}).some((v) => isLink(v) && v[0] === current && Number(v[1]) === 0);
                if (!feeds) continue;
                if (node.class_type === "SECoursesMiniMaxH3InitAudio") {
                    const audio = await resolver.input(node, "init_audio");
                    const mode = (await resolver.input(node, "audio_conditioning")) || "lock soundtrack + guide";
                    if (audio && mode !== "lock soundtrack only") extras.audioGuide = true;
                    next.push(consumer);
                } else if (node.class_type === "MiniMaxH3AddGuide") {
                    if (await resolver.input(node, "image")) extras.keyframeImages += 1;
                    if (await resolver.input(node, "audio")) extras.audioKeyframes += 1;
                    next.push(consumer);
                }
            }
        }
        frontier = next;
    }
    return extras;
}

async function h3Canvas(resolver, node) {
    const width = asNumber(await resolver.input(node, "width"));
    const height = asNumber(await resolver.input(node, "height"));
    const length = asNumber(await resolver.input(node, "length"));
    return { width, height, length, complete: width != null && height != null && length != null };
}

async function h3CoreImageToVideo(resolver, node, active, id) {
    const canvas = await h3Canvas(resolver, node);
    const prompt = await resolver.input(node, "prompt");
    let keyframes = 0;
    if (await resolver.input(node, "first_frame")) keyframes += 1;
    if (await resolver.input(node, "last_frame")) keyframes += 1;
    const extras = await downstreamConditioningExtras(resolver, id, active);
    return {
        ...canvas, approximate: typeof prompt !== "string",
        spec: {
            width: canvas.width, height: canvas.height, frames: canvas.length, prompt: typeof prompt === "string" ? prompt : "",
            keyframeImages: keyframes + extras.keyframeImages, audioGuide: extras.audioGuide || extras.audioKeyframes > 0,
            pipeline: "core",
        },
        label: keyframes ? "image to video" : "text to video",
    };
}

async function h3CoreReferenceToVideo(resolver, node, active, id) {
    const canvas = await h3Canvas(resolver, node);
    const prompt = await resolver.input(node, "prompt");
    const refImages = [], refVideos = [], refAudios = [];
    let approximate = typeof prompt !== "string";
    for (const name of Object.keys(node.inputs || {})) {
        if (name.startsWith("ref_images.")) {
            const image = await resolver.input(node, name);
            if (image === null) continue;
            refImages.push(image && image.kind === "image" ? image : {});
            if (!image) approximate = true;
        } else if (name.startsWith("ref_videos.")) {
            const video = await resolver.input(node, name);
            if (video === null) continue;
            const index = name.slice(name.lastIndexOf("_") + 1);
            const soundtrack = await resolver.input(node, `ref_video_audios.ref_video_audio_${index}`);
            refVideos.push({ ...(video && video.kind === "video" ? video : {}), hasAudio: Boolean(soundtrack) });
            if (!video) approximate = true;
        } else if (name.startsWith("ref_audios.")) {
            const audio = await resolver.input(node, name);
            if (audio === null) continue;
            refAudios.push(audio && audio.kind === "audio" ? audio : {});
            if (!audio) approximate = true;
        }
    }
    const extras = await downstreamConditioningExtras(resolver, id, active);
    return {
        ...canvas, approximate,
        spec: {
            width: canvas.width, height: canvas.height, frames: canvas.length, prompt: typeof prompt === "string" ? prompt : "",
            refImages, refVideos, refAudios, refImageSize: (await resolver.input(node, "ref_image_size")) || "match",
            keyframeImages: extras.keyframeImages, audioGuide: extras.audioGuide || extras.audioKeyframes > 0, pipeline: "core",
        },
        label: "reference to video",
    };
}

async function h3GalleryAdapter(resolver, node, active, id, mode) {
    const canvas = await h3Canvas(resolver, node);
    const pack = await resolver.input(node, "references");
    const override = await resolver.input(node, "prompt_override");
    let prompt = pack?.prompt ?? "";
    if (typeof override === "string" && override.trim()) prompt = override;
    const extras = await downstreamConditioningExtras(resolver, id, active);
    const continuation = mode === "text" ? await resolver.input(node, "first_frame") : await resolver.input(node, "continuation_frame");
    const hasRefs = Boolean(pack && pack.images.length + pack.videos.length + pack.audios.length > 0);
    const useRefs = mode === "refs" || (mode === "auto" && hasRefs);
    let spec;
    let label;
    if (useRefs) {
        const audioOnly = asBool(await resolver.input(node, "audio_only_mode")) === true;
        const images = selectMentioned(prompt, pack?.images ?? [], H3.MAX_IMAGES, "image");
        const videos = selectMentioned(prompt, pack?.videos ?? [], H3.MAX_VIDEOS, "video");
        const audios = selectMentioned(prompt, pack?.audios ?? [], H3.MAX_AUDIOS, "audio");
        spec = {
            width: canvas.width, height: canvas.height, frames: canvas.length, prompt,
            refImages: continuation ? [...images, { width: canvas.width, height: canvas.height }].slice(0, H3.MAX_IMAGES) : images,
            refVideos: videos, refAudios: audios,
            refImageSize: (await resolver.input(node, "ref_image_size")) || "match",
            maxSeconds: pack?.maxSeconds ?? 15, audioOnly,
            keyframeImages: extras.keyframeImages, audioGuide: extras.audioGuide || extras.audioKeyframes > 0, pipeline: "secourses",
        };
        label = audioOnly ? "audio only with references" : "reference to video";
    } else {
        spec = {
            width: canvas.width, height: canvas.height, frames: canvas.length, prompt,
            keyframeImages: (continuation ? 1 : 0) + extras.keyframeImages,
            audioGuide: extras.audioGuide || extras.audioKeyframes > 0, pipeline: "secourses",
        };
        label = continuation ? "image to video" : "text to video";
    }
    return { ...canvas, approximate: !pack, spec, label };
}

async function h3EmptyLatent(resolver, node, active, id) {
    const canvas = await h3Canvas(resolver, node);
    return {
        ...canvas, approximate: true,
        spec: { width: canvas.width, height: canvas.height, frames: canvas.length, prompt: "", pipeline: "core" },
        label: "text to video",
    };
}

/** conditioning node class -> async (resolver, node, activeSet, id) => {width, height, length, complete, approximate, spec, label} */
export const CONDITIONING_ESTIMATORS = {
    MiniMaxH3ImageToVideo: h3CoreImageToVideo,
    MiniMaxH3ReferenceToVideo: h3CoreReferenceToVideo,
    EmptyMiniMaxH3LatentAV: h3EmptyLatent,
    SECoursesMiniMaxH3References: (r, n, a, id) => h3GalleryAdapter(r, n, a, id, "refs"),
    SECoursesMiniMaxH3TextOnly: (r, n, a, id) => h3GalleryAdapter(r, n, a, id, "text"),
    SECoursesMiniMaxH3Auto: (r, n, a, id) => h3GalleryAdapter(r, n, a, id, "auto"),
};

const H3_DEFAULTS = { width: 1344, height: 768, length: 124 };

/**
 * Estimates the token count of the generation the gallery node `galleryNodeId` feeds, from a
 * flat API prompt. Returns {estimate, label, candidates, reason} (estimate null when nothing
 * could be resolved).
 */
export async function estimateFromPrompt(output, galleryNodeId) {
    output = output || {};
    const resolver = new PromptResolver(output);
    const nodeId = String(galleryNodeId);
    let galleryId = output[nodeId]?.class_type === GALLERY_CLASS ? nodeId
        : Object.keys(output).find((id) => output[id].class_type === GALLERY_CLASS && id.endsWith(`:${nodeId}`));
    if (!galleryId) return { estimate: null, reason: "gallery is not part of the executable graph" };

    const active = await resolver.activeNodes();
    const downstream = resolver.downstream(galleryId);
    let candidateIds = Object.keys(output).filter((id) => CONDITIONING_ESTIMATORS[output[id].class_type] && active.has(id) && downstream.has(id));
    if (!candidateIds.length) {
        candidateIds = Object.keys(output).filter((id) => CONDITIONING_ESTIMATORS[output[id].class_type] && active.has(id));
    }
    if (!candidateIds.length) return { estimate: null, reason: "no MiniMax H3 conditioning node is on the active route" };

    const results = [];
    for (const id of candidateIds) {
        const node = output[id];
        try {
            const result = await CONDITIONING_ESTIMATORS[node.class_type](resolver, node, active, id);
            const spec = { ...result.spec };
            let approximate = Boolean(result.approximate);
            if (spec.width == null) { spec.width = H3_DEFAULTS.width; approximate = true; }
            if (spec.height == null) { spec.height = H3_DEFAULTS.height; approximate = true; }
            if (spec.frames == null) { spec.frames = H3_DEFAULTS.length; approximate = true; }
            const estimate = H3.estimate(spec);
            estimate.approximate = estimate.approximate || approximate;
            results.push({ id, node, complete: result.complete, estimate, label: result.label });
        } catch (error) {
            // one unreadable candidate must not hide the others
        }
    }
    if (!results.length) return { estimate: null, reason: "could not resolve the conditioning inputs" };
    const complete = results.filter((r) => r.complete);
    const pool = complete.length ? complete : results;
    pool.sort((a, b) => b.estimate.total - a.estimate.total);
    const best = pool[0];
    // A 32x32 canvas is the audio-only trick: the video stream is disposable, only the soundtrack matters.
    const label = best.estimate.width <= 32 && best.estimate.height <= 32 && !/audio only/.test(best.label) ? "audio only" : best.label;
    return { estimate: best.estimate, label, candidates: results.length, reason: null };
}
