/**
 * Live "current / budget" token meter for the SECourses Reference Gallery (ComfyUI glue).
 *
 * `TokenMeter` is a generic label + bar + tooltip. `GalleryTokenEstimator` re-runs the graph
 * estimate (see secourses_token_graph.mjs) whenever the workflow changes, feeding it the flat
 * prompt from `app.graphToPrompt()` and media metadata from the FoleyExtension server route
 * `/secourses/media_info` (batched and cached per file).
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import "./minimax_h3_tokens.js";
import { estimateFromPrompt, setMediaInfoProvider } from "./secourses_token_graph.mjs";

const H3 = globalThis.SECoursesMiniMaxH3Tokens;
const REFRESH_DELAY_MS = 250;

// ==================== Media metadata (server side, cached) ====================

const mediaInfoCache = new Map();
let mediaInfoQueue = null;

/** Dimensions / duration of an input-directory file via /secourses/media_info (batched, cached). */
function mediaInfo(file) {
    if (!file) return Promise.resolve(null);
    let cached = mediaInfoCache.get(file);
    if (cached) return cached;
    if (!mediaInfoQueue) {
        mediaInfoQueue = { files: new Set(), promise: null };
        mediaInfoQueue.promise = new Promise((resolve) => setTimeout(resolve, 0)).then(async () => {
            const batch = mediaInfoQueue;
            mediaInfoQueue = null;
            try {
                const response = await api.fetchApi("/secourses/media_info", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ files: [...batch.files] }),
                });
                return response.status === 200 ? await response.json() : {};
            } catch (error) {
                return {};
            }
        });
    }
    const queue = mediaInfoQueue;
    queue.files.add(file);
    cached = queue.promise.then((result) => {
        const info = result?.[file];
        if (!info || info.error) {
            mediaInfoCache.delete(file); // retry next time (eg the upload finished later)
            return null;
        }
        return info;
    });
    mediaInfoCache.set(file, cached);
    return cached;
}

setMediaInfoProvider(mediaInfo);

/** Estimates the token count of the generation this gallery node feeds. */
export async function estimateForGallery(galleryNode) {
    const prompt = await app.graphToPrompt();
    return estimateFromPrompt(prompt?.output || {}, galleryNode.id);
}

// ==================== Meter UI ====================

export class TokenMeter {
    constructor(prefix = "secourses-tokenmeter") {
        this.prefix = prefix;
        this.root = document.createElement("div");
        this.root.className = prefix;
        this.root.setAttribute("role", "status");
        this.bar = document.createElement("span");
        this.bar.className = `${prefix}-bar`;
        this.text = document.createElement("span");
        this.text.className = `${prefix}-text`;
        this.detail = document.createElement("span");
        this.detail.className = `${prefix}-detail`;
        this.root.append(this.bar, this.text, this.detail);
        this.setUnavailable("Tokens: …");
    }

    get element() {
        return this.root;
    }

    setUnavailable(message) {
        this.root.dataset.level = "none";
        this.bar.style.width = "0%";
        this.text.textContent = message;
        this.detail.textContent = "";
        this.root.title = "Estimated packed-sequence tokens (text + references + audio + video) of the generation this prompt feeds.";
    }

    setEstimate(estimate, label = "") {
        if (!estimate) {
            this.setUnavailable("Tokens: n/a");
            return;
        }
        const ratio = estimate.total / estimate.budget;
        const percent = Math.round(ratio * 100);
        this.root.dataset.level = estimate.total > estimate.sageLimit ? "critical" : ratio > 1 ? "over" : ratio > 0.75 ? "warn" : "ok";
        this.bar.style.width = `${Math.max(0, Math.min(100, ratio * 100))}%`;
        const approx = estimate.approximate ? "~" : "≈";
        this.text.textContent = `Tokens ${approx}${H3.formatTokens(estimate.total)} / ${H3.formatTokens(estimate.budget)} (${percent}%)`;
        this.detail.textContent = `${estimate.width}×${estimate.height} · ${estimate.frames}f · ${estimate.seconds.toFixed(1)}s${label ? ` · ${label}` : ""}`;
        this.root.title = [`Estimated MiniMax H3 packed sequence: ${estimate.total.toLocaleString()} tokens`, ...H3.describe(estimate)].join("\n");
    }
}

/** Debounced graph-driven estimator bound to one gallery node. */
export class GalleryTokenEstimator {
    constructor(node, meter) {
        this.node = node;
        this.meter = meter;
        this.timer = null;
        this.running = false;
        this.pending = false;
        this.disposed = false;
        this.onGraphChanged = () => this.schedule();
        api.addEventListener("graphChanged", this.onGraphChanged);
        this.schedule(50);
    }

    dispose() {
        this.disposed = true;
        api.removeEventListener("graphChanged", this.onGraphChanged);
        if (this.timer) clearTimeout(this.timer);
    }

    schedule(delay = REFRESH_DELAY_MS) {
        if (this.disposed) return;
        if (this.timer) clearTimeout(this.timer);
        this.timer = setTimeout(() => {
            this.timer = null;
            this.refresh();
        }, delay);
    }

    async refresh() {
        if (this.disposed) return;
        if (this.running) {
            this.pending = true;
            return;
        }
        this.running = true;
        try {
            if (!this.node.graph) {
                return; // removed from the graph
            }
            const result = await estimateForGallery(this.node);
            if (this.disposed) return;
            if (result.estimate) {
                this.meter.setEstimate(result.estimate, result.label);
            } else {
                this.meter.setUnavailable("Tokens: n/a");
                this.meter.element.title = `Token estimate unavailable: ${result.reason || "unknown"}`;
            }
        } catch (error) {
            this.meter.setUnavailable("Tokens: n/a");
            this.meter.element.title = `Token estimate failed: ${error?.message || error}`;
        } finally {
            this.running = false;
            if (this.pending) {
                this.pending = false;
                this.schedule();
            }
        }
    }
}
