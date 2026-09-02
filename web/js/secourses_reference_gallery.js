/**
 * SECourses Reference Gallery widget: SwarmUI-style media references for ComfyUI.
 *
 * Renders an "Add references" gallery with thumbnail cards, colored @image1 /
 * @video1 / @audio1 tokens, a prompt box with colored token pills, and an '@'
 * autocomplete popover, all inside the SECoursesReferenceGallery node. Files
 * upload through ComfyUI's native /upload/image endpoint into the input
 * directory; the ordered manifest is stored in the node's hidden "references"
 * widget so the Python side can load every file.
 *
 * The optional "Load + trim" loader previews one video or audio file, lets the
 * user drag a start/end window on a timeline, and then adds the file to the
 * references with `trim_start` / `trim_end` seconds in its manifest entry. The
 * file itself is never re-encoded; the Python side decodes only the selected
 * window at generation time. Entries without trim fields behave exactly as
 * before, so existing workflows and presets are unaffected.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import {
    buildSequentialBatchPlan,
    createBatchRunId,
    folderBatchTargets,
    injectNextSequentialBatchItem,
    promptChainTargets,
    supportsSequentialFolderQueue,
} from "./secourses_folder_batch_queue.mjs";
import { reconcilePortableComboPaths } from "./secourses_portable_combo_paths.mjs";
import { resolveConfiguredPrompt } from "./secourses_reference_gallery_state.mjs";
import { GalleryTokenEstimator, TokenMeter } from "./secourses_token_meter.mjs";

const NODE_CLASS = "SECoursesReferenceGallery";
const AUDIO_RESULT_CLASS = "SECoursesBatchAudioSaveMerge";
const VIDEO_RESULT_CLASS = "SECoursesBatchVideoSaveMerge";
const QUEUE_ONLY_INPUTS = ["batch_run_id", "batch_item_index", "batch_item_count"];
const UPLOAD_SUBFOLDER = "reference_gallery";
const REORDER_MIME = "application/x-secourses-reference";
const PROMPT_CHAIN_MAX = 1000;

let sequentialQueueState = null;
let sequentialQueueInstalled = false;

/** Per-type '@' aliases, upload limits, and pill palettes. Uploads accept a
 * roster larger than the model's per-run caps (9 images, 3 videos, 3 audios):
 * the backend attaches only the files each prompt mentions (first-mention
 * order, capped at the model limit), so folder-batch prompts can address eg
 * '@image12', '@video4', or '@audio5'. */
const REFERENCE_TYPES = {
    image: {
        aliases: ["image", "img", "picture", "pic"],
        label: "Picture",
        max: 99,
        colors: ["#4dabf7", "#ffa94d", "#69db7c", "#f783ac", "#b197fc", "#ffd43b", "#3bc9db", "#ff8787", "#a9e34b"],
        stateKey: "images",
    },
    video: {
        aliases: ["video", "vid"],
        label: "Video",
        max: 99,
        colors: ["#ff6b6b", "#748ffc", "#38d9a9"],
        stateKey: "videos",
    },
    audio: {
        aliases: ["audio", "aud", "sound"],
        label: "Audio",
        max: 99,
        colors: ["#fcc419", "#da77f2", "#66d9e8"],
        stateKey: "audios",
    },
};

/**
 * Every computed property that can move a glyph or a soft-wrap point. The pill
 * overlay copies them off the textarea so its text lands exactly on the
 * (transparent) text the caret is placed in.
 */
const OVERLAY_METRIC_PROPERTIES = [
    "direction", "fontFamily", "fontFeatureSettings", "fontKerning", "fontOpticalSizing",
    "fontSize", "fontSizeAdjust", "fontStretch", "fontStyle", "fontVariantCaps",
    "fontVariantEastAsian", "fontVariantLigatures", "fontVariantNumeric", "fontWeight",
    "hyphens", "letterSpacing", "lineHeight", "overflowWrap", "paddingBottom", "paddingLeft",
    "paddingRight", "paddingTop", "tabSize", "textAlign", "textIndent", "textOrientation",
    "textRendering", "textTransform", "unicodeBidi", "whiteSpace", "wordBreak", "wordSpacing",
    "writingMode",
];

const ALIAS_TO_TYPE = {};
for (const type in REFERENCE_TYPES) {
    for (const alias of REFERENCE_TYPES[type].aliases) {
        ALIAS_TO_TYPE[alias] = type;
    }
}

const TYPE_ORDER = ["image", "video", "audio"];

function chainCallback(object, property, callback) {
    const original = object[property];
    object[property] = function () {
        original?.apply(this, arguments);
        callback.apply(this, arguments);
    };
}

function colorFor(type, n) {
    const colors = REFERENCE_TYPES[type].colors;
    return colors[(n - 1) % colors.length];
}

function tokenFor(type, n) {
    return `@${type}${n}`;
}

function escapeHtml(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#039;");
}

function mediaKind(file) {
    if (file.type) {
        if (file.type.startsWith("image/")) return "image";
        if (file.type.startsWith("video/")) return "video";
        if (file.type.startsWith("audio/")) return "audio";
    }
    const extension = file.name.toLowerCase().split(".").pop();
    if (["png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff", "avif"].includes(extension)) return "image";
    if (["mp4", "webm", "mov", "m4v", "mkv", "avi"].includes(extension)) return "video";
    if (["wav", "mp3", "aac", "ogg", "flac", "m4a", "opus"].includes(extension)) return "audio";
    return null;
}

function referenceStorageName(filename) {
    const words = new Uint32Array(4);
    if (globalThis.crypto?.getRandomValues) {
        globalThis.crypto.getRandomValues(words);
    } else {
        words[0] = Date.now();
        words[1] = Math.random() * 0xffffffff;
    }
    const id = Array.from(words, (word) => word.toString(16).padStart(8, "0")).join("");
    const extension = /\.([a-zA-Z0-9]{1,10})$/.exec(String(filename))?.[1]?.toLowerCase();
    return `reference-${id}${extension ? `.${extension}` : ""}`;
}

function notify(severity, summary, detail) {
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({ severity, summary, detail, life: 6000 });
    } else {
        console.warn(`[SECoursesReferenceGallery] ${summary}: ${detail}`);
        if (severity === "error") alert(`${summary}\n${detail}`);
    }
}

async function inspectFolderBatch(batchFolder) {
    const response = await api.fetchApi("/secourses/folder_batch/inspect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_folder: batchFolder }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.error || `Folder batch inspection failed (${response.status}).`);
    }
    return payload;
}

function installSequentialFolderQueue() {
    if (sequentialQueueInstalled) return;
    sequentialQueueInstalled = true;

    const originalGraphToPrompt = app.graphToPrompt.bind(app);
    const originalQueuePrompt = app.queuePrompt.bind(app);
    const originalApiQueuePrompt = api.queuePrompt.bind(api);

    // Inject at the final API boundary, which is called exactly once per queued
    // job. Graph serialization is also used by token meters and previews, so
    // consuming the plan there can skip items while a batch is being queued.
    api.queuePrompt = async function (number, prompt, options) {
        injectNextSequentialBatchItem(prompt, sequentialQueueState);
        return originalApiQueuePrompt(number, prompt, options);
    };

    app.queuePrompt = async function (number, batchCount = 1, queueNodeIds) {
        if (sequentialQueueState || queueNodeIds?.length) {
            return originalQueuePrompt(number, batchCount, queueNodeIds);
        }

        const snapshot = await originalGraphToPrompt();
        const folderTargets = folderBatchTargets(snapshot.output);
        const chainTargets = promptChainTargets(snapshot.output);
        if ((!folderTargets.length && !chainTargets.length) || !supportsSequentialFolderQueue(snapshot.output)) {
            return originalQueuePrompt(number, batchCount, queueNodeIds);
        }

        if (folderTargets.length && chainTargets.length) {
            notify(
                "error",
                "Prompt queue stopped",
                "Use either Folder batch or added prompt boxes, not both at once.",
            );
            return false;
        }

        try {
            let promptCount;
            let targetNodeIds;
            if (folderTargets.length) {
                const folders = [...new Set(folderTargets.map((target) => target.batchFolder))];
                if (folders.length !== 1) {
                    throw new Error("A workflow can sequentially process only one folder path at a time.");
                }
                promptCount = (await inspectFolderBatch(folders[0])).count;
                targetNodeIds = folderTargets.map((target) => target.nodeId);
            } else {
                const counts = [...new Set(chainTargets.map((target) => target.prompts.length))];
                if (counts.length !== 1) {
                    throw new Error("Every connected prompt chain must contain the same number of prompts.");
                }
                for (const target of chainTargets) {
                    const emptyIndex = target.prompts.findIndex(
                        (prompt) => typeof prompt !== "string" || !prompt.trim(),
                    );
                    if (emptyIndex !== -1) {
                        throw new Error(`Prompt ${emptyIndex + 1} is empty. Fill it or remove it.`);
                    }
                }
                promptCount = counts[0];
                targetNodeIds = chainTargets.map((target) => target.nodeId);
            }
            const plan = buildSequentialBatchPlan(
                promptCount,
                Number(batchCount ?? 1),
                createBatchRunId,
            );
            sequentialQueueState = {
                cursor: 0,
                plan,
                targetNodeIds,
            };
            // A normal append queue gives every item a monotonically increasing
            // server priority. This guarantees item 1 saves before item 2 starts.
            return await originalQueuePrompt(0, plan.length, queueNodeIds);
        } catch (error) {
            notify("error", "Prompt queue stopped", String(error?.message || error));
            return false;
        } finally {
            sequentialQueueState = null;
        }
    };
}

function viewURL(annotated) {
    const match = /^(.*?) \[(input|output|temp)\]$/.exec(annotated);
    let name = annotated, type = "input", subfolder = "";
    if (match) {
        name = match[1];
        type = match[2];
    }
    const slash = name.lastIndexOf("/");
    if (slash !== -1) {
        subfolder = name.substring(0, slash);
        name = name.substring(slash + 1);
    }
    return api.apiURL(`/view?filename=${encodeURIComponent(name)}&type=${type}&subfolder=${encodeURIComponent(subfolder)}`);
}

async function uploadReferenceFile(file) {
    const body = new FormData();
    body.append("image", file, referenceStorageName(file.name));
    body.append("type", "input");
    body.append("subfolder", UPLOAD_SUBFOLDER);
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (response.status !== 200) {
        throw new Error(`Upload failed (${response.status} ${response.statusText})`);
    }
    const data = await response.json();
    const subfolder = data.subfolder ? `${data.subfolder}/` : "";
    return { file: `${subfolder}${data.name} [input]`, name: file.name };
}

function hideWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    const element = widget.inputEl ?? widget.element;
    if (element) element.style.display = "none";
}

class ReferenceGalleryUI {
    constructor(node) {
        this.node = node;
        this.promptWidget = node.widgets?.find((w) => w.name === "prompt");
        this.manifestWidget = node.widgets?.find((w) => w.name === "references");
        this.batchFolderWidget = node.widgets?.find((w) => w.name === "batch_folder");
        this.mergeBatchWidget = node.widgets?.find((w) => w.name === "merge_batch_videos");
        this.continueLastFrameWidget = node.widgets?.find((w) => w.name === "continue_batch_with_last_frame");
        this.contextFramesWidget = node.widgets?.find((w) => w.name === "continuation_context_frames");
        this.state = { images: [], videos: [], audios: [] };
        this.hasHydratedPrompt = false;
        this.promptTouched = false;
        this.dragContext = null;
        this.promptEditors = [];
        this.activeEditor = null;
        this.buildDOM();
        hideWidget(this.promptWidget);
        hideWidget(this.manifestWidget);
        hideWidget(this.batchFolderWidget);
        hideWidget(this.mergeBatchWidget);
        hideWidget(this.continueLastFrameWidget);
        hideWidget(this.contextFramesWidget);
        const ui = this;
        this.widget = node.addDOMWidget("gallery_ui", "secourses_gallery", this.root, {
            hideOnZoom: false,
            selectOn: [],
            getValue: () => ui.manifestWidget?.value ?? "{}",
            setValue: () => {},
        });
        // Current ComfyUI checks the widget property, not options.serialize.
        this.widget.serialize = false;
        this.widget.computeSize = (width) => [width, ui.computeHeight(width ?? ui.node.size[0])];
        this.syncFromWidgets({ hydratePrompt: true });
        if (node.size[0] < 420) {
            node.setSize([420, node.size[1]]);
        }
        this.refreshLayout();
        // Live "current / budget" token estimate of the generation this gallery feeds.
        this.tokenEstimator = new GalleryTokenEstimator(
            node,
            this.tokenMeter,
            () => (this.activeEditor ?? this.promptEditors[0])?.textarea.value ?? "",
        );
    }

    dispose() {
        this.tokenEstimator?.dispose();
        this.tokenEstimator = null;
        for (const editor of this.promptEditors) {
            editor.resizeObserver?.disconnect();
            editor.resizeObserver = null;
        }
    }

    /** Re-estimates the packed token count shortly (debounced; also fired by graph changes). */
    scheduleTokenEstimate() {
        this.tokenEstimator?.schedule();
    }

    // ==================== DOM construction ====================

    buildDOM() {
        this.root = document.createElement("div");
        this.root.className = "secourses-refgal";

        this.toolbar = document.createElement("div");
        this.toolbar.className = "secourses-refgal-toolbar";
        this.addButton = document.createElement("button");
        this.addButton.type = "button";
        this.addButton.className = "secourses-refgal-add";
        this.addButton.textContent = "➕ Add references";
        this.addButton.title = "Add reference images, videos, or audio (or drag & drop / paste them)";
        this.trimToggle = document.createElement("button");
        this.trimToggle.type = "button";
        this.trimToggle.className = "secourses-refgal-add secourses-refgal-trimtoggle";
        this.trimToggle.textContent = "✂ Load + trim";
        this.trimToggle.title = "Optional loader: preview a video or audio file, pick a start/end window, then add it to the references. Leave the full range selected to add it untrimmed.";
        this.addPromptButton = document.createElement("button");
        this.addPromptButton.type = "button";
        this.addPromptButton.className = "secourses-refgal-add secourses-refgal-addprompt";
        this.addPromptButton.textContent = "+ Prompt";
        this.addPromptButton.title = "Add another prompt. Added prompts run in order and share these references and batch controls.";
        this.counter = document.createElement("span");
        this.counter.className = "secourses-refgal-counter";
        this.soundtrackHint = document.createElement("span");
        this.soundtrackHint.className = "secourses-refgal-hint";
        this.soundtrackHint.textContent = "For @video1's soundtrack, type <Audio 1>; @audio1 is the first standalone audio file";
        this.soundtrackHint.title = "A reference video's soundtrack uses the native <Audio N> label. Standalone @audioN tokens always refer to standalone audio attachments and are offset after video soundtracks automatically.";
        this.hint = document.createElement("span");
        this.hint.className = "secourses-refgal-hint";
        this.hint.textContent = "Type @ in the prompt to reference";
        this.hint.title = "Type '@' in the prompt for reference autocomplete, eg '@image1'. Click any card to insert its token.";
        this.toolbar.append(this.addButton, this.trimToggle, this.addPromptButton, this.counter, this.soundtrackHint, this.hint);

        this.cards = document.createElement("div");
        this.cards.className = "secourses-refgal-cards";
        this.empty = document.createElement("div");
        this.empty.className = "secourses-refgal-empty";
        this.empty.textContent = "No references yet — click “Add references”, drag & drop, or paste media here.";
        this.cardsWrap = document.createElement("div");
        this.cardsWrap.className = "secourses-refgal-cardswrap";
        this.cardsWrap.append(this.cards, this.empty);

        this.promptList = document.createElement("div");
        this.promptList.className = "secourses-refgal-promptlist";
        this.createPromptEditor("");

        this.batchRow = document.createElement("div");
        this.batchRow.className = "secourses-refgal-batchrow";
        const batchField = document.createElement("label");
        batchField.className = "secourses-refgal-batchfield";
        const batchLabel = document.createElement("span");
        batchLabel.className = "secourses-refgal-batchlabel";
        batchLabel.textContent = "Folder batch (optional)";
        this.batchFolderInput = document.createElement("textarea");
        this.batchFolderInput.className = "secourses-refgal-batchpath";
        this.batchFolderInput.rows = 2;
        this.batchFolderInput.wrap = "soft";
        this.batchFolderInput.spellcheck = false;
        this.batchFolderInput.placeholder = "Paste a local folder path";
        this.batchFolderInput.title = "Local folder containing UTF-8 .txt prompts. Subfolders are scanned recursively.";
        batchField.append(batchLabel, this.batchFolderInput);

        this.mergeToggle = document.createElement("label");
        this.mergeToggle.className = "secourses-refgal-mergetoggle";
        this.mergeToggle.title = "With an init video, append the generated segment to it. With Folder batch, save each prompt's MP4 and merge each prompt directory after the final job.";
        this.mergeCheckbox = document.createElement("input");
        this.mergeCheckbox.type = "checkbox";
        this.mergeCheckbox.setAttribute("role", "switch");
        const mergeTrack = document.createElement("span");
        mergeTrack.className = "secourses-refgal-mergetrack";
        mergeTrack.setAttribute("aria-hidden", "true");
        this.mergeLabel = document.createElement("span");
        this.mergeLabel.className = "secourses-refgal-mergelabel";
        this.mergeLabel.textContent = "Merge videos";
        this.mergeToggle.append(this.mergeCheckbox, mergeTrack, this.mergeLabel);
        this.batchRow.append(batchField, this.mergeToggle);

        this.continuationRow = document.createElement("div");
        this.continuationRow.className = "secourses-refgal-continuationrow";
        this.tokenMeter = new TokenMeter("secourses-refgal-tokens");
        this.continuationControls = document.createElement("div");
        this.continuationControls.className = "secourses-refgal-continuationcontrols";
        this.lastFrameToggle = document.createElement("label");
        this.lastFrameToggle.className = "secourses-refgal-mergetoggle secourses-refgal-continuationtoggle";
        this.lastFrameToggle.title = "Use the selected final frames from an init video, or from the preceding Folder batch result. One frame preserves the original behavior; 5, 22, 39, and 56 use MiniMax H3's native clip guide.";
        this.lastFrameCheckbox = document.createElement("input");
        this.lastFrameCheckbox.type = "checkbox";
        this.lastFrameCheckbox.setAttribute("role", "switch");
        this.lastFrameCheckbox.setAttribute("aria-label", "Continue From Last Video Frames");
        const lastFrameTrack = document.createElement("span");
        lastFrameTrack.className = "secourses-refgal-mergetrack";
        lastFrameTrack.setAttribute("aria-hidden", "true");
        const lastFrameLabel = document.createElement("span");
        lastFrameLabel.className = "secourses-refgal-mergelabel";
        lastFrameLabel.textContent = "Continue From Last Video Frames";
        this.lastFrameToggle.append(this.lastFrameCheckbox, lastFrameTrack, lastFrameLabel);
        this.contextFramesSelect = document.createElement("select");
        this.contextFramesSelect.className = "secourses-refgal-contextframes";
        this.contextFramesSelect.setAttribute("aria-label", "Continuation context frames");
        this.contextFramesSelect.title = "Video frames used as continuation context: 1 preserves the original behavior; 5, 22, 39, and 56 use native multi-frame guides.";
        for (const value of [1, 5, 22, 39, 56]) {
            const option = document.createElement("option");
            option.value = String(value);
            option.textContent = String(value);
            this.contextFramesSelect.appendChild(option);
        }
        this.continuationControls.append(this.lastFrameToggle, this.contextFramesSelect);
        this.continuationRow.append(this.tokenMeter.element, this.continuationControls);

        this.fileInput = document.createElement("input");
        this.fileInput.type = "file";
        this.fileInput.multiple = true;
        this.fileInput.accept = "image/*,video/*,audio/*";
        this.fileInput.hidden = true;

        this.buildTrimLoader();
        this.root.append(
            this.toolbar,
            this.cardsWrap,
            this.loader,
            this.promptList,
            this.batchRow,
            this.continuationRow,
            this.fileInput,
            this.loaderFileInput,
        );
        this.bindEvents();
    }

    createPromptEditor(value) {
        const row = document.createElement("div");
        row.className = "secourses-refgal-promptrow";
        const header = document.createElement("div");
        header.className = "secourses-refgal-promptheader";
        const label = document.createElement("span");
        label.className = "secourses-refgal-promptlabel";
        const controls = document.createElement("span");
        controls.className = "secourses-refgal-promptcontrols";
        const moveUp = document.createElement("button");
        moveUp.type = "button";
        moveUp.textContent = "↑";
        moveUp.title = "Move this prompt earlier";
        moveUp.setAttribute("aria-label", moveUp.title);
        const moveDown = document.createElement("button");
        moveDown.type = "button";
        moveDown.textContent = "↓";
        moveDown.title = "Move this prompt later";
        moveDown.setAttribute("aria-label", moveDown.title);
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.title = "Remove this prompt";
        remove.setAttribute("aria-label", remove.title);
        controls.append(moveUp, moveDown, remove);
        header.append(label, controls);

        const wrap = document.createElement("div");
        wrap.className = "secourses-refgal-promptwrap";
        const overlay = document.createElement("div");
        overlay.className = "secourses-refgal-overlay";
        overlay.setAttribute("aria-hidden", "true");
        // The pills live in an inner box that is moved with a transform to follow the
        // textarea's scroll. Unlike scrollTop, a transform never clamps, so the overlay
        // can never come to rest a line away from the text the caret is actually in.
        const overlayInner = document.createElement("div");
        overlayInner.className = "secourses-refgal-overlay-inner";
        overlay.appendChild(overlayInner);
        const textarea = document.createElement("textarea");
        textarea.className = "secourses-refgal-prompt";
        textarea.placeholder = "Prompt — type @ to reference attachments, eg @image1 …";
        textarea.spellcheck = false;
        textarea.value = value;
        const suggest = document.createElement("div");
        suggest.className = "secourses-refgal-suggest";
        suggest.hidden = true;
        wrap.append(overlay, textarea, suggest);
        row.append(header, wrap);

        const editor = {
            row, header, label, controls, moveUp, moveDown, remove,
            wrap, overlay, overlayInner, textarea, suggest,
            suggestIndex: 0,
            suggestMatches: null,
            suggestContext: null,
            resizeObserver: null,
        };
        this.promptEditors.push(editor);
        this.promptList.appendChild(row);
        this.setPrimaryEditorAliases();

        moveUp.addEventListener("click", () => this.movePromptEditor(editor, -1));
        moveDown.addEventListener("click", () => this.movePromptEditor(editor, 1));
        remove.addEventListener("click", () => this.removePromptEditor(editor));
        textarea.addEventListener("focus", () => {
            this.activeEditor = editor;
            this.updatePromptEditorLabels();
            this.scheduleTokenEstimate();
        });
        textarea.addEventListener("paste", async (event) => {
            const files = [...(event.clipboardData?.items || [])]
                .filter((item) => item.kind === "file")
                .map((item) => item.getAsFile())
                .filter((file) => file && mediaKind(file));
            if (!files.length) return;
            event.preventDefault();
            event.stopPropagation();
            await this.addFiles(files);
        });
        textarea.addEventListener("input", () => this.onPromptInput(editor));
        textarea.addEventListener("scroll", () => this.syncOverlayScroll(editor));
        textarea.addEventListener("click", () => this.updateSuggestions(editor));
        textarea.addEventListener("blur", () => window.setTimeout(() => this.closeSuggestions(editor), 150));
        textarea.addEventListener("keydown", (event) => this.onPromptKeydown(event, editor));
        suggest.addEventListener("mousedown", (event) => event.preventDefault());
        // Resizing the node re-wraps the textarea; the overlay has to be re-measured
        // against its new content width or every pill drifts off the text again.
        if (typeof ResizeObserver !== "undefined") {
            editor.resizeObserver = new ResizeObserver(() => this.syncOverlayGeometry(editor));
            editor.resizeObserver.observe(textarea);
        }
        this.updatePromptEditorLabels();
        this.renderOverlay(null, editor);
        return editor;
    }

    destroyPromptEditor(editor) {
        editor.resizeObserver?.disconnect();
        editor.resizeObserver = null;
        editor.row.remove();
    }

    setPrimaryEditorAliases() {
        const primary = this.promptEditors[0];
        this.promptWrap = primary?.wrap;
        this.overlay = primary?.overlay;
        this.textarea = primary?.textarea;
        this.suggest = primary?.suggest;
        if (!this.activeEditor || !this.promptEditors.includes(this.activeEditor)) {
            this.activeEditor = primary;
        }
    }

    updatePromptEditorLabels() {
        const chained = this.promptEditors.length > 1;
        this.promptEditors.forEach((editor, index) => {
            editor.header.hidden = !chained;
            editor.label.textContent = `Prompt ${index + 1}`;
            editor.moveUp.disabled = index === 0;
            editor.moveDown.disabled = index === this.promptEditors.length - 1;
            editor.remove.disabled = !chained;
            editor.row.classList.toggle("secourses-refgal-promptrow-active", editor === this.activeEditor);
        });
        this.updateBatchModeAvailability();
    }

    addPromptEditor() {
        if (this.promptEditors.length >= PROMPT_CHAIN_MAX || this.batchFolderInput?.value.trim()) return;
        const editor = this.createPromptEditor("");
        this.activeEditor = editor;
        this.savePromptState({ forceManifest: true });
        this.updatePromptEditorLabels();
        this.refreshLayout();
        editor.textarea.focus();
    }

    removePromptEditor(editor) {
        if (this.promptEditors.length <= 1) return;
        const index = this.promptEditors.indexOf(editor);
        if (index === -1) return;
        this.promptEditors.splice(index, 1);
        this.destroyPromptEditor(editor);
        if (this.activeEditor === editor) {
            this.activeEditor = this.promptEditors[Math.min(index, this.promptEditors.length - 1)];
        }
        this.setPrimaryEditorAliases();
        this.savePromptState({ forceManifest: true });
        this.updatePromptEditorLabels();
        this.renderAllOverlays();
        this.refreshLayout();
    }

    movePromptEditor(editor, offset) {
        const index = this.promptEditors.indexOf(editor);
        const target = index + offset;
        if (index < 0 || target < 0 || target >= this.promptEditors.length) return;
        this.promptEditors.splice(index, 1);
        this.promptEditors.splice(target, 0, editor);
        for (const item of this.promptEditors) this.promptList.appendChild(item.row);
        this.setPrimaryEditorAliases();
        this.savePromptState({ forceManifest: true });
        this.updatePromptEditorLabels();
    }

    setAdditionalPrompts(prompts) {
        for (const editor of this.promptEditors.slice(1)) this.destroyPromptEditor(editor);
        this.promptEditors.splice(1);
        for (const prompt of prompts) this.createPromptEditor(prompt);
        this.setPrimaryEditorAliases();
        this.updatePromptEditorLabels();
    }

    savePromptState({ forceManifest = false } = {}) {
        this.promptTouched = true;
        this.syncPromptToWidget();
        if (forceManifest || this.promptEditors.length > 1) this.saveManifest();
        this.node.setDirtyCanvas(true, true);
        this.scheduleTokenEstimate();
    }

    onPromptInput(editor) {
        this.activeEditor = editor;
        this.savePromptState();
        this.renderOverlay(null, editor);
        this.updateSuggestions(editor);
        this.updatePromptEditorLabels();
    }

    updateBatchModeAvailability() {
        if (!this.addPromptButton || !this.batchFolderInput) return;
        const chained = this.promptEditors.length > 1;
        const folder = Boolean(this.batchFolderInput.value.trim());
        this.addPromptButton.disabled = folder || this.promptEditors.length >= PROMPT_CHAIN_MAX;
        this.addPromptButton.title = folder
            ? "Clear Folder batch before adding prompt boxes."
            : "Add another prompt. Added prompts run in order and share these references and batch controls.";
        this.batchFolderInput.disabled = chained;
        this.batchFolderInput.title = chained
            ? "Remove the added prompt boxes before using Folder batch."
            : "Local folder containing UTF-8 .txt prompts. Subfolders are scanned recursively.";
        if (this.mergeLabel) this.updateMergeAvailability();
    }

    /** The optional "Load + trim" panel: preview one video/audio file and pick a start/end window. */
    buildTrimLoader() {
        this.loader = document.createElement("div");
        this.loader.className = "secourses-refgal-loader";
        this.loader.hidden = true;
        this.loaderMedia = null;
        this.trimPreviewActive = false;

        const head = document.createElement("div");
        head.className = "secourses-refgal-loader-head";
        const title = document.createElement("span");
        title.className = "secourses-refgal-loader-title";
        title.textContent = "✂ Trim loader";
        const headHint = document.createElement("span");
        headHint.className = "secourses-refgal-loader-hint";
        headHint.textContent = "optional — trim, then add; full range adds untrimmed";
        const close = document.createElement("button");
        close.type = "button";
        close.className = "secourses-refgal-loader-close";
        close.innerHTML = "&times;";
        close.title = "Close the trim loader";
        head.append(title, headHint, close);

        const pickRow = document.createElement("div");
        pickRow.className = "secourses-refgal-loader-pickrow";
        this.loaderPick = document.createElement("button");
        this.loaderPick.type = "button";
        this.loaderPick.className = "secourses-refgal-add";
        this.loaderPick.textContent = "🎬 Choose video / audio…";
        this.loaderPick.title = "Pick a video or audio file to preview and trim. Images never need trimming — add them with “Add references”.";
        this.loaderName = document.createElement("span");
        this.loaderName.className = "secourses-refgal-loader-file";
        this.loaderName.textContent = "No file loaded yet";
        pickRow.append(this.loaderPick, this.loaderName);

        this.loaderPreview = document.createElement("div");
        this.loaderPreview.className = "secourses-refgal-loader-preview";
        this.loaderPreview.hidden = true;

        this.trimBody = document.createElement("div");
        this.trimBody.className = "secourses-refgal-trimbody";
        this.trimBody.hidden = true;
        this.trimTrack = document.createElement("div");
        this.trimTrack.className = "secourses-refgal-trim-track";
        this.trimTrack.title = "Click to seek the preview. Drag the handles to set the trim window.";
        this.trimFill = document.createElement("div");
        this.trimFill.className = "secourses-refgal-trim-fill";
        this.trimPlayhead = document.createElement("div");
        this.trimPlayhead.className = "secourses-refgal-trim-playhead";
        this.trimStartHandle = document.createElement("div");
        this.trimStartHandle.className = "secourses-refgal-trim-handle secourses-refgal-trim-handle-start";
        this.trimStartHandle.title = "Drag to set the trim start (the preview follows)";
        this.trimEndHandle = document.createElement("div");
        this.trimEndHandle.className = "secourses-refgal-trim-handle secourses-refgal-trim-handle-end";
        this.trimEndHandle.title = "Drag to set the trim end (the preview follows)";
        this.trimTrack.append(this.trimFill, this.trimPlayhead, this.trimStartHandle, this.trimEndHandle);

        const fields = document.createElement("div");
        fields.className = "secourses-refgal-trim-fields";
        const makeTimeField = (labelText, titleText) => {
            const label = document.createElement("label");
            label.className = "secourses-refgal-trim-label";
            label.append(labelText);
            const input = document.createElement("input");
            input.type = "number";
            input.className = "secourses-refgal-trim-input";
            input.min = "0";
            input.step = "0.05";
            input.title = titleText;
            label.appendChild(input);
            fields.appendChild(label);
            return input;
        };
        this.trimStartInput = makeTimeField("Start", "Trim start in seconds");
        this.trimEndInput = makeTimeField("End", "Trim end in seconds");
        const makeToolButton = (text, titleText) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "secourses-refgal-trim-tool";
            button.textContent = text;
            button.title = titleText;
            fields.appendChild(button);
            return button;
        };
        this.trimSetStart = makeToolButton("⇤ Start", "Set the trim start to the current playback position");
        this.trimSetEnd = makeToolButton("End ⇥", "Set the trim end to the current playback position");
        this.trimPreviewBtn = makeToolButton("▶ Preview", "Play only the selected trim window");
        this.trimBadge = document.createElement("span");
        this.trimBadge.className = "secourses-refgal-trim-length";
        fields.appendChild(this.trimBadge);
        this.trimBody.append(this.trimTrack, fields);

        const actions = document.createElement("div");
        actions.className = "secourses-refgal-loader-actions";
        this.loaderAdd = document.createElement("button");
        this.loaderAdd.type = "button";
        this.loaderAdd.className = "secourses-refgal-loader-add";
        this.loaderAdd.textContent = "➕ Add to references";
        this.loaderAdd.title = "Add this file to the references. Only the selected window is used at generation time.";
        this.loaderAdd.disabled = true;
        this.loaderNote = document.createElement("span");
        this.loaderNote.className = "secourses-refgal-loader-hint";
        actions.append(this.loaderAdd, this.loaderNote);

        this.loader.append(head, pickRow, this.loaderPreview, this.trimBody, actions);

        this.loaderFileInput = document.createElement("input");
        this.loaderFileInput.type = "file";
        this.loaderFileInput.accept = "video/*,audio/*";
        this.loaderFileInput.hidden = true;

        close.addEventListener("click", () => this.toggleTrimLoader(false));
        this.loaderPick.addEventListener("click", () => this.loaderFileInput.click());
        this.loaderFileInput.addEventListener("change", async () => {
            const file = this.loaderFileInput.files?.[0];
            this.loaderFileInput.value = "";
            if (file) await this.loaderPickFile(file);
        });
        this.loaderAdd.addEventListener("click", () => this.addTrimmedReference());
        this.bindTrimHandle(this.trimStartHandle, true);
        this.bindTrimHandle(this.trimEndHandle, false);
        this.trimTrack.addEventListener("pointerdown", (event) => {
            if (event.target === this.trimStartHandle || event.target === this.trimEndHandle) return;
            const media = this.loaderMedia;
            if (!media?.duration || !media.element) return;
            event.preventDefault();
            media.element.currentTime = this.timelineTime(event);
        });
        this.trimStartInput.addEventListener("change", () => {
            const media = this.loaderMedia;
            if (!media?.duration) return;
            const value = parseFloat(this.trimStartInput.value);
            this.setTrimRange(value, media.end, { seek: value });
        });
        this.trimEndInput.addEventListener("change", () => {
            const media = this.loaderMedia;
            if (!media?.duration) return;
            const value = parseFloat(this.trimEndInput.value);
            this.setTrimRange(media.start, value, { seek: value });
        });
        for (const input of [this.trimStartInput, this.trimEndInput]) {
            input.addEventListener("keydown", (event) => {
                if (event.key === "Enter") input.blur();
                if (!(event.ctrlKey || event.metaKey)) event.stopPropagation();
            });
        }
        this.trimSetStart.addEventListener("click", () => {
            const media = this.loaderMedia;
            if (!media?.duration || !media.element) return;
            this.setTrimRange(media.element.currentTime, media.end);
        });
        this.trimSetEnd.addEventListener("click", () => {
            const media = this.loaderMedia;
            if (!media?.duration || !media.element) return;
            this.setTrimRange(media.start, media.element.currentTime);
        });
        this.trimPreviewBtn.addEventListener("click", () => {
            const media = this.loaderMedia;
            if (!media?.duration || !media.element) return;
            media.element.currentTime = media.start;
            this.trimPreviewActive = true;
            media.element.play().catch(() => {
                this.trimPreviewActive = false;
            });
        });
    }

    bindEvents() {
        this.addButton.addEventListener("click", () => this.fileInput.click());
        this.trimToggle.addEventListener("click", () => this.toggleTrimLoader());
        this.addPromptButton.addEventListener("click", () => this.addPromptEditor());
        this.batchFolderInput.addEventListener("input", () => {
            if (this.batchFolderWidget) {
                this.batchFolderWidget.value = this.batchFolderInput.value;
                this.batchFolderWidget.callback?.(this.batchFolderWidget.value);
            }
            this.updateBatchModeAvailability();
            this.node.setDirtyCanvas(true, true);
            this.scheduleTokenEstimate();
        });
        this.batchFolderInput.addEventListener("keydown", (event) => {
            if (!(event.ctrlKey || event.metaKey)) event.stopPropagation();
        });
        this.mergeCheckbox.addEventListener("change", () => {
            if (this.mergeBatchWidget) {
                this.mergeBatchWidget.value = this.mergeCheckbox.checked;
                this.mergeBatchWidget.callback?.(this.mergeBatchWidget.value);
            }
            this.node.setDirtyCanvas(true, true);
        });
        this.mergeToggle.addEventListener("click", (event) => {
            if (event.target === this.mergeCheckbox) return;
            event.preventDefault();
            this.mergeCheckbox.checked = !this.mergeCheckbox.checked;
            this.mergeCheckbox.dispatchEvent(new Event("change", { bubbles: true }));
        });
        this.lastFrameCheckbox.addEventListener("change", () => {
            if (this.continueLastFrameWidget) {
                this.continueLastFrameWidget.value = this.lastFrameCheckbox.checked;
                this.continueLastFrameWidget.callback?.(this.continueLastFrameWidget.value);
            }
            this.contextFramesSelect.disabled = !this.lastFrameCheckbox.checked;
            this.node.setDirtyCanvas(true, true);
            this.scheduleTokenEstimate();
        });
        this.lastFrameToggle.addEventListener("click", (event) => {
            if (event.target === this.lastFrameCheckbox) return;
            event.preventDefault();
            this.lastFrameCheckbox.checked = !this.lastFrameCheckbox.checked;
            this.lastFrameCheckbox.dispatchEvent(new Event("change", { bubbles: true }));
        });
        this.contextFramesSelect.addEventListener("change", () => {
            if (this.contextFramesWidget) {
                this.contextFramesWidget.value = this.contextFramesSelect.value;
                this.contextFramesWidget.callback?.(this.contextFramesWidget.value);
            }
            this.node.setDirtyCanvas(true, true);
            this.scheduleTokenEstimate();
        });
        this.fileInput.addEventListener("change", async () => {
            await this.addFiles([...this.fileInput.files]);
            this.fileInput.value = "";
        });

        for (const eventName of ["pointerdown", "pointerup", "dblclick", "contextmenu", "wheel"]) {
            this.root.addEventListener(eventName, (event) => event.stopPropagation());
        }

        this.root.addEventListener("dragover", (event) => {
            if (![...(event.dataTransfer?.items || [])].some((item) => item.kind === "file")) return;
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = "copy";
            this.root.classList.add("secourses-refgal-dragging");
        });
        this.root.addEventListener("dragleave", (event) => {
            if (!this.root.contains(event.relatedTarget)) {
                this.root.classList.remove("secourses-refgal-dragging");
            }
        });
        this.root.addEventListener("drop", async (event) => {
            this.root.classList.remove("secourses-refgal-dragging");
            const files = [...(event.dataTransfer?.files || [])].filter((file) => mediaKind(file));
            if (!files.length) return;
            event.preventDefault();
            event.stopPropagation();
            await this.addFiles(files);
        });

        // Internal card reordering (never carries files, so the file-drop
        // handlers above ignore it and vice versa).
        this.cardsWrap.addEventListener("dragover", (event) => {
            if (!this.dragContext) return;
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = "move";
            this.updateDropMarker(this.reorderInsertPos(event));
        });
        this.cardsWrap.addEventListener("drop", (event) => {
            if (!this.dragContext) return;
            event.preventDefault();
            event.stopPropagation();
            const { type, index } = this.dragContext;
            this.moveReference(type, index, this.reorderInsertPos(event));
            this.dragContext = null;
            this.clearDropMarkers();
        });

    }

    // ==================== State & widgets sync ====================

    syncFromWidgets({ hydratePrompt = false } = {}) {
        let parsed = {};
        try {
            parsed = JSON.parse(this.manifestWidget?.value || "{}");
        } catch (error) {
            parsed = {};
        }
        this.state = {
            images: Array.isArray(parsed.images) ? parsed.images.filter((e) => e && e.file) : [],
            videos: Array.isArray(parsed.videos) ? parsed.videos.filter((e) => e && e.file) : [],
            audios: Array.isArray(parsed.audios) ? parsed.audios.filter((e) => e && e.file) : [],
            prompts: Array.isArray(parsed.prompts) ? parsed.prompts.filter((prompt) => typeof prompt === "string") : [],
        };
        this.setAdditionalPrompts(this.state.prompts);
        this.batchFolderInput.value = String(this.batchFolderWidget?.value ?? "");
        const mergeValue = this.mergeBatchWidget?.value;
        this.mergeCheckbox.checked = mergeValue === true || mergeValue === "true" || mergeValue === 1;
        const continuationValue = this.continueLastFrameWidget?.value;
        this.lastFrameCheckbox.checked = continuationValue === true || continuationValue === "true" || continuationValue === 1;
        const contextValue = String(this.contextFramesWidget?.value ?? "1");
        this.contextFramesSelect.value = ["1", "5", "22", "39", "56"].includes(contextValue) ? contextValue : "1";
        this.updateBatchModeAvailability();
        this.updateMergeAvailability();
        this.updateContinuationAvailability();
        if (hydratePrompt) {
            this.textarea.value = this.promptWidget?.value ?? "";
        }
        this.render();
    }

    updateMergeAvailability() {
        const output = this.node.outputs?.find((item) => item.name === "merge_batch_videos");
        const available = Boolean(output?.links?.length);
        const targetTypes = (output?.links ?? []).map((linkId) => {
            const links = app.graph?.links;
            const link = links?.[linkId] ?? links?.get?.(linkId);
            const targetId = link?.target_id ?? link?.[3];
            return app.graph?.getNodeById?.(targetId)?.type;
        });
        const hasAudio = targetTypes.some((type) =>
            type === "SECoursesBatchAudioMerge" || type === "SECoursesBatchAudioSaveMerge"
        );
        const hasVideo = targetTypes.some((type) =>
            type === "SECoursesBatchVideoMerge" || type === VIDEO_RESULT_CLASS
        );
        const chained = this.promptEditors.length > 1;
        if (hasAudio && !hasVideo) {
            this.mergeLabel.textContent = chained ? "Merge chain audio" : "Merge audio";
            this.mergeCheckbox.setAttribute("aria-label", "Merge audio");
            this.mergeToggle.title = chained
                ? "Save each prompt's FLAC, then create one lossless merged FLAC after the final prompt."
                : "Save each prompt's FLAC before the next queued prompt starts, then create one lossless merged FLAC per prompt directory after the final job. The complete last merge is returned.";
        } else if (hasAudio && hasVideo) {
            this.mergeLabel.textContent = chained ? "Merge chain outputs" : "Merge outputs";
            this.mergeCheckbox.setAttribute("aria-label", "Merge outputs");
            this.mergeToggle.title = chained
                ? "Save each prompt's outputs, then merge them in prompt order after the final job."
                : "Save each prompt's outputs before the next queued prompt starts, then merge each prompt directory after the final job.";
        } else {
            this.mergeLabel.textContent = chained ? "Merge chain videos" : "Merge videos";
            this.mergeCheckbox.setAttribute("aria-label", "Merge videos");
            this.mergeToggle.title = chained
                ? "Save each prompt's MP4, then merge them in prompt order after the final job."
                : "With an init video, append the generated segment to it. With Folder batch, save each prompt's MP4 and merge each prompt directory after the final job.";
        }
        this.mergeToggle.hidden = !available;
        this.mergeCheckbox.disabled = !available;
    }

    updateContinuationAvailability() {
        const output = this.node.outputs?.find((item) => item.name === "continue_batch_with_last_frame");
        const available = Boolean(output?.links?.length);
        this.continuationControls.hidden = !available;
        this.lastFrameCheckbox.disabled = !available;
        this.contextFramesSelect.disabled = !available || !this.lastFrameCheckbox.checked;
        this.scheduleTokenEstimate();
    }

    configureFromWidgets() {
        const configured = resolveConfiguredPrompt(
            this.textarea.value,
            this.promptWidget?.value,
            this.hasHydratedPrompt,
            this.promptTouched,
        );
        this.syncFromWidgets();
        this.textarea.value = configured.value;
        if (!configured.hydrateFromWidget) {
            // Attachment-only configuration may have restored a stale hidden
            // widget value. Put the live editor value back immediately.
            this.syncPromptToWidget();
        }
        this.hasHydratedPrompt = true;
        this.renderAllOverlays();
        this.scheduleTokenEstimate();
    }

    saveManifest() {
        if (this.manifestWidget) {
            this.state.prompts = this.promptEditors.slice(1).map((editor) => editor.textarea.value);
            const serialized = { ...this.state };
            if (!serialized.prompts.length) delete serialized.prompts;
            this.manifestWidget.value = JSON.stringify(serialized);
            this.manifestWidget.callback?.(this.manifestWidget.value);
        }
        this.node.setDirtyCanvas(true, true);
    }

    syncPromptToWidget() {
        if (this.promptWidget) {
            this.promptWidget.value = this.textarea.value;
            this.promptWidget.callback?.(this.promptWidget.value);
        }
    }

    entriesOf(type) {
        return this.state[REFERENCE_TYPES[type].stateKey];
    }

    /** Flat descriptor list of all references for cards, autocomplete, and the pill overlay. */
    referenceEntries() {
        const entries = [];
        for (const type of TYPE_ORDER) {
            this.entriesOf(type).forEach((entry, index) => {
                const n = index + 1;
                entries.push({
                    type, n, entry,
                    token: tokenFor(type, n),
                    color: colorFor(type, n),
                    filename: entry.name || entry.file,
                    thumbSrc: type === "image" ? viewURL(entry.file) : null,
                    keys: REFERENCE_TYPES[type].aliases.map((alias) => `${alias}${n}`),
                });
            });
        }
        return entries;
    }

    counts() {
        return {
            image: this.state.images.length,
            video: this.state.videos.length,
            audio: this.state.audios.length,
        };
    }

    // ==================== Adding & removing references ====================

    async addFiles(files) {
        const rejected = [];
        let added = false;
        for (const file of files) {
            const type = mediaKind(file);
            if (!type || this.entriesOf(type).length >= REFERENCE_TYPES[type].max) {
                rejected.push(file.name);
                continue;
            }
            try {
                const uploaded = await uploadReferenceFile(file);
                this.entriesOf(type).push(uploaded);
                added = true;
            } catch (error) {
                notify("error", "Reference upload failed", `${file.name}: ${error}`);
            }
        }
        if (added) {
            this.saveManifest();
            this.render();
        }
        if (rejected.length) {
            notify("warn", "Reference limits reached",
                `Limits are ${REFERENCE_TYPES.image.max} images, ${REFERENCE_TYPES.video.max} videos, and ${REFERENCE_TYPES.audio.max} audio files. Not added: ${rejected.join(", ")}`);
        }
    }

    removeReference(type, index) {
        const list = this.entriesOf(type);
        if (index < 0 || index >= list.length) return;
        list.splice(index, 1);
        // The prompt text is never touched: surviving references renumber by
        // position, and a token now pointing past the list renders as inactive
        // and is omitted at execution time instead of erroring.
        this.saveManifest();
        this.render();
    }

    moveReference(type, from, insertPos) {
        const list = this.entriesOf(type);
        if (from < 0 || from >= list.length) return;
        let to = insertPos > from ? insertPos - 1 : insertPos;
        to = Math.max(0, Math.min(list.length - 1, to));
        if (to === from) return;
        const [entry] = list.splice(from, 1);
        list.splice(to, 0, entry);
        // Reference numbers follow position; the prompt text stays untouched.
        this.saveManifest();
        this.render();
    }

    /** Cards of the same type as the current drag, in display order. */
    reorderPeers() {
        return [...this.cards.querySelectorAll(`.secourses-refgal-card[data-ref-type="${this.dragContext.type}"]`)];
    }

    /** Insertion slot [0..n] among same-type cards for the pointer position. */
    reorderInsertPos(event) {
        let pos = 0;
        for (const card of this.reorderPeers()) {
            const rect = card.getBoundingClientRect();
            if (event.clientY > rect.bottom || (event.clientY >= rect.top && event.clientX > rect.left + rect.width / 2)) {
                pos++;
            }
        }
        return pos;
    }

    updateDropMarker(insertPos) {
        this.clearDropMarkers();
        const peers = this.reorderPeers();
        if (!peers.length) return;
        if (insertPos < peers.length) {
            peers[insertPos].classList.add("secourses-refgal-drop-before");
        } else {
            peers[peers.length - 1].classList.add("secourses-refgal-drop-after");
        }
    }

    clearDropMarkers() {
        for (const card of this.cards.querySelectorAll(".secourses-refgal-drop-before, .secourses-refgal-drop-after")) {
            card.classList.remove("secourses-refgal-drop-before", "secourses-refgal-drop-after");
        }
    }

    // ==================== Optional trim loader ====================

    toggleTrimLoader(open = this.loader.hidden) {
        this.loader.hidden = !open;
        this.trimToggle.classList.toggle("secourses-refgal-trimtoggle-active", open);
        if (!open) {
            this.resetTrimLoader();
        } else {
            this.refreshLayout();
        }
    }

    resetTrimLoader() {
        const element = this.loaderMedia?.element;
        if (element) {
            element.pause?.();
            element.removeAttribute("src");
            element.load?.();
        }
        this.loaderMedia = null;
        this.trimPreviewActive = false;
        this.loaderPreview.textContent = "";
        this.loaderPreview.hidden = true;
        this.trimBody.hidden = true;
        this.loaderName.textContent = "No file loaded yet";
        this.loaderName.title = "";
        this.loaderNote.textContent = "";
        this.loaderAdd.disabled = true;
        this.trimPlayhead.style.left = "0%";
        this.refreshLayout();
    }

    async loaderPickFile(file) {
        const kind = mediaKind(file);
        if (kind !== "video" && kind !== "audio") {
            notify("warn", "Trim loader", "Pick a video or audio file. Images never need trimming — add them with “Add references”.");
            return;
        }
        this.resetTrimLoader();
        this.loaderName.textContent = `Uploading ${file.name}…`;
        let uploaded;
        try {
            uploaded = await uploadReferenceFile(file);
        } catch (error) {
            this.loaderName.textContent = "No file loaded yet";
            notify("error", "Reference upload failed", `${file.name}: ${error}`);
            return;
        }
        const element = document.createElement(kind === "video" ? "video" : "audio");
        element.className = `secourses-refgal-loader-media secourses-refgal-loader-media-${kind}`;
        element.controls = true;
        element.preload = "metadata";
        if (kind === "video") element.playsInline = true;
        element.src = viewURL(uploaded.file);
        this.loaderMedia = { uploaded, kind, element, duration: null, start: 0, end: null };
        this.loaderName.textContent = file.name;
        this.loaderName.title = file.name;
        this.loaderPreview.appendChild(element);
        this.loaderPreview.hidden = false;
        this.loaderAdd.disabled = false;
        this.loaderNote.textContent = "Loading duration…";
        element.addEventListener("loadedmetadata", () => {
            if (this.loaderMedia?.element !== element) return;
            const duration = Number(element.duration);
            if (Number.isFinite(duration) && duration > 0) {
                this.loaderMedia.duration = duration;
                this.loaderMedia.start = 0;
                this.loaderMedia.end = duration;
                this.loaderNote.textContent = "";
                this.updateTrimUI();
            } else {
                this.loaderNote.textContent = "Duration unavailable — this file can only be added untrimmed.";
            }
            this.refreshLayout();
        });
        element.addEventListener("timeupdate", () => {
            const media = this.loaderMedia;
            if (media?.element !== element || !media.duration) return;
            const position = Math.min(element.currentTime, media.duration);
            this.trimPlayhead.style.left = `${(position / media.duration) * 100}%`;
            if (this.trimPreviewActive && element.currentTime >= media.end - 0.02) {
                element.pause();
                this.trimPreviewActive = false;
            }
        });
        element.addEventListener("pause", () => {
            this.trimPreviewActive = false;
        });
        element.addEventListener("error", () => {
            if (this.loaderMedia?.element !== element) return;
            this.loaderNote.textContent = "Preview failed — the file can still be added untrimmed.";
        });
        this.refreshLayout();
    }

    /** Minimum trim window; reference videos need ≥5 frames (~0.2s at 24 fps). */
    trimMinRange() {
        return this.loaderMedia?.kind === "video" ? 0.25 : 0.05;
    }

    timelineTime(event) {
        const rect = this.trimTrack.getBoundingClientRect();
        const ratio = rect.width ? (event.clientX - rect.left) / rect.width : 0;
        return Math.max(0, Math.min(1, ratio)) * (this.loaderMedia?.duration ?? 0);
    }

    bindTrimHandle(handle, isStart) {
        handle.addEventListener("pointerdown", (event) => {
            const media = this.loaderMedia;
            if (!media?.duration) return;
            event.preventDefault();
            event.stopPropagation();
            try {
                handle.setPointerCapture(event.pointerId);
            } catch (error) {
                // Dragging still works without capture; it just stops at the widget edge.
            }
            const move = (moveEvent) => {
                const time = this.timelineTime(moveEvent);
                if (isStart) {
                    this.setTrimRange(Math.min(time, media.end - this.trimMinRange()), media.end, { seek: time });
                } else {
                    this.setTrimRange(media.start, Math.max(time, media.start + this.trimMinRange()), { seek: time });
                }
            };
            const stop = () => {
                handle.removeEventListener("pointermove", move);
                handle.removeEventListener("pointerup", stop);
                handle.removeEventListener("pointercancel", stop);
            };
            handle.addEventListener("pointermove", move);
            handle.addEventListener("pointerup", stop);
            handle.addEventListener("pointercancel", stop);
            move(event);
        });
    }

    setTrimRange(start, end, { seek } = {}) {
        const media = this.loaderMedia;
        if (!media?.duration) return;
        const minRange = Math.min(this.trimMinRange(), media.duration);
        start = Math.max(0, Math.min(Number.isFinite(start) ? start : 0, media.duration));
        end = Math.max(0, Math.min(Number.isFinite(end) ? end : media.duration, media.duration));
        if (end - start < minRange) {
            end = Math.min(media.duration, start + minRange);
            start = Math.max(0, Math.min(start, end - minRange));
        }
        media.start = start;
        media.end = end;
        if (seek != null && Number.isFinite(seek) && media.element) {
            media.element.currentTime = Math.max(0, Math.min(seek, media.duration));
        }
        this.updateTrimUI();
    }

    isTrimmed() {
        const media = this.loaderMedia;
        if (!media?.duration) return false;
        return media.start > 0.01 || media.end < media.duration - 0.01;
    }

    updateTrimUI() {
        const media = this.loaderMedia;
        if (!media?.duration) {
            this.trimBody.hidden = true;
            return;
        }
        const wasHidden = this.trimBody.hidden;
        this.trimBody.hidden = false;
        const startPct = (media.start / media.duration) * 100;
        const endPct = (media.end / media.duration) * 100;
        this.trimStartHandle.style.left = `${startPct}%`;
        this.trimEndHandle.style.left = `${endPct}%`;
        this.trimFill.style.left = `${startPct}%`;
        this.trimFill.style.width = `${Math.max(0, endPct - startPct)}%`;
        if (document.activeElement !== this.trimStartInput) this.trimStartInput.value = media.start.toFixed(2);
        if (document.activeElement !== this.trimEndInput) this.trimEndInput.value = media.end.toFixed(2);
        const trimmed = this.isTrimmed();
        this.trimBadge.textContent = trimmed
            ? `✂ ${(media.end - media.start).toFixed(2)}s of ${media.duration.toFixed(2)}s`
            : `full ${media.duration.toFixed(2)}s (untrimmed)`;
        this.trimBadge.classList.toggle("secourses-refgal-trim-length-active", trimmed);
        if (wasHidden) this.refreshLayout();
    }

    addTrimmedReference() {
        const media = this.loaderMedia;
        if (!media) return;
        const type = media.kind;
        if (this.entriesOf(type).length >= REFERENCE_TYPES[type].max) {
            notify("warn", "Reference limits reached",
                `The gallery already has ${REFERENCE_TYPES[type].max} ${type} references. Remove one first.`);
            return;
        }
        const entry = { ...media.uploaded };
        if (this.isTrimmed()) {
            entry.trim_start = Math.round(media.start * 1000) / 1000;
            entry.trim_end = Math.round(media.end * 1000) / 1000;
        }
        this.entriesOf(type).push(entry);
        this.saveManifest();
        this.render();
        notify("success", "Reference added", entry.trim_end != null
            ? `${entry.name} (${entry.trim_start}s → ${entry.trim_end}s)`
            : entry.name);
        this.resetTrimLoader();
    }

    /** Gallery height consumed by the trim loader in its current state. */
    trimLoaderHeight() {
        if (this.loader.hidden) return 0;
        if (!this.loaderMedia) return 104;
        const preview = this.loaderMedia.kind === "video" ? 168 : 66;
        const timeline = this.loaderMedia.duration ? 66 : 0;
        return 104 + preview + timeline;
    }

    // ==================== Rendering ====================

    render() {
        this.cards.textContent = "";
        const entries = this.referenceEntries();
        for (const item of entries) {
            this.cards.appendChild(this.buildCard(item));
        }
        const total = entries.length;
        const c = this.counts();
        this.counter.textContent =
            `${c.image}/${REFERENCE_TYPES.image.max} images | ${c.video}/${REFERENCE_TYPES.video.max} videos | ${c.audio}/${REFERENCE_TYPES.audio.max} audio`;
        this.hint.style.display = total ? "" : "none";
        this.empty.style.display = total ? "none" : "";
        this.renderAllOverlays();
        this.refreshLayout();
        this.scheduleTokenEstimate();
    }

    buildCard(item) {
        const card = document.createElement("div");
        card.className = "secourses-refgal-card";
        card.style.setProperty("--ref-color", item.color);
        card.dataset.refType = item.type;
        card.title = `${item.filename}\nClick to insert ${item.token} into the prompt.\nDrag left/right to reorder (tokens renumber by position).`;
        card.draggable = true;
        card.addEventListener("dragstart", (event) => {
            this.dragContext = { type: item.type, index: item.n - 1 };
            event.dataTransfer.effectAllowed = "move";
            event.dataTransfer.setData(REORDER_MIME, `${item.type}:${item.n - 1}`);
            card.classList.add("secourses-refgal-card-dragging");
        });
        card.addEventListener("dragend", () => {
            this.dragContext = null;
            this.clearDropMarkers();
            card.classList.remove("secourses-refgal-card-dragging");
        });

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "secourses-refgal-remove";
        remove.innerHTML = "&times;";
        remove.title = `Remove this ${item.type} reference`;
        remove.addEventListener("click", (event) => {
            event.stopPropagation();
            this.removeReference(item.type, item.n - 1);
        });

        let media;
        const url = viewURL(item.entry.file);
        if (item.type === "image") {
            media = document.createElement("img");
            media.src = url;
            media.loading = "lazy";
        } else if (item.type === "video") {
            media = document.createElement("video");
            media.src = url;
            media.muted = true;
            media.loop = true;
            media.playsInline = true;
            media.preload = "metadata";
            media.addEventListener("mouseenter", () => media.play().catch(() => {}));
            media.addEventListener("mouseleave", () => media.pause());
        } else {
            media = document.createElement("audio");
            media.src = url;
            media.controls = true;
            media.preload = "metadata";
        }
        media.className = `secourses-refgal-media secourses-refgal-media-${item.type}`;
        media.draggable = false;
        media.addEventListener("error", () => card.classList.add("secourses-refgal-card-missing"));

        const label = document.createElement("span");
        label.className = "secourses-refgal-token";
        label.textContent = item.token;
        const name = document.createElement("span");
        name.className = "secourses-refgal-name";
        name.textContent = item.filename;
        name.title = item.filename;

        card.append(remove, media, label, name);
        if (item.entry.trim_start != null || item.entry.trim_end != null) {
            const trim = document.createElement("span");
            trim.className = "secourses-refgal-trimbadge";
            const fmt = (value) => {
                const rounded = Math.round(value * 10) / 10;
                return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
            };
            trim.textContent = item.entry.trim_end != null
                ? `✂ ${fmt(item.entry.trim_start ?? 0)}–${fmt(item.entry.trim_end)}s`
                : `✂ from ${fmt(item.entry.trim_start)}s`;
            trim.title = "Trimmed reference — only this time window is used at generation time.";
            card.appendChild(trim);
        }
        card.addEventListener("click", (event) => {
            if (event.target.closest("button, audio")) return;
            this.insertToken(item.type, item.n);
        });
        return card;
    }

    /** Node height consumed by everything that is not the gallery widget. */
    fixedHeight() {
        let fixed = 10;
        for (const w of this.node.widgets ?? []) {
            if (w === this.widget || w.hidden) continue;
            fixed += (w.computeSize ? w.computeSize(this.node.size[0])[1] : LiteGraph.NODE_WIDGET_HEIGHT) + 4;
        }
        fixed += Math.max(this.node.inputs?.length ?? 0, this.node.outputs?.length ?? 0) * LiteGraph.NODE_SLOT_HEIGHT;
        return fixed;
    }

    /** Smallest gallery height that still shows the toolbar, cards, prompt, and two-line batch path. */
    minContentHeight(width) {
        const total = this.state.images.length + this.state.videos.length + this.state.audios.length;
        const perRow = Math.max(1, Math.floor((width - 12) / 128));
        const rows = total ? Math.ceil(total / perRow) : 0;
        const cardsH = total ? Math.min(rows, 2) * 124 : 30;
        const continuationH = 40;
        const promptH = Math.min(360, 116 + Math.max(0, this.promptEditors.length - 1) * 132);
        return 34 + cardsH + this.trimLoaderHeight() + promptH + 66 + continuationH + 14;
    }

    computeHeight(width) {
        // Fill whatever height the node offers, but never collapse below the minimum.
        const available = this.node.size[1] - this.fixedHeight() - 14;
        return Math.max(this.minContentHeight(width), available);
    }

    refreshLayout() {
        const totalNeeded = this.minContentHeight(this.node.size[0]) + this.fixedHeight() + 14;
        if (this.node.size[1] < totalNeeded) {
            this.node.setSize([this.node.size[0], totalNeeded]);
        }
        this.node.setDirtyCanvas(true, true);
    }

    // ==================== Token insertion ====================

    insertToken(type, n) {
        const editor = this.activeEditor ?? this.promptEditors[0];
        const box = editor.textarea;
        const token = tokenFor(type, n);
        const start = box.selectionStart ?? box.value.length;
        const end = box.selectionEnd ?? box.value.length;
        const before = box.value.substring(0, start);
        const after = box.value.substring(end);
        let insert = token;
        if (before.length && !/[\s([{"']$/.test(before)) {
            insert = " " + insert;
        }
        if (!/^[\s,.)\]}!?;:]/.test(after)) {
            insert = insert + " ";
        }
        box.value = before + insert + after;
        const position = before.length + insert.length;
        box.focus();
        box.setSelectionRange(position, position);
        this.onPromptInput(editor);
    }

    // ==================== Colored pill overlay ====================

    /**
     * Mirrors the textarea's text metrics and content box onto the overlay.
     *
     * The overlay only lines up with the invisible text if it wraps at exactly the
     * same column. It already covers the textarea's border box (both fill the wrap),
     * so copying the textarea's border widths puts their content boxes on the same
     * subpixel, and clientWidth gives the text column in layout pixels: it excludes
     * the border and the scrollbar gutter, and (unlike getBoundingClientRect) ignores
     * the canvas zoom transform ComfyUI puts on the DOM widget layer.
     */
    syncOverlayGeometry(editor = this.promptEditors[0]) {
        const box = editor.textarea;
        const inner = editor.overlayInner;
        // Before the node is laid out (and while a collapsed DOM widget has no size)
        // there is nothing to measure; the ResizeObserver re-runs this on the first
        // real size.
        if (!box.clientWidth) {
            this.syncOverlayScroll(editor);
            return;
        }
        const style = getComputedStyle(box);
        // Every property that can move a glyph or a wrap point has to match, or the
        // pills drift away from the text and the caret lands on the wrong character.
        for (const property of OVERLAY_METRIC_PROPERTIES) inner.style[property] = style[property];
        editor.overlay.style.borderTopWidth = style.borderTopWidth;
        editor.overlay.style.borderRightWidth = style.borderRightWidth;
        editor.overlay.style.borderBottomWidth = style.borderBottomWidth;
        editor.overlay.style.borderLeftWidth = style.borderLeftWidth;
        inner.style.width = `${box.clientWidth}px`;
        // A right-to-left textarea keeps its scrollbar gutter on the left instead, so
        // its text column starts that much further in.
        const gutter = Math.max(0, box.offsetWidth - box.clientWidth
            - (parseFloat(style.borderLeftWidth) || 0) - (parseFloat(style.borderRightWidth) || 0));
        inner.style.marginLeft = style.direction === "rtl" ? `${gutter}px` : "0px";
        this.syncOverlayScroll(editor);
    }

    syncOverlayScroll(editor = this.promptEditors[0]) {
        editor.overlayInner.style.transform =
            `translate(${-editor.textarea.scrollLeft}px, ${-editor.textarea.scrollTop}px)`;
    }

    renderAllOverlays() {
        for (const editor of this.promptEditors) this.renderOverlay(null, editor);
    }

    renderOverlay(caretIndex = null, editor = this.promptEditors[0]) {
        const text = editor.textarea.value;
        const counts = this.counts();
        // Fold the extra Unicode forms JavaScript case-insensitive matching does not
        // equate with ASCII, so the pills mark exactly what reference_gallery_nodes.py
        // will translate. Every replacement is one UTF-16 code unit, so match offsets
        // still address the user's own text.
        const matchText = text.replace(/[İı]/g, "i").replace(/ſ/g, "s");
        const tokenRegex = /(?<![\w@])@[ \t]*(image|img|picture|pic|video|vid|audio|aud|sound)[ \t]*#?[ \t]*(\d{1,2})(?![0-9a-zA-Z])|<[ \t]*(image|img|picture|pic|video|vid|audio|aud|sound)[ \t]*#?[ \t]*(\d{1,2})[ \t]*>/gi;
        let html = "";
        let last = 0;
        let match;
        const emitPlain = (upTo) => {
            let chunk = text.substring(last, upTo);
            if (caretIndex !== null && caretIndex >= last && caretIndex <= upTo) {
                const split = caretIndex - last;
                html += escapeHtml(chunk.substring(0, split))
                    + '<span class="secourses-refgal-caret-marker"></span>'
                    + escapeHtml(chunk.substring(split));
            } else {
                html += escapeHtml(chunk);
            }
        };
        while ((match = tokenRegex.exec(matchText)) !== null) {
            emitPlain(match.index);
            last = match.index + match[0].length;
            let type, n;
            let legacyAudio = false;
            if (match[1] !== undefined) {
                type = ALIAS_TO_TYPE[match[1].toLowerCase()];
                n = parseInt(match[2]);
            } else {
                type = ALIAS_TO_TYPE[match[3].toLowerCase()];
                n = parseInt(match[4]);
                legacyAudio = type === "audio";
            }
            let valid, color = null;
            if (legacyAudio) {
                // Legacy <Audio n> labels index video soundtracks first, then standalone audio.
                if (n >= 1 && n <= counts.video) {
                    valid = true;
                    color = colorFor("video", n);
                } else if (n > counts.video && n <= counts.video + counts.audio) {
                    valid = true;
                    color = colorFor("audio", n - counts.video);
                } else {
                    valid = false;
                }
            } else {
                valid = n >= 1 && n <= counts[type];
                if (valid) color = colorFor(type, n);
            }
            const cls = valid ? "secourses-refgal-pill" : "secourses-refgal-pill secourses-refgal-pill-invalid";
            const style = valid ? ` style="--ref-color:${color};"` : "";
            html += `<span class="${cls}"${style}>${escapeHtml(text.substring(match.index, last))}</span>`;
        }
        emitPlain(text.length);
        // A textarea keeps an empty last line after a trailing newline; a block box
        // does not, so pin one down and the two stay exactly the same height.
        editor.overlayInner.innerHTML = html + "​";
        this.syncOverlayGeometry(editor);
    }

    // ==================== '@' autocomplete ====================

    /** Returns {start, end, partial} when the text before the caret ends in a partial '@' reference. */
    getSuggestContext(editor = this.activeEditor ?? this.promptEditors[0]) {
        const box = editor.textarea;
        const start = box.selectionStart;
        if (start == null || box.selectionEnd !== start) return null;
        const before = box.value.substring(0, start);
        const match = /(?<![\w@])@([a-zA-Z]{0,7}#?\d{0,2})$/.exec(before);
        if (!match) return null;
        return { start: start - match[0].length, end: start, partial: match[1] };
    }

    updateSuggestions(editor = this.activeEditor ?? this.promptEditors[0]) {
        const context = this.getSuggestContext(editor);
        if (!context) {
            this.closeSuggestions(editor);
            return;
        }
        const entries = this.referenceEntries();
        if (!entries.length) {
            this.closeSuggestions(editor);
            return;
        }
        const partial = context.partial.toLowerCase().replace("#", "");
        const matches = partial === "" ? entries
            : entries.filter((entry) => entry.keys.some((key) => key.startsWith(partial)));
        if (!matches.length) {
            this.closeSuggestions(editor);
            return;
        }
        editor.suggestContext = context;
        editor.suggestMatches = matches;
        editor.suggestIndex = Math.min(editor.suggestIndex, matches.length - 1);
        this.renderSuggestions(editor);
    }

    renderSuggestions(editor = this.activeEditor ?? this.promptEditors[0]) {
        editor.suggest.textContent = "";
        const header = document.createElement("div");
        header.className = "secourses-refgal-suggest-header";
        header.textContent = "REFERENCES";
        editor.suggest.appendChild(header);
        editor.suggestMatches.forEach((entry, index) => {
            const row = document.createElement("div");
            row.className = "secourses-refgal-suggest-item" + (index === editor.suggestIndex ? " selected" : "");
            let thumb;
            if (entry.thumbSrc) {
                thumb = document.createElement("img");
                thumb.className = "secourses-refgal-suggest-thumb";
                thumb.src = entry.thumbSrc;
            } else {
                thumb = document.createElement("span");
                thumb.className = "secourses-refgal-suggest-thumb secourses-refgal-suggest-icon";
                thumb.textContent = entry.type === "video" ? "▶" : "♪";
            }
            thumb.style.borderColor = entry.color;
            thumb.style.color = entry.color;
            const token = document.createElement("span");
            token.className = "secourses-refgal-suggest-token";
            token.style.color = entry.color;
            token.textContent = entry.token;
            const name = document.createElement("span");
            name.className = "secourses-refgal-suggest-name";
            name.textContent = entry.filename;
            row.append(thumb, token, name);
            row.addEventListener("mouseenter", () => {
                editor.suggestIndex = index;
                this.renderSuggestions(editor);
            });
            row.addEventListener("click", () => this.applyCompletion(entry, editor));
            editor.suggest.appendChild(row);
        });
        editor.suggest.hidden = false;
        this.positionSuggestions(editor);
    }

    positionSuggestions(editor = this.activeEditor ?? this.promptEditors[0]) {
        // The overlay mirrors the textarea exactly, so a marker span at the caret
        // index gives the caret's pixel position for popover anchoring.
        this.renderOverlay(editor.suggestContext?.end ?? null, editor);
        const marker = editor.overlay.querySelector(".secourses-refgal-caret-marker");
        if (!marker) return;
        const lineHeight = parseFloat(getComputedStyle(editor.textarea).lineHeight) || 16;
        // The marker's offsets are relative to the overlay's content box; add the
        // overlay's own position and border back to land in the wrap's coordinates.
        let top = editor.overlay.offsetTop + editor.overlay.clientTop + marker.offsetTop
            - editor.textarea.scrollTop + lineHeight + 2;
        let left = editor.overlay.offsetLeft + editor.overlay.clientLeft + marker.offsetLeft
            - editor.textarea.scrollLeft;
        const maxLeft = Math.max(0, editor.wrap.clientWidth - 240);
        left = Math.min(Math.max(0, left), maxLeft);
        top = Math.min(Math.max(0, top), editor.wrap.clientHeight - 8);
        editor.suggest.style.left = `${left}px`;
        editor.suggest.style.top = `${top}px`;
        this.renderOverlay(null, editor);
    }

    closeSuggestions(editor = this.activeEditor ?? this.promptEditors[0]) {
        editor.suggest.hidden = true;
        editor.suggestMatches = null;
        editor.suggestIndex = 0;
    }

    applyCompletion(entry, editor = this.activeEditor ?? this.promptEditors[0]) {
        const context = editor.suggestContext ?? this.getSuggestContext(editor);
        if (!context) return;
        const box = editor.textarea;
        const before = box.value.substring(0, context.start);
        const after = box.value.substring(context.end);
        const insert = entry.token + (/^[\s,.)\]}!?;:]/.test(after) ? "" : " ");
        box.value = before + insert + after;
        const position = before.length + insert.length;
        box.focus();
        box.setSelectionRange(position, position);
        this.onPromptInput(editor);
        this.closeSuggestions(editor);
    }

    onPromptKeydown(event, editor = this.activeEditor ?? this.promptEditors[0]) {
        const open = editor.suggestMatches && !editor.suggest.hidden;
        if (open) {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                event.preventDefault();
                event.stopPropagation();
                const delta = event.key === "ArrowDown" ? 1 : -1;
                editor.suggestIndex = (editor.suggestIndex + delta + editor.suggestMatches.length) % editor.suggestMatches.length;
                this.renderSuggestions(editor);
                return;
            }
            if (event.key === "Enter" || event.key === "Tab") {
                event.preventDefault();
                event.stopPropagation();
                this.applyCompletion(editor.suggestMatches[editor.suggestIndex], editor);
                return;
            }
            if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                this.closeSuggestions(editor);
                return;
            }
        }
        // Keep normal typing away from canvas hotkeys, but let queue shortcuts through.
        if (!(event.ctrlKey || event.metaKey)) {
            event.stopPropagation();
        }
    }
}

app.registerExtension({
    name: "SECourses.ReferenceGallery",
    init() {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = new URL("./secourses_reference_gallery.css", import.meta.url).href;
        document.head.appendChild(link);
    },
    setup() {
        installSequentialFolderQueue();
    },
    loadedGraphNode(node) {
        reconcilePortableComboPaths(node);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === AUDIO_RESULT_CLASS) {
            nodeData.input.required.audioUI = ["AUDIO_UI", {}];
            return;
        }
        if (nodeData.name !== NODE_CLASS) return;
        for (const inputName of QUEUE_ONLY_INPUTS) {
            delete nodeData.input?.optional?.[inputName];
        }
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            this.__refGallery = new ReferenceGalleryUI(this);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            this.__refGallery?.configureFromWidgets();
        });
        chainCallback(nodeType.prototype, "onConnectionsChange", function () {
            this.__refGallery?.updateMergeAvailability();
            this.__refGallery?.updateContinuationAvailability();
        });
        chainCallback(nodeType.prototype, "onRemoved", function () {
            this.__refGallery?.dispose();
        });
    },
});
