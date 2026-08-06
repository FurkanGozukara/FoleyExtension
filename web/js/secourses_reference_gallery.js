/**
 * SECourses Reference Gallery widget: SwarmUI-style media references for ComfyUI.
 *
 * Renders an "Add references" gallery with thumbnail cards, colored @image1 /
 * @video1 / @audio1 tokens, a prompt box with colored token pills, and an '@'
 * autocomplete popover, all inside the SECoursesReferenceGallery node. Files
 * upload through ComfyUI's native /upload/image endpoint into the input
 * directory; the ordered manifest is stored in the node's hidden "references"
 * widget so the Python side can load every file.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { resolveConfiguredPrompt } from "./secourses_reference_gallery_state.mjs";

const NODE_CLASS = "SECoursesReferenceGallery";
const UPLOAD_SUBFOLDER = "reference_gallery";
const REORDER_MIME = "application/x-secourses-reference";

/** Per-type '@' aliases, limits, and pill palettes, matching the SwarmUI extension. */
const REFERENCE_TYPES = {
    image: {
        aliases: ["image", "img", "picture", "pic"],
        label: "Picture",
        max: 9,
        colors: ["#4dabf7", "#ffa94d", "#69db7c", "#f783ac", "#b197fc", "#ffd43b", "#3bc9db", "#ff8787", "#a9e34b"],
        stateKey: "images",
    },
    video: {
        aliases: ["video", "vid"],
        label: "Video",
        max: 3,
        colors: ["#ff6b6b", "#748ffc", "#38d9a9"],
        stateKey: "videos",
    },
    audio: {
        aliases: ["audio", "aud", "sound"],
        label: "Audio",
        max: 3,
        colors: ["#fcc419", "#da77f2", "#66d9e8"],
        stateKey: "audios",
    },
};

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

function notify(severity, summary, detail) {
    const toast = app.extensionManager?.toast;
    if (toast?.add) {
        toast.add({ severity, summary, detail, life: 6000 });
    } else {
        console.warn(`[SECoursesReferenceGallery] ${summary}: ${detail}`);
        if (severity === "error") alert(`${summary}\n${detail}`);
    }
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
    body.append("image", file, file.name);
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
        this.state = { images: [], videos: [], audios: [] };
        this.suggestIndex = 0;
        this.suggestMatches = null;
        this.hasHydratedPrompt = false;
        this.promptTouched = false;
        this.dragContext = null;
        this.buildDOM();
        hideWidget(this.promptWidget);
        hideWidget(this.manifestWidget);
        const ui = this;
        this.widget = node.addDOMWidget("gallery_ui", "secourses_gallery", this.root, {
            hideOnZoom: false,
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
        this.counter = document.createElement("span");
        this.counter.className = "secourses-refgal-counter";
        this.hint = document.createElement("span");
        this.hint.className = "secourses-refgal-hint";
        this.hint.textContent = "Type @ in the prompt to reference";
        this.hint.title = "Type '@' in the prompt for reference autocomplete, eg '@image1'. Click any card to insert its token.";
        this.toolbar.append(this.addButton, this.counter, this.hint);

        this.cards = document.createElement("div");
        this.cards.className = "secourses-refgal-cards";
        this.empty = document.createElement("div");
        this.empty.className = "secourses-refgal-empty";
        this.empty.textContent = "No references yet — click “Add references”, drag & drop, or paste media here.";
        this.cardsWrap = document.createElement("div");
        this.cardsWrap.className = "secourses-refgal-cardswrap";
        this.cardsWrap.append(this.cards, this.empty);

        this.promptWrap = document.createElement("div");
        this.promptWrap.className = "secourses-refgal-promptwrap";
        this.overlay = document.createElement("div");
        this.overlay.className = "secourses-refgal-overlay";
        this.overlay.setAttribute("aria-hidden", "true");
        this.textarea = document.createElement("textarea");
        this.textarea.className = "secourses-refgal-prompt";
        this.textarea.placeholder = "Prompt — type @ to reference attachments, eg @image1 …";
        this.textarea.spellcheck = false;
        this.suggest = document.createElement("div");
        this.suggest.className = "secourses-refgal-suggest";
        this.suggest.hidden = true;
        this.promptWrap.append(this.overlay, this.textarea, this.suggest);

        this.fileInput = document.createElement("input");
        this.fileInput.type = "file";
        this.fileInput.multiple = true;
        this.fileInput.accept = "image/*,video/*,audio/*";
        this.fileInput.hidden = true;

        this.root.append(this.toolbar, this.cardsWrap, this.promptWrap, this.fileInput);
        this.bindEvents();
    }

    bindEvents() {
        this.addButton.addEventListener("click", () => this.fileInput.click());
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

        this.textarea.addEventListener("paste", async (event) => {
            const files = [...(event.clipboardData?.items || [])]
                .filter((item) => item.kind === "file")
                .map((item) => item.getAsFile())
                .filter((file) => file && mediaKind(file));
            if (!files.length) return;
            event.preventDefault();
            event.stopPropagation();
            await this.addFiles(files);
        });

        this.textarea.addEventListener("input", () => {
            this.promptTouched = true;
            this.syncPromptToWidget();
            this.renderOverlay();
            this.updateSuggestions();
        });
        this.textarea.addEventListener("scroll", () => this.syncOverlayScroll());
        this.textarea.addEventListener("click", () => this.updateSuggestions());
        this.textarea.addEventListener("blur", () => window.setTimeout(() => this.closeSuggestions(), 150));
        this.textarea.addEventListener("keydown", (event) => this.onPromptKeydown(event));
        this.suggest.addEventListener("mousedown", (event) => event.preventDefault());
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
        };
        if (hydratePrompt) {
            this.textarea.value = this.promptWidget?.value ?? "";
        }
        this.render();
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
        this.renderOverlay();
    }

    saveManifest() {
        if (this.manifestWidget) {
            this.manifestWidget.value = JSON.stringify(this.state);
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
        this.renderOverlay();
        this.refreshLayout();
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

    /** Smallest gallery height that still shows the toolbar, one cards row, and the prompt. */
    minContentHeight(width) {
        const total = this.state.images.length + this.state.videos.length + this.state.audios.length;
        const perRow = Math.max(1, Math.floor((width - 12) / 128));
        const rows = total ? Math.ceil(total / perRow) : 0;
        const cardsH = total ? Math.min(rows, 2) * 124 : 30;
        return 34 + cardsH + 116 + 14;
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
        const box = this.textarea;
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
        this.promptTouched = true;
        this.syncPromptToWidget();
        this.renderOverlay();
    }

    // ==================== Colored pill overlay ====================

    syncOverlayScroll() {
        this.overlay.scrollTop = this.textarea.scrollTop;
        this.overlay.scrollLeft = this.textarea.scrollLeft;
    }

    renderOverlay(caretIndex = null) {
        const text = this.textarea.value;
        const counts = this.counts();
        const tokenRegex = /(?<![\w@])@(image|img|picture|pic|video|vid|audio|aud|sound)#?(\d{1,2})(?![0-9a-zA-Z])|<(Picture|Video|Audio)[ ]?(\d{1,2})>/gi;
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
        while ((match = tokenRegex.exec(text)) !== null) {
            emitPlain(match.index);
            last = match.index + match[0].length;
            let type, n;
            let legacyAudio = false;
            if (match[1] !== undefined) {
                type = ALIAS_TO_TYPE[match[1].toLowerCase()];
                n = parseInt(match[2]);
            } else {
                const label = match[3].toLowerCase();
                type = label === "picture" ? "image" : label;
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
            html += `<span class="${cls}"${style}>${escapeHtml(match[0])}</span>`;
        }
        emitPlain(text.length);
        this.overlay.innerHTML = html + "​";
        this.syncOverlayScroll();
    }

    // ==================== '@' autocomplete ====================

    /** Returns {start, end, partial} when the text before the caret ends in a partial '@' reference. */
    getSuggestContext() {
        const box = this.textarea;
        const start = box.selectionStart;
        if (start == null || box.selectionEnd !== start) return null;
        const before = box.value.substring(0, start);
        const match = /(?<![\w@])@([a-zA-Z]{0,7}#?\d{0,2})$/.exec(before);
        if (!match) return null;
        return { start: start - match[0].length, end: start, partial: match[1] };
    }

    updateSuggestions() {
        const context = this.getSuggestContext();
        if (!context) {
            this.closeSuggestions();
            return;
        }
        const entries = this.referenceEntries();
        if (!entries.length) {
            this.closeSuggestions();
            return;
        }
        const partial = context.partial.toLowerCase().replace("#", "");
        const matches = partial === "" ? entries
            : entries.filter((entry) => entry.keys.some((key) => key.startsWith(partial)));
        if (!matches.length) {
            this.closeSuggestions();
            return;
        }
        this.suggestContext = context;
        this.suggestMatches = matches;
        this.suggestIndex = Math.min(this.suggestIndex, matches.length - 1);
        this.renderSuggestions();
    }

    renderSuggestions() {
        this.suggest.textContent = "";
        const header = document.createElement("div");
        header.className = "secourses-refgal-suggest-header";
        header.textContent = "REFERENCES";
        this.suggest.appendChild(header);
        this.suggestMatches.forEach((entry, index) => {
            const row = document.createElement("div");
            row.className = "secourses-refgal-suggest-item" + (index === this.suggestIndex ? " selected" : "");
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
                this.suggestIndex = index;
                this.renderSuggestions();
            });
            row.addEventListener("click", () => this.applyCompletion(entry));
            this.suggest.appendChild(row);
        });
        this.suggest.hidden = false;
        this.positionSuggestions();
    }

    positionSuggestions() {
        // The overlay mirrors the textarea exactly, so a marker span at the caret
        // index gives the caret's pixel position for popover anchoring.
        this.renderOverlay(this.suggestContext?.end ?? null);
        const marker = this.overlay.querySelector(".secourses-refgal-caret-marker");
        if (!marker) return;
        const lineHeight = parseFloat(getComputedStyle(this.textarea).lineHeight) || 16;
        let top = marker.offsetTop - this.textarea.scrollTop + lineHeight + 2;
        let left = marker.offsetLeft - this.textarea.scrollLeft;
        const maxLeft = Math.max(0, this.promptWrap.clientWidth - 240);
        left = Math.min(Math.max(0, left), maxLeft);
        top = Math.min(Math.max(0, top), this.promptWrap.clientHeight - 8);
        this.suggest.style.left = `${left}px`;
        this.suggest.style.top = `${top}px`;
        this.renderOverlay();
    }

    closeSuggestions() {
        this.suggest.hidden = true;
        this.suggestMatches = null;
        this.suggestIndex = 0;
    }

    applyCompletion(entry) {
        const context = this.suggestContext ?? this.getSuggestContext();
        if (!context) return;
        const box = this.textarea;
        const before = box.value.substring(0, context.start);
        const after = box.value.substring(context.end);
        const insert = entry.token + (/^[\s,.)\]}!?;:]/.test(after) ? "" : " ");
        box.value = before + insert + after;
        const position = before.length + insert.length;
        box.focus();
        box.setSelectionRange(position, position);
        this.promptTouched = true;
        this.syncPromptToWidget();
        this.renderOverlay();
        this.closeSuggestions();
    }

    onPromptKeydown(event) {
        const open = this.suggestMatches && !this.suggest.hidden;
        if (open) {
            if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                event.preventDefault();
                event.stopPropagation();
                const delta = event.key === "ArrowDown" ? 1 : -1;
                this.suggestIndex = (this.suggestIndex + delta + this.suggestMatches.length) % this.suggestMatches.length;
                this.renderSuggestions();
                return;
            }
            if (event.key === "Enter" || event.key === "Tab") {
                event.preventDefault();
                event.stopPropagation();
                this.applyCompletion(this.suggestMatches[this.suggestIndex]);
                return;
            }
            if (event.key === "Escape") {
                event.preventDefault();
                event.stopPropagation();
                this.closeSuggestions();
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
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            this.__refGallery = new ReferenceGalleryUI(this);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            this.__refGallery?.configureFromWidgets();
        });
    },
});
