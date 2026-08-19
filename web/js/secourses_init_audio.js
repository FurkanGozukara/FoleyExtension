/**
 * Init Audio (Optional, Auto Enable): audio *or video* file picker, player, and trim panel.
 *
 * Core ComfyUI gives the node an audio player plus an upload button that only
 * accepts `audio/*`. This extension re-targets the upload button, drag & drop,
 * and paste to accept video files too (their soundtrack is used), previews a
 * selected video in place of the audio player, and adds a trim panel under the
 * player: a timeline with draggable start/end handles, numeric start/end
 * fields, playhead capture buttons, a trim-window preview, and a reset. The
 * window is stored in the node's (hidden) `trim_start` / `trim_end` widgets,
 * so it serializes like any other widget value and reaches the Python side,
 * which decodes only that window. Everything stays blank while the node says
 * "(none - disabled)".
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_CLASS = "SECoursesInitAudio";
const NO_AUDIO = "(none - disabled)";
const PANEL_WIDGET = "init_audio_trim";
const UPLOAD_ACCEPT = "audio/*,video/*";
const MIN_TRIM_RANGE = 0.05;
const MEDIA_INFO_ROUTE = "/secourses/media_info";

// Mirrors media_extensions.py: video containers preview with a <video>, everything else with the audio player.
const VIDEO_EXTENSIONS = new Set([
    "3g2", "3gp", "avi", "f4v", "flv", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts", "ogv", "rm",
    "ts", "vob", "webm", "wmv",
]);
const AUDIO_EXTENSIONS = new Set([
    "302", "aa", "aa3", "aac", "aax", "ac3", "ac4", "adts", "aea", "afc", "aif", "aifc", "aiff", "al", "alac",
    "amr", "ape", "apl", "aptx", "aptxhd", "ast", "au", "aud", "avr", "bcstm", "bfstm", "binka", "caf", "daud",
    "dff", "dsf", "dts", "dtshd", "eac3", "ec3", "fap", "flac", "g722", "gsm", "iamf", "it", "laf", "loas", "m2a",
    "m4a", "m4b", "mac", "mca", "mka", "mlp", "mod", "mp1", "mp2", "mp3", "mpa", "mpc", "oga", "ogg", "oma", "omg",
    "opus", "paf", "pvf", "ra", "ram", "rka", "s3m", "sb", "sbc", "sds", "shn", "sln", "snd", "sox", "spx", "sw",
    "tak", "tta", "ub", "ul", "uw", "voc", "w64", "wa", "wav", "wave", "wma", "wv", "xm", "xwma",
]);

function chainCallback(object, property, callback) {
    const original = object[property];
    object[property] = function () {
        const result = original?.apply(this, arguments);
        callback.apply(this, arguments);
        return result;
    };
}

function extensionOf(name) {
    const base = String(name ?? "").split(/[\\/]/).pop() ?? "";
    const dot = base.lastIndexOf(".");
    return dot === -1 ? "" : base.slice(dot + 1).toLowerCase();
}

function fileKind(name) {
    return VIDEO_EXTENSIONS.has(extensionOf(name)) ? "video" : "audio";
}

function isMediaFile(file) {
    const type = file?.type ?? "";
    if (type.startsWith("audio/") || type.startsWith("video/")) return true;
    const extension = extensionOf(file?.name);
    return VIDEO_EXTENSIONS.has(extension) || AUDIO_EXTENSIONS.has(extension);
}

function splitFilePath(name) {
    const slash = name.lastIndexOf("/");
    return slash === -1 ? ["", name] : [name.substring(0, slash), name.substring(slash + 1)];
}

/** Same /view URL the core audio widget uses (input folder, cache-busting rand param). */
function viewURL(name) {
    const [subfolder, filename] = splitFilePath(name);
    const rand = app.getRandParam?.() ?? "";
    return api.apiURL(
        `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}${rand}`,
    );
}

function formatSeconds(value) {
    return Number.isFinite(value) ? value.toFixed(2) : "—";
}

function notify(severity, summary, detail) {
    try {
        app.extensionManager?.toast?.add?.({ severity, summary, detail, life: 6000 });
    } catch (error) {
        console[severity === "error" ? "error" : "log"](`[SECourses InitAudio] ${summary}: ${detail}`);
    }
}

/** Insert the core AUDIO_UI player right after the file combo, before the core upload button
 * (which expects the player to already exist), keeping the same required-inputs object. */
function addAudioPlayer(nodeData) {
    const required = nodeData.input?.required;
    if (!required || required.audioUI) return;
    const entries = Object.entries(required);
    for (const [key] of entries) delete required[key];
    for (const [key, value] of entries) {
        required[key] = value;
        if (key === "audio") required.audioUI = ["AUDIO_UI", {}];
    }
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function clearDisabledPlayer(node) {
    const audioWidget = findWidget(node, "audio");
    const player = findWidget(node, "audioUI");
    if (!audioWidget || !player || audioWidget.value !== NO_AUDIO) return;
    if (player.element) {
        player.element.removeAttribute("src");
        player.element.classList.add("empty-audio-widget");
    }
    player.value = "";
}

function hideValueWidget(widget) {
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
}

/**
 * Nudge every reference gallery's live token meter: the init audio selection and its trim
 * window drive the video duration (and the audio-guide tokens), so the estimate must follow
 * immediately instead of waiting for the next canvas interaction.
 */
function scheduleTokenEstimates(node) {
    const stack = [node?.graph, app.graph];
    const seen = new Set();
    while (stack.length) {
        const graph = stack.pop();
        if (!graph || seen.has(graph)) continue;
        seen.add(graph);
        for (const other of graph._nodes ?? []) {
            other.__refGallery?.scheduleTokenEstimate?.();
            if (other.subgraph) stack.push(other.subgraph);
        }
    }
}

// ==================== Upload (audio or video) ====================

async function uploadInitMedia(node, file) {
    const audioWidget = findWidget(node, "audio");
    if (!audioWidget || !file) return false;
    if (!isMediaFile(file)) {
        notify("warn", "Init audio", `${file.name} is not an audio or video file.`);
        return false;
    }
    if (node.isUploading) {
        notify("warn", "Init audio", "An upload is already in progress.");
        return false;
    }
    node.isUploading = true;
    const previous = audioWidget.value;
    try {
        const body = new FormData();
        body.append("image", file);
        body.append("type", "input");
        const response = await api.fetchApi("/upload/image", { method: "POST", body });
        if (response.status !== 200) {
            notify("error", "Init audio upload failed", `${response.status} - ${response.statusText}`);
            return false;
        }
        const data = await response.json();
        const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
        const values = audioWidget.options.values;
        if (Array.isArray(values) && !values.includes(name)) values.push(name);
        audioWidget.value = name;
        audioWidget.callback?.(name);
        node.onWidgetChanged?.(audioWidget.name, name, previous, audioWidget);
        return true;
    } catch (error) {
        notify("error", "Init audio upload failed", String(error?.message ?? error));
        return false;
    } finally {
        node.isUploading = false;
        node.graph?.setDirtyCanvas(true, true);
    }
}

/** Re-target the core upload button, drag & drop, and paste from audio-only to audio + video. */
function installMediaUpload(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = UPLOAD_ACCEPT;
    input.onchange = () => {
        const file = input.files?.[0];
        input.value = "";
        if (file) void uploadInitMedia(node, file);
    };
    const uploadWidget = findWidget(node, "upload");
    if (uploadWidget) {
        uploadWidget.callback = () => input.click();
        uploadWidget.label = "choose audio / video to upload";
    }
    const hasMediaFile = (event) => {
        const files = event?.dataTransfer?.files;
        if (files?.length) return Array.from(files).some(isMediaFile);
        const items = event?.dataTransfer?.items;
        return items ? Array.from(items).some((item) => item.kind === "file") : false;
    };
    node.onDragOver = (event) => hasMediaFile(event);
    node.onDragDrop = async (event) => {
        const file = Array.from(event?.dataTransfer?.files ?? []).find(isMediaFile);
        if (!file) return false;
        await uploadInitMedia(node, file);
        return true;
    };
    node.pasteFiles = (files) => {
        const file = Array.from(files ?? []).find(isMediaFile);
        if (!file) return false;
        void uploadInitMedia(node, file);
        return true;
    };
    const originalRemoved = node.onRemoved;
    node.onRemoved = function () {
        input.onchange = null;
        return originalRemoved?.apply(this, arguments);
    };
}

// ==================== Trim panel ====================

class InitAudioTrimPanel {
    constructor(node) {
        this.node = node;
        this.audioWidget = findWidget(node, "audio");
        this.playerWidget = findWidget(node, "audioUI");
        this.startWidget = findWidget(node, "trim_start");
        this.endWidget = findWidget(node, "trim_end");
        this.file = null;
        this.kind = null;
        this.duration = null;
        this.previewActive = false;
        this.measuredHeight = null;
        this.metadataRequest = 0;
        this.buildDOM();

        const panel = this;
        this.widget = node.addDOMWidget(PANEL_WIDGET, "secourses_init_audio_trim", this.root, {
            hideOnZoom: false,
            serialize: false,
            margin: 8,
            getValue: () => "",
            setValue: () => {},
        });
        // Current ComfyUI checks the widget property, not options.serialize.
        this.widget.serialize = false;
        this.widget.computeSize = (width) => [width ?? panel.node.size[0], panel.panelHeight(width)];
        this.placeAfterPlayer();
        hideValueWidget(this.startWidget);
        hideValueWidget(this.endWidget);
        this.bindMedia(this.video);
        if (this.playerWidget?.element) this.bindMedia(this.playerWidget.element);
        this.observeHeight();
        this.hide();
    }

    // ---------- DOM ----------

    buildDOM() {
        this.root = document.createElement("div");
        this.root.className = "secourses-ia-panel";
        this.inner = document.createElement("div");
        this.inner.className = "secourses-ia-inner";
        this.root.appendChild(this.inner);

        this.video = document.createElement("video");
        this.video.className = "secourses-ia-video";
        this.video.controls = true;
        this.video.preload = "metadata";
        this.video.playsInline = true;
        this.video.hidden = true;

        this.note = document.createElement("div");
        this.note.className = "secourses-ia-note";
        this.note.hidden = true;

        this.track = document.createElement("div");
        this.track.className = "secourses-ia-track";
        this.track.title = "Click to seek the player. Drag the handles to choose the part of the file to use.";
        this.fill = document.createElement("div");
        this.fill.className = "secourses-ia-fill";
        this.playhead = document.createElement("div");
        this.playhead.className = "secourses-ia-playhead";
        this.startHandle = document.createElement("div");
        this.startHandle.className = "secourses-ia-handle secourses-ia-handle-start";
        this.startHandle.title = "Drag to set the trim start (the player follows)";
        this.endHandle = document.createElement("div");
        this.endHandle.className = "secourses-ia-handle secourses-ia-handle-end";
        this.endHandle.title = "Drag to set the trim end (the player follows)";
        this.track.append(this.fill, this.playhead, this.startHandle, this.endHandle);

        const rowA = document.createElement("div");
        rowA.className = "secourses-ia-row";
        const makeField = (labelText, titleText) => {
            const label = document.createElement("label");
            label.className = "secourses-ia-label";
            label.append(labelText);
            const field = document.createElement("input");
            field.type = "number";
            field.className = "secourses-ia-input";
            field.min = "0";
            field.step = "0.01";
            field.title = titleText;
            label.appendChild(field);
            rowA.appendChild(label);
            return field;
        };
        this.startInput = makeField("Start", "Trim start in seconds (0 = from the beginning)");
        this.endInput = makeField("End", "Trim end in seconds (empty or the full length = until the end of the file)");
        const makeButton = (parent, text, titleText) => {
            const button = document.createElement("button");
            button.type = "button";
            button.className = "secourses-ia-tool";
            button.textContent = text;
            button.title = titleText;
            parent.appendChild(button);
            return button;
        };
        this.setStartButton = makeButton(rowA, "⇤ Start here", "Set the trim start to the player's current position");
        this.setEndButton = makeButton(rowA, "End here ⇥", "Set the trim end to the player's current position");

        const rowB = document.createElement("div");
        rowB.className = "secourses-ia-row";
        this.previewButton = makeButton(rowB, "▶ Preview part", "Play only the selected part");
        this.resetButton = makeButton(rowB, "↺ Whole file", "Use the whole file (no trim)");
        this.badge = document.createElement("span");
        this.badge.className = "secourses-ia-badge";
        rowB.appendChild(this.badge);

        this.inner.append(this.video, this.note, this.track, rowA, rowB);
        this.bindEvents();
    }

    bindEvents() {
        this.bindHandle(this.startHandle, true);
        this.bindHandle(this.endHandle, false);
        this.track.addEventListener("pointerdown", (event) => {
            if (event.target === this.startHandle || event.target === this.endHandle) return;
            const element = this.activeElement();
            if (!this.duration || !element) return;
            event.preventDefault();
            event.stopPropagation();
            element.currentTime = this.timelineTime(event);
        });
        this.startInput.addEventListener("change", () => {
            const { end } = this.readTrim();
            const value = parseFloat(this.startInput.value);
            this.setRange(Number.isFinite(value) ? value : 0, end, { seek: value });
        });
        this.endInput.addEventListener("change", () => {
            const { start } = this.readTrim();
            const raw = this.endInput.value.trim();
            const value = raw === "" ? null : parseFloat(raw);
            this.setRange(start, value != null && Number.isFinite(value) ? value : null, { seek: value ?? undefined });
        });
        for (const field of [this.startInput, this.endInput]) {
            field.addEventListener("keydown", (event) => {
                if (event.key === "Enter") field.blur();
                if (!(event.ctrlKey || event.metaKey)) event.stopPropagation();
            });
            field.addEventListener("pointerdown", (event) => event.stopPropagation());
        }
        this.setStartButton.addEventListener("click", () => {
            const element = this.activeElement();
            if (!element) return;
            this.setRange(element.currentTime, this.readTrim().end);
        });
        this.setEndButton.addEventListener("click", () => {
            const element = this.activeElement();
            if (!element) return;
            this.setRange(this.readTrim().start, element.currentTime);
        });
        this.previewButton.addEventListener("click", () => {
            const element = this.activeElement();
            if (!element) return;
            const { start } = this.readTrim();
            element.currentTime = start;
            this.previewActive = true;
            element.play?.()?.catch?.(() => {
                this.previewActive = false;
            });
        });
        this.resetButton.addEventListener("click", () => this.setRange(0, null));
        // Buttons must not start a canvas drag.
        for (const button of [this.setStartButton, this.setEndButton, this.previewButton, this.resetButton]) {
            button.addEventListener("pointerdown", (event) => event.stopPropagation());
        }
    }

    bindMedia(element) {
        element.addEventListener("loadedmetadata", () => this.onMetadata(element));
        element.addEventListener("durationchange", () => this.onMetadata(element));
        element.addEventListener("timeupdate", () => this.onTimeUpdate(element));
        element.addEventListener("pause", () => {
            if (element === this.activeElement()) this.previewActive = false;
        });
        element.addEventListener("error", () => this.onMediaError(element));
    }

    observeHeight() {
        if (typeof ResizeObserver === "undefined") return;
        this.resizeObserver = new ResizeObserver(() => {
            if (this.widget.hidden) return;
            const height = this.inner.offsetHeight;
            if (!height) return; // display:none while the widget is off-screen; keep the last size
            const total = height + 2 * this.innerPadding();
            if (total !== this.measuredHeight) {
                this.measuredHeight = total;
                this.refreshLayout();
            }
        });
        this.resizeObserver.observe(this.inner);
    }

    dispose() {
        this.resizeObserver?.disconnect();
        this.resizeObserver = null;
        clearTimeout(this.metadataTimer);
        this.video.pause?.();
        this.video.removeAttribute("src");
        this.video.load?.();
    }

    // ---------- layout ----------

    /** Put the panel right under the player: value widgets stay first, so the index-based widgets_values
     * save/restore keeps working (every serialize:false widget still trails them). */
    placeAfterPlayer() {
        const widgets = this.node.widgets;
        if (!widgets) return;
        const from = widgets.indexOf(this.widget);
        const playerIndex = widgets.indexOf(this.playerWidget);
        if (from === -1 || playerIndex === -1 || from === playerIndex + 1) return;
        widgets.splice(from, 1);
        widgets.splice(widgets.indexOf(this.playerWidget) + 1, 0, this.widget);
    }

    innerPadding() {
        return 6;
    }

    videoHeight(width) {
        return Math.max(90, Math.min(220, Math.round((width - 16) * 9 / 16)));
    }

    estimateHeight(width) {
        let height = 2 * this.innerPadding() + 16 + 6 + 24 + 4 + 24;
        if (this.kind === "video") height += this.videoHeight(width) + 6;
        if (!this.note.hidden) height += 18;
        return height;
    }

    panelHeight(width) {
        if (this.widget?.hidden) return 0;
        return this.measuredHeight ?? this.estimateHeight(width ?? this.node.size[0]);
    }

    refreshLayout() {
        const node = this.node;
        if (!node.graph) return;
        const [, needed] = node.computeSize();
        if (node.size[1] < needed) node.setSize([node.size[0], needed]);
        node.setDirtyCanvas(true, true);
    }

    show() {
        if (!this.widget.hidden) return;
        this.widget.hidden = false;
        this.root.style.display = "";
        this.refreshLayout();
    }

    hide() {
        if (this.widget.hidden) return;
        this.widget.hidden = true;
        this.root.style.display = "none";
        const node = this.node;
        if (node.graph) {
            const [, minimum] = node.computeSize();
            if (node.size[1] > minimum) node.setSize([node.size[0], minimum]);
            node.setDirtyCanvas(true, true);
        }
    }

    setPlayerVisible(visible) {
        const player = this.playerWidget;
        if (!player) return;
        player.hidden = !visible;
        if (player.element) player.element.style.display = visible ? "" : "none";
    }

    // ---------- file state ----------

    activeElement() {
        if (!this.file) return null;
        return this.kind === "video" ? this.video : this.playerWidget?.element ?? null;
    }

    /** Show the panel for `name` (an input-folder audio or video file), or blank everything for the disabled value. */
    setFile(name, { resetTrim = false } = {}) {
        if (!name || name === NO_AUDIO) {
            this.clear();
            return;
        }
        const kind = fileKind(name);
        const changed = name !== this.file || kind !== this.kind;
        this.file = name;
        this.kind = kind;
        if (resetTrim) this.writeTrim(0, null);
        if (changed) {
            this.duration = null;
            this.previewActive = false;
            this.playhead.style.left = "0%";
            this.setNote("Loading duration…");
            // Codecs the browser cannot demux stall without ever firing error; the server probe still knows.
            clearTimeout(this.metadataTimer);
            this.metadataTimer = setTimeout(() => {
                if (this.file === name && !this.duration) void this.fetchServerDuration();
            }, 2500);
        }
        if (kind === "video") {
            this.setPlayerVisible(false);
            const player = this.playerWidget?.element;
            if (player) {
                // The core widget pointed the audio player at the video; one preview element is enough.
                player.pause?.();
                player.removeAttribute("src");
                player.load?.();
            }
            const url = viewURL(name);
            if (changed || !this.video.getAttribute("src")) {
                this.video.src = url;
            }
            this.video.hidden = false;
        } else {
            this.video.pause?.();
            this.video.removeAttribute("src");
            this.video.load?.();
            this.video.hidden = true;
            this.setPlayerVisible(true);
        }
        this.show();
        if (changed) {
            const element = this.activeElement();
            if (element && Number.isFinite(element.duration) && element.duration > 0) this.onMetadata(element);
        }
        this.updateUI();
    }

    clear() {
        this.file = null;
        this.kind = null;
        this.duration = null;
        this.previewActive = false;
        clearTimeout(this.metadataTimer);
        this.video.pause?.();
        this.video.removeAttribute("src");
        this.video.load?.();
        this.video.hidden = true;
        this.setNote("");
        this.setPlayerVisible(true);
        this.hide();
    }

    setNote(text) {
        this.note.textContent = text;
        this.note.hidden = !text;
    }

    onMetadata(element) {
        if (element !== this.activeElement()) return;
        const duration = Number(element.duration);
        if (Number.isFinite(duration) && duration > 0) {
            this.duration = duration;
            this.setNote("");
            this.updateUI();
        } else if (!this.duration) {
            void this.fetchServerDuration();
        }
    }

    onMediaError(element) {
        if (element !== this.activeElement()) return;
        this.setNote("No browser preview for this file — trim by the numbers below; the server decodes it.");
        void this.fetchServerDuration();
    }

    async fetchServerDuration() {
        const file = this.file;
        const request = ++this.metadataRequest;
        try {
            const response = await api.fetchApi(MEDIA_INFO_ROUTE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ files: [file] }),
            });
            if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
            const info = (await response.json())?.[file];
            if (request !== this.metadataRequest || file !== this.file) return;
            const duration = Number(info?.duration);
            if (info?.error) {
                this.setNote(`Cannot read this file: ${info.error}`);
            } else if (Number.isFinite(duration) && duration > 0) {
                this.duration = duration;
                if (this.note.textContent.startsWith("Loading")) this.setNote("");
            } else {
                this.setNote("Duration unknown — trim by the numbers below.");
            }
        } catch (error) {
            if (request !== this.metadataRequest || file !== this.file) return;
            this.setNote("Duration unknown — trim by the numbers below.");
        }
        this.updateUI();
    }

    onTimeUpdate(element) {
        if (element !== this.activeElement()) return;
        const duration = this.duration || element.duration;
        if (Number.isFinite(duration) && duration > 0) {
            const position = Math.min(element.currentTime, duration);
            this.playhead.style.left = `${(position / duration) * 100}%`;
        }
        if (this.previewActive) {
            const { end } = this.readTrim();
            const stop = end ?? this.duration;
            if (stop != null && element.currentTime >= stop - 0.02) {
                element.pause();
                this.previewActive = false;
            }
        }
    }

    // ---------- trim values (the hidden trim_start / trim_end widgets are the source of truth) ----------

    readTrim() {
        const start = Number(this.startWidget?.value);
        const end = Number(this.endWidget?.value);
        return {
            start: Number.isFinite(start) && start > 0 ? start : 0,
            end: Number.isFinite(end) && end > 0 ? end : null,
        };
    }

    writeTrim(start, end) {
        const round = (value) => Math.round(value * 100) / 100;
        const next = { start: round(Math.max(0, start || 0)), end: end == null ? 0 : round(Math.max(0, end)) };
        const apply = (widget, value) => {
            if (!widget || widget.value === value) return false;
            const previous = widget.value;
            widget.value = value;
            widget.callback?.(value, app.canvas, this.node);
            this.node.onWidgetChanged?.(widget.name, value, previous, widget);
            return true;
        };
        const changed = [apply(this.startWidget, next.start), apply(this.endWidget, next.end)].some(Boolean);
        if (changed) scheduleTokenEstimates(this.node);
        this.node.graph?.setDirtyCanvas(true, true);
    }

    timelineTime(event) {
        const rect = this.track.getBoundingClientRect();
        const ratio = rect.width ? (event.clientX - rect.left) / rect.width : 0;
        return Math.max(0, Math.min(1, ratio)) * (this.duration ?? 0);
    }

    bindHandle(handle, isStart) {
        handle.addEventListener("pointerdown", (event) => {
            if (!this.duration) return;
            event.preventDefault();
            event.stopPropagation();
            try {
                handle.setPointerCapture(event.pointerId);
            } catch (error) {
                // Dragging still works without capture; it just stops at the widget edge.
            }
            const move = (moveEvent) => {
                const time = this.timelineTime(moveEvent);
                const { start, end } = this.readTrim();
                const endValue = end ?? this.duration;
                if (isStart) {
                    this.setRange(Math.min(time, endValue - MIN_TRIM_RANGE), end, { seek: time });
                } else {
                    this.setRange(start, Math.max(time, start + MIN_TRIM_RANGE), { seek: time });
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

    /** Clamp and store a window; `end` null means "until the end of the file". */
    setRange(start, end, { seek } = {}) {
        const duration = this.duration;
        start = Number.isFinite(start) ? Math.max(0, start) : 0;
        if (end != null && !Number.isFinite(end)) end = null;
        if (duration) {
            start = Math.min(start, duration);
            if (end != null) {
                end = Math.max(0, Math.min(end, duration));
                if (end >= duration - 0.005) end = null;
            }
            const minRange = Math.min(MIN_TRIM_RANGE, duration);
            const endValue = end ?? duration;
            if (endValue - start < minRange) {
                if (end == null) {
                    start = Math.max(0, duration - minRange);
                } else {
                    end = Math.min(duration, start + minRange);
                    start = Math.max(0, Math.min(start, end - minRange));
                    if (end >= duration - 0.005) end = null;
                }
            }
        } else if (end != null && end <= start) {
            end = start + MIN_TRIM_RANGE;
        }
        this.writeTrim(start, end);
        const element = this.activeElement();
        if (seek != null && Number.isFinite(seek) && element) {
            element.currentTime = Math.max(0, duration ? Math.min(seek, duration) : seek);
        }
        this.updateUI();
    }

    isTrimmed() {
        const { start, end } = this.readTrim();
        return start > 0.005 || end != null;
    }

    updateUI() {
        const { start, end } = this.readTrim();
        const duration = this.duration;
        const known = Number.isFinite(duration) && duration > 0;
        this.track.classList.toggle("secourses-ia-track-disabled", !known);
        if (known) {
            const startPct = Math.max(0, Math.min(100, (start / duration) * 100));
            const endPct = end == null ? 100 : Math.max(0, Math.min(100, (end / duration) * 100));
            this.startHandle.style.left = `${startPct}%`;
            this.endHandle.style.left = `${endPct}%`;
            this.fill.style.left = `${startPct}%`;
            this.fill.style.width = `${Math.max(0, endPct - startPct)}%`;
        } else {
            this.startHandle.style.left = "0%";
            this.endHandle.style.left = "100%";
            this.fill.style.left = "0%";
            this.fill.style.width = "100%";
        }
        if (document.activeElement !== this.startInput) this.startInput.value = start.toFixed(2);
        if (document.activeElement !== this.endInput) {
            this.endInput.value = end != null ? end.toFixed(2) : known ? duration.toFixed(2) : "";
        }
        if (known) this.endInput.max = duration.toFixed(2);
        else this.endInput.removeAttribute("max");
        const trimmed = this.isTrimmed();
        let text;
        if (known) {
            const length = (end ?? duration) - start;
            text = trimmed
                ? `✂ using ${formatSeconds(length)}s of ${formatSeconds(duration)}s`
                : `whole file (${formatSeconds(duration)}s)`;
        } else {
            text = trimmed ? `✂ using ${formatSeconds(start)}s → ${end == null ? "end" : `${formatSeconds(end)}s`}` : "whole file";
        }
        this.badge.textContent = text;
        this.badge.classList.toggle("secourses-ia-badge-active", trimmed);
        const element = this.activeElement();
        const canPlay = !!element && !element.error && (known || Number.isFinite(element.duration));
        this.previewButton.disabled = !canPlay;
        this.setStartButton.disabled = !canPlay;
        this.setEndButton.disabled = !canPlay;
        this.resetButton.disabled = !trimmed;
    }
}

// ==================== Node wiring ====================

function coerceTrimWidgets(node) {
    // Workflows saved before the trim inputs existed restore null into them; treat that as "no trim".
    for (const name of ["trim_start", "trim_end"]) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        const value = widget.value;
        if (!(typeof value === "number" && Number.isFinite(value) && value >= 0)) widget.value = 0;
    }
}

function installInitAudioUi(node) {
    if (node.__secoursesInitAudioInstalled) return;
    const audioWidget = findWidget(node, "audio");
    if (!audioWidget) return;
    node.__secoursesInitAudioInstalled = true;

    installMediaUpload(node);
    const panel = new InitAudioTrimPanel(node);
    node.__secoursesInitAudioPanel = panel;

    const originalCallback = audioWidget.callback;
    audioWidget.callback = function (value) {
        if (value === NO_AUDIO) {
            clearDisabledPlayer(node);
            panel.setFile(null);
            scheduleTokenEstimates(node);
            node.setDirtyCanvas?.(true, true);
            return;
        }
        const result = originalCallback?.apply(this, arguments);
        // A different file gets a fresh (whole-file) window; re-picking the same file keeps its trim.
        panel.setFile(value, { resetTrim: value !== panel.file });
        scheduleTokenEstimates(node);
        return result;
    };
    for (const name of ["trim_start", "trim_end", "duration_mode", "duration_seconds"]) {
        const widget = findWidget(node, name);
        if (!widget) continue;
        const isTrim = name.startsWith("trim_");
        const original = widget.callback;
        widget.callback = function () {
            const result = original?.apply(this, arguments);
            if (isTrim) panel.updateUI();
            scheduleTokenEstimates(node);
            return result;
        };
    }
    // The core upload widget re-points the player at the combo value once the graph is configured
    // (whichever hook order the frontend uses); sync again afterwards so a disabled node keeps a blank
    // player and a selected file gets its preview and trim panel.
    const originalGraphConfigured = node.onGraphConfigured;
    node.onGraphConfigured = function () {
        const result = originalGraphConfigured?.apply(this, arguments);
        syncNode(this);
        setTimeout(() => syncNode(this), 0);
        return result;
    };
    const originalRemoved = node.onRemoved;
    node.onRemoved = function () {
        panel.dispose();
        return originalRemoved?.apply(this, arguments);
    };
    syncNode(node);
}

function syncNode(node) {
    coerceTrimWidgets(node);
    clearDisabledPlayer(node);
    const panel = node.__secoursesInitAudioPanel;
    const audioWidget = findWidget(node, "audio");
    if (panel && audioWidget) panel.setFile(audioWidget.value);
}

app.registerExtension({
    name: "SECourses.InitAudio",
    init() {
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.href = new URL("./secourses_init_audio.css", import.meta.url).href;
        document.head.appendChild(link);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        addAudioPlayer(nodeData);
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            installInitAudioUi(this);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            installInitAudioUi(this);
            syncNode(this);
            setTimeout(() => syncNode(this), 0);
        });
    },
});
