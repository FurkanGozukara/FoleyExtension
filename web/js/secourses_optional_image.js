/** Keep the optional image preview consistent with its disabled value. */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const OPTIONAL_NODE_CLASS = "SECoursesOptionalImage";
const INIT_MEDIA_NODE_CLASS = "SECoursesLoadImage";
const NO_IMAGE = "(none - disabled)";
const VIDEO_EXTENSIONS = new Set([
    "3g2", "3gp", "avi", "f4v", "flv", "m2ts", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "mts",
    "ogv", "rm", "ts", "vob", "webm", "wmv",
]);

function chainCallback(object, property, callback) {
    const original = object[property];
    object[property] = function () {
        const result = original?.apply(this, arguments);
        callback.apply(this, arguments);
        return result;
    };
}

function clearDisabledPreview(node) {
    const imageWidget = node.widgets?.find((widget) => widget.name === "image");
    if (!imageWidget || imageWidget.value !== NO_IMAGE) return;

    const nodeId = String(node.id);
    delete app.nodeOutputs?.[nodeId];
    delete app.nodePreviewImages?.[nodeId];
    node.imgs = null;
    node.images = null;
    node.imageIndex = null;
    node.preview = null;
}

function extensionOf(name) {
    const clean = String(name ?? "").replace(/ \[(input|output|temp)\]$/i, "");
    const base = clean.split(/[\\/]/).pop() ?? "";
    const dot = base.lastIndexOf(".");
    return dot === -1 ? "" : base.slice(dot + 1).toLowerCase();
}

function isVideo(name) {
    return VIDEO_EXTENSIONS.has(extensionOf(name));
}

function isInitMediaFile(file) {
    const type = file?.type ?? "";
    return type.startsWith("image/") || type.startsWith("video/") || isVideo(file?.name);
}

function viewURL(name) {
    const clean = String(name ?? "").replace(/ \[(input|output|temp)\]$/i, "");
    const slash = clean.lastIndexOf("/");
    const subfolder = slash === -1 ? "" : clean.substring(0, slash);
    const filename = slash === -1 ? clean : clean.substring(slash + 1);
    const rand = app.getRandParam?.() ?? "";
    return api.apiURL(
        `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}${rand}`,
    );
}

function clearImagePreview(node) {
    const nodeId = String(node.id);
    delete app.nodeOutputs?.[nodeId];
    delete app.nodePreviewImages?.[nodeId];
    node.imgs = null;
    node.images = null;
    node.imageIndex = null;
    node.preview = null;
}

async function uploadInitMedia(node, file) {
    const imageWidget = node.widgets?.find((widget) => widget.name === "image");
    if (!imageWidget || !file || !isInitMediaFile(file) || node.isUploading) return false;
    node.isUploading = true;
    const previous = imageWidget.value;
    try {
        const body = new FormData();
        body.append("image", file);
        body.append("type", "input");
        const response = await api.fetchApi("/upload/image", { method: "POST", body });
        if (response.status !== 200) throw new Error(`${response.status} - ${response.statusText}`);
        const data = await response.json();
        const name = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
        const values = imageWidget.options.values;
        if (Array.isArray(values) && !values.includes(name)) values.push(name);
        imageWidget.value = name;
        imageWidget.callback?.(name);
        node.onWidgetChanged?.(imageWidget.name, name, previous, imageWidget);
        return true;
    } catch (error) {
        app.extensionManager?.toast?.add?.({
            severity: "error",
            summary: "Init media upload failed",
            detail: String(error?.message ?? error),
            life: 6000,
        });
        return false;
    } finally {
        node.isUploading = false;
        node.graph?.setDirtyCanvas(true, true);
    }
}

function installInitMediaUi(node) {
    if (node.__secoursesInitMediaInstalled) return;
    const imageWidget = node.widgets?.find((widget) => widget.name === "image");
    if (!imageWidget) return;
    node.__secoursesInitMediaInstalled = true;

    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.style.width = "100%";
    video.style.aspectRatio = "16 / 9";
    video.style.objectFit = "contain";
    video.style.background = "#000";
    const previewWidget = node.addDOMWidget?.("init_video_preview", "init_video_preview", video, {
        hideOnZoom: false,
        serialize: false,
        getValue: () => "",
        setValue: () => {},
    });
    if (previewWidget) {
        previewWidget.serialize = false;
        previewWidget.computeSize = (width) => [width ?? node.size[0], Math.max(140, Math.round((width ?? 320) * 9 / 16))];
    }

    const syncPreview = () => {
        const selectedVideo = isVideo(imageWidget.value);
        if (previewWidget) previewWidget.hidden = !selectedVideo;
        if (selectedVideo) {
            clearImagePreview(node);
            const selected = String(imageWidget.value ?? "");
            if (video.dataset.file !== selected) {
                video.dataset.file = selected;
                video.src = viewURL(selected);
            }
        } else {
            video.pause?.();
            delete video.dataset.file;
            video.removeAttribute("src");
            video.load?.();
        }
        node.setDirtyCanvas?.(true, true);
        node.graph?.setDirtyCanvas?.(true, true);
    };

    const originalCallback = imageWidget.callback;
    imageWidget.callback = function (value) {
        const result = isVideo(value) ? undefined : originalCallback?.apply(this, arguments);
        syncPreview();
        return result;
    };

    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*,video/*";
    input.onchange = () => {
        const file = input.files?.[0];
        input.value = "";
        if (file) void uploadInitMedia(node, file);
    };
    const uploadWidget = node.widgets?.find((widget) => widget.name === "upload");
    if (uploadWidget) {
        uploadWidget.callback = () => input.click();
        uploadWidget.label = "choose image / video to upload";
    }

    const hasMediaFile = (event) => Array.from(event?.dataTransfer?.files ?? []).some(isInitMediaFile);
    node.onDragOver = (event) => hasMediaFile(event);
    node.onDragDrop = async (event) => {
        const file = Array.from(event?.dataTransfer?.files ?? []).find(isInitMediaFile);
        if (!file) return false;
        await uploadInitMedia(node, file);
        return true;
    };
    node.pasteFiles = (files) => {
        const file = Array.from(files ?? []).find(isInitMediaFile);
        if (!file) return false;
        void uploadInitMedia(node, file);
        return true;
    };

    const originalGraphConfigured = node.onGraphConfigured;
    node.onGraphConfigured = function () {
        const result = originalGraphConfigured?.apply(this, arguments);
        syncPreview();
        setTimeout(syncPreview, 0);
        return result;
    };
    const originalRemoved = node.onRemoved;
    node.onRemoved = function () {
        input.onchange = null;
        video.pause?.();
        video.removeAttribute("src");
        return originalRemoved?.apply(this, arguments);
    };
    syncPreview();
}

function installOptionalImageUi(node) {
    if (node.__secoursesOptionalImageInstalled) return;

    const imageWidget = node.widgets?.find((widget) => widget.name === "image");
    if (!imageWidget) return;

    node.__secoursesOptionalImageInstalled = true;
    const originalWidgetCallback = imageWidget.callback;
    imageWidget.callback = function (value) {
        if (value === NO_IMAGE) {
            clearDisabledPreview(node);
            node.setDirtyCanvas?.(true, true);
            node.graph?.setDirtyCanvas?.(true, true);
            return;
        }
        return originalWidgetCallback?.apply(this, arguments);
    };

    const originalDrawBackground = node.onDrawBackground;
    node.onDrawBackground = function () {
        clearDisabledPreview(this);
        return originalDrawBackground?.apply(this, arguments);
    };

    clearDisabledPreview(node);
}

app.registerExtension({
    name: "SECourses.OptionalImage",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === OPTIONAL_NODE_CLASS) {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                installOptionalImageUi(this);
            });
            chainCallback(nodeType.prototype, "onConfigure", function () {
                installOptionalImageUi(this);
                clearDisabledPreview(this);
            });
        }
        if (nodeData.name === INIT_MEDIA_NODE_CLASS) {
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                installInitMediaUi(this);
            });
            chainCallback(nodeType.prototype, "onConfigure", function () {
                installInitMediaUi(this);
            });
        }
    },
});
