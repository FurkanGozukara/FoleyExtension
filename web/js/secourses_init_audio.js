/** Init Audio (Optional, Auto Enable): audio player + upload button, blank while disabled. */

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "SECoursesInitAudio";
const NO_AUDIO = "(none - disabled)";

function chainCallback(object, property, callback) {
    const original = object[property];
    object[property] = function () {
        const result = original?.apply(this, arguments);
        callback.apply(this, arguments);
        return result;
    };
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

function clearDisabledPlayer(node) {
    const audioWidget = node.widgets?.find((widget) => widget.name === "audio");
    const player = node.widgets?.find((widget) => widget.name === "audioUI");
    if (!audioWidget || !player || audioWidget.value !== NO_AUDIO) return;
    if (player.element) {
        player.element.removeAttribute("src");
        player.element.classList.add("empty-audio-widget");
    }
    player.value = "";
}

function installInitAudioUi(node) {
    if (node.__secoursesInitAudioInstalled) return;
    const audioWidget = node.widgets?.find((widget) => widget.name === "audio");
    if (!audioWidget) return;
    node.__secoursesInitAudioInstalled = true;

    const originalCallback = audioWidget.callback;
    audioWidget.callback = function (value) {
        if (value === NO_AUDIO) {
            clearDisabledPlayer(node);
            node.setDirtyCanvas?.(true, true);
            return;
        }
        return originalCallback?.apply(this, arguments);
    };
    // The core upload widget re-points the player at the combo value once the graph is configured
    // (whichever hook order the frontend uses); clear again afterwards so a disabled node keeps a blank player.
    const originalGraphConfigured = node.onGraphConfigured;
    node.onGraphConfigured = function () {
        const result = originalGraphConfigured?.apply(this, arguments);
        clearDisabledPlayer(this);
        setTimeout(() => clearDisabledPlayer(this), 0);
        return result;
    };
    clearDisabledPlayer(node);
}

app.registerExtension({
    name: "SECourses.InitAudio",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        addAudioPlayer(nodeData);
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            installInitAudioUi(this);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            installInitAudioUi(this);
            clearDisabledPlayer(this);
            setTimeout(() => clearDisabledPlayer(this), 0);
        });
    },
});
