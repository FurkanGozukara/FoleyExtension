/** Keep the optional image preview consistent with its disabled value. */

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "SECoursesOptionalImage";
const NO_IMAGE = "(none - disabled)";

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
        if (nodeData.name !== NODE_CLASS) return;
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            installOptionalImageUi(this);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            installOptionalImageUi(this);
            clearDisabledPreview(this);
        });
    },
});
