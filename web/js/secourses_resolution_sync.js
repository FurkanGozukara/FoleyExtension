/** Keep the SECourses resolution widgets synchronized in both directions. */

import { app } from "../../../scripts/app.js";

const NODE_CLASS = "SECoursesResolutionSync";
const CUSTOM_ASPECT = "Custom";
const PIXELS_PER_MEGAPIXEL = 1024 * 1024;

const ASPECT_RATIOS = new Map([
    ["1:1 (Square)", [1, 1]],
    ["2:3 (Portrait Photo)", [2, 3]],
    ["3:2 (Photo)", [3, 2]],
    ["3:4 (Portrait Standard)", [3, 4]],
    ["4:3 (Standard)", [4, 3]],
    ["9:16 (Portrait Widescreen)", [9, 16]],
    ["16:9 (Widescreen)", [16, 9]],
    ["21:9 (Ultrawide)", [21, 9]],
]);

function chainCallback(object, property, callback) {
    const original = object[property];
    object[property] = function () {
        original?.apply(this, arguments);
        callback.apply(this, arguments);
    };
}

function finiteNumber(value, fallback) {
    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
}

function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
}

function snapDimension(value, multiple) {
    const safeMultiple = clamp(Math.round(finiteNumber(multiple, 1)), 1, 128);
    const snapped = Math.round(finiteNumber(value, safeMultiple) / safeMultiple) * safeMultiple;
    return clamp(snapped, safeMultiple, 16384);
}

function roundedMegapixels(width, height) {
    return Math.max(0.01, Math.round((width * height / PIXELS_PER_MEGAPIXEL) * 100) / 100);
}

function closestAspect(width, height) {
    const actual = width / height;
    let closest = CUSTOM_ASPECT;
    let smallestError = Number.POSITIVE_INFINITY;
    for (const [label, [ratioWidth, ratioHeight]] of ASPECT_RATIOS) {
        const expected = ratioWidth / ratioHeight;
        const relativeError = Math.abs(actual - expected) / expected;
        if (relativeError < smallestError) {
            smallestError = relativeError;
            closest = label;
        }
    }
    return smallestError <= 0.035 ? closest : CUSTOM_ASPECT;
}

function dimensionsFor(megapixels, ratio, multiple) {
    const targetPixels = clamp(finiteNumber(megapixels, 0.4), 0.01, 64) * PIXELS_PER_MEGAPIXEL;
    const [ratioWidth, ratioHeight] = ratio;
    const scale = Math.sqrt(targetPixels / (ratioWidth * ratioHeight));
    return [
        snapDimension(ratioWidth * scale, multiple),
        snapDimension(ratioHeight * scale, multiple),
    ];
}

function installResolutionSync(node) {
    if (node.__secoursesResolutionSyncInstalled) return;

    const widgets = Object.fromEntries((node.widgets ?? []).map((widget) => [widget.name, widget]));
    const aspectWidget = widgets.aspect_ratio;
    const megapixelsWidget = widgets.megapixels;
    const widthWidget = widgets.width;
    const heightWidget = widgets.height;
    const multipleWidget = widgets.multiple;
    if (!aspectWidget || !megapixelsWidget || !widthWidget || !heightWidget || !multipleWidget) return;

    node.__secoursesResolutionSyncInstalled = true;
    let synchronizing = false;

    const updateValues = (updates) => {
        let changed = false;
        synchronizing = true;
        try {
            for (const [widget, value] of updates) {
                if (widget.value === value) continue;
                widget.value = value;
                changed = true;
            }
        } finally {
            synchronizing = false;
        }
        if (changed) {
            node.setDirtyCanvas?.(true, true);
            node.graph?.setDirtyCanvas?.(true, true);
        }
    };

    const syncFromPreset = () => {
        if (synchronizing) return;
        let ratio = ASPECT_RATIOS.get(aspectWidget.value);
        if (!ratio) {
            ratio = [
                Math.max(1, finiteNumber(widthWidget.value, 1)),
                Math.max(1, finiteNumber(heightWidget.value, 1)),
            ];
        }
        const [width, height] = dimensionsFor(megapixelsWidget.value, ratio, multipleWidget.value);
        updateValues([[widthWidget, width], [heightWidget, height]]);
    };

    const syncFromDimensions = () => {
        if (synchronizing) return;
        const multiple = multipleWidget.value;
        const width = snapDimension(widthWidget.value, multiple);
        const height = snapDimension(heightWidget.value, multiple);
        updateValues([
            [widthWidget, width],
            [heightWidget, height],
            [megapixelsWidget, roundedMegapixels(width, height)],
            [aspectWidget, closestAspect(width, height)],
        ]);
    };

    const wrapWidget = (widget, synchronize) => {
        const original = widget.callback;
        widget.callback = function () {
            original?.apply(this, arguments);
            synchronize();
        };
    };

    wrapWidget(aspectWidget, syncFromPreset);
    wrapWidget(megapixelsWidget, syncFromPreset);
    wrapWidget(widthWidget, syncFromDimensions);
    wrapWidget(heightWidget, syncFromDimensions);
    wrapWidget(multipleWidget, syncFromDimensions);
}

app.registerExtension({
    name: "SECourses.ResolutionSync",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            installResolutionSync(this);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            installResolutionSync(this);
        });
    },
});
