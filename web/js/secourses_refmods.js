import { app } from "../../../scripts/app.js";
import "./refmod_picker.js";

app.registerExtension({
    name: "SECourses.H3RefMods",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "SECoursesH3RefModStack") return;
        let created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            let selection = this.widgets.find(w => w.name === "selection");
            selection.hidden = true;
            selection.computeSize = () => [0, -4];
            if (selection.inputEl) selection.inputEl.style.display = "none";
            let root = document.createElement("div");
            let widget = this.addDOMWidget("refmods_ui", "refmods", root, { hideOnZoom: false, selectOn: [] });
            widget.serialize = false;
            widget.computeSize = width => [width, Math.min(620, 155 + (this.__refmods?.rows().length || 0) * 86)];
            this.__refmods = new globalThis.SECoursesRefModPicker(root, {
                getValue: () => selection.value,
                setValue: value => { selection.value = value; this.graph?.change(); },
                onResize: () => { this.setSize([Math.max(490, this.size[0]), Math.min(690, Math.max(220, 215 + (this.__refmods?.rows().length || 0) * 86))]); this.setDirtyCanvas(true, true); },
            });
        };
        let configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            configure?.apply(this, arguments);
            this.__refmods?.render();
        };
    },
});
