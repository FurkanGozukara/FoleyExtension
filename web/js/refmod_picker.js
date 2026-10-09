/** Shared RefMod selector for ComfyUI and SwarmUI. Values are portable backend filenames. */
globalThis.SECoursesRefModPicker = class {
    constructor(root, { base = "", getValue, setValue, onResize = () => {} }) {
        this.root = root;
        this.base = base;
        this.getValue = getValue;
        this.setValue = setValue;
        this.onResize = onResize;
        this.files = [];
        this.message = "";
        root.classList.add("secourses-refmod-picker");
        if (!document.getElementById("secourses-refmod-picker-style")) {
            let style = document.createElement("style");
            style.id = "secourses-refmod-picker-style";
            style.textContent = `.secourses-refmod-picker{box-sizing:border-box;padding:10px;background:#20242b;color:#e4e7ed;font:13px/1.5 system-ui;border-radius:8px;overflow:auto;min-width:0}.secourses-refmod-picker *{box-sizing:border-box}.secourses-refmod-picker button,.secourses-refmod-picker select,.secourses-refmod-picker input{color:#e4e7ed;background:#303641;border:1px solid #536174;border-radius:4px;padding:5px;font:inherit;min-width:0}.secourses-refmod-picker button{cursor:pointer}.secourses-refmod-picker button:hover{background:#465365}.secourses-refmod-picker .refmod-toolbar{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.secourses-refmod-picker .refmod-row{display:grid;grid-template-columns:22px 1fr 60px;gap:6px;padding:7px 0;border-bottom:1px solid #435064}.secourses-refmod-picker .refmod-details{grid-column:2/4;display:flex;gap:5px;align-items:center;flex-wrap:wrap}.secourses-refmod-picker .refmod-status{font-size:12px;color:#b8c6d8;overflow-wrap:anywhere;margin-top:8px}.secourses-refmod-picker select{width:100%}.secourses-refmod-picker .refmod-details select{width:auto}.secourses-refmod-picker input[type=number]{width:60px}.secourses-refmod-picker input[type=checkbox]{width:16px}.secourses-refmod-picker .refmod-label{color:#9ed8f7}`;
            document.head.append(style);
        }
        this.render();
        this.refresh();
    }
    rows() {
        let value = this.getValue() || "[]";
        try {
            let rows = value.trim().startsWith("[") ? JSON.parse(value) : value.split("\n").filter(s => s.trim());
            return rows.map(row => typeof row == "string" ? { file: row, enabled: true, strength: 1, components: "all" } : row);
        } catch (error) {
            this.message = `Selection needs valid JSON or one filename per line: ${error.message}`;
            return [];
        }
    }
    save(rows) {
        this.setValue(JSON.stringify(rows));
        this.render();
    }
    button(text, action, title = text) {
        let button = document.createElement("button");
        button.type = "button";
        button.textContent = text;
        button.title = title;
        button.addEventListener("click", action);
        return button;
    }
    async refresh() {
        try {
            let response = await fetch(`${this.base}/secourses/h3/refmods`);
            if (!response.ok) throw new Error(await response.text());
            let data = await response.json();
            this.files = data.files;
            this.message = data.errors.length ? data.errors.map(e => `${e.file}: ${e.error}`).join("; ") : `${data.files.length} files · ${data.folders.join("; ")}`;
        } catch (error) {
            this.message = `Library unavailable: ${error.message}`;
        }
        this.render();
    }
    async upload(files) {
        try {
            for (let file of files) {
                this.message = `Importing ${file.name}…`;
                this.render();
                let form = new FormData();
                form.append("file", file, file.name);
                let response = await fetch(`${this.base}/secourses/h3/refmods/upload`, { method: "POST", body: form });
                if (!response.ok) throw new Error(await response.text());
                let result = await response.json();
                this.save([...this.rows(), { file: result.file, enabled: true, strength: 1, components: "all" }]);
            }
            await this.refresh();
        } catch (error) {
            this.message = `Import failed: ${error.message}`;
            this.render();
        }
    }
    render() {
        this.root.replaceChildren();
        let rows = this.rows();
        let toolbar = document.createElement("div");
        toolbar.className = "refmod-toolbar";
        toolbar.append(this.button("+ Add RefMod", () => this.save([...this.rows(), { file: "", enabled: true, strength: 1, components: "all" }])));
        toolbar.append(this.button("Refresh", () => this.refresh()));
        toolbar.append(this.button("Import files", () => {
            let input = document.createElement("input");
            input.type = "file";
            input.accept = ".safetensors";
            input.multiple = true;
            input.onchange = () => this.upload(input.files);
            input.click();
        }));
        this.root.append(toolbar);
        for (let [index, row] of rows.entries()) {
            let holder = document.createElement("div");
            holder.className = "refmod-row";
            let enabled = document.createElement("input");
            enabled.type = "checkbox";
            enabled.checked = row.enabled !== false;
            enabled.setAttribute("aria-label", `Enable RefMod ${index + 1}`);
            enabled.onchange = () => { row.enabled = enabled.checked; this.save(rows); };
            let select = document.createElement("select");
            select.setAttribute("aria-label", `RefMod ${index + 1} file`);
            select.add(new Option("None — optional", ""));
            for (let file of this.files) select.add(new Option(`${file.file} (${file.kinds.join(" + ")})`, file.file));
            if (row.file && !this.files.some(f => f.file == row.file)) select.add(new Option(`${row.file} (not in library)`, row.file));
            select.value = row.file || "";
            select.onchange = () => { row.file = select.value; this.save(rows); };
            let strength = document.createElement("input");
            strength.type = "number";
            strength.min = "0"; strength.max = "1"; strength.step = "0.05";
            strength.value = row.strength ?? 1;
            strength.title = "Reference detail retention, 0 disables the row";
            strength.setAttribute("aria-label", `RefMod ${index + 1} strength`);
            strength.onchange = () => { row.strength = Math.max(0, Math.min(1, Number(strength.value))); this.save(rows); };
            let details = document.createElement("div");
            details.className = "refmod-details";
            let label = document.createElement("span");
            label.className = "refmod-label";
            label.textContent = `@refmod${index + 1}`;
            let components = document.createElement("select");
            components.setAttribute("aria-label", `RefMod ${index + 1} components`);
            for (let kind of ["all", "visual", "audio"]) components.add(new Option(kind, kind));
            components.value = row.components || "all";
            components.onchange = () => { row.components = components.value; this.save(rows); };
            details.append(label, components, this.button("↑", () => {
                if (index > 0) { [rows[index - 1], rows[index]] = [rows[index], rows[index - 1]]; this.save(rows); }
            }, "Move up"), this.button("↓", () => {
                if (index < rows.length - 1) { [rows[index + 1], rows[index]] = [rows[index], rows[index + 1]]; this.save(rows); }
            }, "Move down"), this.button("Remove", () => { rows.splice(index, 1); this.save(rows); }));
            holder.append(enabled, select, strength, details);
            this.root.append(holder);
        }
        let help = document.createElement("div");
        help.className = "refmod-status";
        help.textContent = rows.length ? "Mention @refmod1, @refmod2… in the prompt. A bundle may contain several numbered references. More references use more VRAM." : "Optional: no RefMods selected. The preset runs normally.";
        let status = document.createElement("div");
        status.className = "refmod-status";
        let tokens = rows.reduce((sum, row) => {
            if (row.enabled === false || Number(row.strength ?? 1) <= 0) return sum;
            let file = this.files.find(f => f.file === row.file);
            let kind = row.components || "all";
            return sum + (kind !== "audio" ? file?.tokens?.visual || 0 : 0) + (kind !== "visual" ? file?.tokens?.audio || 0 : 0);
        }, 0);
        status.textContent = (tokens ? `RefMods: ${tokens.toLocaleString()} reference tokens, additional to the gallery estimate. ` : "") + this.message;
        this.root.append(help, status);
        this.onResize();
    }
};
