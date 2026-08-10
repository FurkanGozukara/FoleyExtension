function portablePath(value) {
    return value.replaceAll("\\", "/");
}

export function reconcilePortableComboPaths(node) {
    let changed = false;

    for (const widget of node?.widgets ?? []) {
        const values = widget?.options?.values;
        if (!Array.isArray(values) || typeof widget.value !== "string" || values.includes(widget.value)) {
            continue;
        }

        const portableValue = portablePath(widget.value);
        const matches = values.filter(
            (value) => typeof value === "string" && portablePath(value) === portableValue,
        );
        if (matches.length === 1) {
            widget.value = matches[0];
            changed = true;
        }
    }

    return changed;
}
