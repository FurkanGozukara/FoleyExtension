const GALLERY_CLASS = "SECoursesReferenceGallery";
const LEGACY_MERGE_CLASSES = new Set([
    "SECoursesBatchVideoMerge",
    "SECoursesBatchAudioMerge",
]);
const SEQUENTIAL_OUTPUT_CLASSES = new Set([
    "SECoursesBatchVideoSaveMerge",
    "SECoursesBatchAudioSaveMerge",
]);

let fallbackRunCounter = 0;

export function createBatchRunId(cryptoProvider = globalThis.crypto) {
    if (typeof cryptoProvider?.randomUUID === "function") {
        return cryptoProvider.randomUUID();
    }
    if (typeof cryptoProvider?.getRandomValues === "function") {
        const bytes = new Uint8Array(16);
        cryptoProvider.getRandomValues(bytes);
        return `batch_${[...bytes].map((value) => value.toString(16).padStart(2, "0")).join("")}`;
    }

    fallbackRunCounter += 1;
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).slice(2);
    return `batch_${timestamp}_${fallbackRunCounter.toString(36)}_${random}`;
}

export function folderBatchTargets(promptOutput) {
    return Object.entries(promptOutput ?? {})
        .filter(([, node]) => node?.class_type === GALLERY_CLASS)
        .map(([nodeId, node]) => ({
            nodeId,
            batchFolder: String(node.inputs?.batch_folder ?? "").trim(),
        }))
        .filter((target) => target.batchFolder.length > 0);
}

export function supportsSequentialFolderQueue(promptOutput) {
    const classes = new Set(
        Object.values(promptOutput ?? {}).map((node) => node?.class_type),
    );
    const hasSequentialOutput = [...SEQUENTIAL_OUTPUT_CLASSES].some((name) => classes.has(name));
    const hasLegacyMerge = [...LEGACY_MERGE_CLASSES].some((name) => classes.has(name));
    return hasSequentialOutput || !hasLegacyMerge;
}

export function buildSequentialBatchPlan(promptCount, repetitions, makeRunId) {
    const count = Number(promptCount);
    const runs = Number(repetitions);
    if (!Number.isInteger(count) || count < 1) {
        throw new Error("Folder batch inspection returned an invalid prompt count.");
    }
    if (!Number.isInteger(runs) || runs < 1) {
        throw new Error("Folder batch queue count must be a positive integer.");
    }
    const plan = [];
    for (let repetition = 0; repetition < runs; repetition++) {
        const runId = makeRunId();
        for (let itemIndex = 0; itemIndex < count; itemIndex++) {
            plan.push({
                runId,
                itemIndex,
                itemCount: count,
            });
        }
    }
    return plan;
}

export function injectSequentialBatchItem(promptOutput, targetNodeIds, item) {
    for (const nodeId of targetNodeIds) {
        const node = promptOutput?.[nodeId];
        if (!node?.inputs) {
            throw new Error(`Folder batch gallery node ${nodeId} is missing from the queued prompt.`);
        }
        node.inputs.batch_run_id = item.runId;
        node.inputs.batch_item_index = item.itemIndex;
        node.inputs.batch_item_count = item.itemCount;
    }
    return promptOutput;
}
