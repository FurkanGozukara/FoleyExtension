import assert from "node:assert/strict";
import test from "node:test";

import {
    buildSequentialBatchPlan,
    createBatchRunId,
    folderBatchTargets,
    injectSequentialBatchItem,
    supportsSequentialFolderQueue,
} from "../web/js/secourses_folder_batch_queue.mjs";

test("creates a portable run ID when randomUUID is unavailable", () => {
    const provider = {
        getRandomValues(bytes) {
            bytes.forEach((_, index) => { bytes[index] = index; });
            return bytes;
        },
    };

    assert.equal(
        createBatchRunId(provider),
        "batch_000102030405060708090a0b0c0d0e0f",
    );
});

test("finds active gallery folder paths", () => {
    const output = {
        10: { class_type: "SECoursesReferenceGallery", inputs: { batch_folder: " C:/batch " } },
        11: { class_type: "SECoursesReferenceGallery", inputs: { batch_folder: "" } },
    };

    assert.deepEqual(folderBatchTargets(output), [
        { nodeId: "10", batchFolder: "C:/batch" },
    ]);
});

test("builds separate ordered jobs for every prompt", () => {
    let run = 0;
    const plan = buildSequentialBatchPlan(3, 2, () => `run_${++run}`);

    assert.deepEqual(plan, [
        { runId: "run_1", itemIndex: 0, itemCount: 3 },
        { runId: "run_1", itemIndex: 1, itemCount: 3 },
        { runId: "run_1", itemIndex: 2, itemCount: 3 },
        { runId: "run_2", itemIndex: 0, itemCount: 3 },
        { runId: "run_2", itemIndex: 1, itemCount: 3 },
        { runId: "run_2", itemIndex: 2, itemCount: 3 },
    ]);
});

test("injects only the current folder item into every gallery target", () => {
    const output = {
        10: { class_type: "SECoursesReferenceGallery", inputs: {} },
        "20:30": { class_type: "SECoursesReferenceGallery", inputs: {} },
    };
    injectSequentialBatchItem(output, ["10", "20:30"], {
        runId: "run_12345678",
        itemIndex: 1,
        itemCount: 4,
    });

    for (const node of Object.values(output)) {
        assert.equal(node.inputs.batch_run_id, "run_12345678");
        assert.equal(node.inputs.batch_item_index, 1);
        assert.equal(node.inputs.batch_item_count, 4);
    }
});

test("uses sequential queuing for combined outputs but not legacy merge graphs", () => {
    assert.equal(supportsSequentialFolderQueue({
        1: { class_type: "SECoursesBatchVideoSaveMerge" },
    }), true);
    assert.equal(supportsSequentialFolderQueue({
        1: { class_type: "SECoursesBatchVideoMerge" },
    }), false);
    assert.equal(supportsSequentialFolderQueue({
        1: { class_type: "SaveImage" },
    }), true);
});
