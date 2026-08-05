import test from "node:test";
import assert from "node:assert/strict";

import { resolveConfiguredPrompt } from "../web/js/secourses_reference_gallery_state.mjs";

test("initial workflow configuration hydrates the saved prompt", () => {
    const result = resolveConfiguredPrompt("", "preset prompt", false, false);

    assert.deepEqual(result, {
        value: "preset prompt",
        hydrateFromWidget: true,
    });
});

test("attachment reconfiguration cannot reset an edited prompt", () => {
    const result = resolveConfiguredPrompt(
        "user edited prompt with @image1",
        "preset prompt",
        true,
        true,
    );

    assert.deepEqual(result, {
        value: "user edited prompt with @image1",
        hydrateFromWidget: false,
    });
});

test("a first configure pass preserves text entered on a new node", () => {
    const result = resolveConfiguredPrompt(
        "new prompt typed before adding a reference",
        "",
        false,
        true,
    );

    assert.deepEqual(result, {
        value: "new prompt typed before adding a reference",
        hydrateFromWidget: false,
    });
});
