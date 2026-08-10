import assert from "node:assert/strict";
import test from "node:test";

import { reconcilePortableComboPaths } from "../web/js/secourses_portable_combo_paths.mjs";

function combo(value, values) {
    return { type: "combo", value, options: { values } };
}

test("matches Windows preset paths to Linux combo values", () => {
    const widget = combo(
        "MiniMaxH3\\minimax_h3_video_vae_fp16.safetensors",
        ["MiniMaxH3/minimax_h3_video_vae_fp16.safetensors"],
    );

    assert.equal(reconcilePortableComboPaths({ widgets: [widget] }), true);
    assert.equal(widget.value, "MiniMaxH3/minimax_h3_video_vae_fp16.safetensors");
});

test("matches Linux preset paths to Windows combo values", () => {
    const widget = combo(
        "MiniMaxH3/minimax_h3_audio_vae_fp32.safetensors",
        ["MiniMaxH3\\minimax_h3_audio_vae_fp32.safetensors"],
    );

    assert.equal(reconcilePortableComboPaths({ widgets: [widget] }), true);
    assert.equal(widget.value, "MiniMaxH3\\minimax_h3_audio_vae_fp32.safetensors");
});

test("leaves valid and unrelated combo values unchanged", () => {
    const valid = combo("euler", ["euler", "dpmpp_2m"]);
    const missing = combo("Other\\missing.safetensors", ["MiniMaxH3/model.safetensors"]);

    assert.equal(reconcilePortableComboPaths({ widgets: [valid, missing] }), false);
    assert.equal(valid.value, "euler");
    assert.equal(missing.value, "Other\\missing.safetensors");
});

test("does not guess when normalized options are ambiguous", () => {
    const widget = combo("MiniMaxH3\\folder/model.safetensors", [
        "MiniMaxH3/folder/model.safetensors",
        "MiniMaxH3\\folder\\model.safetensors",
    ]);

    assert.equal(reconcilePortableComboPaths({ widgets: [widget] }), false);
    assert.equal(widget.value, "MiniMaxH3\\folder/model.safetensors");
});
