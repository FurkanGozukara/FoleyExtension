import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

// The token model is a classic script (shared byte-for-byte with the SwarmUI extension); it
// publishes a global instead of exporting.
import "../web/js/minimax_h3_tokens.js";

const H3 = globalThis.SECoursesMiniMaxH3Tokens;
const fixtures = JSON.parse(readFileSync(new URL("./fixtures_minimax_h3_tokens.json", import.meta.url), "utf8"));

test("publishes the shared model", () => {
    assert.ok(H3);
    assert.equal(H3.BUDGET_TOKENS, 109062);
    assert.equal(H3.SAGE_ATTENTION_MAX_TOKENS, 299593);
});

test("frame grid and latent shapes match ComfyUI", () => {
    assert.equal(H3.alignFrames(120), 124);
    assert.equal(H3.alignFrames(124), 124);
    assert.equal(H3.alignFrames(1), 5);
    assert.equal(H3.alignFrames(360), 362);
    assert.equal(H3.alignFramesDown(360), 345);
    assert.equal(H3.videoLatentT(124), 37);
    assert.equal(H3.videoLatentT(5), 2);
    assert.equal(H3.audioLatentT(124), 207);
    assert.equal(H3.audioLatentForSeconds(8), 320);
    assert.equal(H3.audioLatentForSeconds(3.37), 135);
    assert.equal(H3.patchRows(1344, 768), 1008);
    assert.equal(H3.patchRows(864, 480), 405);
    assert.equal(H3.patchRows(1000, 560), Math.ceil(62 / 2) * Math.ceil(35 / 2));
    assert.equal(H3.pyRound(2.5), 2);
    assert.equal(H3.pyRound(3.5), 4);
    assert.equal(H3.pyRound(2.51), 3);
});

test("label and text token heuristics", () => {
    assert.equal(H3.labelTokens("image", 1), 6);
    assert.equal(H3.labelTokens("audio", 1), 5);
    assert.equal(H3.labelTokens("audio", 12), 6);
    assert.equal(H3.labelTokens("video", 3), 6);
    assert.equal(H3.timestampTokens(0.25), 6);
    assert.equal(H3.timestampTokens(12.25), 7);
    assert.equal(H3.textTokens(""), 0);
    assert.equal(H3.textTokens("A cat walks across a sunny kitchen, purring softly."), 11);
    assert.equal(H3.qwenVisionTokens(864, 480), 405);
});

for (const fixture of fixtures) {
    test(`matches ComfyUI PackedLayout: ${fixture.name}`, () => {
        const est = H3.estimate(fixture.spec);
        const expected = fixture.expected;
        const nonText = est.total - est.parts.text;
        if (expected.total != null) assert.equal(est.total, expected.total);
        if (expected.nontext != null) assert.equal(nonText, expected.nontext);
        if (expected.video != null) assert.equal(est.parts.video, expected.video);
        if (expected.audio != null) assert.equal(est.parts.audio, expected.audio);
        if (expected.frames != null) assert.equal(est.frames, expected.frames);
        if (expected.keyframes != null) assert.equal(est.parts.keyframes, expected.keyframes);
        if (expected.refImages != null) assert.equal(est.parts.refImages, expected.refImages);
        if (expected.refVideos != null) assert.equal(est.parts.refVideos, expected.refVideos);
        if (expected.refAudios != null) assert.equal(est.parts.refAudios, expected.refAudios);
    });
}

test("unknown reference metadata is assumed at the caps and flagged", () => {
    const est = H3.estimate({ width: 864, height: 480, frames: 124, refImages: [{}], refVideos: [{}], refAudios: [{}] });
    assert.equal(est.approximate, true);
    assert.ok(est.parts.refImages > 0 && est.parts.refVideos > 0 && est.parts.refAudios > 0);
});

test("formatting", () => {
    assert.equal(H3.formatTokens(999), "999");
    assert.equal(H3.formatTokens(38144), "38.1k");
    assert.equal(H3.formatTokens(109062), "109k");
    assert.equal(H3.formatTokens(4080), "4.08k");
    assert.ok(H3.describe(H3.estimate({ width: 864, height: 480, frames: 124 })).length >= 3);
});
