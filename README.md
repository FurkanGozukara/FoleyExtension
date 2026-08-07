# FoleyExtension

SECourses custom nodes for ComfyUI, installed automatically by the SECourses
ComfyUI installers into `ComfyUI/StandaloneCustomNodes/FoleyExtension`.

## SECourses Reference Gallery

SwarmUI-style dynamic media references for reference-driven models such as
MiniMax H3, without banks of LoadImage / LoadVideo / LoadAudio nodes.

- **SECourses Reference Gallery (Images / Videos / Audio)** — one node with an
  *Add references* button, drag & drop, and clipboard paste. Every reference
  shows as a thumbnail card with its own colored `@image1` / `@video1` /
  `@audio1` token. The built-in prompt box renders the tokens as colored pills,
  autocompletes them when you type `@`, and clicking a card inserts its token
  at the cursor. Cards can be dragged left/right to reorder; tokens renumber by
  card position. The prompt text itself is never modified by removing or
  reordering cards — a token without a matching attachment shows in red and is
  simply ignored at generation time. Files upload through ComfyUI's native
  `/upload/image` endpoint into `input/reference_gallery/`.
  - Up to 9 images, 3 videos, 3 audio files.
  - Videos are resampled to `video_fps` (24 for MiniMax H3), trimmed to
    `max_seconds`, and their soundtrack is paired automatically. The default
    is 15 seconds because clean 2-15 second references are quality-tested, but
    this is guidance rather than a hard cap; longer values are accepted.
  - Large media is decoded only after the adapter knows the required canvas
    and duration. Images are downscaled before float conversion; video frames
    are scaled during FFmpeg decode and trimmed to the usable H3 frame count.
    The original aspect ratio is retained on H3's 32-pixel canvas grid, and
    shared RAM budgets prevent high-megapixel attachments from causing an OOM.
    Inputs above 100 MP per image or 40 MP per video frame fail with a clear
    validation error instead of attempting an unsafe allocation.
  - Token aliases: `@img1`, `@pic1`, `@picture1`, `@vid1`, `@aud1`, `@sound1`,
    and `@image#1` all work.
- **MiniMax H3 References (Gallery)** — adapts the gallery pack to MiniMax H3.
  It translates `@` tokens into the `<Picture i>` / `<Video k>` / `<Audio j>`
  labels the model expects (audio labels are offset past video soundtracks
  automatically) and then runs ComfyUI's native `MiniMaxH3ReferenceToVideo`
  conditioning, so upstream improvements apply automatically. Outputs
  `positive` conditioning plus the AV latent. Legacy `<Picture 1>` labels typed
  directly still pass through unchanged. Its audio-only mode extracts each
  reference video's soundtrack without decoding or conditioning on the video
  frames; in that mode `@video1` maps to `<Audio 1>`.
- **Load Video Soundtrack (Base64, No Frames)** and **Trim Reference Audio**
  provide the same soundtrack-only, user-controlled reference duration path
  to the SwarmUI extension. No hidden aggregate 15-second limit is applied.
- **MiniMax H3 Reference Mode** and **MiniMax H3 Text Only (Gallery Prompt)**
  let one workflow choose the task-specific checkpoint lazily: FL2VA when the
  current prompt has no media and Ref2VA when it does. This also works per item
  in recursive folder batches.

Future reference-driven models only need another small adapter node; the
gallery node, its manifest format, and the `@` token grammar stay the same.

## Synchronized resolution controls

**SECourses Resolution Sync** keeps aspect ratio, megapixels, width, and height
synchronized in both directions. Width and height are persisted as the
authoritative workflow values, so browser edits and API execution resolve to
the same dimensions. The advanced `multiple` setting controls dimension
snapping; use `32` for MiniMax H3 generation canvases or `1` when a workflow
aligns the model canvas separately from an exact final output size.

## LTX 2.3 Foley nodes

Low-memory streaming helpers for the LTX 2.3 Foley video-to-audio workflow:
frame streaming with the 8n+1 cap, overlapping window planning/selection for
long videos, and audio muxing onto the original compressed video. See
`THIRD_PARTY_NOTICES.md` for upstream attribution.
