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
  - **Optional folder batch** uses a two-line, wrapping local-path field. The
    adjacent merge toggle leaves the normal per-prompt outputs intact and
    additionally concatenates each prompt directory. Video presets produce an
    MP4 in `output/video`; the audio-only preset produces a lossless FLAC in
    `output/audio`. A batch with no subfolders produces one root merge, while
    multiple prompt directories produce one merge per directory. Only the
    complete final merge is previewed. Nothing is written into the batch input
    tree, where media files are treated as references. Prompt and media
    filenames use Windows-style natural ordering (`1`, `2`, `10`) on Windows
    and Linux alike; that order assigns `@image1`, `@image2`, and the other
    numbered reference slots.
  - Video soundtracks take the first native audio slots. With `@video1` and
    standalone `@audio1` attached, use `<Audio 1>` for `@video1`'s soundtrack;
    `@audio1` remains the standalone file and is translated to `<Audio 2>`.
  - **Optional trim loader** — the *✂ Load + trim* toolbar button opens an
    inline loader that previews one video or audio file, with draggable
    start/end handles on a timeline (the preview seeks while dragging),
    numeric start/end fields, playhead capture buttons, and a trim-window
    preview. *Add to references* stores the file with `trim_start` /
    `trim_end` seconds in its manifest entry; only that window is decoded at
    generation time (video frames and soundtrack alike), with a keyframe seek
    so trimming the tail of a long file stays fast. Nothing is re-encoded,
    leaving the full range selected adds the file untrimmed, and trimmed
    cards show a `✂ 2–9.5s` badge. Entries without trim fields behave
    exactly as before, so existing workflows, presets, and the SwarmUI
    extension are unaffected.
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

- **Merge MiniMax H3 Folder Batch Videos** is the optional output companion
  used by the video presets. It groups generated videos using the gallery's
  validated batch metadata and saves flat `MiniMax_H3_Merged_*.mp4` files in
  `output/video`, beside the individual generations saved by Save Video.

- **Save + Merge MiniMax H3 Folder Batch Audio** is the audio preset's single
  result node. It saves all individual lossless FLACs, optionally writes flat
  `MiniMax_H3_Audio_Merged_*.flac` files in `output/audio`, and returns only
  the complete final merge through the result node's audio output and player
  when merging is enabled. The same merged file appears in Job Queue. Using one
  result node prevents an individual Save Audio preview from replacing it.

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
