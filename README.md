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
  - Up to 99 images, 99 videos, and 99 audio files. MiniMax H3 itself accepts
    at most 9 images, 3 videos, and 3 standalone audios per run, so a gallery
    above a cap becomes a roster for that modality: each run attaches only the
    files its prompt actually mentions (first-mention order, capped at the
    model limit) and discards the rest for that run. A folder-batch prompt
    saying `@image12`, `@video4`, or `@audio5` really receives that attached
    file; if a prompt mentions more than the cap, the first mentioned ones are
    used. Within the caps nothing changes: every attachment is passed exactly
    as before, mentioned or not.
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
    frontend queues each prompt as a separate job in natural order. Each job is
    fully generated, decoded, and saved before the next job starts. The adjacent
    merge toggle retains those per-prompt files and, after the final job, also
    concatenates each prompt directory. A prompt named like `scene_8.txt` uses
    8 seconds for that item; the final underscore-separated part must be a
    positive integer. Names such as `8.txt`, `scene.txt`, and `scene_0.txt` use
    the workflow's duration control. Video presets produce MP4 files in
    `output/video`; the audio-only preset produces lossless FLAC files in
    `output/audio`. A batch with no subfolders produces one root merge, while
    multiple prompt directories produce one merge per directory. Only the
    complete final merge is returned and previewed. Nothing is written into the
    batch input tree, where media files are treated as references. Prompt and media
    filenames use Windows-style natural ordering (`1`, `2`, `10`) on Windows
    and Linux alike; that order assigns `@image1`, `@image2`, and the other
    numbered reference slots.
  - **Continue from last frame** is available beside the video merge control
    and is off by default. The first item starts normally. After every later
    prompt, only the preceding saved video's final frame is decoded and used as
    the next starting image. Items with media references remain Ref2VA; items
    without references use FL2VA. Ref2VA continuation reserves one of its nine
    picture slots, so the prompt may use at most eight additional images.
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
- **MiniMax H3 Auto (Gallery)** combines those paths for video presets. Its
  mode output is diagnostic; **MiniMax H3 Reference Mode** selects the
  checkpoint before model-dependent VAE optimization. Auto applies a
  continuation frame as an FL2VA first-frame keyframe when no other references
  exist, or as an explicit Ref2VA picture reference when they do.
- **MiniMax H3 Batch Duration** resolves the per-item filename suffix while
  preserving the user-set duration for normal prompts and unmatched names.
- **MiniMax H3 Previous Batch Final Frame** validates strict sequential order
  and loads the saved frame used by the optional continuation path.

- **Save + Merge MiniMax H3 Folder Batch Videos** is the video presets' single
  result node. It saves the current individual MP4 before the next queued prompt
  starts. On the final job it groups those saved files using the gallery's
  validated metadata, writes flat `MiniMax_H3_Merged_*.mp4` files in
  `output/video`, and returns the complete last merge.

- **Save + Merge MiniMax H3 Folder Batch Audio** is the audio preset's single
  result node. It saves the current lossless FLAC before the next queued prompt
  starts, optionally writes flat `MiniMax_H3_Audio_Merged_*.flac` files in
  `output/audio` after the final job, and returns only the complete final merge
  through the result node's audio output and player. The same merged file appears
  in Job Queue.

Future reference-driven models only need another small adapter node; the
gallery node, its manifest format, and the `@` token grammar stay the same.

## Optional init audio (MiniMax H3)

An optional soundtrack the generated video must follow, the same idea as the
optional init image: nothing happens until a file is selected, and every node
passes its inputs through untouched without one, so a single preset covers
text-only, first-frame, and reference generation with or without init audio.

- **Init Audio (Optional, Auto Enable)** — an audio file selector with upload
  button and player (video files contribute their soundtrack). It emits the
  loaded audio (or nothing) plus the duration the workflow should use: by
  default the audio's own length (`match init audio length`), otherwise the
  connected workflow duration (`keep workflow duration`, longer audio is cut,
  shorter audio is padded with silence).
- **MiniMax H3 Init Audio (Optional)** sits between any MiniMax H3 conditioning
  node (Image to Video, the gallery Auto / References adapters, Reference to
  Video) and the guider/sampler. With init audio connected it encodes the
  soundtrack with the audio VAE, locks it into the joint AV latent with a
  nested noise mask (video denoised, audio kept exact at every step) and adds
  a t=1.0 audio guide at frame 0 through ComfyUI's native
  `MiniMaxH3AddGuide` keyframe mechanism, so the transformer reads the clean
  soundtrack from the first step and generates lipsync, action timing, and
  ambience to match it. This mirrors the `multimodalart/minimax-h3-audio-to-video`
  Space's locked audio rows using only mechanisms core ComfyUI already has, and
  it works identically for FL2VA (text / first frame) and Ref2VA (references),
  including folder batches. `audio_conditioning` also offers `lock soundtrack
  only` and `guide only (model re-voices)`. The `init_audio` output is the
  normalized 32 kHz stereo soundtrack cut to the video length, so the final
  MP4 carries the user's audio instead of a VAE round trip.
- **Audio Fallback (Optional Override)** returns that init audio when present
  and otherwise the decoded generated audio (which is then not even decoded).
- **MiniMax H3 Frames From Init Audio** turns an optional init audio into the
  24 FPS `17k+5` frame count (used by the SwarmUI extension).

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
