## Genjutsu task adapters and aligned guides

The Character Swap v1.1 / Motion Transfer v1.0 recipes and aligned-guide routing follow
[Henge](https://github.com/alisson-anjos/Henge), reviewed at `df72d72`, by Alisson Anjos.
Our implementation calls native ComfyUI loaders and Add Guide; it does not install Henge
or ComfyUI-BFSNodes. Task model weights are downloaded unchanged and retain their separate
[asset licenses](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI). Training adapter V2
is by [ostris](https://huggingface.co/ostris/minimax_h3_training_adapter).

## FuzzPuppy LTX 2.3 Foley Workflow

`fuzzpuppy_nodes.py` is derived from the `ltx_foley_v2a/nodes.py` helper in:

https://huggingface.co/FuzzPuppy/LTX-2.3-Foley-Workflow

Source revision reviewed: `b40231e`.

The upstream helper and workflow are licensed under Apache License 2.0. The
loop mechanics inside that helper are adapted upstream from
`akatz-ai/ComfyUI-Execution-Inversion`, licensed under the MIT License and
Copyright (c) 2025 akatz-ai.


## MiniMax H3 RefMod compatibility

`refmod_nodes.py` implements the MiniMaxH3Mod v4/v5 safetensors format and Fantastic encoder-frame records. Reference-retention low-pass math and reference presentation were adapted from these MIT-licensed projects:

- https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod - reviewed f9462081e28794389b5a6c5067eb327412ad8ee7.
- https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder - reviewed 1e8df670cc212c692e422690829731ae403906f4.

```text
MIT License

Copyright (c) 2026 Luisa (luisacaotica)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

```text
MIT License

Copyright (c) 2026 Adudeguyman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

## H3 community methods (November 2026)

Written as our own code from the published descriptions of these projects, reviewed on 2026-11-03:

- Masked edit and grey-figure guide (`genjutsu_nodes.py`): the approach of [openjutsu](https://github.com/D3voz/openjutsu) by D3voz (reviewed at its 2026-10-30 commit).
- Relight prompt (`relight_nodes.py`): the lighting-plan approach of [H3-Relight](https://github.com/NyckM/H3-Relight-Minimax-ComfyUI) by NyckM / Bruxos do VFX, reviewed at `f6019c0`; the 20 look presets and 25 atmosphere phrases are adapted from its `web/relight_data.js`.
- Face swap method: uses the [FaceSwap LoRA](https://huggingface.co/UntMods/FaceSwap_MiniMaxH3_REF2VA) by UntMods, downloaded unchanged.

