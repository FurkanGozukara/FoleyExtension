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

