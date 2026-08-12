# Krea2 Training Free Improvements

Composable training-free guidance nodes for Krea2 in ComfyUI.

## Nodes

### Krea2 Geometry-Aware Attention Guidance

An adaptation of [Geometry-Aware Attention Guidance](https://arxiv.org/abs/2603.02531v2)
for Krea2's single-stream transformer. It applies alpha=1.5 Entmax and the
parallel-only GAG correction to image-query -> positive-text retrieval while
leaving image self-attention dense.

The paper proves GAG for explicit cross-attention. Krea2 uses joint attention,
so this implementation preserves the native text-attention mass and replaces
only the conditional distribution within the positive-text partition. It is a
Krea2 adaptation, not official code from the paper authors.

### Krea2 NAG Negative Prompt

Krea2 negative prompting based on
[Normalized Attention Guidance](https://arxiv.org/abs/2505.21179).

## Composition

The two nodes are independent and can be connected in either order:

```text
MODEL -> Krea2 GAG -> Krea2 NAG Negative Prompt -> KSampler
```

GAG is computed only from the positive prompt's shared sparse/dense pair. NAG
adds a separate positive-vs-negative correction afterward; GAG projection is
never applied to the NAG residual.

An importable example is available at
[`workflows/Krea2 GAG and NAG - Raw FP8 Turbo.json`](workflows/Krea2%20GAG%20and%20NAG%20-%20Raw%20FP8%20Turbo.json).

Use `CFG = 1.0`. The initial release targets native Krea2 text-to-image only;
Krea2Edit/reference latents are rejected explicitly.

## Installation

Clone this repository into `ComfyUI/custom_nodes` and restart ComfyUI.

## Default settings

- GAG: `guidance_scale=10`, `eta=15`, `strength=1`
- NAG: `phi=4`, `tau=2.5`, `alpha=0.25`

The GAG defaults follow the paper, but Krea2 is a different architecture.
Compare against the same seed and reduce `strength` first if the result is too
strong.

## Verified example

Prompt: `A llama-bird hybrid creature flying in the sky. Llama head, bird body.`

NAG negative: `big wings`

All four images below were generated successfully with the same seed, Krea2
Raw FP8, the Raw-to-Turbo LoRA, 10 Euler/simple steps, and CFG 1.0.

| Baseline | GAG |
| --- | --- |
| ![Baseline](examples/llama_bird_baseline.png) | ![GAG](examples/llama_bird_gag.png) |

| NAG | GAG + NAG |
| --- | --- |
| ![NAG](examples/llama_bird_nag.png) | ![GAG and NAG](examples/llama_bird_gag_nag.png) |
