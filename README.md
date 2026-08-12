# Krea2 Training Free Improvements

Composable training-free guidance and acceleration nodes for Krea2 in ComfyUI.

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

### Krea2 Prompt Reinjection

An adaptation of [Prompt Reinjection](https://github.com/fudan-generative-vision/PromptReinjection)
that saves Krea2's shallow text features and adds them back before deeper
single-stream blocks. The defaults (`origin=1`, `targets=2..27`, `weight=0.025`,
no anchoring) follow the official recommendation for a new MMDiT architecture.

### Krea2 TaylorSeer Lite

An adaptation of the official
[TaylorSeer Lite HunyuanImage implementation](https://github.com/Shenyi-Z/Cache4Diffusion/blob/main/HunyuanImage-2.1/run_hyimage_taylorseer_lite.py).
It periodically computes the full Krea2 output and keeps Taylor factors for
that one final feature only. Forecast steps skip all 28 transformer blocks.
Unlike the earlier per-block implementation, cache memory is independent of
the number and hidden width of transformer blocks.

This follows the released Lite code: the cached feature is the denoiser output
after Krea2's final layer, not the hidden state before it. Finite differences
and forecasts use integer denoising-step distance as in equations 7-9 of the
[TaylorSeer paper](https://arxiv.org/abs/2503.06923) and the official code.

### Krea2 Adaptive Progressive Sampler

A Krea2-aware progressive-resolution Euler sampler. It was informed by the
fixed-noise hierarchy in [Fresco](https://arxiv.org/abs/2601.07462), but it is
an independent implementation rather than a reproduction: Fresco's exact
promotion schedule and mixed-resolution attention code have not been released.

The sampler makes the transition verifiable instead of generating fresh random
padding. It decomposes ComfyUI's incoming seeded noise with an orthonormal 2D
Haar transform, denoises the low-frequency parent at a smaller spatial size,
then restores the untouched child detail coefficients at the current flow
sigma. For Krea2's rectified-flow marginal this is
`x_sigma = sigma * noise + (1 - sigma) * clean`. Each resolution uses Krea2's
native dense position grid; spreading coarse tokens across the final canvas
creates an out-of-distribution sparse RoPE grid and can duplicate anatomy.

#### Why a smaller start can still produce duplicated anatomy

A smaller latent does not automatically make a large final canvas safe. The
first denoising calls establish global topology: subject count, pose, head and
limb placement. Full-resolution calls performed later usually refine that
topology rather than replace it.

An earlier prototype scaled the coarse token coordinates across the final
canvas. For example, a half-resolution 32x32 token grid was positioned sparsely
over the range of a 64x64 grid. Although only 32x32 tokens were evaluated, the
transformer therefore saw an unfamiliar wide coordinate field. This can look
to the model like unsupported oversized generation and produce extra heads,
legs, or repeated subjects.

The current sampler does **not** scale or spread Krea2's position coordinates.
Every resolution uses its native contiguous coordinate grid. On promotion, the
sampler reconstructs the next resolution from the same seeded Haar noise
hierarchy and the current clean prediction; it does not add newly sampled
padding noise. A promotion also invalidates TaylorSeer Lite's cached output.

#### Recommended 10-step setup

Use the sampler with `SamplerCustomAdvanced`, `BasicGuider`, and a
`BasicScheduler` set to `simple`:

| Setting | Recommended value | Effect |
| --- | --- | --- |
| `initial_scale` | `0.5` | Starts at half latent width and height, then promotes once. |
| `mode` | `adaptive` | May promote after stable predictions, while preserving the required full-resolution tail. |
| `stability_threshold` | `0.08` | A larger value makes early promotion easier; a smaller value delays it. |
| `minimum_steps_per_level` | `2` | Prevents promotion before two model calls at the current resolution. |
| `full_resolution_steps` | `7` | Reserves at least the final seven of ten calls at full resolution. |

With these defaults, a 10-step run performs at most the first three calls at
half resolution and at least the last seven at full resolution. Adaptive
stability can promote earlier. This is the quality-first preset and the
recommended starting point for 1024x1024 Krea2 Turbo generation.

`full_resolution_steps=4` is an aggressive speed preset. It leaves much more of
the global structure to the coarse stage, so composition drift and duplicated
anatomy are more likely. `initial_scale=0.25` adds a second promotion and is
also experimental; compare it against the same seed before relying on it.

In `manual` mode, `transition_sigmas` controls promotion directly and requires
one descending value for each promotion (`0.5` needs one value; `0.25` needs
two). `stability_threshold` is ignored in manual mode. For ordinary 10-step
Turbo workflows, adaptive mode with the reserved tail is easier to reason
about than a schedule-specific sigma value.

The sampler intentionally supports only Krea2 CONST/rectified-flow models and
Euler sampling with a simple/Flux sigma schedule. It accepts Krea2's 4D image
latents and 5D single-frame image latents. Inpainting masks are rejected, and
the latent dimensions must be divisible by 2 for `0.5` or by 4 for `0.25`.
Progressive output is not expected to be pixel-identical to a full-resolution
baseline because the early model evaluations operate on a different token
grid.

## Composition

The four MODEL nodes are independent and can be connected in any order:

```text
MODEL -> Krea2 GAG -> Krea2 NAG -> Prompt Reinjection -> TaylorSeer Lite -> KSampler
```

GAG is computed only from the positive prompt's shared sparse/dense pair. NAG
adds a separate positive-vs-negative correction afterward; GAG projection is
never applied to the NAG residual. TaylorSeer Lite forces a fresh computation when
GAG or NAG crosses a configured sigma boundary, so it does not reuse a cache
from a different guidance regime.

The progressive sampler is a `SAMPLER`, not a MODEL patch. It can therefore be
used with the MODEL chain above through `SamplerCustomAdvanced`. A resolution
promotion invalidates TaylorSeer's cached output before the next model call.

An importable example is available at
[`workflows/Krea2 GAG and NAG - Raw FP8 Turbo.json`](workflows/Krea2%20GAG%20and%20NAG%20-%20Raw%20FP8%20Turbo.json).
The custom-sampler graph is available at
[`workflows/Krea2 Adaptive Progressive Sampler - Raw FP8 Turbo.json`](workflows/Krea2%20Adaptive%20Progressive%20Sampler%20-%20Raw%20FP8%20Turbo.json).

Use `CFG = 1.0`. TaylorSeer Lite is intended for Euler/simple, where each scheduled
sigma has one model evaluation. Unsupported intermediate sampler evaluations
fall back to a full computation. These nodes currently target native Krea2
text-to-image only; Krea2Edit/reference latents are rejected explicitly.

## Installation

Clone this repository into `ComfyUI/custom_nodes` and restart ComfyUI.

## Default settings

- GAG: `guidance_scale=5`, `eta=15`, `strength=0.6`, `sigma_start=0.881`, `sigma_end=0.678`
- NAG: `phi=4`, `tau=2.5`, `alpha=0.25`
- Prompt Reinjection: `origin_layer=1`, `target_start=2`, `target_end=27`, `weight=0.025`, `anchoring=false`
- TaylorSeer Lite: `warmup_steps=3`, `fresh_interval=2`, `tail_full_steps=2`, `max_order=0`
- Adaptive Progressive Sampler: `initial_scale=0.5`, `mode=adaptive`,
  `stability_threshold=0.08`, `minimum_steps_per_level=2`,
  `full_resolution_steps=7`

TaylorSeer Lite trades quality for speed. Krea2 Turbo has only 10 large
denoising intervals; direct testing found first-order extrapolation produces
visible colored squares. The released implementations support order 0, and it
is the stable Krea2 Turbo default: cached steps reuse the latest full output
instead of extrapolating it. Orders 1 and 2 remain available for longer
schedules through an integer input (`0` = latest output, `1` = linear trend,
`2` = linear trend plus curvature). The final two steps always use the full model. At 1024x1024, batch
1, the one BF16 order-0 feature requires about 0.5 MiB, instead of more than 5
GiB for the former per-block cache.

On an RTX 3060 Ti 8 GB, the included four-node workflow completed at 1024x1024
without OOM or block noise. GAG + NAG + Prompt Reinjection took 105.7 seconds
in testing; order-0 TaylorSeer Lite with the final two steps kept full took
67.5 seconds. Absolute times vary by offloading configuration.

The GAG defaults follow the paper, but Krea2 is a different architecture.
Compare against the same seed and reduce `strength` first if the result is too
strong.

### Progressive sampler performance

The speed-up comes from evaluating fewer image tokens during the early calls;
it is not an upscaler or a high-resolution fix pass. The default seven-step
full-resolution tail deliberately favors structure over the maximum possible
speed-up. Shortening the tail or starting at quarter resolution is faster but
changes a larger portion of the denoising trajectory.

Measure performance with the same seed after one warm-up run. Model loading,
DynamicVRAM, asynchronous weight offloading, VAE decode, and other MODEL
patches can otherwise dominate the timing and make a cold baseline comparison
misleading.

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
