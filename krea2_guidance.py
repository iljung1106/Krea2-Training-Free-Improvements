from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from threading import RLock

import torch
import torch.nn.functional as F
from einops import rearrange

import comfy.utils
from comfy.ldm.flux.layers import timestep_embedding
from comfy.ldm.flux.math import apply_rope
from comfy.ldm.modules.attention import optimized_attention_masked

from .guidance_math import entmax15, geometry_aware_attention, normalized_attention_guidance


@dataclass(frozen=True)
class GAGState:
    guidance_scale: float
    eta: float
    strength: float
    sigma_start: float
    sigma_end: float
    key_chunk_size: int


@dataclass(frozen=True)
class NAGState:
    negative_context: torch.Tensor
    phi: float
    tau: float
    alpha: float
    sigma_start: float
    sigma_end: float


@dataclass(frozen=True)
class PromptReinjectionState:
    origin_layer: int
    target_start: int
    target_end: int
    weight: float
    anchoring: bool


class TaylorRuntime:
    def __init__(self):
        self.lock = RLock()
        self.run_key = None
        self.last_step = -1
        self.last_full_step = None
        self.last_full_signature = None
        self.cache = {}

    def reset(self, run_key):
        self.run_key = run_key
        self.last_step = -1
        self.last_full_step = None
        self.last_full_signature = None
        self.cache = {}


@dataclass(frozen=True)
class TaylorSeerState:
    warmup_steps: int
    fresh_interval: int
    tail_full_steps: int
    max_order: int
    runtime: TaylorRuntime


@dataclass(frozen=True)
class GuidanceState:
    gag: GAGState | None = None
    nag: NAGState | None = None
    prompt_reinjection: PromptReinjectionState | None = None
    taylorseer: TaylorSeerState | None = None


def _in_sigma_range(transformer_options: dict, start: float, end: float) -> bool:
    sigmas = transformer_options.get("sigmas")
    if sigmas is None:
        return True
    # Node inputs use a 0.001 step, while scheduler values retain more precision
    # (for example, the displayed 0.678 boundary is 0.677987 internally).
    # Compare within half one UI unit so the selected displayed boundary is inclusive.
    tolerance = 0.0005
    return bool(
        torch.all((sigmas >= end - tolerance) & (sigmas <= start + tolerance)).item()
    )


def gag_is_active(transformer_options: dict, state: GAGState | None) -> bool:
    return bool(
        state
        and state.strength != 0.0
        and _in_sigma_range(transformer_options, state.sigma_start, state.sigma_end)
    )


def nag_is_active(transformer_options: dict, state: NAGState | None) -> bool:
    return bool(
        state
        and state.alpha != 0.0
        and state.phi != 0.0
        and _in_sigma_range(transformer_options, state.sigma_start, state.sigma_end)
    )


def _repeat_batch(tensor: torch.Tensor, batch: int) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    return comfy.utils.repeat_to_batch_size(tensor, batch)


def _project_attention(attn, x, freqs):
    q = rearrange(attn.wq(x), "B L (H D) -> B H L D", H=attn.heads)
    k = rearrange(attn.wk(x), "B L (H D) -> B H L D", H=attn.kvheads)
    v = rearrange(attn.wv(x), "B L (H D) -> B H L D", H=attn.kvheads)
    gate = attn.gate(x)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = apply_rope(q, k, freqs)
    if attn.kvheads != attn.heads:
        repeat = attn.heads // attn.kvheads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    return q, k, v, gate


def _native_attention(attn, q, k, v, transformer_options):
    return optimized_attention_masked(
        q,
        k,
        v,
        attn.heads,
        mask=None,
        skip_reshape=True,
        transformer_options=transformer_options,
    )


def _text_mass(q_target, k_all, text_logits, chunk_size: int):
    """Softmax probability mass assigned to text keys without storing QxK_all."""
    scale = 1.0 / sqrt(q_target.shape[-1])
    text_logsumexp = torch.logsumexp(text_logits, dim=-1, keepdim=True)
    all_logsumexp = None
    for start in range(0, k_all.shape[-2], chunk_size):
        keys = k_all[:, :, start:start + chunk_size].float()
        logits = torch.matmul(q_target.float(), keys.transpose(-2, -1)) * scale
        chunk_logsumexp = torch.logsumexp(logits, dim=-1, keepdim=True)
        all_logsumexp = (
            chunk_logsumexp
            if all_logsumexp is None
            else torch.logaddexp(all_logsumexp, chunk_logsumexp)
        )
    return torch.exp(text_logsumexp - all_logsumexp).clamp_(0.0, 1.0)


def apply_gag_to_image_queries(raw, q, k, v, text_len: int, state: GAGState):
    """Replace only target->text retrieval; preserve all image/self-attention."""
    q_target = q[:, :, text_len:]
    k_text = k[:, :, :text_len]
    v_text = v[:, :, :text_len]
    scale = 1.0 / sqrt(q.shape[-1])
    logits = torch.matmul(q_target.float(), k_text.float().transpose(-2, -1)) * scale
    dense_probabilities = torch.softmax(logits, dim=-1)
    sparse_probabilities = entmax15(logits, dim=-1).float()
    dense = torch.matmul(dense_probabilities, v_text.float())
    sparse = torch.matmul(sparse_probabilities, v_text.float())
    gag = geometry_aware_attention(
        dense,
        sparse,
        guidance_scale=state.guidance_scale,
        eta=state.eta,
        strength=state.strength,
    ).float()
    mass = _text_mass(q_target, k, logits, state.key_chunk_size)
    correction = rearrange(
        mass * (gag - dense), "B H L D -> B L (H D)"
    )
    corrected = raw[:, text_len:].float() + correction
    return torch.cat((raw[:, :text_len], corrected.to(raw.dtype)), dim=1)


def apply_prompt_reinjection(
    target: torch.Tensor,
    origin: torch.Tensor,
    weight: float,
    anchoring: bool,
) -> torch.Tensor:
    """Basic Prompt Reinjection, with the paper's optional token-wise anchoring."""
    if target.shape != origin.shape:
        raise ValueError(
            f"Prompt Reinjection shapes must match, got {target.shape} and {origin.shape}"
        )
    if not anchoring:
        return target + float(weight) * origin

    target32 = target.float()
    origin32 = origin.float()
    eps = 1e-6
    target_mean = target32.mean(dim=-1, keepdim=True)
    target_std = target32.std(dim=-1, keepdim=True).clamp_min(eps)
    target_norm = (target32 - target_mean) / (target_std + eps)
    origin_mean = origin32.mean(dim=-1, keepdim=True)
    origin_std = origin32.std(dim=-1, keepdim=True).clamp_min(eps)
    origin_norm = (origin32 - origin_mean) / (origin_std + eps)
    mixed = F.layer_norm(
        target_norm + float(weight) * origin_norm,
        normalized_shape=(target.shape[-1],),
        eps=eps,
    )
    return (mixed * target_std + target_mean).to(target.dtype)


def taylor_update(
    factors: dict[int, torch.Tensor],
    feature: torch.Tensor,
    distance: int,
    max_order: int,
    allow_derivatives: bool = True,
):
    """Update finite-difference Taylor factors as in the official TaylorSeer code."""
    updated = {0: feature.detach()}
    if not allow_derivatives:
        return updated
    for order in range(max_order):
        previous = factors.get(order)
        if previous is None:
            break
        updated[order + 1] = (updated[order] - previous) / float(distance)
    return updated


def taylor_forecast(factors: dict[int, torch.Tensor], distance: int) -> torch.Tensor:
    output = torch.zeros_like(factors[0])
    factorial = 1
    for order in range(len(factors)):
        if order > 0:
            factorial *= order
        output = output + factors[order] * (float(distance) ** order) / factorial
    return output


def _guided_block(
    block,
    positive_text,
    negative_text,
    image,
    vec,
    positive_freqs,
    negative_freqs,
    transformer_options,
    gag_state,
    nag_state,
):
    prescale, preshift, pregate, postscale, postshift, postgate = block.mod(vec)
    positive_len = positive_text.shape[1]
    positive = torch.cat((positive_text, image), dim=1)
    positive_pre = (1 + prescale) * block.prenorm(positive) + preshift
    positive_q, positive_k, positive_v, positive_gate = _project_attention(
        block.attn, positive_pre, positive_freqs
    )
    positive_dense = _native_attention(
        block.attn, positive_q, positive_k, positive_v, transformer_options
    )
    positive_raw = positive_dense

    if gag_state is not None:
        positive_raw = apply_gag_to_image_queries(
            positive_raw, positive_q, positive_k, positive_v, positive_len, gag_state
        )

    if nag_state is not None:
        negative_len = negative_text.shape[1]
        negative = torch.cat((negative_text, image), dim=1)
        negative_pre = (1 + prescale) * block.prenorm(negative) + preshift
        negative_q, negative_k, negative_v, negative_gate = _project_attention(
            block.attn, negative_pre, negative_freqs
        )
        negative_raw = _native_attention(
            block.attn, negative_q, negative_k, negative_v, transformer_options
        )
        nag_target = normalized_attention_guidance(
            positive_dense[:, positive_len:],
            negative_raw[:, negative_len:],
            phi=nag_state.phi,
            tau=nag_state.tau,
            alpha=nag_state.alpha,
        )
        positive_raw = torch.cat(
            (
                positive_raw[:, :positive_len],
                positive_raw[:, positive_len:] + nag_target - positive_dense[:, positive_len:],
            ),
            dim=1,
        )
        negative_text_attention = block.attn.wo(
            negative_raw[:, :negative_len] * F.sigmoid(negative_gate[:, :negative_len])
        )
        negative_text = negative_text + pregate * negative_text_attention
        negative_text = negative_text + postgate * block.mlp(
            (1 + postscale) * block.postnorm(negative_text) + postshift
        )

    positive = positive + pregate * block.attn.wo(
        positive_raw * F.sigmoid(positive_gate)
    )
    positive = positive + postgate * block.mlp(
        (1 + postscale) * block.postnorm(positive) + postshift
    )
    return positive[:, :positive_len], negative_text, positive[:, positive_len:]


def _sampling_step(transformer_options: dict):
    current = transformer_options.get("sigmas")
    schedule = transformer_options.get("sample_sigmas")
    if current is None or schedule is None or current.numel() != 1 or schedule.numel() < 2:
        return None
    sigma = float(current.flatten()[0].detach().cpu())
    scheduled = schedule[:-1].detach().float().cpu()
    distances = (scheduled - sigma).abs()
    step = int(distances.argmin())
    tolerance = max(1e-5, abs(sigma) * 1e-4)
    if float(distances[step]) > tolerance:
        return None
    run_key = (
        tuple(round(float(value), 7) for value in schedule.detach().float().cpu()),
        tuple(transformer_options.get("cond_or_uncond", [0])),
        int(transformer_options.get("krea2_progressive_level", 0)),
    )
    return step, run_key


def _taylor_mode(transformer_options: dict, state: TaylorSeerState | None):
    if state is None:
        return "full", None, None
    resolved = _sampling_step(transformer_options)
    if resolved is None:
        return "full", None, None
    step, run_key = resolved
    runtime = state.runtime
    if runtime.run_key != run_key or step <= runtime.last_step:
        runtime.reset(run_key)
    runtime.last_step = step
    total_steps = len(run_key[0]) - 1
    full = step < state.warmup_steps or step >= total_steps - state.tail_full_steps
    if not full:
        full = runtime.last_full_step is None or step - runtime.last_full_step >= state.fresh_interval
    return ("full" if full else "forecast"), step, runtime


def krea2_guided_forward(
    model,
    x,
    timesteps,
    context,
    transformer_options,
    state: GuidanceState,
    gag_active: bool,
    nag_active: bool,
):
    temporal = x.ndim == 5
    if temporal:
        batch_5d, channels_5d, frames_5d, height_5d, width_5d = x.shape
        x = x.reshape(batch_5d * frames_5d, channels_5d, height_5d, width_5d)
    elif x.ndim != 4:
        raise RuntimeError(f"Krea2 guidance expected a 4D or 5D latent, got rank {x.ndim}.")

    batch, _, original_height, original_width = x.shape
    height_tokens = (original_height + model.patch - 1) // model.patch
    width_tokens = (original_width + model.patch - 1) // model.patch
    image_tokens = height_tokens * width_tokens
    taylor_mode, taylor_step, taylor_runtime = _taylor_mode(
        transformer_options, state.taylorseer
    )
    guidance_signature = (gag_active, nag_active)
    if taylor_mode == "forecast" and not taylor_runtime.cache:
        taylor_mode = "full"
    if (
        taylor_mode == "forecast"
        and taylor_runtime.last_full_signature != guidance_signature
    ):
        # A cache produced on one side of a guidance sigma boundary cannot
        # represent the model function on the other side of that boundary.
        taylor_mode = "full"

    if taylor_mode == "forecast":
        distance = taylor_step - taylor_runtime.last_full_step
        final = taylor_forecast(taylor_runtime.cache, distance)
        return _unpack_output(
            final,
            image_tokens,
            height_tokens,
            width_tokens,
            original_height,
            original_width,
            model,
            temporal,
            batch_5d if temporal else None,
            frames_5d if temporal else None,
        )

    positive_context = model._unpack_context(context)
    negative_context = None
    if nag_active:
        negative_context = model._unpack_context(
            _repeat_batch(state.nag.negative_context, batch).to(context)
        )

    image, image_pos, height_tokens, width_tokens = model.process_img(x)
    image = model.first(image)
    timestep = model.tmlp(
        timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(image.dtype)
    )
    timestep_vector = model.tproj(timestep)
    positive_text = model.txtmlp(
        model.txtfusion(positive_context, mask=None, transformer_options=transformer_options)
    )
    negative_text = None
    if nag_active:
        negative_text = model.txtmlp(
            model.txtfusion(negative_context, mask=None, transformer_options=transformer_options)
        )

    positive_len = positive_text.shape[1]
    device = image.device
    positive_ids = torch.cat(
        (
            torch.zeros(batch, positive_len, 3, device=device, dtype=torch.float32),
            image_pos,
        ),
        dim=1,
    )
    positive_freqs = model.pe_embedder(positive_ids)
    negative_freqs = None
    if nag_active:
        negative_len = negative_text.shape[1]
        negative_ids = torch.cat(
            (
                torch.zeros(batch, negative_len, 3, device=device, dtype=torch.float32),
                image_pos,
            ),
            dim=1,
        )
        negative_freqs = model.pe_embedder(negative_ids)

    transformer_options = transformer_options.copy()
    transformer_options["total_blocks"] = len(model.blocks)
    transformer_options["block_type"] = "single"
    transformer_options["img_slice"] = [positive_len, positive_len + image_tokens]
    gag_state = state.gag if gag_active else None
    nag_state = state.nag if nag_active else None
    reinjection = state.prompt_reinjection
    positive_origin = None
    negative_origin = None
    for index, block in enumerate(model.blocks):
        transformer_options["block_index"] = index
        if reinjection is not None:
            if index == reinjection.origin_layer:
                positive_origin = positive_text.detach()
                if nag_active:
                    negative_origin = negative_text.detach()
            if reinjection.target_start <= index <= reinjection.target_end:
                if positive_origin is None:
                    raise RuntimeError(
                        "Prompt Reinjection origin_layer must be before all target layers."
                    )
                positive_text = apply_prompt_reinjection(
                    positive_text,
                    positive_origin,
                    reinjection.weight,
                    reinjection.anchoring,
                )
                if nag_active:
                    negative_text = apply_prompt_reinjection(
                        negative_text,
                        negative_origin,
                        reinjection.weight,
                        reinjection.anchoring,
                    )

        positive_text, negative_text, image = _guided_block(
            block,
            positive_text,
            negative_text,
            image,
            timestep_vector,
            positive_freqs,
            negative_freqs,
            transformer_options,
            gag_state,
            nag_state,
        )

    combined = torch.cat((positive_text, image), dim=1)
    final = model.last(combined, timestep)
    final = final[:, positive_len:positive_len + image_tokens]
    if taylor_runtime is not None:
        distance = (
            1
            if taylor_runtime.last_full_step is None
            else taylor_step - taylor_runtime.last_full_step
        )
        taylor_runtime.cache = taylor_update(
            taylor_runtime.cache,
            final,
            distance,
            state.taylorseer.max_order,
            allow_derivatives=(
                taylor_step > state.taylorseer.warmup_steps - 2
            ),
        )
        taylor_runtime.last_full_step = taylor_step
        taylor_runtime.last_full_signature = guidance_signature

    return _unpack_output(
        final,
        image_tokens,
        height_tokens,
        width_tokens,
        original_height,
        original_width,
        model,
        temporal,
        batch_5d if temporal else None,
        frames_5d if temporal else None,
    )


def _unpack_output(
    final,
    image_tokens,
    height_tokens,
    width_tokens,
    original_height,
    original_width,
    model,
    temporal,
    batch_5d,
    frames_5d,
):
    output = final[:, :image_tokens]
    output = rearrange(
        output,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=height_tokens,
        w=width_tokens,
        ph=model.patch,
        pw=model.patch,
        c=model.channels,
    )[:, :, :original_height, :original_width]
    if temporal:
        output = output.reshape(
            batch_5d, frames_5d, model.channels, original_height, original_width
        ).movedim(1, 2)
    return output


def guidance_wrapper(executor, *args, state: GuidanceState, **kwargs):
    if len(args) < 3:
        raise RuntimeError("Krea2 guidance encountered an unexpected diffusion-model signature.")
    x, timesteps, context = args[:3]
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        transformer_options = next(
            (value for value in reversed(args[3:]) if isinstance(value, dict)), {}
        )
    gag_active = gag_is_active(transformer_options, state.gag)
    nag_active = nag_is_active(transformer_options, state.nag)
    if (
        not gag_active
        and not nag_active
        and state.prompt_reinjection is None
        and state.taylorseer is None
    ):
        return executor(*args, **kwargs)

    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
    if any(branch != 0 for branch in cond_or_uncond):
        raise RuntimeError("Krea2 training-free guidance requires CFG 1.0 (positive branch only).")
    ref_latents = kwargs.get("ref_latents")
    if ref_latents is None and len(args) >= 5 and not isinstance(args[4], dict):
        ref_latents = args[4]
    if ref_latents:
        raise RuntimeError(
            "Krea2 Training Free Improvements currently does not support "
            "Krea2Edit reference latents."
        )

    def run():
        return krea2_guided_forward(
            executor.class_obj,
            x,
            timesteps,
            context,
            transformer_options,
            state,
            gag_active,
            nag_active,
        )

    if state.taylorseer is not None:
        with state.taylorseer.runtime.lock:
            return run()
    return run()
