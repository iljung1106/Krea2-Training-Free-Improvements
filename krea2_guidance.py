from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

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
class GuidanceState:
    gag: GAGState | None = None
    nag: NAGState | None = None


def _in_sigma_range(transformer_options: dict, start: float, end: float) -> bool:
    sigmas = transformer_options.get("sigmas")
    if sigmas is None:
        return True
    return bool(torch.all((sigmas >= end) & (sigmas <= start)).item())


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

    positive_attention = block.attn.wo(positive_raw * F.sigmoid(positive_gate))
    positive = positive + pregate * positive_attention
    positive = positive + postgate * block.mlp(
        (1 + postscale) * block.postnorm(positive) + postshift
    )
    return positive[:, :positive_len], negative_text, positive[:, positive_len:]


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
    positive_context = model._unpack_context(context)
    negative_context = None
    if nag_active:
        negative_context = model._unpack_context(
            _repeat_batch(state.nag.negative_context, batch).to(context)
        )

    image, image_pos, height_tokens, width_tokens = model.process_img(x)
    image_tokens = image.shape[1]
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
    for index, block in enumerate(model.blocks):
        transformer_options["block_index"] = index
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
    output = final[:, positive_len:positive_len + image_tokens]
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
    if not gag_active and not nag_active:
        return executor(*args, **kwargs)

    cond_or_uncond = transformer_options.get("cond_or_uncond", [0])
    if any(branch != 0 for branch in cond_or_uncond):
        raise RuntimeError("Krea2 training-free guidance requires CFG 1.0 (positive branch only).")
    ref_latents = kwargs.get("ref_latents")
    if ref_latents is None and len(args) >= 5 and not isinstance(args[4], dict):
        ref_latents = args[4]
    if ref_latents:
        raise RuntimeError("The initial GAG/NAG nodes do not support Krea2Edit reference latents.")
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
