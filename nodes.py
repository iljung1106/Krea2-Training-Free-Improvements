from __future__ import annotations

from dataclasses import replace
from functools import partial

import comfy.patcher_extension
from comfy.ldm.krea2.model import SingleStreamDiT

from .adaptive_resolution import Krea2AdaptiveProgressiveSampler
from .krea2_guidance import (
    GAGState,
    GuidanceState,
    NAGState,
    PromptReinjectionState,
    TaylorRuntime,
    TaylorSeerState,
    guidance_wrapper,
)


ATTACHMENT_KEY = "krea2_training_free_improvements_state"
WRAPPER_KEY = "krea2_training_free_improvements"


def _validate_krea2(model):
    diffusion_model = model.model.diffusion_model
    original_model = getattr(diffusion_model, "_orig_mod", diffusion_model)
    if not isinstance(original_model, SingleStreamDiT):
        raise ValueError(
            f"Krea2 Training Free Improvements requires SingleStreamDiT, got {type(original_model)}"
        )
    wrappers = getattr(model, "wrappers", {}).get(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {}
    )
    incompatible = {
        "krea2_normalized_attention_guidance",
        "krea2_edit_normalized_attention_guidance",
        "krea2_edit",
    }.intersection(wrappers)
    if incompatible:
        raise ValueError(
            "Do not stack this node pack with legacy Krea2 NAG or Krea2Edit wrappers. "
            f"Found: {', '.join(sorted(incompatible))}"
        )


def _validate_sigma(start, end):
    if start < end:
        raise ValueError("sigma_start must be greater than or equal to sigma_end.")


def _state(model):
    return model.get_attachment(ATTACHMENT_KEY) or GuidanceState()


def _install(model, state):
    patched = model.clone()
    patched.set_attachments(ATTACHMENT_KEY, state)
    patched.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY
    )
    patched.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
        WRAPPER_KEY,
        partial(guidance_wrapper, state=state),
    )
    return (patched,)


class Krea2GeometryAwareAttentionGuidance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "eta": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.01, "max": 100.0, "step": 0.1},
                ),
                "strength": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "sigma_start": (
                    "FLOAT",
                    {"default": 0.881, "min": 0.0, "max": 1000.0, "step": 0.001},
                ),
                "sigma_end": (
                    "FLOAT",
                    {"default": 0.678, "min": 0.0, "max": 1000.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Krea2 Training Free Improvements/guidance"
    DESCRIPTION = (
        "Krea2 adaptation of Geometry-Aware Attention Guidance. It applies alpha=1.5 "
        "Entmax and parallel-only GAG to image-query -> positive-text retrieval while "
        "preserving image self-attention. Use CFG 1.0."
    )

    def patch(self, model, guidance_scale, eta, strength, sigma_start, sigma_end):
        _validate_krea2(model)
        _validate_sigma(sigma_start, sigma_end)
        gag = GAGState(
            guidance_scale=float(guidance_scale),
            eta=float(eta),
            strength=float(strength),
            sigma_start=float(sigma_start),
            sigma_end=float(sigma_end),
            key_chunk_size=256,
        )
        return _install(model, replace(_state(model), gag=gag))


class Krea2NAGNegativePrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "nag_negative": ("CONDITIONING",),
                "phi": (
                    "FLOAT",
                    {"default": 4.0, "min": 0.0, "max": 20.0, "step": 0.1},
                ),
                "tau": (
                    "FLOAT",
                    {"default": 2.5, "min": 0.01, "max": 20.0, "step": 0.05},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "sigma_start": (
                    "FLOAT",
                    {"default": 1000.0, "min": 0.0, "max": 1000.0, "step": 0.001},
                ),
                "sigma_end": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Krea2 Training Free Improvements/guidance"
    DESCRIPTION = (
        "Normalized Attention Guidance used only for negative prompting. The node "
        "composes with Krea2 GAG in either connection order. Use CFG 1.0."
    )

    def patch(self, model, nag_negative, phi, tau, alpha, sigma_start, sigma_end):
        _validate_krea2(model)
        _validate_sigma(sigma_start, sigma_end)
        if not nag_negative:
            raise ValueError("nag_negative conditioning is empty.")
        nag = NAGState(
            negative_context=nag_negative[0][0],
            phi=float(phi),
            tau=float(tau),
            alpha=float(alpha),
            sigma_start=float(sigma_start),
            sigma_end=float(sigma_end),
        )
        return _install(model, replace(_state(model), nag=nag))


class Krea2PromptReinjection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "origin_layer": (
                    "INT",
                    {"default": 1, "min": 0, "max": 27, "step": 1},
                ),
                "target_start": (
                    "INT",
                    {"default": 2, "min": 0, "max": 27, "step": 1},
                ),
                "target_end": (
                    "INT",
                    {"default": 27, "min": 0, "max": 27, "step": 1},
                ),
                "weight": (
                    "FLOAT",
                    {"default": 0.025, "min": -1.0, "max": 1.0, "step": 0.001},
                ),
                "anchoring": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Krea2 Training Free Improvements/guidance"
    DESCRIPTION = (
        "Reinjects shallow prompt features into deeper Krea2 text tokens to reduce "
        "prompt forgetting. Defaults follow the official recommendation for a new MMDiT."
    )

    def patch(self, model, origin_layer, target_start, target_end, weight, anchoring):
        _validate_krea2(model)
        if not 0 <= origin_layer < 28:
            raise ValueError("origin_layer must be in [0, 27].")
        if not origin_layer < target_start <= target_end < 28:
            raise ValueError(
                "Prompt Reinjection requires origin_layer < target_start <= target_end < 28."
            )
        reinjection = PromptReinjectionState(
            origin_layer=int(origin_layer),
            target_start=int(target_start),
            target_end=int(target_end),
            weight=float(weight),
            anchoring=bool(anchoring),
        )
        return _install(
            model, replace(_state(model), prompt_reinjection=reinjection)
        )


class Krea2TaylorSeer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "warmup_steps": (
                    "INT",
                    {"default": 3, "min": 2, "max": 100, "step": 1},
                ),
                "fresh_interval": (
                    "INT",
                    {"default": 2, "min": 2, "max": 100, "step": 1},
                ),
                "tail_full_steps": (
                    "INT",
                    {"default": 2, "min": 0, "max": 100, "step": 1},
                ),
                "max_order": (
                    "INT",
                    {"default": 0, "min": 0, "max": 2, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Krea2 Training Free Improvements/acceleration"
    DESCRIPTION = (
        "TaylorSeer Lite for Krea2. It forecasts one final model output between periodic "
        "full refreshes, skipping all transformer blocks with only a tiny cache. "
        "Euler/simple is recommended; increase fresh_interval for more speed at a "
        "larger quality cost. Order 0 is the tested 10-step Krea2 Turbo default."
    )

    def patch(self, model, warmup_steps, fresh_interval, tail_full_steps, max_order):
        _validate_krea2(model)
        taylorseer = TaylorSeerState(
            warmup_steps=int(warmup_steps),
            fresh_interval=int(fresh_interval),
            tail_full_steps=int(tail_full_steps),
            max_order=int(max_order),
            runtime=TaylorRuntime(),
        )
        return _install(model, replace(_state(model), taylorseer=taylorseer))


NODE_CLASS_MAPPINGS = {
    "Krea2GeometryAwareAttentionGuidance": Krea2GeometryAwareAttentionGuidance,
    "Krea2NAGNegativePrompt": Krea2NAGNegativePrompt,
    "Krea2PromptReinjection": Krea2PromptReinjection,
    "Krea2TaylorSeer": Krea2TaylorSeer,
    "Krea2AdaptiveProgressiveSampler": Krea2AdaptiveProgressiveSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2GeometryAwareAttentionGuidance": "Krea2 Geometry-Aware Attention Guidance",
    "Krea2NAGNegativePrompt": "Krea2 NAG Negative Prompt",
    "Krea2PromptReinjection": "Krea2 Prompt Reinjection",
    "Krea2TaylorSeer": "Krea2 TaylorSeer Lite",
    "Krea2AdaptiveProgressiveSampler": "Krea2 Adaptive Progressive Sampler",
}
