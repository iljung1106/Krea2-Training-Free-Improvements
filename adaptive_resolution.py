from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.model_sampling
import comfy.model_patcher
import comfy.samplers
from comfy.k_diffusion.sampling import to_d
from comfy.ldm.krea2.model import SingleStreamDiT
from comfy.utils import model_trange


LOGGER = logging.getLogger(__name__)
GUIDANCE_ATTACHMENT_KEY = "krea2_training_free_improvements_state"


def haar_split_2d(x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    """Split a field into orthonormal 2x2 Haar low/detail coefficients."""
    if x.ndim not in (4, 5) or x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError(
            "Haar splitting requires a 4D image or 5D image sequence with even height and width."
        )
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    low = (a + b + c + d) * 0.5
    horizontal = (a - b + c - d) * 0.5
    vertical = (a + b - c - d) * 0.5
    diagonal = (a - b - c + d) * 0.5
    return low, (horizontal, vertical, diagonal)


def haar_merge_2d(
    low: torch.Tensor, details: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Invert :func:`haar_split_2d` without allocating random transition noise."""
    horizontal, vertical, diagonal = details
    if not (low.shape == horizontal.shape == vertical.shape == diagonal.shape):
        raise ValueError("Haar low and detail coefficients must have identical shapes.")
    a = (low + horizontal + vertical + diagonal) * 0.5
    b = (low - horizontal + vertical - diagonal) * 0.5
    c = (low + horizontal - vertical - diagonal) * 0.5
    d = (low - horizontal - vertical + diagonal) * 0.5
    output = low.new_empty(*low.shape[:-2], low.shape[-2] * 2, low.shape[-1] * 2)
    output[..., 0::2, 0::2] = a
    output[..., 0::2, 1::2] = b
    output[..., 1::2, 0::2] = c
    output[..., 1::2, 1::2] = d
    return output


def relative_prediction_change(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Scale-independent change used by the adaptive promotion gate."""
    current_f = current.detach().float()
    previous_f = previous.detach().float()
    delta = torch.mean((current_f - previous_f).square()).sqrt()
    scale = torch.mean(current_f.square()).sqrt().clamp_min(1e-6)
    return float((delta / scale).item())


def _resize_clean(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if x.shape[-2:] == size:
        return x
    temporal = x.ndim == 5
    if temporal:
        batch, channels, frames, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    if size[0] < x.shape[-2] or size[1] < x.shape[-1]:
        output = F.interpolate(x, size=size, mode="area")
    else:
        output = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if temporal:
        output = output.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4)
    return output


def _model_call_args(extra_args: dict, progressive_level: int) -> dict:
    output = dict(extra_args)
    model_options = comfy.model_patcher.create_model_options_clone(
        extra_args.get("model_options", {})
    )
    model_options.setdefault("transformer_options", {})[
        "krea2_progressive_level"
    ] = progressive_level
    output["model_options"] = model_options
    return output


def _parse_sigmas(value: str, count: int) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise ValueError("transition_sigmas must be comma-separated numbers.") from error
    if len(values) != count:
        raise ValueError(
            f"Expected {count} transition sigma value(s), received {len(values)}."
        )
    if any(sigma <= 0.0 for sigma in values):
        raise ValueError("transition_sigmas must be greater than zero.")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError("transition_sigmas must be in strictly descending order.")
    return values


@dataclass(frozen=True)
class ProgressiveOptions:
    levels: int
    mode: str
    transition_sigmas: tuple[float, ...]
    stability_threshold: float
    minimum_steps_per_level: int
    full_resolution_steps: int
    stability_patience: int = 2


def _model_sampling(model):
    try:
        return model.inner_model.inner_model.model_sampling
    except AttributeError as error:
        raise RuntimeError(
            "Adaptive Progressive Resolution could not access ComfyUI model sampling."
        ) from error


def _validate_krea2_model(model) -> None:
    guider = getattr(model, "inner_model", None)
    patcher = getattr(guider, "model_patcher", None)
    diffusion_model = getattr(getattr(patcher, "model", None), "diffusion_model", None)
    original_model = getattr(diffusion_model, "_orig_mod", diffusion_model)
    if not isinstance(original_model, SingleStreamDiT):
        raise ValueError(
            "Krea2 Adaptive Progressive Sampler requires Krea2 SingleStreamDiT, "
            f"got {type(original_model)}."
        )


def _taylorseer_is_active(model) -> bool:
    guider = getattr(model, "inner_model", None)
    patcher = getattr(guider, "model_patcher", None)
    if patcher is None or not hasattr(patcher, "get_attachment"):
        return False
    state = patcher.get_attachment(GUIDANCE_ATTACHMENT_KEY)
    return bool(state is not None and getattr(state, "taylorseer", None) is not None)


def _normalized_input_noise(
    x: torch.Tensor,
    latent_image: torch.Tensor | None,
    sigma: torch.Tensor,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent = torch.zeros_like(x) if latent_image is None else latent_image
    if latent.shape != x.shape:
        raise ValueError("The latent image and sampling noise must have the same shape.")
    sigma_value = float(sigma)
    if sigma_value <= 0.0:
        raise ValueError("The initial sigma must be greater than zero.")
    noise = (x - (1.0 - sigma) * latent) / (sigma * noise_scale)
    return noise, latent


def _noise_pyramid(
    noise: torch.Tensor, levels: int
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, ...]]]:
    details_fine_to_coarse = []
    current = noise
    for _ in range(levels):
        current, details = haar_split_2d(current)
        details_fine_to_coarse.append(details)
    return current, list(reversed(details_fine_to_coarse))


def _should_promote(
    *,
    options: ProgressiveOptions,
    level_index: int,
    steps_on_level: int,
    stable_count: int,
    sigma_next: float,
    future_steps: int,
) -> tuple[bool, str | None]:
    if options.mode == "manual":
        threshold = options.transition_sigmas[level_index]
        return (sigma_next <= threshold, "manual sigma" if sigma_next <= threshold else None)

    transitions_after_this = options.levels - level_index - 1
    calls_needed = (
        options.full_resolution_steps
        + transitions_after_this * options.minimum_steps_per_level
    )
    forced = future_steps <= calls_needed
    stable = (
        steps_on_level >= options.minimum_steps_per_level
        and stable_count >= options.stability_patience
    )
    if forced:
        return True, "reserved full-resolution tail"
    if stable:
        return True, "prediction stabilized"
    return False, None


@torch.no_grad()
def sample_adaptive_progressive(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    options: ProgressiveOptions,
):
    """Euler flow sampling with seed-preserving progressive spatial resolution."""
    extra_args = {} if extra_args is None else extra_args
    if extra_args.get("denoise_mask") is not None:
        raise ValueError("Adaptive Progressive Resolution does not support inpainting masks.")
    if x.ndim not in (4, 5):
        raise ValueError(
            f"Expected a 4D image or 5D image sequence latent, received shape {tuple(x.shape)}."
        )

    _validate_krea2_model(model)
    sampling = _model_sampling(model)
    if not isinstance(sampling, comfy.model_sampling.CONST):
        raise ValueError(
            "Adaptive Progressive Resolution requires a CONST/rectified-flow model such as Krea2."
        )
    divisor = 2 ** options.levels
    if x.shape[-2] % divisor or x.shape[-1] % divisor:
        raise ValueError(
            f"Latent height and width must be divisible by {divisor} for the selected scale."
        )

    noise_scale = float(getattr(sampling, "noise_scale", 1.0))
    full_size = x.shape[-2:]
    full_noise, latent_image = _normalized_input_noise(
        x, getattr(model, "latent_image", None), sigmas[0], noise_scale
    )
    noise, details_by_level = _noise_pyramid(full_noise, options.levels)
    current_size = noise.shape[-2:]
    clean = _resize_clean(latent_image, current_size)
    x = sigmas[0] * noise_scale * noise + (1.0 - sigmas[0]) * clean

    total_steps = len(sigmas) - 1
    level_index = 0
    steps_on_level = 0
    stable_count = 0
    previous_denoised = None
    last_change = None
    stability_allowed = not _taylorseer_is_active(model)

    LOGGER.info(
        "Krea2 adaptive progressive sampling: %sx%s -> %sx%s, %s steps",
        current_size[1], current_size[0], full_size[1], full_size[0], total_steps,
    )

    for step in model_trange(total_steps, disable=disable):
        sigma = sigmas[step]
        sigma_batch = x.new_ones([x.shape[0]]) * sigma
        denoised = model(
            x,
            sigma_batch,
            **_model_call_args(extra_args, level_index),
        )
        steps_on_level += 1

        if previous_denoised is not None and stability_allowed:
            last_change = relative_prediction_change(denoised, previous_denoised)
            if last_change <= options.stability_threshold:
                stable_count += 1
            else:
                stable_count = 0
        previous_denoised = denoised.detach()

        if callback is not None:
            callback(
                {
                    "x": x,
                    "i": step,
                    "sigma": sigma,
                    "sigma_hat": sigma,
                    "denoised": denoised,
                }
            )

        sigma_next = sigmas[step + 1]
        x = x + to_d(x, sigma, denoised) * (sigma_next - sigma)

        if level_index >= options.levels or float(sigma_next) <= 0.0:
            continue

        future_steps = total_steps - step - 1
        promote, reason = _should_promote(
            options=options,
            level_index=level_index,
            steps_on_level=steps_on_level,
            stable_count=stable_count,
            sigma_next=float(sigma_next),
            future_steps=future_steps,
        )
        if not promote:
            continue

        # The Euler update above follows the current clean prediction to sigma_next.
        # Recover its parent noise coefficient, restore the untouched child details
        # from the original seed, and reconstruct the exact flow marginal.
        parent_noise = (
            x - (1.0 - sigma_next) * denoised
        ) / (sigma_next * noise_scale)
        noise = haar_merge_2d(parent_noise, details_by_level[level_index])
        clean = _resize_clean(denoised, noise.shape[-2:])
        x = sigma_next * noise_scale * noise + (1.0 - sigma_next) * clean
        level_index += 1
        steps_on_level = 0
        stable_count = 0
        previous_denoised = None
        LOGGER.info(
            "Krea2 progressive promotion at step %s, sigma %.6f (%s, change=%s): %sx%s",
            step + 1,
            float(sigma_next),
            reason,
            "n/a" if last_change is None else f"{last_change:.5f}",
            x.shape[-1],
            x.shape[-2],
        )

    if x.shape[-2:] != full_size:
        raise RuntimeError(
            "Sampling ended before full resolution. Increase full_resolution_steps or "
            "use earlier manual transition sigmas."
        )
    return x


class Krea2AdaptiveProgressiveSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "initial_scale": (["0.5", "0.25"], {"default": "0.5"}),
                "mode": (["adaptive", "manual"], {"default": "adaptive"}),
                "transition_sigmas": (
                    "STRING",
                    {
                        "default": "0.60",
                        "tooltip": "Manual mode only. Use one descending sigma per 2x promotion.",
                    },
                ),
                "stability_threshold": (
                    "FLOAT",
                    {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "minimum_steps_per_level": (
                    "INT",
                    {"default": 2, "min": 1, "max": 100, "step": 1},
                ),
                "full_resolution_steps": (
                    "INT",
                    {"default": 7, "min": 1, "max": 100, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "build"
    CATEGORY = "Krea2 Training Free Improvements/acceleration"
    DESCRIPTION = (
        "Experimental seed-preserving progressive Euler sampler for Krea2. It denoises "
        "at half or quarter latent resolution, then restores the original seed's "
        "orthogonal high-frequency noise at the correct flow sigma. Use with "
        "SamplerCustomAdvanced and a simple/Flux sigma schedule."
    )

    def build(
        self,
        initial_scale,
        mode,
        transition_sigmas,
        stability_threshold,
        minimum_steps_per_level,
        full_resolution_steps,
    ):
        levels = {"0.5": 1, "0.25": 2}[initial_scale]
        manual_sigmas = _parse_sigmas(transition_sigmas, levels) if mode == "manual" else ()
        options = ProgressiveOptions(
            levels=levels,
            mode=mode,
            transition_sigmas=manual_sigmas,
            stability_threshold=float(stability_threshold),
            minimum_steps_per_level=int(minimum_steps_per_level),
            full_resolution_steps=int(full_resolution_steps),
        )
        return (
            comfy.samplers.KSAMPLER(
                sample_adaptive_progressive,
                extra_options={"options": options},
            ),
        )
