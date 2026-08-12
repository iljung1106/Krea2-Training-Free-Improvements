from __future__ import annotations

import torch


def entmax15(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact alpha=1.5 Entmax with float32 internal arithmetic."""
    dtype = logits.dtype
    values = logits.float() / 2.0
    sorted_values, _ = torch.sort(values, dim=dim, descending=True)
    count = values.shape[dim]
    rho_shape = [1] * values.ndim
    rho_shape[dim] = count
    rho = torch.arange(1, count + 1, device=values.device, dtype=torch.float32).view(rho_shape)

    mean = sorted_values.cumsum(dim) / rho
    mean_square = sorted_values.square().cumsum(dim) / rho
    variance_sum = rho * (mean_square - mean.square())
    delta = (1.0 - variance_sum) / rho
    taus = mean - delta.clamp_min(0.0).sqrt()
    support = (taus <= sorted_values).sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = taus.gather(dim, support - 1)
    probabilities = (values - threshold).clamp_min(0.0).square()
    probabilities = probabilities / probabilities.sum(dim=dim, keepdim=True).clamp_min(
        torch.finfo(torch.float32).eps
    )
    return probabilities.to(dtype=dtype)


def geometry_aware_attention(
    dense: torch.Tensor,
    sparse: torch.Tensor,
    guidance_scale: float,
    eta: float,
    strength: float = 1.0,
) -> torch.Tensor:
    """GAG equation 13 with zeta=0, blended against dense attention."""
    if dense.shape != sparse.shape:
        raise ValueError(f"GAG attention shapes must match, got {dense.shape} and {sparse.shape}")

    dtype = dense.dtype
    dense32 = dense.float()
    sparse32 = sparse.float()
    residual = sparse32 - dense32
    eps = torch.finfo(torch.float32).eps
    sparse_norm_sq = sparse32.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    parallel = (
        (residual * sparse32).sum(dim=-1, keepdim=True) / sparse_norm_sq
    ) * sparse32
    parallel_norm = parallel.square().sum(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    capped_parallel = parallel * (float(eta) / parallel_norm).clamp_max(1.0)
    gag = sparse32 + float(guidance_scale) * capped_parallel
    return (dense32 + float(strength) * (gag - dense32)).to(dtype=dtype)


def normalized_attention_guidance(
    positive: torch.Tensor,
    negative: torch.Tensor,
    phi: float,
    tau: float,
    alpha: float,
) -> torch.Tensor:
    """NAG equations 7-10 with float32 norm evaluation."""
    if positive.shape != negative.shape:
        raise ValueError(
            f"NAG attention shapes must match, got {positive.shape} and {negative.shape}"
        )

    dtype = positive.dtype
    positive32 = positive.float()
    negative32 = negative.float()
    guided = positive32 + float(phi) * (positive32 - negative32)
    eps = torch.finfo(torch.float32).eps
    positive_norm = positive32.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    guided_norm = guided.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    ratio = guided_norm / positive_norm
    normalized = guided * (ratio.clamp_max(float(tau)) / ratio)
    return (
        float(alpha) * normalized + (1.0 - float(alpha)) * positive32
    ).to(dtype=dtype)
