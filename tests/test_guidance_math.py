from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guidance_math import entmax15, geometry_aware_attention, normalized_attention_guidance


def test_entmax15_is_normalized_and_sparse():
    logits = torch.tensor([[5.0, 1.0, -2.0]], dtype=torch.float16)
    probabilities = entmax15(logits)
    assert probabilities.dtype == logits.dtype
    assert torch.allclose(probabilities.float().sum(-1), torch.ones(1), atol=1e-4)
    assert probabilities[0, 2] == 0


def test_gag_strength_zero_is_dense_and_parallel_only_is_finite():
    dense = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float16)
    sparse = torch.tensor([[[[2.0, 1.0]]]], dtype=torch.float16)
    disabled = geometry_aware_attention(dense, sparse, 10.0, 15.0, strength=0.0)
    enabled = geometry_aware_attention(dense, sparse, 2.0, 0.5, strength=1.0)
    assert torch.equal(disabled, dense)
    assert torch.isfinite(enabled).all()
    assert not torch.equal(enabled, dense)


def test_nag_alpha_zero_is_positive():
    positive = torch.randn(1, 2, 3, 4, dtype=torch.float16)
    negative = torch.randn_like(positive)
    result = normalized_attention_guidance(positive, negative, 4.0, 2.5, 0.0)
    assert torch.equal(result, positive)
