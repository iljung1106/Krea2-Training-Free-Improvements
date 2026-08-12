from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch


root = Path(__file__).resolve().parents[1]
package = types.ModuleType("krea2_tfi")
package.__path__ = [str(root)]
sys.modules.setdefault("krea2_tfi", package)

from krea2_tfi.adaptive_resolution import (
    ProgressiveOptions,
    _taylorseer_is_active,
    _should_promote,
    haar_merge_2d,
    haar_split_2d,
)


def test_haar_noise_hierarchy_is_exact_and_energy_preserving():
    generator = torch.Generator().manual_seed(2026)
    noise = torch.randn(2, 4, 64, 64, generator=generator)
    low, details = haar_split_2d(noise)
    restored = haar_merge_2d(low, details)
    assert torch.allclose(restored, noise, rtol=1e-6, atol=5e-7)
    child_energy = noise.square().sum()
    coefficient_energy = low.square().sum() + sum(x.square().sum() for x in details)
    assert torch.allclose(coefficient_energy, child_energy, rtol=1e-6, atol=1e-5)
    assert abs(float(low.std()) - 1.0) < 0.03


def test_flow_transition_preserves_clean_scale_and_seed_noise():
    generator = torch.Generator().manual_seed(7)
    full_noise = torch.randn(1, 2, 8, 8, generator=generator)
    parent_noise, details = haar_split_2d(full_noise)
    clean_low = torch.randn(1, 2, 4, 4, generator=generator)
    sigma = torch.tensor(0.6)
    x_low = sigma * parent_noise + (1.0 - sigma) * clean_low

    recovered_parent = (x_low - (1.0 - sigma) * clean_low) / sigma
    recovered_noise = haar_merge_2d(recovered_parent, details)
    clean_high = torch.nn.functional.interpolate(
        clean_low, scale_factor=2.0, mode="bilinear", align_corners=False
    )
    transitioned = sigma * recovered_noise + (1.0 - sigma) * clean_high

    expected = sigma * full_noise + (1.0 - sigma) * clean_high
    assert torch.allclose(transitioned, expected, atol=1e-6)


def test_haar_hierarchy_preserves_krea2_single_frame_latent_shape():
    noise = torch.randn(1, 16, 1, 8, 8)
    low, details = haar_split_2d(noise)
    assert low.shape == (1, 16, 1, 4, 4)
    assert torch.allclose(haar_merge_2d(low, details), noise, rtol=1e-6, atol=5e-7)


def test_adaptive_gate_reserves_a_real_full_resolution_tail():
    options = ProgressiveOptions(
        levels=1,
        mode="adaptive",
        transition_sigmas=(),
        stability_threshold=0.08,
        minimum_steps_per_level=2,
        full_resolution_steps=4,
    )
    promote, reason = _should_promote(
        options=options,
        level_index=0,
        steps_on_level=6,
        stable_count=0,
        sigma_next=0.4,
        future_steps=4,
    )
    assert promote
    assert reason == "reserved full-resolution tail"


def test_taylorseer_disables_a_false_stability_signal():
    state = SimpleNamespace(taylorseer=object())
    patcher = SimpleNamespace(get_attachment=lambda _key: state)
    model = SimpleNamespace(inner_model=SimpleNamespace(model_patcher=patcher))
    assert _taylorseer_is_active(model)
