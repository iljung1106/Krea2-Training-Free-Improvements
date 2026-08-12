from dataclasses import replace
from pathlib import Path
import sys
import types
from types import SimpleNamespace

import torch

root = Path(__file__).resolve().parents[1]
package = types.ModuleType("krea2_tfi")
package.__path__ = [str(root)]
sys.modules.setdefault("krea2_tfi", package)
from krea2_tfi.krea2_guidance import (
    GAGState,
    GuidanceState,
    NAGState,
    PromptReinjectionState,
    TaylorRuntime,
    TaylorSeerState,
    _in_sigma_range,
    _taylor_mode,
    _unpack_output,
    apply_gag_to_image_queries,
    apply_prompt_reinjection,
    taylor_forecast,
    taylor_update,
)


def test_gag_default_range_selects_the_intended_ten_step_simple_calls():
    schedule = torch.tensor(
        [1.0, 0.9660, 0.9266, 0.8805, 0.8257, 0.7595, 0.6780, 0.5751, 0.4412, 0.2598]
    )
    active = [
        index
        for index, sigma in enumerate(schedule)
        if _in_sigma_range({"sigmas": sigma.reshape(1)}, 0.881, 0.678)
    ]
    assert active == [3, 4, 5, 6]


def test_displayed_sigma_boundary_includes_scheduler_rounding_residue():
    assert _in_sigma_range({"sigmas": torch.tensor([0.677987])}, 0.881, 0.678)


def test_all_improvement_states_compose_without_replacing_each_other():
    gag = GAGState(10.0, 15.0, 1.0, 1000.0, 0.0, 256)
    nag = NAGState(torch.zeros(1, 2, 3), 4.0, 2.5, 0.25, 1000.0, 0.0)
    reinjection = PromptReinjectionState(1, 2, 27, 0.025, False)
    taylorseer = TaylorSeerState(3, 2, 2, 1, TaylorRuntime())
    state = replace(GuidanceState(), gag=gag)
    state = replace(state, nag=nag)
    state = replace(state, prompt_reinjection=reinjection)
    state = replace(state, taylorseer=taylorseer)
    assert state.gag is gag
    assert state.nag is nag
    assert state.prompt_reinjection is reinjection
    assert state.taylorseer is taylorseer


def test_gag_preserves_native_attention_shape_and_text_prefix():
    torch.manual_seed(7)
    batch, heads, text, image, head_dim = 1, 2, 3, 4, 5
    q = torch.randn(batch, heads, text + image, head_dim)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw = torch.randn(batch, text + image, heads * head_dim)
    state = GAGState(2.0, 1.0, 1.0, 1000.0, 0.0, 2)
    result = apply_gag_to_image_queries(raw, q, k, v, text, state)
    assert result.shape == raw.shape
    assert torch.equal(result[:, :text], raw[:, :text])
    assert not torch.equal(result[:, text:], raw[:, text:])


def test_prompt_reinjection_matches_basic_rule_and_anchoring_statistics():
    torch.manual_seed(11)
    target = torch.randn(2, 3, 8)
    origin = torch.randn_like(target)
    basic = apply_prompt_reinjection(target, origin, 0.025, False)
    assert torch.allclose(basic, target + 0.025 * origin)

    anchored = apply_prompt_reinjection(target, origin, 0.2, True)
    assert torch.allclose(anchored.mean(-1), target.mean(-1), atol=1e-6)
    # The official implementation records torch.std's sample deviation, then
    # restores it after LayerNorm (which uses population variance).
    assert torch.allclose(
        anchored.std(-1, unbiased=False), target.std(-1), atol=2e-6
    )


def test_taylor_factors_forecast_linear_feature_progression():
    factors = taylor_update({}, torch.tensor([1.0]), distance=1, max_order=1)
    factors = taylor_update(factors, torch.tensor([3.0]), distance=2, max_order=1)
    forecast = taylor_forecast(factors, distance=1)
    assert torch.equal(forecast, torch.tensor([4.0]))


def test_taylor_derivatives_start_only_after_official_warmup_gate():
    factors = taylor_update(
        {0: torch.tensor([1.0])},
        torch.tensor([2.0]),
        distance=1,
        max_order=2,
        allow_derivatives=False,
    )
    assert list(factors) == [0]
    factors = taylor_update(
        factors,
        torch.tensor([3.0]),
        distance=1,
        max_order=2,
        allow_derivatives=True,
    )
    assert list(factors) == [0, 1]


def test_taylor_lite_final_tokens_unpack_to_latent_shape():
    final = torch.arange(16.0).reshape(1, 4, 4)
    model = SimpleNamespace(patch=2, channels=1)
    output = _unpack_output(final, 4, 2, 2, 4, 4, model, False, None, None)
    assert output.shape == (1, 1, 4, 4)
    expected = torch.tensor(
        [[[[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]]]],
        dtype=torch.float32,
    )
    assert torch.equal(output, expected)


def test_taylor_lite_keeps_last_two_steps_full():
    state = TaylorSeerState(3, 2, 2, 1, TaylorRuntime())
    schedule = torch.linspace(1.0, 0.0, 11)
    modes = []
    for step in range(10):
        options = {
            "sigmas": schedule[step:step + 1],
            "sample_sigmas": schedule,
            "cond_or_uncond": [0],
        }
        mode, _, _ = _taylor_mode(options, state)
        modes.append(mode)
        if mode == "full":
            state.runtime.last_full_step = step
    assert modes == [
        "full", "full", "full", "forecast", "full",
        "forecast", "full", "forecast", "full", "full",
    ]
