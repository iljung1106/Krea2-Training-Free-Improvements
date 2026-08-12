from dataclasses import replace
from pathlib import Path
import sys
import types

import torch

root = Path(__file__).resolve().parents[1]
package = types.ModuleType("krea2_tfi")
package.__path__ = [str(root)]
sys.modules.setdefault("krea2_tfi", package)
from krea2_tfi.krea2_guidance import (
    GAGState,
    GuidanceState,
    NAGState,
    apply_gag_to_image_queries,
)


def test_gag_and_nag_states_compose_without_replacing_each_other():
    gag = GAGState(10.0, 15.0, 1.0, 1000.0, 0.0, 256)
    nag = NAGState(torch.zeros(1, 2, 3), 4.0, 2.5, 0.25, 1000.0, 0.0)
    state = replace(GuidanceState(), gag=gag)
    state = replace(state, nag=nag)
    assert state.gag is gag
    assert state.nag is nag


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
