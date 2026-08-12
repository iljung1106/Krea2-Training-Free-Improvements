from pathlib import Path
import sys
import types


root = Path(__file__).resolve().parents[1]
package = types.ModuleType("krea2_tfi")
package.__path__ = [str(root)]
sys.modules.setdefault("krea2_tfi", package)

from krea2_tfi.nodes import (
    Krea2AdaptiveProgressiveSampler,
    Krea2GeometryAwareAttentionGuidance,
    Krea2NAGNegativePrompt,
    Krea2PromptReinjection,
    Krea2TaylorSeer,
)


def test_sigma_inputs_support_three_decimal_places():
    for node in (Krea2GeometryAwareAttentionGuidance, Krea2NAGNegativePrompt):
        required = node.INPUT_TYPES()["required"]
        assert required["sigma_start"][1]["step"] == 0.001
        assert required["sigma_end"][1]["step"] == 0.001


def test_requested_gag_defaults_and_new_node_defaults():
    gag = Krea2GeometryAwareAttentionGuidance.INPUT_TYPES()["required"]
    assert gag["guidance_scale"][1]["default"] == 5.0
    assert gag["eta"][1]["default"] == 15.0
    assert gag["strength"][1]["default"] == 0.6
    assert gag["sigma_start"][1]["default"] == 0.881
    assert gag["sigma_end"][1]["default"] == 0.678

    reinjection = Krea2PromptReinjection.INPUT_TYPES()["required"]
    assert reinjection["origin_layer"][1]["default"] == 1
    assert reinjection["target_start"][1]["default"] == 2
    assert reinjection["target_end"][1]["default"] == 27
    assert reinjection["weight"][1]["default"] == 0.025

    taylorseer = Krea2TaylorSeer.INPUT_TYPES()["required"]
    assert taylorseer["warmup_steps"][1]["default"] == 3
    assert taylorseer["fresh_interval"][1]["default"] == 2
    assert taylorseer["tail_full_steps"][1]["default"] == 2
    assert taylorseer["max_order"][0] == "INT"
    assert taylorseer["max_order"][1]["default"] == 0
    assert taylorseer["max_order"][1]["min"] == 0
    assert taylorseer["max_order"][1]["max"] == 2
    assert taylorseer["max_order"][1]["step"] == 1

    progressive = Krea2AdaptiveProgressiveSampler.INPUT_TYPES()["required"]
    assert progressive["initial_scale"][1]["default"] == "0.5"
    assert progressive["full_resolution_steps"][1]["default"] == 7
