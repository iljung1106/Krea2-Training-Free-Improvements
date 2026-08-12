from pathlib import Path
import sys
import types


root = Path(__file__).resolve().parents[1]
package = types.ModuleType("krea2_tfi")
package.__path__ = [str(root)]
sys.modules.setdefault("krea2_tfi", package)

from krea2_tfi.nodes import (
    Krea2GeometryAwareAttentionGuidance,
    Krea2NAGNegativePrompt,
)


def test_sigma_inputs_support_three_decimal_places():
    for node in (Krea2GeometryAwareAttentionGuidance, Krea2NAGNegativePrompt):
        required = node.INPUT_TYPES()["required"]
        assert required["sigma_start"][1]["step"] == 0.001
        assert required["sigma_end"][1]["step"] == 0.001
