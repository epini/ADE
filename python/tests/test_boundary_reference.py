import json
from pathlib import Path

import pytest

from generalized_ade import bc_ade

REFERENCE = Path(__file__).parent / "reference" / "bc_reference.json"


def rel_err(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


@pytest.mark.reference
def test_boundary_against_matlab_reference():
    if not REFERENCE.exists():
        pytest.skip("bc_reference.json not found.")

    data = json.loads(REFERENCE.read_text())
    max_rel = 0.0
    for case in data["cases"]:
        ze, z0 = bc_ade(
            case["n_in"], case["n_ext"], case["musx"], case["musy"], case["musz"], case["g"]
        )
        max_rel = max(max_rel, rel_err(ze, case["ze"]))
        max_rel = max(max_rel, rel_err(z0, case["z0"]))

    assert max_rel < 1e-10
