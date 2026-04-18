import json
from pathlib import Path

import pytest

from generalized_ade import d_tensor_ade

REFERENCE = Path(__file__).parent / "reference" / "d_tensor_reference.json"


def rel_err(a, b):
    denom = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denom


@pytest.mark.reference
def test_diffusion_against_matlab_reference():
    if not REFERENCE.exists():
        pytest.skip("d_tensor_reference.json not found.")

    # export_d_tensor_reference.m writes a top-level JSON array of cases.
    cases = json.loads(REFERENCE.read_text())
    max_rel = 0.0
    for case in cases:
        dx, dy, dz = d_tensor_ade(
            case["n_in"], case["musx"], case["musy"], case["musz"], case["g"]
        )
        for py_val, mat_val in zip((dx, dy, dz), (case["Dx"], case["Dy"], case["Dz"])):
            max_rel = max(max_rel, rel_err(py_val, mat_val))

    assert max_rel < 1e-10
