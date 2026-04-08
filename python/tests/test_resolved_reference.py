import json
from pathlib import Path

import numpy as np
import pytest

from generalized_ade import r_ade, rt_ade, rxy_ade, rxyt_ade, t_ade, tt_ade, txy_ade, txyt_ade

REFERENCE = Path(__file__).parent / "reference" / "resolved_reference.json"


def rel_err_array(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.maximum.reduce([np.abs(a), np.abs(b), np.full_like(a, 1e-300, dtype=float)])
    return np.max(np.abs(a - b) / denom)


def _maybe_fix_orientation(arr, target_shape):
    arr = np.asarray(arr)
    if arr.shape == tuple(target_shape):
        return arr
    if arr.ndim == 2 and arr.T.shape == tuple(target_shape):
        return arr.T
    if arr.ndim == 3 and arr.shape[:2] == tuple(target_shape[:2][::-1]) and arr.shape[2] == target_shape[2]:
        return np.transpose(arr, (1, 0, 2))
    raise AssertionError(f"Unexpected shape {arr.shape}, expected {target_shape}")


@pytest.mark.reference
def test_resolved_against_matlab_reference():
    if not REFERENCE.exists():
        pytest.skip("resolved_reference.json not found.")

    data = json.loads(REFERENCE.read_text())
    p = data["params"]
    x = np.asarray(data["x_mm"], dtype=float)
    y = np.asarray(data["y_mm"], dtype=float)
    t = np.asarray(data["t_ns"], dtype=float)

    max_rel = 0.0

    R = r_ade(p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["mua"])
    T = t_ade(p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["mua"])
    max_rel = max(max_rel, rel_err_array(R, data["R"]))
    max_rel = max(max_rel, rel_err_array(T, data["T"]))

    Rt = rt_ade(t, p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["mua"])
    Tt = tt_ade(t, p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["mua"])
    max_rel = max(max_rel, rel_err_array(Rt, data["Rt"]))
    max_rel = max(max_rel, rel_err_array(Tt, data["Tt"]))

    Rxy = _maybe_fix_orientation(
        rxy_ade(x, y, p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["mua"]),
        np.asarray(data["Rxy"]).shape,
    )
    Txy = _maybe_fix_orientation(
        txy_ade(x, y, p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["mua"]),
        np.asarray(data["Txy"]).shape,
    )
    max_rel = max(max_rel, rel_err_array(Rxy, data["Rxy"]))
    max_rel = max(max_rel, rel_err_array(Txy, data["Txy"]))

    Rxyt = _maybe_fix_orientation(
        rxyt_ade(x, y, t, p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["sx"], p["sy"], p["mua"]),
        np.asarray(data["Rxyt"]).shape,
    )
    Txyt = _maybe_fix_orientation(
        txyt_ade(x, y, t, p["L"], p["n_in"], p["n_ext"], p["musx"], p["musy"], p["musz"], p["g"], p["sx"], p["sy"], p["mua"]),
        np.asarray(data["Txyt"]).shape,
    )
    max_rel = max(max_rel, rel_err_array(Rxyt, data["Rxyt"]))
    max_rel = max(max_rel, rel_err_array(Txyt, data["Txyt"]))

    assert max_rel < 1e-9
