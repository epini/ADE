import numpy as np
import pytest

from generalized_ade import (
    d_tensor_ade,
    bc_ade,
    r_ade,
    rt_ade,
    rxy_ade,
    rxyt_ade,
    t_ade,
    tt_ade,
    txy_ade,
    txyt_ade,
)


@pytest.mark.smoke
def test_smoke_all_functions_run():
    n_in = 1.4
    n_ext = 1.0
    musx, musy, musz = 12.0, 8.0, 5.0
    g = 0.85
    mua = 0.01
    L = 20.0
    x = np.linspace(-2.0, 2.0, 9)
    y = np.linspace(-2.0, 2.0, 7)
    t = np.linspace(0.02, 0.3, 11)

    dx, dy, dz = d_tensor_ade(n_in, musx, musy, musz, g)
    ze, z0 = bc_ade(n_in, n_ext, musx, musy, musz, g)
    R = r_ade(L, n_in, n_ext, musx, musy, musz, g, mua)
    T = t_ade(L, n_in, n_ext, musx, musy, musz, g, mua)
    Rt = rt_ade(t, L, n_in, n_ext, musx, musy, musz, g, mua)
    Tt = tt_ade(t, L, n_in, n_ext, musx, musy, musz, g, mua)
    Rxy = rxy_ade(x, y, L, n_in, n_ext, musx, musy, musz, g, mua)
    Txy = txy_ade(x, y, L, n_in, n_ext, musx, musy, musz, g, mua)
    Rxyt = rxyt_ade(x, y, t, L, n_in, n_ext, musx, musy, musz, g, 0.05, 0.05, mua)
    Txyt = txyt_ade(x, y, t, L, n_in, n_ext, musx, musy, musz, g, 0.05, 0.05, mua)

    assert dx > 0 and dy > 0 and dz > 0
    assert ze > 0 and z0 > 0
    assert np.isfinite(R) and np.isfinite(T)
    assert Rt.shape == t.shape
    assert Tt.shape == t.shape
    assert Rxy.shape == (y.size, x.size)
    assert Txy.shape == (y.size, x.size)
    assert Rxyt.shape == (y.size, x.size, t.size)
    assert Txyt.shape == (y.size, x.size, t.size)
