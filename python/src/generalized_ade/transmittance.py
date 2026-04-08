from __future__ import annotations

import math
from typing import Any

import numpy as np

from .boundary import bc_ade
from .diffusion import C0_MM_PER_NS, _require_nonnegative_scalar, _require_positive_scalar, d_tensor_ade

_NUM_VIRTUAL_SOURCES = 10000


def _require_real_1d_array(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a non-empty finite 1D array.")
    return arr



def t_ade(
    L: float,
    n_in: float,
    n_ext: float,
    musx: float,
    musy: float,
    musz: float,
    g: float,
    mua: float,
    *,
    return_info: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Total diffuse transmittance through an anisotropic turbid slab.

    Parameters
    ----------
    L : float
        Slab thickness [mm].
    n_in, n_ext : float
        Refractive indices of the slab and external medium [-].
    musx, musy, musz : float
        Principal-axis scattering coefficients [mm^-1].
    g : float
        Henyey-Greenstein asymmetry factor, with -1 < g < 1.
    mua : float
        Absorption coefficient [mm^-1].
    return_info : bool, default=False
        If True, also return numerical diagnostics.

    Returns
    -------
    t : float
        Total diffuse transmittance [-].
    info : dict, optional
        Numerical diagnostics.
    """
    L = _require_positive_scalar("L", L)
    n_in = _require_positive_scalar("n_in", n_in)
    n_ext = _require_positive_scalar("n_ext", n_ext)
    musx = _require_positive_scalar("musx", musx)
    musy = _require_positive_scalar("musy", musy)
    musz = _require_positive_scalar("musz", musz)
    mua = _require_nonnegative_scalar("mua", mua)

    g = float(g)
    if not np.isfinite(g) or not (-1.0 < g < 1.0):
        raise ValueError("g must satisfy -1 < g < 1.")

    _, _, dz = d_tensor_ade(n_in, musx, musy, musz, g)
    ze, z0 = bc_ade(n_in, n_ext, musx, musy, musz, g)

    v = C0_MM_PER_NS / n_in  # [mm/ns]

    if mua * z0 < 1e-10:
        t = (z0 + ze) / (L + 2.0 * ze)
        info = {
            "v": float(v),
            "Dz": float(dz),
            "ze": float(ze),
            "z0": float(z0),
            "num_virtual_sources": 0,
            "regime": "low_absorption_shortcut",
        }
        return (float(t), info) if return_info else float(t)

    kappa = math.sqrt(mua * v / dz)
    m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
    z1 = L * (1.0 - 2.0 * m) - 4.0 * m * ze - z0
    z2 = L * (1.0 - 2.0 * m) - (4.0 * m - 2.0) * ze + z0

    s = np.sign(z1) * np.exp(-np.abs(z1) * kappa) - np.sign(z2) * np.exp(-np.abs(z2) * kappa)
    t = 0.5 * np.sum(s)

    info = {
        "v": float(v),
        "Dz": float(dz),
        "ze": float(ze),
        "z0": float(z0),
        "kappa": float(kappa),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "regime": "series",
    }
    return (float(t), info) if return_info else float(t)



def tt_ade(
    t: Any,
    L: float,
    n_in: float,
    n_ext: float,
    musx: float,
    musy: float,
    musz: float,
    g: float,
    mua: float,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Time-resolved total diffuse transmittance through an anisotropic slab.

    Returns an array with the same shape as the input 1D time grid.
    """
    t = _require_real_1d_array("t", t)
    L = _require_positive_scalar("L", L)
    n_in = _require_positive_scalar("n_in", n_in)
    n_ext = _require_positive_scalar("n_ext", n_ext)
    musx = _require_positive_scalar("musx", musx)
    musy = _require_positive_scalar("musy", musy)
    musz = _require_positive_scalar("musz", musz)
    mua = _require_nonnegative_scalar("mua", mua)

    g = float(g)
    if not np.isfinite(g) or not (-1.0 < g < 1.0):
        raise ValueError("g must satisfy -1 < g < 1.")

    _, _, dz = d_tensor_ade(n_in, musx, musy, musz, g)
    ze, z0 = bc_ade(n_in, n_ext, musx, musy, musz, g)
    v = C0_MM_PER_NS / n_in

    tt = np.zeros_like(t, dtype=float)
    pos = t > 0.0
    if np.any(pos):
        tp = t[pos]
        tsum = np.zeros_like(tp)
        m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
        for mm in m:
            z1 = L * (1.0 - 2.0 * mm) - 4.0 * mm * ze - z0
            z2 = L * (1.0 - 2.0 * mm) - (4.0 * mm - 2.0) * ze + z0
            tsum += z1 * np.exp(-(z1**2) / (4.0 * dz * tp)) - z2 * np.exp(-(z2**2) / (4.0 * dz * tp))

        tt[pos] = 0.25 * (np.pi * dz * tp**3) ** (-0.5) * tsum * np.exp(-v * tp * mua)

    info = {
        "v": float(v),
        "Dz": float(dz),
        "ze": float(ze),
        "z0": float(z0),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "shape": tuple(tt.shape),
    }
    return (tt, info) if return_info else tt



def txy_ade(
    x: Any,
    y: Any,
    L: float,
    n_in: float,
    n_ext: float,
    musx: float,
    musy: float,
    musz: float,
    g: float,
    mua: float,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Space-resolved steady-state diffuse transmittance through an anisotropic slab.

    Returns an array shaped as (ny, nx), with rows corresponding to y and columns to x.
    """
    x = _require_real_1d_array("x", x)
    y = _require_real_1d_array("y", y)
    L = _require_positive_scalar("L", L)
    n_in = _require_positive_scalar("n_in", n_in)
    n_ext = _require_positive_scalar("n_ext", n_ext)
    musx = _require_positive_scalar("musx", musx)
    musy = _require_positive_scalar("musy", musy)
    musz = _require_positive_scalar("musz", musz)
    mua = _require_nonnegative_scalar("mua", mua)

    g = float(g)
    if not np.isfinite(g) or not (-1.0 < g < 1.0):
        raise ValueError("g must satisfy -1 < g < 1.")

    dx, dy, dz = d_tensor_ade(n_in, musx, musy, musz, g)
    ze, z0 = bc_ade(n_in, n_ext, musx, musy, musz, g)
    v = C0_MM_PER_NS / n_in
    dgeom = (dx * dy * dz) ** (1.0 / 3.0)

    xx = x[None, :] ** 2 / dx
    yy = y[:, None] ** 2 / dy
    txy = np.zeros((y.size, x.size), dtype=float)

    m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
    for mm in m:
        z1 = L * (1.0 - 2.0 * mm) - 4.0 * mm * ze - z0
        z2 = L * (1.0 - 2.0 * mm) - (4.0 * mm - 2.0) * ze + z0

        arg1 = z1**2 / dz + xx + yy
        arg2 = z2**2 / dz + xx + yy
        s1 = np.sqrt(mua * v * arg1)
        s2 = np.sqrt(mua * v * arg2)

        txy += z1 * arg1 ** (-1.5) * (1.0 + s1) * np.exp(-s1) - z2 * arg2 ** (-1.5) * (1.0 + s2) * np.exp(-s2)

    txy *= 1.0 / (4.0 * np.pi * dgeom ** 1.5)

    info = {
        "v": float(v),
        "Dx": float(dx),
        "Dy": float(dy),
        "Dz": float(dz),
        "D": float(dgeom),
        "ze": float(ze),
        "z0": float(z0),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "shape": tuple(txy.shape),
    }
    return (txy, info) if return_info else txy



def txyt_ade(
    x: Any,
    y: Any,
    t: Any,
    L: float,
    n_in: float,
    n_ext: float,
    musx: float,
    musy: float,
    musz: float,
    g: float,
    sx: float,
    sy: float,
    mua: float,
    *,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Time- and space-resolved diffuse transmittance through an anisotropic slab.

    Returns an array shaped as (ny, nx, nt), with axes corresponding to y, x, and t.
    """
    x = _require_real_1d_array("x", x)
    y = _require_real_1d_array("y", y)
    t = _require_real_1d_array("t", t)
    L = _require_positive_scalar("L", L)
    n_in = _require_positive_scalar("n_in", n_in)
    n_ext = _require_positive_scalar("n_ext", n_ext)
    musx = _require_positive_scalar("musx", musx)
    musy = _require_positive_scalar("musy", musy)
    musz = _require_positive_scalar("musz", musz)
    sx = _require_nonnegative_scalar("sx", sx)
    sy = _require_nonnegative_scalar("sy", sy)
    mua = _require_nonnegative_scalar("mua", mua)

    g = float(g)
    if not np.isfinite(g) or not (-1.0 < g < 1.0):
        raise ValueError("g must satisfy -1 < g < 1.")

    dx, dy, dz = d_tensor_ade(n_in, musx, musy, musz, g)
    ze, z0 = bc_ade(n_in, n_ext, musx, musy, musz, g)
    v = C0_MM_PER_NS / n_in

    txyt = np.zeros((y.size, x.size, t.size), dtype=float)
    pos = t > 0.0
    if np.any(pos):
        tp = t[pos]
        tz = np.zeros_like(tp)
        m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
        for mm in m:
            z1 = L * (1.0 - 2.0 * mm) - 4.0 * mm * ze - z0
            z2 = L * (1.0 - 2.0 * mm) - (4.0 * mm - 2.0) * ze + z0
            tz += z1 * np.exp(-(z1**2) / (4.0 * dz * tp)) - z2 * np.exp(-(z2**2) / (4.0 * dz * tp))

        x2 = x[None, :] ** 2
        y2 = y[:, None] ** 2
        for kk, (k, tk) in enumerate(zip(np.nonzero(pos)[0], tp)):
            denx = 2.0 * sx**2 + 4.0 * dx * tk
            deny = 2.0 * sy**2 + 4.0 * dy * tk
            gx = np.exp(-x2 / denx)
            gy = np.exp(-y2 / deny)
            pref = 1.0 / (2.0 * (4.0 * np.pi) ** 1.5 * tk ** 2.5 * math.sqrt(dx * dy * dz))
            txyt[:, :, k] = pref * (gy * gx) * tz[kk] * np.exp(-v * tk * mua)

    info = {
        "v": float(v),
        "Dx": float(dx),
        "Dy": float(dy),
        "Dz": float(dz),
        "ze": float(ze),
        "z0": float(z0),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "shape": tuple(txyt.shape),
    }
    return (txyt, info) if return_info else txyt
