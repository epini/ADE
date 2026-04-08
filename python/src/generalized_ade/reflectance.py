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



def r_ade(
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
    """Total diffuse reflectance from an anisotropic turbid slab.

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
    r : float
        Total diffuse reflectance [-].
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
        r = 1.0 - (z0 + ze) / (L + 2.0 * ze)
        info = {
            "v": float(v),
            "Dz": float(dz),
            "ze": float(ze),
            "z0": float(z0),
            "num_virtual_sources": 0,
            "regime": "low_absorption_shortcut",
        }
        return (float(r), info) if return_info else float(r)

    kappa = math.sqrt(mua * v / dz)
    m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
    z3 = -2.0 * m * L - 4.0 * m * ze - z0
    z4 = -2.0 * m * L - (4.0 * m - 2.0) * ze + z0

    s = np.sign(z3) * np.exp(-np.abs(z3) * kappa) - np.sign(z4) * np.exp(-np.abs(z4) * kappa)
    r = -0.5 * np.sum(s)

    info = {
        "v": float(v),
        "Dz": float(dz),
        "ze": float(ze),
        "z0": float(z0),
        "kappa": float(kappa),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "regime": "series",
    }
    return (float(r), info) if return_info else float(r)



def rt_ade(
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
    """Time-resolved total diffuse reflectance from an anisotropic slab.

    Parameters
    ----------
    t : array_like
        1D time array [ns].
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
    rt : ndarray
        Time-resolved total reflectance [ns^-1], with the same shape as `t`.
    info : dict, optional
        Numerical diagnostics.
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

    rt = np.zeros_like(t, dtype=float)
    pos = t > 0.0
    if np.any(pos):
        tp = t[pos]
        m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
        z3 = -2.0 * m * L - 4.0 * m * ze - z0
        z4 = -2.0 * m * L - (4.0 * m - 2.0) * ze + z0

        rsum = np.zeros_like(tp)
        for z3m, z4m in zip(z3, z4):
            rsum += z3m * np.exp(-(z3m**2) / (4.0 * dz * tp)) - z4m * np.exp(-(z4m**2) / (4.0 * dz * tp))

        rt[pos] = -0.25 * (np.pi * dz * tp**3) ** (-0.5) * rsum * np.exp(-v * tp * mua)

    info = {
        "v": float(v),
        "Dz": float(dz),
        "ze": float(ze),
        "z0": float(z0),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "shape": tuple(rt.shape),
    }
    return (rt, info) if return_info else rt



def rxy_ade(
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
    """Space-resolved steady-state diffuse reflectance from an anisotropic slab.

    Parameters
    ----------
    x, y : array_like
        1D coordinate arrays [mm].
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
    rxy : ndarray
        Space-resolved steady-state reflectance [mm^-2], shaped as (ny, nx).
        The first axis corresponds to y and the second to x.
    info : dict, optional
        Numerical diagnostics.
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
    rxy = np.zeros((y.size, x.size), dtype=float)

    m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
    for mm in m:
        z3 = -2.0 * mm * L - 4.0 * mm * ze - z0
        z4 = -2.0 * mm * L - (4.0 * mm - 2.0) * ze + z0

        arg3 = z3**2 / dz + xx + yy
        arg4 = z4**2 / dz + xx + yy
        s3 = np.sqrt(mua * v * arg3)
        s4 = np.sqrt(mua * v * arg4)

        rxy += z3 * arg3 ** (-1.5) * (1.0 + s3) * np.exp(-s3) - z4 * arg4 ** (-1.5) * (1.0 + s4) * np.exp(-s4)

    rxy *= -1.0 / (4.0 * np.pi * dgeom ** 1.5)

    info = {
        "v": float(v),
        "Dx": float(dx),
        "Dy": float(dy),
        "Dz": float(dz),
        "D": float(dgeom),
        "ze": float(ze),
        "z0": float(z0),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "shape": tuple(rxy.shape),
    }
    return (rxy, info) if return_info else rxy



def rxyt_ade(
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
    """Time- and space-resolved diffuse reflectance from an anisotropic slab.

    Parameters
    ----------
    x, y : array_like
        1D coordinate arrays [mm].
    t : array_like
        1D time array [ns].
    L : float
        Slab thickness [mm].
    n_in, n_ext : float
        Refractive indices of the slab and external medium [-].
    musx, musy, musz : float
        Principal-axis scattering coefficients [mm^-1].
    g : float
        Henyey-Greenstein asymmetry factor, with -1 < g < 1.
    sx, sy : float
        Initial standard deviations of the lateral Gaussian profile [mm].
    mua : float
        Absorption coefficient [mm^-1].
    return_info : bool, default=False
        If True, also return numerical diagnostics.

    Returns
    -------
    rxyt : ndarray
        Time- and space-resolved reflectance [mm^-2 ns^-1], shaped as (ny, nx, nt).
        The first axis corresponds to y, the second to x, and the third to t.
    info : dict, optional
        Numerical diagnostics.
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

    rxyt = np.zeros((y.size, x.size, t.size), dtype=float)
    pos = t > 0.0
    if np.any(pos):
        tp = t[pos]
        rz = np.zeros_like(tp)
        m = np.arange(-_NUM_VIRTUAL_SOURCES, _NUM_VIRTUAL_SOURCES + 1, dtype=float)
        for mm in m:
            z3 = -2.0 * mm * L - 4.0 * mm * ze - z0
            z4 = -2.0 * mm * L - (4.0 * mm - 2.0) * ze + z0
            rz += z3 * np.exp(-(z3**2) / (4.0 * dz * tp)) - z4 * np.exp(-(z4**2) / (4.0 * dz * tp))

        x2 = x[None, :] ** 2
        y2 = y[:, None] ** 2
        for kk, (k, tk) in enumerate(zip(np.nonzero(pos)[0], tp)):
            denx = 2.0 * sx**2 + 4.0 * dx * tk
            deny = 2.0 * sy**2 + 4.0 * dy * tk
            gx = np.exp(-x2 / denx)
            gy = np.exp(-y2 / deny)
            pref = -1.0 / (2.0 * (4.0 * np.pi) ** 1.5 * tk ** 2.5 * math.sqrt(dx * dy * dz))
            rxyt[:, :, k] = pref * (gy * gx) * rz[kk] * np.exp(-v * tk * mua)

    info = {
        "v": float(v),
        "Dx": float(dx),
        "Dy": float(dy),
        "Dz": float(dz),
        "ze": float(ze),
        "z0": float(z0),
        "num_virtual_sources": _NUM_VIRTUAL_SOURCES,
        "shape": tuple(rxyt.shape),
    }
    return (rxyt, info) if return_info else rxyt
