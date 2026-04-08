from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.special import gammaln, lpmv

from .quadrature import gauss_legendre


C0_MM_PER_NS = 299.792458


def _require_positive_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar.")
    return value


def _require_nonnegative_scalar(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a nonnegative finite scalar.")
    return value


def _require_int(name: str, value: int, *, positive: bool = False) -> int:
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    value = int(value)
    if positive and value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def d_tensor_ade(
    n_in: float,
    musx: float,
    musy: float,
    musz: float,
    g: float,
    *,
    lmax_start: int = 15,
    nchi: int = 200,
    nphi: int = 512,
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-10,
    return_info: bool = False,
) -> tuple[float, float, float] | tuple[float, float, float, dict[str, Any]]:
    """Diffusion tensor for anisotropic scattering in the generalized ADE model.

    Parameters
    ----------
    n_in : float
        Refractive index of the medium [-].
    musx, musy, musz : float
        Principal-axis scattering coefficients [mm^-1].
    g : float
        Henyey-Greenstein asymmetry factor, with -1 < g < 1.
    lmax_start : int, default=15
        Initial odd harmonic order used for convergence checking.
    nchi : int, default=200
        Number of Gauss-Legendre nodes in chi = cos(theta).
    nphi : int, default=512
        Number of azimuthal samples in phi.
    rel_tol : float, default=1e-5
        Relative convergence tolerance for the harmonic series.
    abs_tol : float, default=1e-10
        Absolute convergence tolerance for the harmonic series.
    return_info : bool, default=False
        If True, also return a dictionary with numerical diagnostics.

    Returns
    -------
    dx, dy, dz : float
        Diffusion coefficients [mm^2/ns].
    info : dict, optional
        Numerical diagnostics.
    """
    n_in = _require_positive_scalar("n_in", n_in)
    musx = _require_positive_scalar("musx", musx)
    musy = _require_positive_scalar("musy", musy)
    musz = _require_positive_scalar("musz", musz)

    g = float(g)
    if not np.isfinite(g) or not (-1.0 < g < 1.0):
        raise ValueError("g must satisfy -1 < g < 1.")

    lmax_start = _require_int("lmax_start", lmax_start, positive=True)
    nchi = _require_int("nchi", nchi, positive=True)
    nphi = _require_int("nphi", nphi, positive=True)
    rel_tol = _require_positive_scalar("rel_tol", rel_tol)
    abs_tol = _require_nonnegative_scalar("abs_tol", abs_tol)

    if lmax_start % 2 == 0:
        lmax_start += 1
        warnings.warn(
            "lmax_start was even; incremented to the next odd value.",
            RuntimeWarning,
            stacklevel=2,
        )

    v = C0_MM_PER_NS / n_in  # [mm/ns]

    tol_iso = 1e-12 * max(musx, musy, musz)
    is_iso = abs(musx - musy) <= tol_iso and abs(musx - musz) <= tol_iso
    if is_iso:
        d = v / (3.0 * musx * (1.0 - g))
        info = {
            "v": v,
            "lavg": 1.0 / musx,
            "D0": np.array([d, d, d], dtype=float),
            "LmaxUsed": 0,
            "converged": True,
            "Nchi": nchi,
            "Nphi": nphi,
            "RelTol": rel_tol,
            "AbsTol": abs_tol,
            "LmaxStart": lmax_start,
            "note": "Isotropic shortcut: D = v/(3*musx*(1-g)).",
        }
        return (d, d, d, info) if return_info else (d, d, d)

    chi, wchi = gauss_legendre(nchi)
    phi = np.arange(nphi, dtype=float) * (2.0 * np.pi / nphi)
    wphi = 2.0 * np.pi / nphi

    PHI, CHI = np.meshgrid(phi, chi, indexing="xy")
    S = np.sqrt(np.maximum(0.0, 1.0 - CHI**2))

    cos_phi = np.cos(PHI)
    sin_phi = np.sin(PHI)

    W = wchi[:, None] * wphi

    sx = S * cos_phi
    sy = S * sin_phi
    sz = CHI

    mu = (
        musx * (1.0 - CHI**2) * cos_phi**2
        + musy * (1.0 - CHI**2) * sin_phi**2
        + musz * CHI**2
    )
    invmu = 1.0 / mu

    lavg = (invmu * W).sum() / (4.0 * np.pi)

    ix = ((sx**2) * (invmu**2) * W).sum()
    iy = ((sy**2) * (invmu**2) * W).sum()
    iz = ((sz**2) * (invmu**2) * W).sum()

    prefactor = v / (4.0 * np.pi * lavg)
    dx0 = prefactor * ix
    dy0 = prefactor * iy
    dz0 = prefactor * iz

    dx, dy, dz = dx0, dy0, dz0
    converged = True
    lmax_used = 0

    if g != 0.0:
        lmax_cap = max(lmax_start, 101)

        bx = sx * invmu * W
        by = sy * invmu * W
        bz = sz * invmu * W

        corr_x = 0.0
        corr_y = 0.0
        corr_z = 0.0

        dx_prev = np.nan
        dy_prev = np.nan
        dz_prev = np.nan
        converged = False

        for ell in range(1, lmax_cap + 1, 2):
            denom = 1.0 - g**ell
            coeff = (g**ell) / denom

            mpos = np.arange(ell + 1)
            e = np.exp(1j * np.outer(mpos, phi))

            sx_acc = bx @ e.T
            sy_acc = by @ e.T
            sz_acc = bz @ e.T

            hx = np.zeros(ell + 1, dtype=np.complex128)
            hy = np.zeros(ell + 1, dtype=np.complex128)
            hz = np.zeros(ell + 1, dtype=np.complex128)

            for k, m in enumerate(mpos):
                log_norm = 0.5 * (
                    math.log(2 * ell + 1)
                    - math.log(2.0)
                    + gammaln(ell - m + 1)
                    - gammaln(ell + m + 1)
                )
                a = np.exp(log_norm) * lpmv(m, ell, chi)
                ylm_theta = a / math.sqrt(2.0 * np.pi)

                hx[k] = np.dot(ylm_theta, sx_acc[:, k])
                hy[k] = np.dot(ylm_theta, sy_acc[:, k])
                hz[k] = np.dot(ylm_theta, sz_acc[:, k])

            sum_x = abs(hx[0]) ** 2 + 2.0 * np.sum(abs(hx[1:]) ** 2)
            sum_y = abs(hy[0]) ** 2 + 2.0 * np.sum(abs(hy[1:]) ** 2)
            sum_z = abs(hz[0]) ** 2 + 2.0 * np.sum(abs(hz[1:]) ** 2)

            corr_x += coeff * sum_x
            corr_y += coeff * sum_y
            corr_z += coeff * sum_z

            dx_new = dx0 + prefactor * corr_x
            dy_new = dy0 + prefactor * corr_y
            dz_new = dz0 + prefactor * corr_z

            if ell >= lmax_start and not np.isnan(dx_prev):
                okx = abs(dx_new - dx_prev) <= abs_tol + rel_tol * abs(dx_new)
                oky = abs(dy_new - dy_prev) <= abs_tol + rel_tol * abs(dy_new)
                okz = abs(dz_new - dz_prev) <= abs_tol + rel_tol * abs(dz_new)
                if okx and oky and okz:
                    dx, dy, dz = float(dx_new), float(dy_new), float(dz_new)
                    converged = True
                    lmax_used = ell
                    break

            dx_prev, dy_prev, dz_prev = dx_new, dy_new, dz_new
            dx, dy, dz = float(dx_new), float(dy_new), float(dz_new)
            lmax_used = ell

        if not converged:
            warnings.warn(
                "g-correction did not meet tolerance. Consider increasing "
                "lmax_start (or, secondarily, nchi/nphi).",
                RuntimeWarning,
                stacklevel=2,
            )

    info = {
        "v": v,
        "lavg": float(lavg),
        "D0": np.array([dx0, dy0, dz0], dtype=float),
        "LmaxUsed": lmax_used,
        "converged": bool(converged),
        "Nchi": nchi,
        "Nphi": nphi,
        "RelTol": rel_tol,
        "AbsTol": abs_tol,
        "LmaxStart": lmax_start,
    }

    return (dx, dy, dz, info) if return_info else (dx, dy, dz)
