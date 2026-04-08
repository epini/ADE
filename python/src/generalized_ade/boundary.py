from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy.special import eval_legendre, gammaln, lpmv

from .diffusion import C0_MM_PER_NS, _require_int, _require_nonnegative_scalar, _require_positive_scalar
from .quadrature import gauss_legendre


def _fresnel_r(chi: np.ndarray, n: float) -> np.ndarray:
    """Unpolarized Fresnel reflectance for internal incidence, with TIR handling."""
    chi = np.asarray(chi, dtype=float)
    trans_arg = 1.0 - (1.0 - chi**2) * n**2
    r = np.zeros_like(chi)

    ok = trans_arg >= 0.0
    if np.any(ok):
        t = np.sqrt(trans_arg[ok])
        r1 = (n * chi[ok] - t) / (n * chi[ok] + t)
        r2 = (chi[ok] - n * t) / (chi[ok] + n * t)
        r[ok] = 0.5 * (np.abs(r1) ** 2 + np.abs(r2) ** 2)
    if np.any(~ok):
        r[~ok] = 1.0
    return r


def _norm_assoc_legendre_with_cs(ell: int, m: int, x: np.ndarray) -> np.ndarray:
    """Normalized associated Legendre function with Condon-Shortley phase.

    This matches the MATLAB combination:
        P = legendre(ell, x, 'norm');
        a = ((-1)^m) * P(m+1,:).'
    """
    log_norm = 0.5 * (
        math.log(2 * ell + 1)
        - math.log(2.0)
        + gammaln(ell - m + 1)
        - gammaln(ell + m + 1)
    )
    return np.exp(log_norm) * lpmv(m, ell, x)



def bc_ade(
    n_in: float,
    n_ext: float,
    musx: float,
    musy: float,
    musz: float,
    g: float,
    *,
    lmax_start: int = 15,
    rel_tol: float = 1e-5,
    abs_tol: float = 1e-10,
    nchi: int = 200,
    nphi: int = 512,
    lmax_cap: int = 101,
    return_info: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, Any]]:
    """Boundary-condition lengths for the generalized ADE slab diffusion model.

    Parameters
    ----------
    n_in : float
        Refractive index of the medium [-].
    n_ext : float
        Refractive index of the external medium [-].
    musx, musy, musz : float
        Principal-axis scattering coefficients [mm^-1].
    g : float
        Henyey-Greenstein asymmetry factor, with -1 < g < 1.
    lmax_start : int, default=15
        Initial odd harmonic order used for convergence checking.
    rel_tol : float, default=1e-5
        Relative convergence tolerance for the series expansions.
    abs_tol : float, default=1e-10
        Absolute convergence tolerance for the series expansions.
    nchi : int, default=200
        Number of Gauss-Legendre nodes in chi = cos(theta).
    nphi : int, default=512
        Number of azimuthal samples in phi.
    lmax_cap : int, default=101
        Maximum odd harmonic order allowed in the series expansions.
    return_info : bool, default=False
        If True, also return a dictionary with numerical diagnostics.

    Returns
    -------
    ze, z0 : float
        Extrapolated boundary length and source depth [mm].
    info : dict, optional
        Numerical diagnostics.
    """
    n_in = _require_positive_scalar("n_in", n_in)
    n_ext = _require_positive_scalar("n_ext", n_ext)
    musx = _require_positive_scalar("musx", musx)
    musy = _require_positive_scalar("musy", musy)
    musz = _require_positive_scalar("musz", musz)

    g = float(g)
    if not np.isfinite(g) or not (-1.0 < g < 1.0):
        raise ValueError("g must satisfy -1 < g < 1.")

    lmax_start = _require_int("lmax_start", lmax_start, positive=True)
    nchi = _require_int("nchi", nchi, positive=True)
    nphi = _require_int("nphi", nphi, positive=True)
    lmax_cap = _require_int("lmax_cap", lmax_cap, positive=True)
    rel_tol = _require_positive_scalar("rel_tol", rel_tol)
    abs_tol = _require_nonnegative_scalar("abs_tol", abs_tol)

    if lmax_start % 2 == 0:
        lmax_start += 1
        warnings.warn(
            "lmax_start was even; incremented to the next odd value.",
            RuntimeWarning,
            stacklevel=2,
        )
    if lmax_cap % 2 == 0:
        lmax_cap += 1
        warnings.warn(
            "lmax_cap was even; incremented to the next odd value.",
            RuntimeWarning,
            stacklevel=2,
        )
    lmax_cap = max(lmax_cap, lmax_start)

    n = n_in / n_ext
    v = C0_MM_PER_NS / n_in  # [mm/ns]

    tol_iso = 1e-12 * max(musx, musy, musz)
    is_iso = abs(musx - musy) <= tol_iso and abs(musx - musz) <= tol_iso
    if is_iso:
        lt = (1.0 / musx) / (1.0 - g)
        if np.isclose(n, 1.0, rtol=0.0, atol=1e-15):
            ze = 2.0 * lt / 3.0
        else:
            x, w = gauss_legendre(nchi)
            chi = (x + 1.0) / 2.0
            w = w / 2.0
            rchi = _fresnel_r(chi, n)
            i1 = np.sum(w * (chi * rchi))
            i2 = np.sum(w * (chi**2 * rchi))
            a = (1.0 + 3.0 * i2) / (1.0 - 2.0 * i1)
            ze = 2.0 * a * lt / 3.0

        z0 = lt
        dz_iso = v * lt / 3.0

        info = {
            "case": "isotropic",
            "n": n,
            "v": v,
            "ze": float(ze),
            "z0": float(z0),
            "Dz": float(dz_iso),
            "Dz0": float(dz_iso),
            "Y": 0.0,
            "Y0": 0.0,
            "C": float((2.0 * np.pi / v) * dz_iso),
            "B": None,
            "X": None,
            "denBC": None,
            "I2": None,
            "I2R": None,
            "LmaxStart": lmax_start,
            "LmaxCap": lmax_cap,
            "LmaxUsedDz": 0,
            "convergedDz": True,
            "LmaxUsedY": 0,
            "convergedY": True,
            "LmaxUsedZ0": 0,
            "convergedZ0": True,
            "Nchi": nchi,
            "Nphi": nphi,
            "RelTol": rel_tol,
            "AbsTol": abs_tol,
        }
        return (float(ze), float(z0), info) if return_info else (float(ze), float(z0))

    chi_full, wchi = gauss_legendre(nchi)
    phi = np.arange(nphi, dtype=float) * (2.0 * np.pi / nphi)
    wphi = 2.0 * np.pi / nphi

    cos2 = np.cos(phi) ** 2
    sin2 = np.sin(phi) ** 2

    chif = chi_full[:, None]
    w_full = wchi[:, None] * wphi
    mu_f = musx * (1.0 - chif**2) * cos2[None, :] + musy * (1.0 - chif**2) * sin2[None, :] + musz * (chif**2)
    invmu_f = 1.0 / mu_f

    lavg = np.sum(invmu_f * w_full) / (4.0 * np.pi)

    chi_hemi = (chi_full + 1.0) / 2.0
    w_hemi = wchi / 2.0
    chih = chi_hemi[:, None]
    w_h = w_hemi[:, None] * wphi
    mu_h = musx * (1.0 - chih**2) * cos2[None, :] + musy * (1.0 - chih**2) * sin2[None, :] + musz * (chih**2)
    invmu_h = 1.0 / mu_h

    rchi = np.zeros_like(chi_hemi) if np.isclose(n, 1.0, rtol=0.0, atol=1e-15) else _fresnel_r(chi_hemi, n)
    r_h = rchi[:, None]

    b = np.sum((chih * invmu_h) * w_h) / lavg
    x_term = np.sum((chih * invmu_h * r_h) * w_h) / lavg
    den_bc = b - x_term
    if not (den_bc > 0.0):
        raise ValueError(f"B - X must be positive; got B={b}, X={x_term}.")

    i2 = np.sum((chih**2 * invmu_h**2) * w_h) / lavg
    i2r = np.sum((chih**2 * invmu_h**2 * r_h) * w_h) / lavg
    y0 = i2r
    y = y0

    iz_full = np.sum((chif**2) * (invmu_f**2) * w_full)
    dz0 = v / (4.0 * np.pi * lavg) * iz_full
    dz = dz0

    lmax_used_dz = 0
    lmax_used_y = 0
    converged_dz = True
    converged_y = True

    if g != 0.0:
        converged_dz = False
        converged_y = False
        corr_z = 0.0
        corr_y = 0.0
        dz_prev = np.nan
        y_prev = np.nan
        normfac = 1.0 / math.sqrt(2.0 * np.pi)

        bz_f = (chif * invmu_f) * w_full
        bz_hr = (chih * invmu_h * r_h) * w_h

        for ell in range(1, lmax_cap + 1, 2):
            coeff = (g**ell) / (1.0 - g**ell)
            mpos = np.arange(ell + 1)

            eplus = np.exp(1j * np.outer(mpos, phi))
            eminus = np.exp(-1j * np.outer(mpos, phi))

            sz_f = bz_f @ eplus.T
            sz_h = bz_hr @ eminus.T

            hz = np.zeros(ell + 1, dtype=np.complex128)
            ht = np.zeros(ell + 1, dtype=np.complex128)

            for k, m in enumerate(mpos):
                af_full = _norm_assoc_legendre_with_cs(ell, int(m), chi_full)
                af_hemi = _norm_assoc_legendre_with_cs(ell, int(m), chi_hemi)
                hz[k] = normfac * np.dot(af_full, sz_f[:, k])
                ht[k] = normfac * np.dot(af_hemi, sz_h[:, k])

            idx_even = np.arange(0, ell + 1, 2)
            hz_even = hz[idx_even]
            ht_even = ht[idx_even]

            sum_z = abs(hz_even[0]) ** 2
            if hz_even.size > 1:
                sum_z += 2.0 * np.sum(np.abs(hz_even[1:]) ** 2)
            corr_z += coeff * sum_z
            dz_new = dz0 + v / (4.0 * np.pi * lavg) * corr_z

            sum_y = float(np.real(hz_even[0] * ht_even[0]))
            if hz_even.size > 1:
                sum_y += 2.0 * float(np.real(np.sum(hz_even[1:] * ht_even[1:])))
            corr_y += coeff * sum_y
            y_new = y0 + corr_y / lavg

            if ell >= lmax_start:
                if not np.isnan(dz_prev) and not converged_dz:
                    if abs(dz_new - dz_prev) <= abs_tol + rel_tol * abs(dz_new):
                        converged_dz = True
                        lmax_used_dz = ell
                if not np.isnan(y_prev) and not converged_y:
                    if abs(y_new - y_prev) <= abs_tol + rel_tol * abs(y_new):
                        converged_y = True
                        lmax_used_y = ell
                if converged_dz and converged_y:
                    dz = float(dz_new)
                    y = float(y_new)
                    break

            dz_prev = dz_new
            y_prev = y_new
            dz = float(dz_new)
            y = float(y_new)
            lmax_used_dz = ell
            lmax_used_y = ell

        if not converged_dz:
            warnings.warn(
                "Dz(g) did not meet tolerance. Consider increasing lmax_cap (or nchi/nphi).",
                RuntimeWarning,
                stacklevel=2,
            )
        if not converged_y:
            warnings.warn(
                "Y(g) did not meet tolerance. Consider increasing lmax_cap (or nchi/nphi).",
                RuntimeWarning,
                stacklevel=2,
            )

    c = (2.0 * np.pi / v) * dz
    ze = (c + y) / den_bc

    if g == 0.0:
        z0 = 1.0 / musz
        lmax_used_z0 = 0
        converged_z0 = True
    else:
        invmu_phi_int = np.sum(invmu_f, axis=1) * wphi
        z0acc = 0.0
        z0prev = np.nan
        converged_z0 = False
        lmax_used_z0 = 1

        for ell in range(1, lmax_cap + 1, 2):
            denom = 1.0 - g**ell
            pl0 = eval_legendre(ell, chi_full)
            il = np.sum(wchi * (chi_full * pl0 * invmu_phi_int))
            term = ((2.0 * ell + 1.0) / (4.0 * np.pi)) * (il / denom)
            z0acc = float(np.real(z0acc + term))
            lmax_used_z0 = ell

            if ell >= lmax_start:
                if not np.isnan(z0prev):
                    if abs(z0acc - z0prev) <= abs_tol + rel_tol * abs(z0acc):
                        converged_z0 = True
                        break
                z0prev = z0acc

        z0 = z0acc

        if not converged_z0:
            warnings.warn(
                "z0=lambda_z did not meet tolerance. Consider increasing lmax_cap (or nchi/nphi).",
                RuntimeWarning,
                stacklevel=2,
            )

    info = {
        "case": "anisotropic",
        "n": n,
        "v": v,
        "lavg": float(lavg),
        "B": float(b),
        "X": float(x_term),
        "denBC": float(den_bc),
        "I2": float(i2),
        "I2R": float(i2r),
        "Y0": float(y0),
        "Y": float(y),
        "Dz": float(dz),
        "Dz0": float(dz0),
        "C": float(c),
        "ze": float(ze),
        "z0": float(z0),
        "LmaxStart": lmax_start,
        "LmaxCap": lmax_cap,
        "LmaxUsedDz": int(lmax_used_dz),
        "convergedDz": bool(converged_dz),
        "LmaxUsedY": int(lmax_used_y),
        "convergedY": bool(converged_y),
        "LmaxUsedZ0": int(lmax_used_z0),
        "convergedZ0": bool(converged_z0),
        "Nchi": nchi,
        "Nphi": nphi,
        "RelTol": rel_tol,
        "AbsTol": abs_tol,
    }

    return (float(ze), float(z0), info) if return_info else (float(ze), float(z0))
