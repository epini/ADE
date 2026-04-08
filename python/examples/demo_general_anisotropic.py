#!/usr/bin/env python3
"""Demo for the generalized ADE Python package in the fully anisotropic case.

This example shows:
  1) diffusion tensor and boundary conditions
  2) total steady-state reflectance/transmittance
  3) total time-resolved reflectance/transmittance
  4) space-resolved steady-state maps (linear and log10 colormap)
  5) selected space-time reflectance/transmittance frames
  6) directional anisotropy in the time domain
  7) steady-state 1D cuts

Units convention:
  lengths in mm, optical coefficients in mm^-1, time in ns.

Author:       Ernesto Pini
Affiliation:  Istituto Nazionale di Ricerca Metrologica (INRiM)
Email:        pinie@lens.unifi.it
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Optional fallback: allow running the script directly from a repo checkout
# without installing the package, assuming the structure:
#   python/
#     examples/demo_general_anisotropic.py
#     src/generalized_ade/
_THIS_FILE = Path(__file__).resolve()
_REPO_PYTHON_DIR = _THIS_FILE.parents[1] if len(_THIS_FILE.parents) >= 2 else None
_SRC_DIR = _REPO_PYTHON_DIR / "src" if _REPO_PYTHON_DIR is not None else None
if _SRC_DIR is not None and _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from generalized_ade import (  # noqa: E402
    bc_ade,
    d_tensor_ade,
    r_ade,
    rt_ade,
    rxy_ade,
    rxyt_ade,
    t_ade,
    tt_ade,
    txy_ade,
    txyt_ade,
)


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "lines.linewidth": 1.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def nearest_index(v: np.ndarray, x0: float) -> int:
    """Return the index of the entry of v closest to x0."""
    return int(np.argmin(np.abs(v - x0)))


# -----------------------------------------------------------------------------
# Medium and slab parameters
# -----------------------------------------------------------------------------
L = 20.0      # slab thickness [mm]
n_in = 1.40   # refractive index inside slab [-]
n_ext = 1.00  # refractive index outside slab [-]

# Fully anisotropic scattering coefficients [mm^-1]
musx = 12.0
musy = 5.0
musz = 8.0

g = 0.85     # Henyey-Greenstein asymmetry factor [-]
mua = 0.01   # absorption coefficient [mm^-1]

# Initial lateral widths for space-time solutions [mm]
sx = 0.05
sy = 0.05

# Grids
x = np.linspace(-30.0, 30.0, 121)   # [mm]
y = np.linspace(-30.0, 30.0, 121)   # [mm]
t = np.linspace(0.01, 4.0, 300)     # [ns]


# -----------------------------------------------------------------------------
# Diffusion tensor and boundary conditions
# -----------------------------------------------------------------------------
Dx, Dy, Dz, info_d = d_tensor_ade(n_in, musx, musy, musz, g, return_info=True)
ze, z0, info_bc = bc_ade(n_in, n_ext, musx, musy, musz, g, return_info=True)

print("------------------------------------------------------------")
print("General anisotropic ADE demo (Python)")
print("------------------------------------------------------------")
print("Input parameters:")
print(f"  L    = {L:.3f} mm")
print(f"  n_in = {n_in:.3f}")
print(f"  n_ext= {n_ext:.3f}")
print(f"  musx = {musx:.3f} mm^-1")
print(f"  musy = {musy:.3f} mm^-1")
print(f"  musz = {musz:.3f} mm^-1")
print(f"  g    = {g:.3f}")
print(f"  mua  = {mua:.4f} mm^-1")
print()
print("Computed ADE parameters:")
print(f"  Dx   = {Dx:.6f} mm^2/ns")
print(f"  Dy   = {Dy:.6f} mm^2/ns")
print(f"  Dz   = {Dz:.6f} mm^2/ns")
print(f"  ze   = {ze:.6f} mm")
print(f"  z0   = {z0:.6f} mm")
print()
print("Numerical info:")
print(f"  d_tensor_ade converged: {info_d.get('converged', True)}")
print(f"  d_tensor_ade LmaxUsed:  {info_d.get('LmaxUsed', 0)}")
print(
    "  bc_ade converged:      "
    f"Dz={info_bc.get('convergedDz', True)}  "
    f"Y={info_bc.get('convergedY', True)}  "
    f"z0={info_bc.get('convergedZ0', True)}"
)
print("------------------------------------------------------------")


# -----------------------------------------------------------------------------
# Total steady-state reflectance/transmittance
# -----------------------------------------------------------------------------
R = r_ade(L, n_in, n_ext, musx, musy, musz, g, mua)
T = t_ade(L, n_in, n_ext, musx, musy, musz, g, mua)
A = 1.0 - R - T

print("Energy balance:")
print(f"  R = {R:.6f}")
print(f"  T = {T:.6f}")
print(f"  A = {A:.6f}")
print("------------------------------------------------------------")


# -----------------------------------------------------------------------------
# Total time-resolved reflectance/transmittance
# -----------------------------------------------------------------------------
Rt = rt_ade(t, L, n_in, n_ext, musx, musy, musz, g, mua)
Tt = tt_ade(t, L, n_in, n_ext, musx, musy, musz, g, mua)

fig1, ax1 = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
ax1.semilogy(t, np.maximum(Rt, np.finfo(float).tiny), label=r"$R_t$")
ax1.semilogy(t, np.maximum(Tt, np.finfo(float).tiny), label=r"$T_t$")
ax1.set_xlabel("t [ns]")
ax1.set_ylabel(r"Signal [ns$^{-1}$]")
ax1.set_ylim(1e-7, 1e1)
ax1.legend(loc="best")
ax1.set_title("Total time-resolved reflectance and transmittance")


# -----------------------------------------------------------------------------
# Space-resolved steady-state maps
# -----------------------------------------------------------------------------
Rxy = rxy_ade(x, y, L, n_in, n_ext, musx, musy, musz, g, mua)  # shape (ny, nx)
Txy = txy_ade(x, y, L, n_in, n_ext, musx, musy, musz, g, mua)  # shape (ny, nx)

Rxy_log = np.log10(np.maximum(Rxy, np.finfo(float).tiny))
Txy_log = np.log10(np.maximum(Txy, np.finfo(float).tiny))

fig2, axs2 = plt.subplots(2, 2, figsize=(8.2, 7.4), constrained_layout=True)
extent = (x[0], x[-1], y[0], y[-1])

im = axs2[0, 0].imshow(Rxy, extent=extent, origin="lower", aspect="equal")
axs2[0, 0].set_title("R(x,y)")
axs2[0, 0].set_xlabel("x [mm]")
axs2[0, 0].set_ylabel("y [mm]")
cb = fig2.colorbar(im, ax=axs2[0, 0])
cb.set_label(r"mm$^{-2}$")

im = axs2[0, 1].imshow(Txy, extent=extent, origin="lower", aspect="equal")
axs2[0, 1].set_title("T(x,y)")
axs2[0, 1].set_xlabel("x [mm]")
axs2[0, 1].set_ylabel("y [mm]")
cb = fig2.colorbar(im, ax=axs2[0, 1])
cb.set_label(r"mm$^{-2}$")

im = axs2[1, 0].imshow(Rxy_log, extent=extent, origin="lower", aspect="equal")
axs2[1, 0].set_title(r"$\log_{10} R(x,y)$")
axs2[1, 0].set_xlabel("x [mm]")
axs2[1, 0].set_ylabel("y [mm]")
cb = fig2.colorbar(im, ax=axs2[1, 0])
cb.set_label(r"$\log_{10}$(mm$^{-2}$)")

im = axs2[1, 1].imshow(Txy_log, extent=extent, origin="lower", aspect="equal")
axs2[1, 1].set_title(r"$\log_{10} T(x,y)$")
axs2[1, 1].set_xlabel("x [mm]")
axs2[1, 1].set_ylabel("y [mm]")
cb = fig2.colorbar(im, ax=axs2[1, 1])
cb.set_label(r"$\log_{10}$(mm$^{-2}$)")

fig2.suptitle("Space-resolved steady-state reflectance and transmittance")


# -----------------------------------------------------------------------------
# Time- and space-resolved maps at selected times
# -----------------------------------------------------------------------------
Rxyt = rxyt_ade(x, y, t, L, n_in, n_ext, musx, musy, musz, g, sx, sy, mua)  # (ny, nx, nt)
Txyt = txyt_ade(x, y, t, L, n_in, n_ext, musx, musy, musz, g, sx, sy, mua)  # (ny, nx, nt)

t_sel = np.array([0.05, 0.5, 2.0])
idx_sel = [nearest_index(t, tt) for tt in t_sel]

fig3, axs3 = plt.subplots(2, 3, figsize=(11.0, 6.4), constrained_layout=True)
for k, it in enumerate(idx_sel):
    im = axs3[0, k].imshow(Rxyt[:, :, it], extent=extent, origin="lower", aspect="equal")
    axs3[0, k].set_title(f"R, t = {t[it]:.2f} ns")
    axs3[0, k].set_xlabel("x [mm]")
    axs3[0, k].set_ylabel("y [mm]")
    cb = fig3.colorbar(im, ax=axs3[0, k])
    cb.set_label(r"mm$^{-2}$ ns$^{-1}$")

for k, it in enumerate(idx_sel):
    im = axs3[1, k].imshow(Txyt[:, :, it], extent=extent, origin="lower", aspect="equal")
    axs3[1, k].set_title(f"T, t = {t[it]:.2f} ns")
    axs3[1, k].set_xlabel("x [mm]")
    axs3[1, k].set_ylabel("y [mm]")
    cb = fig3.colorbar(im, ax=axs3[1, k])
    cb.set_label(r"mm$^{-2}$ ns$^{-1}$")

fig3.suptitle("Selected space-time frames")


# -----------------------------------------------------------------------------
# Directional anisotropy in the time domain
# -----------------------------------------------------------------------------
x_probe = np.array([0.0, 10.0, 20.0, 30.0])
y_probe = np.array([0.0, 10.0, 20.0, 30.0])
ix_probe = [nearest_index(x, xx) for xx in x_probe]
iy_probe = [nearest_index(y, yy) for yy in y_probe]
ix0 = nearest_index(x, 0.0)
iy0 = nearest_index(y, 0.0)

fig4, axs4 = plt.subplots(2, 2, figsize=(10.6, 7.2), constrained_layout=True)

for k, ix in enumerate(ix_probe):
    axs4[0, 0].semilogy(t, np.maximum(Rxyt[iy0, ix, :], np.finfo(float).tiny), label=f"x = {x[ix]:.1f} mm")
axs4[0, 0].set_xlabel("t [ns]")
axs4[0, 0].set_ylabel(r"R(x,0,t) [mm$^{-2}$ ns$^{-1}$]")
axs4[0, 0].set_ylim(1e-10, 1e0)
axs4[0, 0].legend(loc="best")
axs4[0, 0].set_title("Reflectance along x")

for k, iy in enumerate(iy_probe):
    axs4[0, 1].semilogy(t, np.maximum(Rxyt[iy, ix0, :], np.finfo(float).tiny), label=f"y = {y[iy]:.1f} mm")
axs4[0, 1].set_xlabel("t [ns]")
axs4[0, 1].set_ylabel(r"R(0,y,t) [mm$^{-2}$ ns$^{-1}$]")
axs4[0, 1].set_ylim(1e-10, 1e0)
axs4[0, 1].legend(loc="best")
axs4[0, 1].set_title("Reflectance along y")

for k, ix in enumerate(ix_probe):
    axs4[1, 0].semilogy(t, np.maximum(Txyt[iy0, ix, :], np.finfo(float).tiny), label=f"x = {x[ix]:.1f} mm")
axs4[1, 0].set_xlabel("t [ns]")
axs4[1, 0].set_ylabel(r"T(x,0,t) [mm$^{-2}$ ns$^{-1}$]")
axs4[1, 0].set_ylim(1e-10, 1e0)
axs4[1, 0].legend(loc="best")
axs4[1, 0].set_title("Transmittance along x")

for k, iy in enumerate(iy_probe):
    axs4[1, 1].semilogy(t, np.maximum(Txyt[iy, ix0, :], np.finfo(float).tiny), label=f"y = {y[iy]:.1f} mm")
axs4[1, 1].set_xlabel("t [ns]")
axs4[1, 1].set_ylabel(r"T(0,y,t) [mm$^{-2}$ ns$^{-1}$]")
axs4[1, 1].set_ylim(1e-10, 1e0)
axs4[1, 1].legend(loc="best")
axs4[1, 1].set_title("Transmittance along y")

fig4.suptitle("Directional anisotropy: x versus y")


# -----------------------------------------------------------------------------
# Simple 1D cuts of the steady-state maps
# -----------------------------------------------------------------------------
fig5, axs5 = plt.subplots(1, 2, figsize=(9.6, 4.2), constrained_layout=True)

axs5[0].plot(x, Rxy[iy0, :], label=r"R(x,0)")
axs5[0].plot(y, Rxy[:, ix0], "--", label=r"R(0,y)")
axs5[0].set_xlabel("position [mm]")
axs5[0].set_ylabel(r"R [mm$^{-2}$]")
axs5[0].legend(loc="best")
axs5[0].set_title("Steady-state reflectance cuts")

axs5[1].plot(x, Txy[iy0, :], label=r"T(x,0)")
axs5[1].plot(y, Txy[:, ix0], "--", label=r"T(0,y)")
axs5[1].set_xlabel("position [mm]")
axs5[1].set_ylabel(r"T [mm$^{-2}$]")
axs5[1].legend(loc="best")
axs5[1].set_title("Steady-state transmittance cuts")

plt.show()
