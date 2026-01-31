# -*- coding: utf-8 -*-
"""
Find_z0_vs_g_reverted.py

Version reverted to COARSE+REFINE without early stopping, but:
 - uses nphotons_fit = 1e6 (per tua richiesta)
 - SKIPS any propose with step < 0.01 µm (no new MC run; returns cached value)
 - excludes t < 150 fs
 - objective: sum-based square relative error
 - per-g MAT and CSV saved; final plot z0(g)
"""

import os
os.environ.setdefault('PYOPENCL_COMPILER_OUTPUT', '1')

import time, csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import savemat

from xopto.mcml import mc
from xopto import make_user_dirs, USER_TMP_PATH

make_user_dirs()
plt.close('all')

# ------------------------ USER SETTINGS ------------------------
lex, ley, lez = 20, 20, 20          # geometry requested
n_sample = 1.2
mua_l = 0.0

g_values = np.linspace(-0.9, 0.9, 19)  # 10 points

# Monte Carlo statistics (fit and final)
nphotons_ref = int(1e7)        # high-stat final runs (per g)
nphotons_fit = int(1e6)        # FIT uses 1e6 photons as requested

# optimizer / step sizing
expected_z0_um = 12.5
coarse_frac = 0.25
refine_frac = 0.02

maxiter_coarse = 80
maxiter_refine = 200

# time exclusion: exclude t < 150 fs
t_min_fit = 50e-15   # 150 fs

# step-skip threshold: skip evaluating if step < 0.01 µm
STEP_SKIP_UM = 0.001
STEP_SKIP_M = STEP_SKIP_UM * 1e-6

# plotting / saving
OUT_MAT = "z0_vs_g_results_reverted_iso.mat"
BASE_LOG_DIR = "."        # where per-g CSV logs are written

# detector/time axis (same across g)
theta = np.deg2rad(0)
det_axis = mc.mcdetector.Axis(0e-12*(3e8), 6001e-12*(3e8), 201)
pl_detectors_template = mc.mcdetector.Detectors(bottom=mc.mcdetector.TotalPl(det_axis))

# OpenCL device and export path
cl_device = mc.clinfo.gpus()[0]
exportsrc = os.path.join(USER_TMP_PATH, 'mcml_fluence_tr.h')

# ------------------------ Utility functions ------------------------
def build_layers_for_g(g, lex, ley, lez, n_sample=n_sample, mua_l=mua_l):
    pf = mc.mcpf.Hga(g)
    lx = lex * (1 - g)
    ly = ley * (1 - g)
    lz = lez * (1 - g)
    layers = mc.mclayer.Layers([
        mc.mclayer.AnisotropicLayer(d=0.0, n=1.0, mua=0.0, mus=0.0, pf=pf),
        mc.mclayer.AnisotropicLayer(d=5e-3, n=n_sample, mua=1e6*mua_l,
                                    mus=[1e6/lx, 1e6/ly, 1e6/lz], pf=pf),
        mc.mclayer.AnisotropicLayer(d=0.0, n=1.0, mua=0.0, mus=0.0, pf=pf),
    ])
    return layers, (lx, ly, lz)

def run_pencil_beam(layers, nphotons=nphotons_ref, verbose=False):
    src = mc.mcsource.Line((0.0,0.0,0.0), (np.sin(theta), 0.0, np.cos(theta)))
    mc_obj = mc.Mc(layers, src, pl_detectors_template, fluence=None, cl_devices=cl_device)
    res = mc_obj.run(int(nphotons), exportsrc=exportsrc, verbose=verbose)[-1]
    t = res.bottom.pl / 3e8
    counts = res.bottom.raw
    return t, counts

def run_isotropic(layers, z0_m, nphotons, verbose=False):
    src = mc.mcsource.IsotropicPoint([0.0, 0.0, float(z0_m)])
    mc_obj = mc.Mc(layers, src, pl_detectors_template, fluence=None, cl_devices=cl_device)
    res = mc_obj.run(int(nphotons), exportsrc=exportsrc, verbose=verbose)[-1]
    return res.bottom.raw

def sum_objective_aligned(counts_iso_raw, counts_ref, mask_bool, N_eval, N_ref, eps=1e-12):
    """
    Align counts_iso to counts_ref length, apply mask_bool (len == len(counts_ref)),
    scale iso by N_ref / N_eval and compute square relative error on sums.
    """
    Lref = len(counts_ref)
    # align (truncate or pad with zeros if iso shorter)
    if len(counts_iso_raw) < Lref:
        iso_al = np.zeros(Lref, dtype=counts_iso_raw.dtype)
        iso_al[:len(counts_iso_raw)] = counts_iso_raw
    else:
        iso_al = counts_iso_raw[:Lref]

    if mask_bool is None:
        mask_use = np.ones(Lref, dtype=bool)
    else:
        if len(mask_bool) != Lref:
            raise ValueError(f"Mask length {len(mask_bool)} != reference length {Lref}")
        mask_use = mask_bool

    s = float(N_ref) / float(N_eval)
    S_iso = float(np.sum(iso_al[mask_use] * s))
    S_ref = float(np.sum(counts_ref[mask_use]))

    if S_ref <= eps:
        return (S_iso - S_ref)**2
    else:
        return ((S_iso - S_ref)**2) / (S_ref**2)

# ------------------------ Loop over g ------------------------
g_list = []
zfit_list_um = []
obj_list = []
meta_per_g = []

for ig, g in enumerate(g_values):
    print("\n" + "="*70)
    print(f"g = {g:.3f}  ({ig+1}/{len(g_values)})")
    print("="*70)

    # build layers and get lz
    layers, (lx, ly, lz) = build_layers_for_g(g, lex, ley, lez)

    # pencil-beam reference
    print("Running pencil-beam reference (high-stat)...", flush=True)
    t_full, counts_full = run_pencil_beam(layers, nphotons=nphotons_ref, verbose=False)
    # drop last 2 bins to match earlier pipeline
    t = t_full[:-2]
    counts_ref = counts_full[:-2]

    # time mask exclude early times
    t_mask = (t >= t_min_fit)
    n_used_bins = int(np.sum(t_mask))
    print(f"Using {n_used_bins} / {len(t)} time bins (t >= {t_min_fit*1e15:.0f} fs) for objective.", flush=True)

    # initial z0: use lz converted to meters (if lz already meters, remove *1e-6)
    z0_init = float(lez) * 1e-6
    zmin = 0.0
    zmax = float(layers[1].d)

    if not (zmin < z0_init < zmax):
        z0_init = 0.5 * zmax
        print(f"Initial z0 candidate out of bounds; using layer center {z0_init:.3e} m", flush=True)
    else:
        print(f"Initial z0 (from lz) = {z0_init*1e6:.3f} µm", flush=True)

    # compute coarse/refine step sizes
    expected_z0_m = expected_z0_um * 1e-6
    distance_to_expected = abs(z0_init - expected_z0_m)
    thickness = zmax
    step_coarse = max(distance_to_expected * 0.5, coarse_frac * thickness)
    step_refine = max(refine_frac * thickness, 0.05 * step_coarse)
    # clip to bounds
    step_coarse = min(step_coarse, max(z0_init - zmin, zmax - z0_init))
    step_refine = min(step_refine, max(z0_init - zmin, zmax - z0_init))

    print(f"Step sizes (coarse={step_coarse*1e6:.3f} µm, refine={step_refine*1e6:.3f} µm)", flush=True)

    # prepare per-g log CSV
    log_csv_path = os.path.join(BASE_LOG_DIR, f"fit_attempts_g_{g:.2f}.csv")
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attempt", "z0_m", "z0_um", "obj_value", "dt_s", "skipped"])

    # caching / last_eval for step-skip logic
    last_eval = {"z": None, "val": None, "attempt": None}

    # counter
    counter = {"n": 0}

    # objective (no early stop; use cache and skip tiny steps)
    def obj_wrapped_cached(x_arr):
        counter["n"] += 1
        ntry = counter["n"]

        zval = float(x_arr[0]) if hasattr(x_arr, "__iter__") else float(x_arr)
        # bounds check
        if zval <= zmin or zval >= zmax:
            print(f"[g={g:.2f}] try {ntry:04d}: z0 = {zval*1e6:8.3f} µm (OUT OF BOUNDS)", flush=True)
            return 1e30

        # check step size relative to last evaluated z
        if last_eval["z"] is not None:
            step = abs(zval - last_eval["z"])
            if step < STEP_SKIP_M:
                # skip running MC, return cached value
                print(f"[g={g:.2f}] try {ntry:04d}: z0 = {zval*1e6:8.3f} µm  (SKIPPED: step {step*1e6:.5f} µm < {STEP_SKIP_UM} µm) -> reuse val {last_eval['val']}", flush=True)
                # log skipped
                with open(log_csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([ntry, zval, zval*1e6, last_eval["val"], 0.0, True])
                return last_eval["val"]

        # otherwise run MC
        print(f"[g={g:.2f}] try {ntry:04d}: z0 = {zval*1e6:8.3f} µm  (running MC, nphotons={nphotons_fit})", flush=True)
        t0 = time.time()
        try:
            counts_iso_raw = run_isotropic(layers, zval, nphotons=nphotons_fit, verbose=False)
        except Exception as e:
            dt = time.time() - t0
            print(f"[g={g:.2f}] ERROR during run_isotropic at z={zval*1e6:.3f} µm: {e}", flush=True)
            with open(log_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ntry, zval, zval*1e6, np.nan, dt, False])
            return 1e30

        dt = time.time() - t0

        # align lengths
        if len(counts_iso_raw) < len(counts_ref):
            counts_iso_al = np.zeros_like(counts_ref)
            counts_iso_al[:len(counts_iso_raw)] = counts_iso_raw
        else:
            counts_iso_al = counts_iso_raw[:len(counts_ref)]

        val = sum_objective_aligned(counts_iso_al, counts_ref, t_mask, nphotons_fit, nphotons_ref)

        # update cache
        last_eval["z"] = zval
        last_eval["val"] = val
        last_eval["attempt"] = ntry

        # log
        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ntry, zval, zval*1e6, val, dt, False])

        print(f"   -> done (obj={val:.3e}, dt={dt:.2f}s)", flush=True)
        return val

    # ----------------- COARSE stage (no early stop) -----------------
    simplex_coarse = np.array([[z0_init], [min(z0_init + step_coarse, zmax * 0.999)]])
    res_coarse = minimize(obj_wrapped_cached, x0=[z0_init], method="Nelder-Mead",
                          options={"initial_simplex": simplex_coarse, "maxiter": maxiter_coarse})
    z_coarse = float(res_coarse.x[0])
    print(f"Coarse result (scipy): z = {z_coarse*1e6:.4f} µm, fun = {res_coarse.fun:.3e}", flush=True)

    # ----------------- REFINE stage (no early stop) -----------------
    # NOTE: we DO NOT reset last_eval (we keep cache across refine)
    simplex_refine = np.array([[z_coarse], [max(z_coarse - step_refine, zmin * 1.001)]])
    res_refine = minimize(obj_wrapped_cached, x0=[z_coarse], method="Nelder-Mead",
                          options={"initial_simplex": simplex_refine, "maxiter": maxiter_refine})
    z_fit_m = float(res_refine.x[0])
    obj_fit_val = float(res_refine.fun)

    # Prefer cached best if available (last_eval), else use res_refine
    if last_eval["z"] is not None:
        # choose which to trust: if last_eval corresponds to something near z_fit, prefer it
        if abs(last_eval["z"] - z_fit_m) < STEP_SKIP_M:
            z_fit_m = last_eval["z"]
            obj_fit_val = last_eval["val"]

    print(f"FINAL for g={g:.3f}: z_fit = {z_fit_m*1e6:.4f} µm  obj={obj_fit_val:.3e}", flush=True)

    # final high-stat run at z_fit to get final curve for plotting/saving
    try:
        counts_iso_final = run_isotropic(layers, z_fit_m, nphotons=nphotons_ref, verbose=False)
        # align
        if len(counts_iso_final) < len(counts_ref):
            counts_iso_final_al = np.zeros_like(counts_ref); counts_iso_final_al[:len(counts_iso_final)] = counts_iso_final
        else:
            counts_iso_final_al = counts_iso_final[:len(counts_ref)]
    except Exception as e:
        print("Error running final high-stat isotropic run:", e, flush=True)
        counts_iso_final_al = np.zeros_like(counts_ref)

    counts_iso_final_scaled = counts_iso_final_al * (float(nphotons_ref) / float(nphotons_ref))

    # save per-g results
    g_list.append(g)
    zfit_list_um.append(z_fit_m * 1e6)
    obj_list.append(obj_fit_val)
    meta_per_g.append({
        "g": g,
        "lx": lx, "ly": ly, "lz": lz,
        "initial_z0_um": z0_init*1e6,
        "z_fit_um": z_fit_m*1e6,
        "obj_fit": obj_fit_val,
        "nphotons_fit": nphotons_fit,
        "nphotons_ref": nphotons_ref
    })

    # save per-g MAT
    out_mat_g = os.path.join(BASE_LOG_DIR, f"tr_fit_g_{g:.2f}.mat")
    try:
        savemat(out_mat_g, {
            "t": t,
            "t_ps": t * 1e12,
            "pencil_beam": counts_ref,
            "isotropic_fit": counts_iso_final_scaled,
            "z_fit_m": float(z_fit_m),
            "obj_fit": float(obj_fit_val),
            "meta": meta_per_g[-1]
        })
        print(f"Saved per-g MAT to {out_mat_g}", flush=True)
    except Exception as e:
        print(f"Warning: could not save MAT for g={g:.2f}: {e}", flush=True)

# ------------------------ Final summary & plot ------------------------
g_arr = np.array(g_list)
z_arr = np.array(zfit_list_um)
obj_arr = np.array(obj_list)

plt.figure(figsize=(7,5))
plt.plot(g_arr, z_arr, 'o-', lw=2)
plt.xlabel('g')
plt.ylabel(r'$z_0$ [$\mu$m]')
plt.title(r'Best-fit $z_0$ vs anisotropy $g$ (t >= 150 fs)')
plt.grid(True)
plt.tight_layout()
plt.show()

# save global results
savemat(OUT_MAT, {
    "g": g_arr,
    "z0_um": z_arr,
    "obj": obj_arr,
    "meta": {
        "lex": lex, "ley": ley, "lez": lez,
        "nphotons_fit": nphotons_fit, "nphotons_ref": nphotons_ref,
        "t_min_fit_fs": int(t_min_fit*1e15),
        "step_skip_um": STEP_SKIP_UM
    }
})

print("All done. Results saved to", OUT_MAT, flush=True)
