# Generalized ADE

MATLAB and Python implementations of the generalized anisotropic diffusion equation (ADE) for radiative transfer in slab geometry, as derived in:

**E. Pini et al.**  
*Generalized diffusion theory for radiative transfer in fully anisotropic scattering media.*  
arXiv preprint arXiv:2602.18963 (2026).

The repository provides routines for fully anisotropic scattering media with principal-axis scattering coefficients `musx`, `musy`, `musz`, scalar Henyey-Greenstein asymmetry factor `g`, refractive-index mismatch at the slab boundaries, and homogeneous absorption `mua`.

## Units convention

All MATLAB and Python functions use the same units:

- lengths in `mm`
- optical coefficients in `mm^-1`
- time in `ns`

Accordingly:

- `Dx`, `Dy`, `Dz` are in `mm^2/ns`
- `ze`, `z0` are in `mm`
- total reflectance and transmittance are dimensionless
- time-resolved signals are in `ns^-1`
- space-resolved signals are in `mm^-2`
- time- and space-resolved signals are in `mm^-2 ns^-1`

## Repository structure

```text
Generalized ADE/
├── README.md
├── LICENSE
├── CITATION.cff
├── matlab/
│   ├── D_Tensor_ADE.m
│   ├── BC_ADE.m
│   ├── R_ADE.m
│   ├── Rt_ADE.m
│   ├── Rxy_ADE.m
│   ├── Rxyt_ADE.m
│   ├── T_ADE.m
│   ├── Tt_ADE.m
│   ├── Txy_ADE.m
│   ├── Txyt_ADE.m
│   ├── gauss_legendre.m
│   └── examples/
│       └── demo_general_anisotropic.m
├── python/
│   ├── README.md
│   ├── pyproject.toml
│   ├── pytest.ini
│   ├── examples/
│   │   └── demo_general_anisotropic.py
│   ├── src/
│   │   └── generalized_ade/
│   └── tests/
│       ├── reference/
│       ├── test_smoke.py
│       ├── test_diffusion_reference.py
│       ├── test_boundary_reference.py
│       └── test_resolved_reference.py
└── validation/
    └── matlab_export/
        └── export_d_tensor_reference.m
```

## MATLAB functions

### Core coefficients
- `D_Tensor_ADE.m` — diffusion tensor components `Dx`, `Dy`, `Dz`
- `BC_ADE.m` — extrapolated boundary length `ze` and source depth `z0`

### Reflectance
- `R_ADE.m` — total steady-state reflectance
- `Rt_ADE.m` — total time-resolved reflectance
- `Rxy_ADE.m` — space-resolved steady-state reflectance
- `Rxyt_ADE.m` — time- and space-resolved reflectance

### Transmittance
- `T_ADE.m` — total steady-state transmittance
- `Tt_ADE.m` — total time-resolved transmittance
- `Txy_ADE.m` — space-resolved steady-state transmittance
- `Txyt_ADE.m` — time- and space-resolved transmittance

### Numerical helper
- `gauss_legendre.m` — Gauss-Legendre quadrature nodes and weights on `[-1,1]`

## Python package

The Python package mirrors the MATLAB implementation and exposes:

- `gauss_legendre`
- `d_tensor_ade`
- `bc_ade`
- `r_ade`, `rt_ade`, `rxy_ade`, `rxyt_ade`
- `t_ade`, `tt_ade`, `txy_ade`, `txyt_ade`

### Installation

From the `python/` folder:

```bash
pip install -e .
```

### Minimal example

```python
from generalized_ade import d_tensor_ade, bc_ade, r_ade, t_ade

Dx, Dy, Dz = d_tensor_ade(1.4, 12.0, 8.0, 5.0, 0.85)
ze, z0 = bc_ade(1.4, 1.0, 12.0, 8.0, 5.0, 0.85)
R = r_ade(20.0, 1.4, 1.0, 12.0, 8.0, 5.0, 0.85, 0.01)
T = t_ade(20.0, 1.4, 1.0, 12.0, 8.0, 5.0, 0.85, 0.01)
```

## Examples

- MATLAB: `matlab/examples/demo_general_anisotropic.m`
- Python: `python/examples/demo_general_anisotropic.py`

Both examples illustrate a fully anisotropic case with `musx ~= musy ~= musz` and `g > 0`, including:

- diffusion tensor and boundary conditions
- total reflectance and transmittance
- time-resolved reflectance and transmittance
- space-resolved maps
- time- and space-resolved signals

## Validation

The Python implementation has been benchmarked against the MATLAB implementation on isotropic and anisotropic test cases, including:

- `d_tensor_ade` vs `D_Tensor_ADE`
- `bc_ade` vs `BC_ADE`
- all reflectance/transmittance functions in steady-state, time-resolved, space-resolved, and time-space-resolved forms

Tests are organized under `python/tests/` and are ready for `pytest`.

## Citation

If you use this repository, please cite the associated preprint and the software metadata provided in `CITATION.cff`.

## Author

**Ernesto Pini**  
Istituto Nazionale di Ricerca Metrologica (INRiM)  
pinie@lens.unifi.it
