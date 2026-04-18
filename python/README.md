# generalized_ade (Python)

Python implementation of the generalized anisotropic diffusion equation (ADE) for radiative transfer in slab geometry.

This package mirrors the MATLAB implementation in the repository and uses the same units convention:

- lengths in `mm`
- optical coefficients in `mm^-1`
- time in `ns`

## Installation

From the `python/` folder of the repository:

```bash
python -m pip install -e . --no-build-isolation
```

This installs the package in editable mode, which is convenient during development.
In a typical networked environment, `python -m pip install -e .` also works.

## Package structure

```text
python/
├── pyproject.toml
├── pytest.ini
├── README.md
├── examples/
│   └── demo_general_anisotropic.py
├── src/
│   └── generalized_ade/
│       ├── __init__.py
│       ├── quadrature.py
│       ├── diffusion.py
│       ├── boundary.py
│       ├── reflectance.py
│       └── transmittance.py
└── tests/
    ├── reference/
    ├── test_smoke.py
    ├── test_diffusion_reference.py
    ├── test_boundary_reference.py
    └── test_resolved_reference.py
```

## Public API

```python
from generalized_ade import (
    gauss_legendre,
    d_tensor_ade,
    bc_ade,
    r_ade, rt_ade, rxy_ade, rxyt_ade,
    t_ade, tt_ade, txy_ade, txyt_ade,
)
```

The installable public API lives in `src/generalized_ade/`. The top-level
`python/` directory is project scaffolding, not a second import package.

## Minimal example

```python
from generalized_ade import d_tensor_ade, bc_ade, r_ade, t_ade

Dx, Dy, Dz = d_tensor_ade(1.4, 12.0, 8.0, 5.0, 0.85)
ze, z0 = bc_ade(1.4, 1.0, 12.0, 8.0, 5.0, 0.85)
R = r_ade(20.0, 1.4, 1.0, 12.0, 8.0, 5.0, 0.85, 0.01)
T = t_ade(20.0, 1.4, 1.0, 12.0, 8.0, 5.0, 0.85, 0.01)

print(Dx, Dy, Dz)
print(ze, z0)
print(R, T)
```

## Conventions for resolved outputs

- `rxy_ade`, `txy_ade` return arrays with shape `(ny, nx)`
- `rxyt_ade`, `txyt_ade` return arrays with shape `(ny, nx, nt)`

That is:

- axis 0 corresponds to `y`
- axis 1 corresponds to `x`
- axis 2 corresponds to `t`

This makes plotting straightforward with `matplotlib.imshow(..., extent=[xmin, xmax, ymin, ymax], origin="lower")`.

## Tests

Run from the `python/` folder after the editable install:

```bash
python -m pytest -q
```

The repository includes:

- smoke tests for import and basic execution
- reference tests against MATLAB benchmark data

Reference JSON files are stored in `python/tests/reference/`.
Their readers are intentionally kept in sync with the exporters in
`validation/matlab_export/`; if one side's JSON schema changes, update the
other in the same change.

## Example script

A full example is provided in:

```text
python/examples/demo_general_anisotropic.py
```

It reproduces the same kind of workflow as the MATLAB demo:

- diffusion tensor and boundary conditions
- total and time-resolved signals
- space-resolved maps
- time-space-resolved slices
- directional anisotropy along `x` and `y`
