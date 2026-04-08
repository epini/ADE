# Generalized ADE MATLAB functions

MATLAB functions for the generalized anisotropic diffusion equation (ADE) in slab geometry, as derived in

**E. Pini et al.,**  
*Generalized diffusion theory for radiative transfer in fully anisotropic scattering media.*  
arXiv preprint arXiv:2602.18963 (2026).

The repository provides analytical and semi-analytical MATLAB routines for computing steady-state, time-resolved, space-resolved, and space-time-resolved diffuse reflectance and transmittance in fully anisotropic scattering slabs.

## Physical model

The slab is assumed to be infinite in the transverse plane `(x, y)` and finite along `z`, which is also the direction of normally incident illumination.

The model supports:

- fully anisotropic scattering with principal-axis scattering coefficients  
  `musx ~= musy ~= musz`
- scalar Henyey-Greenstein asymmetry factor `g`
- refractive-index mismatch at the slab boundaries
- homogeneous absorption coefficient `mua`

## Units convention

All functions follow the same units convention:

- **lengths** in `mm`
- **optical coefficients** in `mm^-1`
- **time** in `ns`

Accordingly:

- diffusion coefficients `Dx, Dy, Dz` are returned in `mm^2/ns`
- boundary-condition lengths `ze, z0` are returned in `mm`
- total reflectance/transmittance are dimensionless
- time-resolved signals are in `ns^-1`
- space-resolved signals are in `mm^-2`
- space-time-resolved signals are in `mm^-2 ns^-1`

## Main functions

### Diffusion tensor and boundary conditions
- `D_Tensor_ADE.m`  
  Computes the diffusion tensor components `Dx`, `Dy`, and `Dz`.

- `BC_ADE.m`  
  Computes the extrapolated boundary length `ze` and source depth `z0`.

### Reflectance
- `R_ADE.m`  
  Total steady-state diffuse reflectance.

- `Rt_ADE.m`  
  Total time-resolved diffuse reflectance.

- `Rxy_ADE.m`  
  Space-resolved steady-state diffuse reflectance.

- `Rxyt_ADE.m`  
  Time- and space-resolved diffuse reflectance.

### Transmittance
- `T_ADE.m`  
  Total steady-state diffuse transmittance.

- `Tt_ADE.m`  
  Total time-resolved diffuse transmittance.

- `Txy_ADE.m`  
  Space-resolved steady-state diffuse transmittance.

- `Txyt_ADE.m`  
  Time- and space-resolved diffuse transmittance.

### Numerical helper
- `gauss_legendre.m`  
  Gauss-Legendre quadrature nodes and weights on `[-1, 1]`.

## Example usage

### Diffusion tensor
```matlab
[Dx, Dy, Dz] = D_Tensor_ADE(1.4, 12.5, 10.0, 5.0, 0.9);