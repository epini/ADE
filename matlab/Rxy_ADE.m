function Rxy = Rxy_ADE(x, y, L, n_in, n_ext, musx, musy, musz, g, mua)
%RXY_ADE Space-resolved steady-state diffuse reflectance from an anisotropic slab.
%
%   Rxy = RXY_ADE(x, y, L, n_in, n_ext, musx, musy, musz, g, mua)
%
%   Computes the space-resolved steady-state diffuse reflectance R(x,y) of
%   a slab described by the anisotropic diffusion equation (ADE). The slab
%   lies in the xy plane, with thickness L along z, and is illuminated by a
%   normally incident pencil beam along the z direction. Refractive-index
%   mismatch at the boundaries is accounted for through extrapolated
%   boundary conditions.
%
%   Units convention:
%     lengths in mm, optical coefficients in mm^-1, time in ns.
%
%   Inputs
%   ------
%   x       - x coordinates in the slab plane [mm, real vector].
%   y       - y coordinates in the slab plane [mm, real vector].
%   L       - Slab thickness [mm, positive scalar].
%   n_in    - Refractive index of the diffusive medium [dimensionless scalar].
%   n_ext   - Refractive index of the external medium [dimensionless scalar].
%   musx    - Scattering coefficient along x [mm^-1, positive scalar].
%   musy    - Scattering coefficient along y [mm^-1, positive scalar].
%   musz    - Scattering coefficient along z [mm^-1, positive scalar].
%   g       - Henyey-Greenstein asymmetry factor [dimensionless scalar,
%             -1 < g < 1].
%   mua     - Absorption coefficient [mm^-1, non-negative scalar].
%
%   Output
%   ------
%   Rxy     - Space-resolved steady-state diffuse reflectance evaluated on
%             the grid defined by x and y [mm^-2, numel(x)-by-numel(y) array].
%
%   Notes
%   -----
%   The diffusion coefficients Dx, Dy, Dz are obtained from D_Tensor_ADE,
%   and the boundary parameters ze and z0 are obtained from BC_ADE.
%
%   Reference
%   ---------
%   E. Pini et al., "Generalized diffusion theory for radiative transfer
%   in fully anisotropic scattering media." arXiv preprint
%   arXiv:2602.18963 (2026).
%
%   Author:       Ernesto Pini
%   Affiliation:  Istituto Nazionale di Ricerca Metrologica (INRiM)
%   Email:        pinie@lens.unifi.it

validateattributes(x,     {'numeric'}, {'real','finite','vector'});
validateattributes(y,     {'numeric'}, {'real','finite','vector'});
validateattributes(L,     {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(n_in,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(n_ext, {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musx,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musy,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musz,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(g,     {'numeric'}, {'real','finite','scalar','>',-1,'<',1});
validateattributes(mua,   {'numeric'}, {'real','finite','scalar','nonnegative'});

[Dx, Dy, Dz] = D_Tensor_ADE(n_in, musx, musy, musz, g);
[ze, z0]     = BC_ADE(n_in, n_ext, musx, musy, musz, g);

v = 299.792458 / n_in;    % speed of light in the medium [mm/ns]
D = (Dx * Dy * Dz)^(1/3); % geometric mean diffusion coefficient [mm^2/ns]

x = x(:);
y = y(:).';

Rxy = zeros(numel(x), numel(y));

M = 10000; % number of virtual sources considered in the expansion
for m = -M:M
    z3 = -2*m*L - 4*m*ze - z0;
    z4 = -2*m*L - (4*m - 2)*ze + z0;

    arg3 = z3^2 / Dz + (x.^2) / Dx + (y.^2) / Dy;
    arg4 = z4^2 / Dz + (x.^2) / Dx + (y.^2) / Dy;

    s3 = sqrt(mua * v * arg3);
    s4 = sqrt(mua * v * arg4);

    Rxy = Rxy ...
        + z3 .* arg3.^(-3/2) .* (1 + s3) .* exp(-s3) ...
        - z4 .* arg4.^(-3/2) .* (1 + s4) .* exp(-s4);
end

Rxy = -abs(Rxy) / (4 * pi * D^(3/2));

end