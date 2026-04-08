function Rt = Rt_ADE(t, L, n_in, n_ext, musx, musy, musz, g, mua)
%RT_ADE Time-resolved diffuse reflectance from an anisotropic turbid slab.
%
%   Rt = RT_ADE(t, L, n_in, n_ext, musx, musy, musz, g, mua)
%
%   Computes the total time-resolved diffuse reflectance Rt(t) of a slab
%   described by the anisotropic diffusion equation (ADE). The slab lies in
%   the xy plane, with thickness L along z, and is illuminated along the z
%   direction. Refractive-index mismatch at the boundaries is accounted for
%   through extrapolated boundary conditions.
%
%   Units convention:
%     lengths in mm, optical coefficients in mm^-1, time in ns.
%
%   Inputs
%   ------
%   t       - Time array [ns, real array].
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
%   Rt      - Total time-resolved diffuse reflectance [array, same size as t].
%
%   Notes
%   -----
%   The diffusion coefficient Dz is obtained from D_Tensor_ADE, and the
%   boundary parameters ze and z0 are obtained from BC_ADE.
%
%   For t <= 0, the function returns Rt = 0.
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

validateattributes(t,     {'numeric'}, {'real','finite'});
validateattributes(L,     {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(n_in,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(n_ext, {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musx,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musy,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musz,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(g,     {'numeric'}, {'real','finite','scalar','>',-1,'<',1});
validateattributes(mua,   {'numeric'}, {'real','finite','scalar','nonnegative'});

[~, ~, Dz] = D_Tensor_ADE(n_in, musx, musy, musz, g);
[ze, z0]   = BC_ADE(n_in, n_ext, musx, musy, musz, g);

v = 299.792458 / n_in;   % speed of light in the medium [mm/ns]

Rt = zeros(size(t));

idx = (t > 0);
if ~any(idx)
    return;
end

tp = t(idx);
Rsum = zeros(size(tp));

M = 10000; % number of virtual sources considered in the expansion
for m = -M:M
    z3 = -2*m*L - 4*m*ze - z0;
    z4 = -2*m*L - (4*m - 2)*ze + z0;
    Rsum = Rsum + ( ...
        z3 .* exp(-(z3.^2) ./ (4 * Dz * tp)) ...
      - z4 .* exp(-(z4.^2) ./ (4 * Dz * tp)) );
end

Rt(idx) = -(1/4) .* (pi .* Dz .* tp.^3).^(-1/2) .* Rsum .* exp(-v .* tp .* mua);

end