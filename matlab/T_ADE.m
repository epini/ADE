function T = T_ADE(L, n_in, n_ext, musx, musy, musz, g, mua)
%T_ADE Total diffuse transmittance through an anisotropic turbid slab.
%
%   T = T_ADE(L, n_in, n_ext, musx, musy, musz, g, mua)
%
%   Computes the total diffuse transmittance T of a slab described by the
%   anisotropic diffusion equation (ADE). The slab lies in the xy plane,
%   with thickness L along z, and is illuminated along the z direction.
%   Refractive-index mismatch at the boundaries is accounted for through
%   the extrapolated boundary conditions.
%
%   Units convention:
%     lengths in mm, optical coefficients in mm^-1, time in ns.
%
%   Inputs
%   ------
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
%   T       - Total diffuse transmittance [dimensionless scalar].
%
%   Notes
%   -----
%   The diffusion coefficient Dz is obtained from D_Tensor_ADE, and the
%   boundary parameters ze and z0 are obtained from BC_ADE.
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

if mua * z0 < 1e-10

    T = (z0 + ze) / (L + 2*ze);

else

    T = 0;
    M = 10000; % number of virtual sources considered in the expansion
    kappa = sqrt(mua * v / Dz);

    for m = -M:M
        z1 = L * (1 - 2*m) - 4*m*ze - z0;
        z2 = L * (1 - 2*m) - (4*m - 2)*ze + z0;
        T = T + sign(z1) * exp(-abs(z1) * kappa) ...
              - sign(z2) * exp(-abs(z2) * kappa);
    end

    T = abs(T / 2);

end

end