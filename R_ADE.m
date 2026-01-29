function R = R_ADE(L, n_in, n_ext, lx, ly, lz, mua)
% R_ADE total reflectance from a turbid slab
%
% Brief: this function returns the total reflectance R
% from an anisotropic slab of thickness L [μm].
% xy is the slab plane, while z is the direction of incidence of the pencil beam.
% If a refractive index contrast is set, the effect of Fresnel
% reflections at the boundaries is considered. Absorption is assumed to be
% uniform, mua [1/μm].
%
% Inputs:
%    L - slab thickness [μm]
%    n_in - refractive index of the diffusive medium
%    n_ext - refractive index of the external medium
%    lx - scattering mean free path along x [μm]
%    ly - scattering mean free path along y [μm]
%    lz - scattering mean free path along z [μm]
%    mua - absorption rate [1/μm]
%
% Outputs:
%    R - total reflectance
%
% See also: total_refl_trans.m

% Author:       Ernesto Pini
% Affiliation:  Department of Physics and Astronomy, Università di Firenze
% Email:        pinie@lens.unifi.it

[~, ~, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);
[ze, z0] = BC_ADE(n_in, n_ext, lx, ly, lz, g);

v = 299.7924589/n_in;

    R = 0;

    M = 10000; % number of virtual sources considered in the expansion
    for m = -M:M
        z3 = - 2*m*L - 4*m*ze - z0;
        z4 = - 2*m*L - (4*m - 2)*ze + z0;
        R = R + (sign(z3)*exp(-abs(z3)*sqrt(mua*v/Dz)) - sign(z4)*exp(-abs(z4)*sqrt(mua*v/Dz)));
    end

    R = -R/2;

end

end