function T = T_ADE(L, n_in, n_ext, lx, ly, lz, g, mua)
% T_ADE total transmittance from a turbid slab
%
% Brief: this function returns the total transmittance T
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
%    T - total transmittance
% See also: total_refl_trans.m

% Author:       Ernesto Pini
% Affiliation:  Department of Physics and Astronomy, Università di Firenze
% Email:        pinie@lens.unifi.it

[~, ~, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);
[ze, z0] = BC_ADE(n_in, n_ext, lx, ly, lz, g);

v = 299.7924589/n_in;

if mua*z0 < 1e-10

    T = (z0 + ze)/(L + 2*ze);

else

    T = 0;

    M = 10000; % number of virtual sources considered in the expansion
    for m = -M:M
        z1 = L*(1-2*m) - 4*m*ze - z0;
        z2 = L*(1-2*m) - (4*m - 2)*ze + z0;
        T = T + (sign(z1)*exp(-abs(z1)*sqrt(mua*v/Dz)) - sign(z2)*exp(-abs(z2)*sqrt(mua*v/Dz)));
    end

    T = T/2;

end

end
