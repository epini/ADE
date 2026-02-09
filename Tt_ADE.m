function Tt = Tt_ADE(t, L, n_in, n_ext, lx, ly, lz, mua, g)
% TT_ADE time-resolved transmittance from a turbid slab
%
% Brief: this function returns the total time-resolved transmittance T(t)
% from an anisotropic slab of thickness L [μm].
% xy is the slab plane, while z is the direction of incidence of the pencil beam.
% If a refractive index contrast is set, the effect of Fresnel
% reflections at the boundaries is considered. Absorption is assumed to be
% uniform, mua [1/μm].
% t is an array of times [ps], while lx, ly and lz are scalars [μm].
%
% Inputs:
%    t - array of times [ps]
%    L - slab thickness [μm]
%    n_in - refractive index of the diffusive medium
%    n_ext - refractive index of the external medium
%    lx - scattering mean free path along x [μm]
%    ly - scattering mean free path along y [μm]
%    lz - scattering mean free path along z [μm]
%    mua - absorption rate [1/μm]
%
% Outputs:
%    Tt - array of total time-resolved transmittance T(t)
%
% See also: total_time_resolved_refl_trans.m

% Author:       Ernesto Pini
% Affiliation:  Department of Physics and Astronomy, Università di Firenze
% Email:        pinie@lens.unifi.it

[~, ~, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);
[ze, z0] = BC_ADE_zeta(n_in, n_ext, lx, ly, lz, g);

v = 299.7924589/n_in;

T = zeros(size(t));

M = 10000; % number of virtual sources considered in the expansion
for m = -M:M
    z1 = L*(1-2*m) - 4*m*ze - z0;
    z2 = L*(1-2*m) - (4*m - 2)*ze + z0;
    T = T + (z1*exp(-(z1)^2./(4*Dz*t)) - z2*exp(-(z2)^2./(4*Dz*t)));
end

Tt = (1/4)*((pi.*Dz.*(t).^3).^(-1/2)).*T.*exp(-v*t*mua);

end