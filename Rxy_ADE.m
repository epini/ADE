function Rxy = Rxy_ADE(x, y, L, n_in, n_ext, lx, ly, lz, mua)
% RXY_ADE space-resolved reflectance from an index-matched turbid slab
%
% Brief: this function returns the space-resolved steady-state reflectance R(x,y)
% for an anisotropic scattering slab of thickness L [μm].
% xy is the slab plane, while z is the direction of incidence of the pencil beam.
% If a refractive index contrast is set, the effect of Fresnel
% reflections at the boundaries is considered. Absorption is assumed to be
% uniform, mua [1/μm].
% lx, ly and lz are scalars [μm]. The source is considered to be
% point-like.
% 
% Inputs:
%    x - array of positions [μm]
%    y - array of positions [μm]
%    L - slab thickness [μm]
%    n_in - refractive index of the diffusive medium
%    n_ext - refractive index of the external medium
%    lx - scattering mean free path along x [μm]
%    ly - scattering mean free path along y [μm]
%    lz - scattering mean free path along z [μm]
%    mua - absorption rate [1/μm]
% 
% Outputs:
%    Rxy - array of space-resolved reflectance R(x,y)
% 
% See also: space_resolved_refl_trans.m

% Author:       Ernesto Pini
% Affiliation:  Department of Physics and Astronomy, Università di Firenze
% Email:        pinie@lens.unifi.it

[Dx, Dy, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);
[ze, z0] = BC_ADE(n_in, n_ext, lx, ly, lz, g);

v = 299.7924589/n_in;
D = (Dx*Dy*Dz)^(1/3);

Rxy = zeros(length(x),length(y));

M = 5000; % number of virtual sources considered in the expansion
for m = -M:M
    z3 = -2*m*L - 4*m*ze - z0;
    z4 = -2*m*L - (4*m - 2)*ze + z0;
    Rxy = Rxy + z3.*(z3^2/Dz + (x.^2).'/Dx + y.^2/Dy).^(-3/2).*(1 + (mua*v)^(1/2)*(z3^2/Dz + (x.^2).'/Dx + y.^2/Dy).^(1/2)).*exp(-((mua*v)^(1/2)*(z3^2/Dz + (x.^2).'/Dx + y.^2/Dy).^(1/2)))...
        - z4.*(z4^2/Dz + (x.^2).'/Dx + y.^2/Dy).^(-3/2).*(1 + (mua*v)^(1/2)*(z4^2/Dz + (x.^2).'/Dx + y.^2/Dy).^(1/2)).*exp(-((mua*v)^(1/2)*(z4^2/Dz + (x.^2).'/Dx + y.^2/Dy).^(1/2)));
end

Rxy = -D^(-3/2)*Rxy/4/pi;

end