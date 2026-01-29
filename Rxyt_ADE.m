function Rxyt = Rxyt_ADE(x, y, t, L, n_in, n_ext, lx, ly, lz, sx, sy, mua)
% RXYT_ADE time- and space-resolved reflectance from an index-matched turbid slab
%
% Brief: this function returns the time- and space-resolved reflectance 
% R(x,y,t) for an anisotropic slab of thickness L [μm].
% xy is the slab plane, while z is the direction of incidence of the pencil beam.
% If a refractive index contrast is set, the effect of Fresnel
% reflections at the boundaries is considered. Absorption is considered to 
% be uniform, mua [1/μm].
% t is an array of times [ps], while lx, ly and lz are scalars [μm].
% sx and sy are the initial standard deviation [μm] of the 2D intensity 
% gaussian distribution at t = 0 along x and y.
% 
% Inputs:
%    x - array of positions [μm]
%    y - array of positions [μm]
%    t - array of times [ps]
%    L - slab thickness [μm]
%    n_in - refractive index of the diffusive medium
%    n_ext - refractive index of the external medium
%    lx - scattering mean free path along x [μm]
%    ly - scattering mean free path along y [μm]
%    lz - scattering mean free path along z [μm]
%    sx - standard deviation a t = 0 along x [μm]
%    sy - standard deviation a t = 0 along y [μm]
%    mua - absorption rate [1/μm]
% 
% Outputs:
%    Rxyt - 2D-array of time- and space-resolved reflectance R(x,y,t)
% 
% See also: time_space_resolved_refl_trans.m

% Author:       Ernesto Pini
% Affiliation:  Department of Physics and Astronomy, Università di Firenze
% Email:        pinie@lens.unifi.it

[Dx, Dy, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);
[ze, z0] = BC_ADE(n_in, n_ext, lx, ly, lz, g);

v = 299.7924589/n_in;

R = zeros(size(t));
Rxyt = zeros(length(x),length(y),length(t));

M = 10000; % number of virtual sources considered in the expansion
for m = -M:M
    z3 = -2*m*L - 4*m*ze - z0;
    z4 = -2*m*L - (4*m - 2)*ze + z0;
    R = R + (z3*exp(-(z3)^2./(4*Dz*t)) - z4*exp(-(z4)^2./(4*Dz*t)));
end

j = 1;
for i = 1:length(t)
    t2 = t(i);
    Rxyt(:,:,j) = -exp(-x.^2./(2*sx^2+4*Dx*t2)).*(exp(-y.^2./(2*sy^2+4*Dy*t2))).'./((2*(4*pi)^(3/2)*t2^(5/2)*(Dx*Dy*Dz)^(1/2))).*R(j).'.*exp(-v*t2*mua);
    j = j + 1;
end
end