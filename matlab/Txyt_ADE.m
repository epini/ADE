function Txyt = Txyt_ADE(x, y, t, L, n_in, n_ext, musx, musy, musz, g, sx, sy, mua)
%TXYT_ADE Time- and space-resolved diffuse transmittance through an anisotropic slab.
%
%   Txyt = TXYT_ADE(x, y, t, L, n_in, n_ext, musx, musy, musz, g, sx, sy, mua)
%
%   Computes the time- and space-resolved diffuse transmittance T(x,y,t)
%   through a slab described by the anisotropic diffusion equation (ADE).
%   The slab lies in the xy plane, with thickness L along z, and is
%   illuminated by a normally incident beam along the z direction.
%   Refractive-index mismatch at the boundaries is accounted for through
%   extrapolated boundary conditions.
%
%   A Gaussian lateral profile at t = 0 is included through the initial
%   standard deviations sx and sy along x and y.
%
%   Units convention:
%     lengths in mm, optical coefficients in mm^-1, time in ns.
%
%   Inputs
%   ------
%   x       - x coordinates in the slab plane [mm, real vector].
%   y       - y coordinates in the slab plane [mm, real vector].
%   t       - Time array [ns, real vector].
%   L       - Slab thickness [mm, positive scalar].
%   n_in    - Refractive index of the diffusive medium [dimensionless scalar].
%   n_ext   - Refractive index of the external medium [dimensionless scalar].
%   musx    - Scattering coefficient along x [mm^-1, positive scalar].
%   musy    - Scattering coefficient along y [mm^-1, positive scalar].
%   musz    - Scattering coefficient along z [mm^-1, positive scalar].
%   g       - Henyey-Greenstein asymmetry factor [dimensionless scalar,
%             -1 < g < 1].
%   sx      - Initial standard deviation along x at t = 0 [mm, non-negative scalar].
%   sy      - Initial standard deviation along y at t = 0 [mm, non-negative scalar].
%   mua     - Absorption coefficient [mm^-1, non-negative scalar].
%
%   Output
%   ------
%   Txyt    - Time- and space-resolved diffuse transmittance evaluated on
%             the grid defined by x, y and t
%             [mm^-2 ns^-1, numel(x)-by-numel(y)-by-numel(t) array].
%
%   Notes
%   -----
%   The diffusion coefficients Dx, Dy, Dz are obtained from D_Tensor_ADE,
%   and the boundary parameters ze and z0 are obtained from BC_ADE.
%
%   For t <= 0, the function returns Txyt = 0.
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
validateattributes(t,     {'numeric'}, {'real','finite','vector'});
validateattributes(L,     {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(n_in,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(n_ext, {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musx,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musy,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(musz,  {'numeric'}, {'real','finite','scalar','positive'});
validateattributes(g,     {'numeric'}, {'real','finite','scalar','>',-1,'<',1});
validateattributes(sx,    {'numeric'}, {'real','finite','scalar','nonnegative'});
validateattributes(sy,    {'numeric'}, {'real','finite','scalar','nonnegative'});
validateattributes(mua,   {'numeric'}, {'real','finite','scalar','nonnegative'});

[Dx, Dy, Dz] = D_Tensor_ADE(n_in, musx, musy, musz, g);
[ze, z0]     = BC_ADE(n_in, n_ext, musx, musy, musz, g);

v = 299.792458 / n_in;   % speed of light in the medium [mm/ns]

x = x(:);
y = y(:).';
t = t(:);

Nx = numel(x);
Ny = numel(y);
Nt = numel(t);

Txyt = zeros(Nx, Ny, Nt);

idx = find(t > 0);
if isempty(idx)
    return;
end

tp = t(idx);
Tz = zeros(size(tp));

M = 10000; % number of virtual sources considered in the expansion
for m = -M:M
    z1 = L*(1 - 2*m) - 4*m*ze - z0;
    z2 = L*(1 - 2*m) - (4*m - 2)*ze + z0;
    Tz = Tz + z1 .* exp(-(z1^2) ./ (4 * Dz * tp)) ...
            - z2 .* exp(-(z2^2) ./ (4 * Dz * tp));
end

for k = 1:numel(tp)
    tk = tp(k);

    denx = 2*sx^2 + 4*Dx*tk;
    deny = 2*sy^2 + 4*Dy*tk;

    Gx = exp(-(x.^2) ./ denx);
    Gy = exp(-(y.^2) ./ deny);

    pref = 1 / (2 * (4*pi)^(3/2) * tk^(5/2) * sqrt(Dx*Dy*Dz));

    Txyt(:,:,idx(k)) = pref .* (Gx * Gy) .* abs(Tz(k)) .* exp(-v * tk * mua);
end

end