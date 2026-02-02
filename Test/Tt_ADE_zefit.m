function Tt = Tt_ADE_zefit(t, L, n_in, n_ext, lx, ly, lz, mua, g, ze, z0)
% Tt_ADE_ZEFIT  total time-resolved transmittance for anisotropic slab
%
% Units (required):
%   L, lx, ly, lz, ze  -> [um]
%   t                  -> [ps]
%   mua                -> [1/um]
%   Outputs Tt in same time-sampling as input t (column vector).
%
% Implementation notes:
%  - uses image-source expansion (vectorized)
%  - prefactor (1/4)*(pi*Dz*t^3)^(-1/2) included
%  - absorption factor exp(-v*mua*t) included (v in um/ps)
%  - adaptive truncation M chosen such that images beyond sqrt(4*Dz*tmax*A) neglected
%  - returns Tt(t<=0) = 0
%
% Example:
%   Tt = Tt_ADE_zefit(t_ps, L_um, n_in, n_ext, lx_um, ly_um, lz_um, mua_per_um, g, ze_um);

% ---------- compute Dz and z0 from your helper functions (units: um,ps) ----------
[~, ~, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);  % expects um -> returns um^2/ps
% [~, z0] = BC_ADE(n_in, n_ext, lx, ly, lz, g);    % returns z0 in um

% ---------- speed in um/ps ----------
v = 299.7924589 / n_in;  % um/ps (299.7924589 um/ps = 299792458.9 m/s scaled)

% ---------- prepare time vector ----------
t = t(:);            % column
Nt = numel(t);

% avoid division by zero: find a tiny safe t for t==0 entries
if any(t <= 0)
    % choose a small fraction of smallest positive time or a tiny fallback
    tpos = t(t>0);
    if isempty(tpos)
        t_eps = 1e-6;   % 1e-6 ps fallback very small
    else
        t_eps = min(tpos) * 1e-6;
    end
    t_safe = t;
    t_safe(t_safe <= 0) = t_eps;
else
    t_safe = t;
end

% ---------- adaptive truncation M ----------
tmax = max(t_safe);
A = 30;  % threshold; contributions ~exp(-A) negligible
period = 2 * (L + 2*ze);  % approx spacing of image sources [um]
if period <= 0, period = eps; end

M = ceil( sqrt(4 * Dz * tmax * A) / period );
M = max(M, 5);   % at least a few images

% ---------- build image-source positions (vectorized) ----------
m = (-M:M).';                       % (2M+1)x1
z1 = L*(1 - 2*m) - 4*m*ze - z0;     % (2M+1)x1
z2 = L*(1 - 2*m) - (4*m - 2)*ze + z0; % (2M+1)x1

% ---------- precompute repeated quantities ----------
% z^2/(4*Dz) has units of ps
z1sq_4Dz = (z1.^2) ./ (4 * Dz);  % (2M+1)x1
z2sq_4Dz = (z2.^2) ./ (4 * Dz);

% make time a row for implicit expansion
trow = t_safe.';  % 1 x Nt

% ---------- exponentials: (2M+1) x Nt ----------
% guard against overflow: use elementwise operations with implicit expansion
E1 = exp( - ( z1sq_4Dz ./ trow ) );   % (2M+1)xNt
E2 = exp( - ( z2sq_4Dz ./ trow ) );

% ---------- image-sum (sum over m) -> 1 x Nt ----------
Img = (z1 .* E1) - (z2 .* E2);   % (2M+1)xNt
Tsum = sum(Img, 1);              % 1 x Nt

% ---------- prefactor and absorption ----------
pref = (1/4) .* ( (pi .* Dz .* (t_safe.^3)).^(-1/2) );  % Nt x 1
absorp = exp( - (v .* t_safe .* mua) );               % Nt x 1

% ---------- final Tt ----------
Tt = (pref .* absorp) .* (Tsum(:));   % column vector

% set Tt exactly zero where t <= 0
Tt(t <= 0) = 0;

end
