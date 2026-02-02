function [ze, z0] = BC_ADE_corrected(n_in, n_ext, lx, ly, lz, g, Lmax, reltol, abstol)
% BC_ADE_corrected Boundary-condition lengths for ADE slab diffusion (with tail correction)
%
% Usage:
%   [ze,z0] = BC_ADE_corrected(n_in,n_ext,lx,ly,lz,g)
%   [ze,z0] = BC_ADE_corrected(..., Lmax, reltol, abstol)
%
% Inputs:
%   same as original. Optional:
%     Lmax  - maximum harmonic order (default 15; only odd l used)
%     reltol, abstol - integrator tolerances (defaults provided)
%
% Behavior:
%   - Keeps exact z0 = lz for g == 0.
%   - For g ~= 0 uses resolvent series for z0, then applies additive "tail correction"
%     computed as (lz - truncated_series_at_g0) to enforce the correct g->0 limit.
%
% Author: adapted from Ernesto Pini's original
% Minor: uses only odd l (1,3,5,...)

if nargin < 7 || isempty(Lmax), Lmax = 15; end
if nargin < 8 || isempty(reltol), reltol = 1e-4; end
if nargin < 9 || isempty(abstol), abstol = 1e-4; end

% refractive index ratio
n = n_in / n_ext;

% quick isotropic shortcut
if lx == lz && lx == ly
    lt = lx/(1 - g);
    Rfun_iso = @(chi) fresnel_R(chi, n);
    Cfun = @(chi) (chi.^2) .* Rfun_iso(chi);
    Bfun = @(chi) chi .* Rfun_iso(chi);
    if n == 1
        ze = 2 * lt / 3;
    else
        A = (1 + 3 * integral(Cfun, 0, 1, 'RelTol',reltol,'AbsTol',abstol)) ...
            / (1 - 2 * integral(Bfun, 0, 1, 'RelTol',reltol,'AbsTol',abstol));
        ze = 2 * A * lt / 3;
    end
    z0 = lt;
    return
end

% anisotropic: convert lengths -> scattering rates
mux = 1 / lx;
muy = 1 / ly;
muz = 1 / lz;

% Rfun with TIR handling
Rfun = @(chi) fresnel_R(chi, n);

% directional scattering rate mu_s(s) for a diagonal tensor
mus = @(chi,phi) mux .* (1 - chi.^2) .* (cos(phi).^2) + ...
                 muy .* (1 - chi.^2) .* (sin(phi).^2) + ...
                 muz .* (chi.^2);

% B and X integrands use true step statistics (P(s) ∝ 1/mu_s)
Bfun = @(chi,phi) chi ./ mus(chi,phi);
Xfun = @(chi,phi) Bfun(chi,phi) .* Rfun(chi);

% g numerical guard
g = min(max(g, -0.999999), 0.999999);

% g == 0 simple branch: keep exact solution (user required)
if g == 0
    % compute ze similarly to original g==0 branch
    Cfun = @(chi,phi) (chi.^2) ./ (mus(chi,phi).^2);
    Yfun = @(chi,phi) Cfun(chi,phi) .* Rfun(chi);
    C = integral2(Cfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    B = integral2(Bfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    if n == 1
        ze = C / B;
    else
        X = integral2(Xfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        Y = integral2(Yfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        ze = (C + Y) / (B - X);
    end
    z0 = lz;  % exact required
    return
end

% For g != 0: compute harmonic coefficients
[Hx, Hy, Hz] = compute_H_coeffs(mux, muy, muz, Lmax, reltol, abstol);

% build Xi(s) function from truncated series
Xi_fun = @(chi,phi) arrayfun(@(c,p) Xi_scalar(c,p,g,Lmax,Hx,Hy,Hz), chi, phi);

% integrals for ze (use Xi in C and Y)
Cfun = @(chi,phi) (chi.^2) .* Xi_fun(chi,phi) ./ mus(chi,phi);
Yfun = @(chi,phi) Cfun(chi,phi) .* Rfun(chi);

C = integral2(Cfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
B = integral2(Bfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);

if n == 1
    ze = C / B;
else
    X = integral2(Xfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    Y = integral2(Yfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    ze = (C + Y) / (B - X);
end

% ========== z0: truncated resolvent series ==========
% compute truncated series at current g
z0_series = 0;
for l = 1:2:Lmax     % only odd l
    denom = (1 - g^l);
    for m = -l:l
        mi = m + l + 1;
        z0_series = z0_series + (Hz{l+1}(mi) / denom) * Ylm(l, m, 0, 0);
    end
end
z0_series = real(z0_series);

% compute truncated series evaluated at g=0 (i.e., with denom = 1)
z0_series_g0 = 0;
for l = 1:2:Lmax
    for m = -l:l
        mi = m + l + 1;
        z0_series_g0 = z0_series_g0 + Hz{l+1}(mi) * Ylm(l, m, 0, 0);
    end
end
z0_series_g0 = real(z0_series_g0);

% additive tail correction: missing contribution from ell > Lmax
tail = lz - z0_series_g0;

% Apply correction fully so that the g->0 limit is exact.
% (We keep g==0 branch exact above; this enforces smoothness near 0.)
z0 = z0_series + tail;

end

% ----------------- helper functions -----------------

function R = fresnel_R(chi, n)
% Fresnel reflectance average with TIR handling.
transArg = 1 - (1 - chi.^2) .* n.^2;
R = zeros(size(chi));
ok = transArg >= 0;
if any(ok(:))
    t = sqrt(transArg(ok));
    r1 = (n .* chi(ok) - t) ./ (n .* chi(ok) + t);
    r2 = (chi(ok) - n .* t) ./ (chi(ok) + n .* t);
    R(ok) = 0.5 * (abs(r1).^2 + abs(r2).^2);
end
if any(~ok(:)), R(~ok) = 1; end
end

function [Hx, Hy, Hz] = compute_H_coeffs(mux, muy, muz, Lmax, reltol, abstol)
% Compute H_i(l,m) = ∫ (s_i / mu_s(s)) * conj(Y_lm(s)) dΩ
Hx = cell(Lmax+1,1);
Hy = cell(Lmax+1,1);
Hz = cell(Lmax+1,1);

mus = @(c,p) mux.*(1 - c.^2).*(cos(p).^2) + ...
             muy.*(1 - c.^2).*(sin(p).^2) + ...
             muz.*(c.^2);

for l = 1:2:Lmax   % only odd l
    Hx{l+1} = zeros(2*l+1,1);
    Hy{l+1} = zeros(2*l+1,1);
    Hz{l+1} = zeros(2*l+1,1);
    for m = -l:l
        mi = m + l + 1;
        HfunX = @(chi,phi) arrayfun(@(c,p) (sqrt(max(0,1-c.^2)) .* cos(p) ./ mus(c,p)) .* conj( Ylm(l,m, acos(c), p ) ), chi, phi);
        HfunY = @(chi,phi) arrayfun(@(c,p) (sqrt(max(0,1-c.^2)) .* sin(p) ./ mus(c,p)) .* conj( Ylm(l,m, acos(c), p ) ), chi, phi);
        HfunZ = @(chi,phi) arrayfun(@(c,p) (c ./ mus(c,p)) .* conj( Ylm(l,m, acos(c), p ) ), chi, phi);

        Hx{l+1}(mi) = integral2(HfunX, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        Hy{l+1}(mi) = integral2(HfunY, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        Hz{l+1}(mi) = integral2(HfunZ, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    end
end
end

function Xi = Xi_scalar(c, p, g, Lmax, Hx, Hy, Hz)
sx = sqrt(max(0,1-c.^2)) .* cos(p);
sy = sqrt(max(0,1-c.^2)) .* sin(p);
sz = c;
th = acos(c);
Xi = 0;
for l = 1:2:Lmax   % only odd l
    denom = (1 - g^l);
    for m = -l:l
        mi = m + l + 1;
        Y = Ylm(l, m, th, p);  % expects scalar inputs
        dotH = sx * Hx{l+1}(mi) + sy * Hy{l+1}(mi) + sz * Hz{l+1}(mi);
        Xi = Xi + (dotH / denom) * Y;
    end
end
Xi = real(Xi);
end
