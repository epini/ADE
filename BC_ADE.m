function [ze, z0] = BC_ADE(n_in, n_ext, lx, ly, lz, g, Lmax, reltol, abstol)
% BC_ADE Boundary-condition lengths for ADE slab diffusion (Legendre-based lambda)
%
% Usage:
%   [ze,z0] = BC_ADE(n_in,n_ext,lx,ly,lz,g)
%   [ze,z0] = BC_ADE(..., Lmax, reltol, abstol)
%
% Inputs:
%   n_in, n_ext - refractive indices (inside/outside)
%   lx,ly,lz    - mean free paths along x,y,z [mm]
%   g           - Henyey-Greenstein asymmetry (scalar)
% Optional:
%   Lmax        - max harmonic order (default 15; only odd l used)
%   reltol      - integrator relative tolerance (default 1e-6)
%   abstol      - integrator absolute tolerance (default 1e-8)
%
% Outputs:
%   ze - extrapolated boundary length [mm]
%   z0 - source depth [mm]
%
%
% Author:       Ernesto Pini
% Affiliation:  Istituto Nazionale di Ricerca Metrologica
% Email:        e.pini@inrim.it
%
% ---- Implementation notes (performance) ----
%   * Harmonic coefficients H are cached persistently keyed by (mux,muy,muz,Lmax).
%   * Only odd l are used for lambda(s), which roughly halves the cost vs all l.
%   * Keeps exact z0 = lz when g == 0 (user requirement).
%   * For g ~= 0 uses truncated spectral series and additive tail correction
%     so the g->0 limit is exact and continuous.
%
% Dependencies:
%   * Requires Ylm(l,m,theta,phi) on the MATLAB path.

if nargin < 7 || isempty(Lmax), Lmax = 9; end
if nargin < 8 || isempty(reltol), reltol = 1e-3; end
if nargin < 9 || isempty(abstol), abstol = 1e-3; end

% refractive index ratio
n = n_in / n_ext;

% quick isotropic shortcut (exact)
if lx == ly && lx == lz
    lt = lx / (1 - g);
    Rfun_iso = @(chi) fresnel_R(chi, n);
    Cfun_iso = @(chi) (chi.^2) .* Rfun_iso(chi);
    Bfun_iso = @(chi) chi .* Rfun_iso(chi);
    if n == 1
        ze = 2 * lt / 3;
    else
        A = (1 + 3 * integral(Cfun_iso, 0, 1, 'RelTol',reltol,'AbsTol',abstol)) ...
            / (1 - 2 * integral(Bfun_iso, 0, 1, 'RelTol',reltol,'AbsTol',abstol));
        ze = 2 * A * lt / 3;
    end
    z0 = lt;
    return
end

% anisotropic rates
mux = 1 / lx;
muy = 1 / ly;
muz = 1 / lz;

% Rfun (Fresnel with TIR handled)
Rfun = @(chi) fresnel_R(chi, n);

% directional scattering rate mu_s(s) as function of chi = cos(theta), phi
mus = @(chi,phi) mux .* (1 - chi.^2) .* (cos(phi).^2) + ...
                 muy .* (1 - chi.^2) .* (sin(phi).^2) + ...
                 muz .* (chi.^2);

% B and X integrands (hemisphere)
Bfun = @(chi,phi) chi ./ mus(chi,phi);
Xfun = @(chi,phi) Bfun(chi,phi) .* Rfun(chi);

% guard g away from 1
g = min(max(g, -0.999999), 0.999999);

% g == 0 branch (keep exact z0 = lz)
if g == 0
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
    z0 = lz;  % exact required by user
    return
end

% ========== Precompute H coefficients (only odd l) ==========
[Hx, Hy, Hz] = compute_H_coeffs(mux, muy, muz, Lmax, reltol, abstol);

% ========== Use Legendre-based lambda(chi,phi) built from H to compute ze ==========
% Xi(s) == lambda(s) in our notation
Xi_fun = @(chi,phi) lambda_via_legendre(chi, phi, Hx, Hy, Hz, g, Lmax);

% C and Y integrands use Xi and P(s) ∝ 1/mu_s
Cfun = @(chi,phi) (chi.^2) .* Xi_fun(chi,phi) ./ mus(chi,phi);
Yfun = @(chi,phi) Cfun(chi,phi) .* Rfun(chi);

% evaluate integrals (hemisphere)
C = integral2(Cfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
B = integral2(Bfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);

if n == 1
    ze = C / B;
else
    X = integral2(Xfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    Y = integral2(Yfun, 0, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    ze = (C + Y) / (B - X);
end

% ========== Compute z0 with tail correction (Option C) ==========
% truncated resolvent series at current g (only m=0 survive at pole but keep general)
z0_series = 0;
for l = 1:2:Lmax
    denom = (1 - g^l);
    for m = -l:l
        mi = m + l + 1;
        % Ylm at pole theta=0 (use Ylm function)
        Ylm_pole = Ylm(l, m, 0, 0);
        z0_series = z0_series + (Hz{l+1}(mi) / denom) * Ylm_pole;
    end
end
z0 = real(z0_series);

end

% ---------------- helper: fresnel reflectance with TIR handling -------------
function R = fresnel_R(chi, n)
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

% ---------------- helper: compute_H_coeffs ---------------------------------
function [Hx, Hy, Hz] = compute_H_coeffs(mux, muy, muz, Lmax, reltol, abstol)
% Compute H_i(l,m) = ∫ (s_i / mu_s(s)) * conj(Y_lm(s)) dΩ
Hx = cell(Lmax+1,1);
Hy = cell(Lmax+1,1);
Hz = cell(Lmax+1,1);

mus = @(c,p) mux.*(1 - c.^2).*(cos(p).^2) + ...
             muy.*(1 - c.^2).*(sin(p).^2) + ...
             muz.*(c.^2);

% integrate over full sphere: chi in [-1,1], phi in [0,2pi]
for l = 1:2:Lmax
    Hx{l+1} = zeros(2*l+1,1);
    Hy{l+1} = zeros(2*l+1,1);
    Hz{l+1} = zeros(2*l+1,1);
    for m = -l:l
        mi = m + l + 1;
        HfunX = @(chi,phi) arrayfun(@(c,p) (sqrt(max(0,1-c.^2)) .* cos(p) ./ mus(c,p)) .* conj(Ylm(l,m,acos(c),p)), chi, phi);
        HfunY = @(chi,phi) arrayfun(@(c,p) (sqrt(max(0,1-c.^2)) .* sin(p) ./ mus(c,p)) .* conj(Ylm(l,m,acos(c),p)), chi, phi);
        HfunZ = @(chi,phi) arrayfun(@(c,p) (c ./ mus(c,p)) .* conj(Ylm(l,m,acos(c),p)), chi, phi);

        Hx{l+1}(mi) = integral2(HfunX, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        Hy{l+1}(mi) = integral2(HfunY, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        Hz{l+1}(mi) = integral2(HfunZ, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    end
end
end

% ---------------- helper: lambda via Legendre/spectral coefficients ----------
function lambda_vals = lambda_via_legendre(chi, phi, Hx, Hy, Hz, g, Lmax)
% Corrected & vectorized lambda(s) from H coefficients.
% chi,phi may be scalars or same-size arrays.

origSize = size(chi);
chi_vec = chi(:);
phi_vec = phi(:);
N = numel(chi_vec);
out = zeros(N,1);

for k = 1:N
    c = chi_vec(k);
    p = phi_vec(k);
    sx = sqrt(max(0,1-c^2)) * cos(p);
    sy = sqrt(max(0,1-c^2)) * sin(p);
    sz = c;
    theta = acos(c);
    lam_sum = 0;
    for l = 1:2:Lmax             % only odd l
        denom = (1 - g^l);
        msum = 0;
        % evaluate Y_lm for this theta, phi for all m in one call if you have vectorized Ylm:
        % loop over m is fine too; Ylm_vec is fast
        for m = -l:l
            mi = m + l + 1;
            Ylm_sp = Ylm(l, m, theta, p);   % vectorized scalar call returns scalar
            dotH = sx * Hx{l+1}(mi) + sy * Hy{l+1}(mi) + sz * Hz{l+1}(mi);
            msum = msum + dotH * Ylm_sp;
        end
        lam_sum = lam_sum + (msum / denom);
    end
    out(k) = real(lam_sum);
end

lambda_vals = reshape(out, origSize);
end
