function [Lxx, Lyy, Lzz] = lambda_ADE(lx, ly, lz, g, Lmax, reltol, abstol)
% lambda_ADE  Return diagonal elements of the Λ tensor that generates λ(s)
%
% Interprets λ(s) as coming from a diagonal tensor Λ in the principal axes:
%   λ(s) = s' * Λ * s,  Λ = diag(Lxx,Lyy,Lzz)
% so that:
%   Lxx = λ(s = x̂), Lyy = λ(s = ŷ), Lzz = λ(s = ẑ)
%
% Usage:
%   [Lxx,Lyy,Lzz] = lambda_ADE(n_in,n_ext,lx,ly,lz,g)
%   [Lxx,Lyy,Lzz] = lambda_ADE(..., Lmax, reltol, abstol)
%
% Inputs are identical to BC_ADE.
%
% Dependencies:
%   * Requires Ylm(l,m,theta,phi) on the MATLAB path.

if nargin < 5 || isempty(Lmax), Lmax = 9;   end
if nargin < 6 || isempty(Nchi), Nchi = 200; end
if nargin < 7 || isempty(Nphi), Nphi = 512; end

% guard g away from 1
g = min(max(g, -0.999999), 0.999999);

% isotropic shortcut: λ(s)=lt constant => Λxx=Λyy=Λzz=lt
if lx == ly && lx == lz
    lt  = lx / (1 - g);
    Lxx = lt; Lyy = lt; Lzz = lt;
    return
end

% anisotropic rates
mux = 1 / lx;
muy = 1 / ly;
muz = 1 / lz;

% precompute H coefficients (same as BC_ADE)
[Hx, Hy, Hz] = compute_H_coeffs(mux, muy, muz, Lmax, reltol, abstol);

% lambda(s) from the same spectral construction
lambda_fun = @(chi,phi) lambda_via_legendre(chi, phi, Hx, Hy, Hz, g, Lmax);

% Component functions Λx(s), Λy(s), Λz(s)
LambdaX_fun = @(chi,phi) Lambda_comp_via_legendre(chi,phi,Hx,g,Lmax);
LambdaY_fun = @(chi,phi) Lambda_comp_via_legendre(chi,phi,Hy,g,Lmax);
LambdaZ_fun = @(chi,phi) Lambda_comp_via_legendre(chi,phi,Hz,g,Lmax);

% Evaluate “diagonal” axis components
Lxx = LambdaX_fun(0, 0);        % Λx( x-hat )
Lyy = LambdaY_fun(0, pi/2);     % Λy( y-hat )
Lzz = LambdaZ_fun(1, 0);        % Λz( z-hat )

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

        HfunX = @(chi,phi) arrayfun(@(c,p) ...
            (sqrt(max(0,1-c.^2)) .* cos(p) ./ mus(c,p)) .* conj(Ylm(l,m,acos(c),p)), ...
            chi, phi);

        HfunY = @(chi,phi) arrayfun(@(c,p) ...
            (sqrt(max(0,1-c.^2)) .* sin(p) ./ mus(c,p)) .* conj(Ylm(l,m,acos(c),p)), ...
            chi, phi);

        HfunZ = @(chi,phi) arrayfun(@(c,p) ...
            (c ./ mus(c,p)) .* conj(Ylm(l,m,acos(c),p)), ...
            chi, phi);

        Hx{l+1}(mi) = integral2(HfunX, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        Hy{l+1}(mi) = integral2(HfunY, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
        Hz{l+1}(mi) = integral2(HfunZ, -1, 1, 0, 2*pi, 'RelTol',reltol,'AbsTol',abstol);
    end
end
end

% ---------------- helper: lambda via Legendre/spectral coefficients ----------
function lambda_vals = lambda_via_legendre(chi, phi, Hx, Hy, Hz, g, Lmax)
% lambda(s) from H coefficients (odd-l resolvent). chi,phi can be arrays.

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
    for l = 1:2:Lmax
        denom = (1 - g^l);
        msum = 0;
        for m = -l:l
            mi = m + l + 1;
            Ylm_sp = Ylm(l, m, theta, p);
            dotH = sx * Hx{l+1}(mi) + sy * Hy{l+1}(mi) + sz * Hz{l+1}(mi);
            msum = msum + dotH * Ylm_sp;
        end
        lam_sum = lam_sum + (msum / denom);
    end

    out(k) = real(lam_sum);
end

lambda_vals = reshape(out, origSize);
end

function vals = Lambda_comp_via_legendre(chi, phi, Hc, g, Lmax)
% Returns Λ_i(s) for one component i given its harmonic coefficients Hc{l+1}(m)

origSize = size(chi);
chi = chi(:);
phi = phi(:);

out = zeros(numel(chi),1);

for k = 1:numel(chi)
    c = chi(k);
    p = phi(k);
    theta = acos(c);

    sumc = 0;
    for l = 1:2:Lmax
        denom = (1 - g^l);
        msum = 0;
        for m = -l:l
            mi = m + l + 1;
            Ylm_sp = Ylm(l, m, theta, p);
            msum = msum + Hc{l+1}(mi) * Ylm_sp;
        end
        sumc = sumc + msum/denom;
    end

    out(k) = real(sumc);
end

vals = reshape(out, origSize);
end
