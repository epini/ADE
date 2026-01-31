function [ze, z0] = BC_ADE_Alerstam(n_in, n_ext, lx, ly, lz, g, reltol, abstol)
% BC_ADE_Alerstam  Boundary conditions using Alerstam approximation (l* method)
%
% Procedure:
%   i)   Compute diffusion tensor elements D_i via D_Tensor_ADE(...)
%   ii)  Convert to reduced scattering coefficients mu_s,i' using D_i = v/(3 mu_s,i')
%   iii) Compute z_e with the standard anisotropic integral formula, replacing mu_s(s) -> mu_s'(s)
%   iv)  Set z0 = lz* = 1/mu_s,z'
%
% Inputs:
%   n_in, n_ext - refractive indices (inside/outside)
%   lx,ly,lz    - microscopic mean free paths along x,y,z [mm]
%   g           - HG asymmetry factor
% Optional:
%   reltol, abstol - tolerances for integral2 (defaults 1e-6, 1e-8)
%
% Outputs:
%   ze - extrapolated length [mm]
%   z0 - equivalent isotropic source depth [mm] (Alerstam: z0 = lz*)
%
% Requires:
%   [Dx, Dy, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g)

if nargin < 7 || isempty(reltol), reltol = 1e-6; end
if nargin < 8 || isempty(abstol), abstol = 1e-8; end

% Speed of light in mm/ns and transport speed in medium
c0 = 299.7924589;         % mm/ns
v  = c0 / n_in;

% refractive index ratio for Fresnel
n = n_in / n_ext;

% --- isotropic shortcut (Alerstam reduces to standard isotropic result) ---
if lx == ly && lx == lz
    lt = lx/(1-g);  % transport mfp
    Rfun = @(chi) fresnel_R(chi, n);
    Cfun = @(chi) (chi.^2).*Rfun(chi);
    Bfun = @(chi) chi.*Rfun(chi);

    if n == 1
        ze = 2*lt/3;
    else
        A = (1 + 3*integral(Cfun, 0, 1, 'RelTol',reltol,'AbsTol',abstol)) / ...
            (1 - 2*integral(Bfun, 0, 1, 'RelTol',reltol,'AbsTol',abstol));
        ze = 2*A*lt/3;
    end

    z0 = lt;  % z0 = l* in isotropic case
    return
end

% --- Step (i): compute diffusion tensor elements from your ADE expression ---
[Dx, Dy, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);

% --- Step (ii): convert to reduced scattering coefficients mu_s,i' ---
% D_i = v / (3 mu_s,i')  -> mu_s,i' = v / (3 D_i)
mux_p = v / (3*Dx);
muy_p = v / (3*Dy);
muz_p = v / (3*Dz);

% --- Build direction-dependent reduced scattering rate mu_s'(s) ---
% using chi = cos(theta), phi azimuth
mus_p = @(chi,phi) mux_p.*(1 - chi.^2).*(cos(phi).^2) + ...
                   muy_p.*(1 - chi.^2).*(sin(phi).^2) + ...
                   muz_p.*(chi.^2);

% Fresnel reflectance (chi = cos(theta))
Rfun = @(chi) fresnel_R(chi, n);

% --- Step (iii): compute ze using the "standard" integral formula but with mus_p ---
% Same structure as your g==0 anisotropic branch, but mus replaced by mus'
Bfun = @(chi,phi) chi ./ mus_p(chi,phi);
Xfun = @(chi,phi) Bfun(chi,phi) .* Rfun(chi);

Cfun = @(chi,phi) (chi.^2) ./ (mus_p(chi,phi).^2);
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

% --- Step (iv): z0 = lz* ---
z0 = 1 / muz_p;

end

% ---------------- helper: Fresnel reflectance with TIR handling -------------
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
