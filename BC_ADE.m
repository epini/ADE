function [ze, z0, info] = BC_ADE(n_in, n_ext, lx, ly, lz, g, varargin)
%BC_ADE  Boundary-condition lengths for ADE slab diffusion (fast quadrature).
%
% Computes:
%   ze : extrapolated boundary length  [same length units as lx,ly,lz]
%   z0 : equivalent isotropic source depth [same units]
%
% Assumptions:
%   * Directional scattering rate:  mu_s(s) = s' * diag(mu_x,mu_y,mu_z) * s
%     with mu_x = 1/lx, mu_y = 1/ly, mu_z = 1/lz (principal axes).
%   * Scalar Henyey–Greenstein phase function with asymmetry factor g, |g|<1.
%   * For g ~= 0, lambda(s) is evaluated from the truncated spherical-harmonic
%     resolvent series (odd l only) with coefficients
%       H_lm^i = ∫ (s_i / mu_s(s)) * conj(Y_lm(s)) dΩ
%     using a Gauss–Legendre grid in chi=cos(theta) and periodic trapezoid in phi.
%   * For g == 0, uses the exact g=0 boundary integrands and enforces z0 = lz.
%
% Usage:
%   [ze,z0] = BC_ADE(n_in,n_ext,lx,ly,lz,g)
%   [ze,z0] = BC_ADE(..., Lmax)
%   [ze,z0] = BC_ADE(..., Lmax, RelTol)
%   [ze,z0] = BC_ADE(..., Lmax, RelTol, AbsTol)
%
% Name/value parameters (optional):
%   'Nchi'    : Gauss-Legendre nodes in chi (default 200)
%   'Nphi'    : phi samples (default 512)
%   'LmaxCap' : max odd l allowed for convergence loop (default max(Lmax,101))
%
% Outputs:
%   info (optional) struct with diagnostics (no printing).

% ---------- input parsing / validation ----------
p = inputParser(); p.FunctionName = mfilename();

p.addRequired('n_in',  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('n_ext', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lx',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('ly',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lz',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('g',     @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','>',-1,'<',1}));

p.addOptional('Lmax',   15,    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive','odd'}));
p.addOptional('RelTol', 1e-5,  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addOptional('AbsTol', 1e-10, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','nonnegative'}));

p.addParameter('Nchi',    200, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('Nphi',    512, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('LmaxCap', 101, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive','odd'}));

p.parse(n_in, n_ext, lx, ly, lz, g, varargin{:});

n_in   = p.Results.n_in;
n_ext  = p.Results.n_ext;
lx     = p.Results.lx;  ly = p.Results.ly;  lz = p.Results.lz;
g      = p.Results.g;
Lmax0  = p.Results.Lmax;
RelTol = p.Results.RelTol;
AbsTol = p.Results.AbsTol;
Nchi   = p.Results.Nchi;
Nphi   = p.Results.Nphi;
LmaxCap = max(Lmax0, p.Results.LmaxCap);

% ---------- refractive-index ratio ----------
n = n_in / n_ext;

% ---------- isotropic shortcut ----------
if lx == ly && lx == lz
    lt = lx/(1-g);
    if n == 1
        ze = 2*lt/3;
    else
        [x,w] = gauss_legendre(Nchi);     % nodes/weights on [-1,1]
        chi = (x + 1)/2;  w = w/2;        % map to [0,1]
        Rchi = fresnel_R(chi, n);
        I1 = sum(w .* (chi   .* Rchi));
        I2 = sum(w .* (chi.^2 .* Rchi));
        A = (1 + 3*I2) / (1 - 2*I1);
        ze = 2*A*lt/3;
    end
    z0 = lt;
    if nargout > 2
        info = struct('case','isotropic','LmaxUsed',0,'converged',true, ...
                      'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol,'n',n);
    end
    return
end

% ---------- anisotropic rates ----------
mux = 1/lx;  muy = 1/ly;  muz = 1/lz;

% ---------- grids ----------
[chi_full, w_full] = gauss_legendre(Nchi);   % chi ∈ [-1,1], column
phi  = (0:Nphi-1) * (2*pi/Nphi);             % phi ∈ [0,2pi), row
wphi = 2*pi/Nphi;

cphi = cos(phi);
sphi = sin(phi);

chi_hemi = (chi_full + 1)/2;                 % chi ∈ [0,1]
w_hemi   = w_full/2;

% ---------- hemisphere geometry ----------
CHIh = chi_hemi * ones(1,Nphi);
S_h  = sqrt(max(0,1-CHIh.^2));

sx_h = S_h .* (ones(Nchi,1) * cphi);
sy_h = S_h .* (ones(Nchi,1) * sphi);
sz_h = CHIh;

COS2_h = ones(Nchi,1) * (cphi.^2);
SIN2_h = ones(Nchi,1) * (sphi.^2);

mu_h = mux.*(1-CHIh.^2).*COS2_h + muy.*(1-CHIh.^2).*SIN2_h + muz.*(CHIh.^2);
invmu_h = 1 ./ mu_h;

W_h = (w_hemi * ones(1,Nphi)) * wphi;

if n == 1
    Rchi = zeros(size(chi_hemi));
else
    Rchi = fresnel_R(chi_hemi, n);
end
R_h = Rchi * ones(1,Nphi);

B = sum(sum( (CHIh .* invmu_h) .* W_h ));
X = sum(sum( (CHIh .* invmu_h .* R_h) .* W_h ));

% ---------- exact g==0 branch ----------
if g == 0
    C = sum(sum( (CHIh.^2 .* (invmu_h.^2)) .* W_h ));
    Y = sum(sum( (CHIh.^2 .* (invmu_h.^2) .* R_h) .* W_h ));
    ze = (C + Y) / (B - X);
    z0 = lz;
    if nargout > 2
        info = struct('case','anisotropic_g0','LmaxUsed',0,'converged',true, ...
                      'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol,'n',n);
    end
    return
end

% ---------- full-sphere fields for H integrals ----------
CHIf = chi_full * ones(1,Nphi);
S_f  = sqrt(max(0,1-CHIf.^2));

sx_f = S_f .* (ones(Nchi,1) * cphi);
sy_f = S_f .* (ones(Nchi,1) * sphi);
sz_f = CHIf;

COS2_f = ones(Nchi,1) * (cphi.^2);
SIN2_f = ones(Nchi,1) * (sphi.^2);

mu_f = mux.*(1-CHIf.^2).*COS2_f + muy.*(1-CHIf.^2).*SIN2_f + muz.*(CHIf.^2);
invmu_f = 1 ./ mu_f;

W_f = (w_full * ones(1,Nphi)) * wphi;

Bx = (sx_f .* invmu_f) .* W_f;
By = (sy_f .* invmu_f) .* W_f;
Bz = (sz_f .* invmu_f) .* W_f;

% Precompute exp(±i m phi) up to LmaxCap
mAll = 0:LmaxCap;
ExpPlus  = exp( 1i * (mAll(:) * phi));  % (LmaxCap+1) x Nphi
ExpMinus = exp(-1i * (mAll(:) * phi));  % (LmaxCap+1) x Nphi

% Storage for H: for each odd l, store m=0..l
Hx = cell(LmaxCap+1,1);
Hy = cell(LmaxCap+1,1);
Hz = cell(LmaxCap+1,1);

normfac = 1/sqrt(2*pi);

% Compute H_lm^i up to LmaxCap
for l = 1:2:LmaxCap
    P = legendre(l, chi_full, 'norm');     % (l+1) x Nchi

    hx = zeros(l+1,1);
    hy = zeros(l+1,1);
    hz = zeros(l+1,1);

    for m = 0:l
        signm = (-1)^m;
        E = ExpMinus(m+1,:).';             % Nphi x 1
        Sx = Bx * E;                       % Nchi x 1
        Sy = By * E;
        Sz = Bz * E;

        Pm = P(m+1,:).';                   % Nchi x 1

        hx(m+1) = normfac * signm * (Pm.' * Sx);
        hy(m+1) = normfac * signm * (Pm.' * Sy);
        hz(m+1) = normfac * signm * (Pm.' * Sz);
    end

    Hx{l+1} = hx;
    Hy{l+1} = hy;
    Hz{l+1} = hz;
end

% ---------- build ze and z0 via truncated resolvent series ----------
WC = (CHIh.^2 .* invmu_h) .* W_h;
WY = WC .* R_h;

Cacc = 0;
Yacc = 0;
z0acc = 0;

ze_prev = NaN;
z0_prev = NaN;
converged = false;
l_used = 1;

for l = 1:2:LmaxCap
    denom = 1 - g^l;

    P_h = legendre(l, chi_hemi, 'norm');   % (l+1) x Nchi

    % pole: only m=0 contributes
    P_pole = legendre(l, 1, 'norm');       % (l+1) x 1
    Ypole0 = normfac * P_pole(1);
    z0acc  = z0acc + (Hz{l+1}(1) * Ypole0) / denom;

    for m = 0:l
        Pm = P_h(m+1,:).';                 % Nchi x 1
        signm = (-1)^m;

        Ym = normfac * signm * (Pm * ExpPlus(m+1,:));   % Nchi x Nphi

        dotH = sx_h * Hx{l+1}(m+1) + sy_h * Hy{l+1}(m+1) + sz_h * Hz{l+1}(m+1);
        prodRe = real(dotH .* Ym);

        fac = (m==0) * (1/denom) + (m>0) * (2/denom);

        Cacc = Cacc + fac * sum(sum(prodRe .* WC));
        Yacc = Yacc + fac * sum(sum(prodRe .* WY));
    end

    l_used = l;

    if l >= Lmax0
        ze_now = (Cacc + Yacc) / (B - X);
        z0_now = real(z0acc);

        if ~isnan(ze_prev)
            dze = abs(ze_now - ze_prev);
            dz0 = abs(z0_now - z0_prev);
            if (dze <= AbsTol + RelTol*abs(ze_now)) && (dz0 <= AbsTol + RelTol*abs(z0_now))
                ze = ze_now;
                z0 = z0_now;
                converged = true;
                break
            end
        end

        ze_prev = ze_now;
        z0_prev = z0_now;
        ze = ze_now;
        z0 = z0_now;
    end
end

if nargout > 2
    info = struct('case','anisotropic_g', 'LmaxUsed',l_used, 'LmaxStart',Lmax0, ...
                  'LmaxCap',LmaxCap,'converged',converged, ...
                  'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol,'n',n);
end

end

% ---- Fresnel reflectance with TIR handling ----
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
