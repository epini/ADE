function [ze, z0, info] = BC_ADE(n_in, n_ext, lx, ly, lz, g, varargin)
%BC_ADE  Boundary-condition lengths for ADE slab diffusion (fast quadrature).
%
% Computes the boundary-condition lengths for anisotropic diffusion in slab
% geometry under:
%   - diagonal scattering tensor (principal-axis mean free paths lx,ly,lz)
%   - scalar Henyey–Greenstein asymmetry factor g (|g|<1)
%
% Outputs:
%   ze   : extrapolated boundary length  [same units as lx,ly,lz]
%   z0   : equivalent isotropic source depth [same units]
%   info : diagnostics struct (optional; no printing)
%
% Theory implemented:
%   - z0 is the persistence length along z:
%       z0 = lambda(z-hat),
%     where lambda(s) is given by the odd-l Legendre-kernel series.
%   - ze depends on g only through Dz(g). Boundary angular integrals are
%     computed using the stationary angular density
%       P(s) ∝ 1/mu_s(s),  with normalization by <ell> = (1/4pi)∫ 1/mu_s dΩ.
%     Using Y = C * Rbar2 and C = (2*pi/v) * Dz, we compute:
%       ze = C * (1 + Rbar2) / (B - X),
%     where
%       B = ∫_{Ωup} P(s) s_z dΩ,  X = ∫_{Ωup} P(s) s_z R(θ) dΩ,
%       Rbar2 = (∫_{Ωup} P(s) s_z^2 R(θ) dΩ) / (∫_{Ωup} P(s) s_z^2 dΩ).
%
% Numerical method:
%   - Sphere/hemisphere integrals via Gauss-Legendre quadrature in
%     chi=cos(theta) and periodic trapezoidal quadrature in phi.
%   - Dz(g) is evaluated with the same odd-l spherical-harmonic series
%     structure used in D_Tensor_ADE (without calling Ylm).
%
% Usage:
%   [ze,z0] = BC_ADE(n_in,n_ext,lx,ly,lz,g)
%   [ze,z0] = BC_ADE(..., LmaxStart)
%   [ze,z0] = BC_ADE(..., LmaxStart, RelTol)
%   [ze,z0] = BC_ADE(..., LmaxStart, RelTol, AbsTol)
%
% Name/value parameters:
%   'Nchi'    : Gauss-Legendre nodes in chi (default 200)
%   'Nphi'    : phi samples (default 512)
%   'LmaxCap' : maximum odd-l used for convergence loops (default 101)
%
% Dependencies:
%   - gauss_legendre(n) must be on the path.

% ---------- parse / validate ----------
p = inputParser(); p.FunctionName = mfilename();

p.addRequired('n_in',  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('n_ext', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lx',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('ly',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lz',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('g',     @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','>',-1,'<',1}));

p.addOptional('LmaxStart', 15,   @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addOptional('RelTol',    1e-5, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addOptional('AbsTol',    1e-10,@(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','nonnegative'}));

p.addParameter('Nchi',    200, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('Nphi',    512, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('LmaxCap', 101, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));

p.parse(n_in, n_ext, lx, ly, lz, g, varargin{:});

n_in      = p.Results.n_in;
n_ext     = p.Results.n_ext;
lx        = p.Results.lx;
ly        = p.Results.ly;
lz        = p.Results.lz;
g         = p.Results.g;
LmaxStart = p.Results.LmaxStart;
RelTol    = p.Results.RelTol;
AbsTol    = p.Results.AbsTol;
Nchi      = p.Results.Nchi;
Nphi      = p.Results.Nphi;
LmaxCap   = p.Results.LmaxCap;

% enforce odd truncations (only odd l contribute / are used)
if mod(LmaxStart,2)==0, LmaxStart = LmaxStart + 1; end
if mod(LmaxCap,2)==0,   LmaxCap   = LmaxCap   + 1; end
LmaxCap = max(LmaxCap, LmaxStart);

% ---------- constants ----------
n = n_in / n_ext;
v = 299.7924589 / n_in;

% ---------- isotropic shortcut ----------
if lx == ly && lx == lz
    lt = lx/(1-g);
    if n == 1
        ze = 2*lt/3;
    else
        % A = (1 + 3∫_0^1 chi^2 R dchi) / (1 - 2∫_0^1 chi R dchi)
        [x,w] = gauss_legendre(Nchi);      % nodes/weights on [-1,1]
        chi = (x + 1)/2;  w = w/2;         % map to [0,1]
        Rchi = fresnel_R(chi, n);
        I1 = sum(w .* (chi   .* Rchi));
        I2 = sum(w .* (chi.^2 .* Rchi));
        A  = (1 + 3*I2) / (1 - 2*I1);
        ze = 2*A*lt/3;
    end
    z0 = lt;

    if nargout > 2
        info = struct('case','isotropic', 'n',n, 'v',v, ...
            'ze',ze,'z0',z0, ...
            'LmaxStart',LmaxStart,'LmaxCap',LmaxCap, ...
            'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol);
    end
    return
end

% ---------- anisotropic rates ----------
mux = 1/lx;  muy = 1/ly;  muz = 1/lz;

% ---------- quadrature grids ----------
[chi_full, wchi] = gauss_legendre(Nchi);            % chi in [-1,1] (column)
phi  = (0:Nphi-1) * (2*pi/Nphi);                    % phi in [0,2pi) (row)
wphi = 2*pi/Nphi;

cphi = cos(phi);
sphi = sin(phi);
COS2 = ones(Nchi,1) * (cphi.^2);
SIN2 = ones(Nchi,1) * (sphi.^2);

% ---------- full sphere fields ----------
CHIf = chi_full * ones(1,Nphi);                     % Nchi x Nphi
mu_f = mux.*(1-CHIf.^2).*COS2 + muy.*(1-CHIf.^2).*SIN2 + muz.*(CHIf.^2);
invmu_f = 1 ./ mu_f;

W_f = (wchi * ones(1,Nphi)) * wphi;                 % dΩ weights on full sphere

% direction-averaged mean free path <ell>
lavg = (1/(4*pi)) * sum(sum(invmu_f .* W_f));

% ---------- hemisphere fields for boundary angular integrals ----------
chi_hemi = (chi_full + 1)/2;                        % map [-1,1] -> [0,1]
w_hemi   = wchi/2;

CHIh = chi_hemi * ones(1,Nphi);
mu_h = mux.*(1-CHIh.^2).*COS2 + muy.*(1-CHIh.^2).*SIN2 + muz.*(CHIh.^2);
invmu_h = 1 ./ mu_h;
W_h = (w_hemi * ones(1,Nphi)) * wphi;

% Fresnel reflectance depends only on chi on hemisphere
if n == 1
    Rchi = zeros(size(chi_hemi));
else
    Rchi = fresnel_R(chi_hemi, n);
end
R_h = Rchi * ones(1,Nphi);

% P(s) ∝ 1/mu, normalized by <ell> (common factor cancels in ratios)
B = (1/lavg) * sum(sum( (CHIh .* invmu_h) .* W_h ));                 % ∫ P s_z dΩ
X = (1/lavg) * sum(sum( (CHIh .* invmu_h .* R_h) .* W_h ));          % ∫ P s_z R dΩ

denBC = (B - X);
if ~(denBC > 0)
    error('BC_ADE:BadDenominator', 'B - X is non-positive (B=%g, X=%g).', B, X);
end

I2  = (1/lavg) * sum(sum( (CHIh.^2 .* invmu_h) .* W_h ));            % ∫ P s_z^2 dΩ
I2R = (1/lavg) * sum(sum( (CHIh.^2 .* invmu_h .* R_h) .* W_h ));     % ∫ P s_z^2 R dΩ
if ~(I2 > 0)
    error('BC_ADE:BadIntegral', 'Denominator integral for Rbar2 is non-positive (I2=%g).', I2);
end
Rbar2 = I2R / I2;

% ============================================================
% Dz(g): g enters ze only through Dz via C = (2*pi/v)*Dz
% ============================================================
Iz_full = sum(sum( (CHIf.^2) .* (invmu_f.^2) .* W_f ));
Dz0 = v/(4*pi*lavg) * Iz_full;

Dz = Dz0;
LmaxUsedDz = 0;
convergedDz = true;

if g ~= 0
    % preweighted integrand for H^z projections: (s_z/mu) dΩ
    Bz = (CHIf .* invmu_f) .* W_f;

    corrZ = 0;
    DzPrev = NaN;
    convergedDz = false;

    normfac = 1/sqrt(2*pi);

    for l = 1:2:LmaxCap
        denom = 1 - g^l;                  % safe since |g|<1
        coeff = (g^l) / denom;

        mpos = 0:l;
        E  = exp(1i*(mpos(:)) * phi);     % (l+1) x Nphi
        Et = E.';

        Sz = Bz * Et;                     % Nchi x (l+1)

        P = legendre(l, chi_full, 'norm'); % (l+1) x Nchi

        Hz = zeros(1, l+1);
        for k = 1:(l+1)
            m = mpos(k);
            a = P(m+1,:).';               % Nchi x 1
            a = ((-1)^m) * a;             % Condon–Shortley phase
            Hz(k) = normfac * (a.' * Sz(:,k));
        end

        sumZ = abs(Hz(1))^2 + 2*sum(abs(Hz(2:end)).^2);
        corrZ = corrZ + coeff * sumZ;

        DzNew = Dz0 + v/(4*pi*lavg) * corrZ;

        if l >= LmaxStart && ~isnan(DzPrev)
            if abs(DzNew - DzPrev) <= AbsTol + RelTol*abs(DzNew)
                Dz = DzNew;
                convergedDz = true;
                LmaxUsedDz = l;
                break
            end
        end

        DzPrev = DzNew;
        Dz = DzNew;
        LmaxUsedDz = l;
    end

    if ~convergedDz
        warning('BC_ADE:NoConvergeDz', ...
            'Dz(g) did not meet tolerance by l=%d. Consider increasing LmaxCap (or Nchi/Nphi).', LmaxUsedDz);
    end
end

C = (2*pi/v) * Dz;
ze = C * (1 + Rbar2) / denBC;

% ============================================================
% z0 = lambda(z-hat): odd-l Legendre-kernel series
% ============================================================
if g == 0
    z0 = lz;                               % exact
    LmaxUsedZ0 = 0;
    convergedZ0 = true;
else
    % For s = z-hat, t = s·s' = chi depends only on chi.
    % Integrate over phi first: J(chi) = ∫_0^{2pi} 1/mu(chi,phi) dphi.
    invmu_phiInt = sum(invmu_f, 2) * wphi; % Nchi x 1

    z0acc = 0;
    z0prev = NaN;
    convergedZ0 = false;
    LmaxUsedZ0 = 1;

    for l = 1:2:LmaxCap
        denom = 1 - g^l;

        Pl = legendre(l, chi_full);        % (l+1) x Nchi, unnormalized
        Pl0 = Pl(1,:).';                   % P_l(chi)

        Il = sum( wchi .* (chi_full .* Pl0 .* invmu_phiInt) );  % ∫ chi P_l(chi)/mu dΩ
        term = ((2*l+1)/(4*pi)) * (Il / denom);

        z0acc = real(z0acc + term);
        LmaxUsedZ0 = l;

        if l >= LmaxStart
            if ~isnan(z0prev)
                if abs(z0acc - z0prev) <= AbsTol + RelTol*abs(z0acc)
                    convergedZ0 = true;
                    break
                end
            end
            z0prev = z0acc;
        end
    end

    z0 = z0acc;

    if ~convergedZ0
        warning('BC_ADE:NoConvergeZ0', ...
            'z0=lambda_z did not meet tolerance by l=%d. Consider increasing LmaxCap (or Nchi/Nphi).', LmaxUsedZ0);
    end
end

% ---------- diagnostics ----------
if nargout > 2
    info = struct( ...
        'case','anisotropic', ...
        'n',n,'v',v,'lavg',lavg, ...
        'B',B,'X',X,'denBC',denBC, ...
        'I2',I2,'I2R',I2R,'Rbar2',Rbar2, ...
        'Dz',Dz,'Dz0',Dz0,'C',C, ...
        'ze',ze,'z0',z0, ...
        'LmaxStart',LmaxStart,'LmaxCap',LmaxCap, ...
        'LmaxUsedDz',LmaxUsedDz,'convergedDz',convergedDz, ...
        'LmaxUsedZ0',LmaxUsedZ0,'convergedZ0',convergedZ0, ...
        'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol ...
    );
end

end

% ---- Fresnel reflectance with TIR handling ----
function R = fresnel_R(chi, n)
transArg = 1 - (1 - chi.^2) .* n.^2;
R = zeros(size(chi));
ok = transArg >= 0;
if any(ok(:))
    t  = sqrt(transArg(ok));
    r1 = (n .* chi(ok) - t) ./ (n .* chi(ok) + t);
    r2 = (chi(ok) - n .* t) ./ (chi(ok) + n .* t);
    R(ok) = 0.5 * (abs(r1).^2 + abs(r2).^2);
end
if any(~ok(:)), R(~ok) = 1; end
end

