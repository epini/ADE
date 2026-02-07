function [ze, z0, info] = BC_ADE(n_in, n_ext, lx, ly, lz, g, varargin)
%BC_ADE  Boundary-condition lengths for ADE slab diffusion (fast quadrature).
%
% Outputs:
%   ze : extrapolation length
%   z0 : source depth
%   info: diagnostics struct (optional)
%
% NOTES (important):
%   - z0 is kept exactly as in your current code (g=0 -> z0=lz; g!=0 -> series).
%   - ze for g!=0 is fixed using the new formula:
%         Y = C * Rbar2,
%     where Rbar2 is the cos^2-weighted effective Fresnel reflectivity computed
%     with P(s) = ell(s)/<ell> = 1/(<ell>*mu_s(s)).
%     This makes ze depend on g only through Dz(g) in C.

% ---------- input parsing / validation ----------
p = inputParser(); p.FunctionName = mfilename();
p.addRequired('n_in',  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('n_ext', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lx',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('ly',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lz',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('g',     @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','>',-1,'<',1}));

p.addOptional('Lmax',   9,    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addOptional('RelTol', 1e-3, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addOptional('AbsTol', 1e-3, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','nonnegative'}));

p.addParameter('Nchi',    200, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('Nphi',    512, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('LmaxCap', 101, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
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

% ensure odd caps
if mod(Lmax0,2)==0,   Lmax0  = Lmax0+1;  end
if mod(LmaxCap,2)==0, LmaxCap = LmaxCap+1; end

% ---------- constants ----------
n = n_in / n_ext;
v = 299.7924589 / n_in;

% ---------- anisotropic rates ----------
mux = 1/lx;  muy = 1/ly;  muz = 1/lz;

% ---------- angular grids ----------
[chi_full, w_full] = gauss_legendre(Nchi);   % chi ∈ [-1,1] column
phi  = (0:Nphi-1) * (2*pi/Nphi);             % phi ∈ [0,2pi), row
wphi = 2*pi/Nphi;
cphi = cos(phi); sphi = sin(phi);

% ---------- full-sphere geometry (for <l> and Hz projections) ----------
CHIf = chi_full * ones(1,Nphi);   % Nchi x Nphi
COS2_f = ones(Nchi,1) * (cphi.^2);
SIN2_f = ones(Nchi,1) * (sphi.^2);

mu_f = mux.*(1-CHIf.^2).*COS2_f + muy.*(1-CHIf.^2).*SIN2_f + muz.*(CHIf.^2);
invmu_f = 1 ./ mu_f;
W_f = (w_full * ones(1,Nphi)) * wphi;        % weights for dΩ on full sphere

% ---------- <l> fast quadrature ----------
lavg = (1/(4*pi)) * sum(sum(invmu_f .* W_f));

% ---------- hemisphere grid (for B, X, and Rbar2) ----------
chi_hemi = (chi_full + 1)/2;    % maps [-1,1] -> [0,1]
w_hemi   = w_full/2;
CHIh = chi_hemi * ones(1,Nphi);

COS2_h = ones(Nchi,1) * (cphi.^2);
SIN2_h = ones(Nchi,1) * (sphi.^2);

mu_h = mux.*(1-CHIh.^2).*COS2_h + muy.*(1-CHIh.^2).*SIN2_h + muz.*(CHIh.^2);
invmu_h = 1 ./ mu_h;
W_h = (w_hemi * ones(1,Nphi)) * wphi;

% ---------- Fresnel reflectance on hemisphere ----------
if n == 1
    Rchi = zeros(size(chi_hemi));
else
    Rchi = fresnel_R(chi_hemi, n);
end
R_h = Rchi * ones(1,Nphi);

% ---------- B and X with P(s)=1/(<l>*mu) (matches your existing convention) ----------
% B = ∫ P cosθ dΩ = (1/<l>) ∫ (cosθ / mu) dΩ
B = (1/lavg) * sum(sum( (CHIh .* invmu_h) .* W_h ));
% X = ∫ P cosθ R(θ) dΩ
X = (1/lavg) * sum(sum( (CHIh .* invmu_h .* R_h) .* W_h ));

% ---------- C from Dz ----------
[~, ~, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g);
C = (2*pi/v) * Dz;

% ---------- g == 0 branch ----------
if g == 0
    % Your existing g=0 formula (kept as-is)
    Y  = (1/lavg) * sum(sum( ((CHIh.^2) .* (invmu_h.^2) .* R_h) .* W_h ));
    ze = (C + Y) / (B - X);
    z0 = lz;

    if nargout > 2
        info = struct('case','anisotropic_g0', 'LmaxUsed',0, 'converged',true, ...
            'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol,'n',n, ...
            'v',v,'lavg',lavg,'Dz',Dz,'B',B,'X',X,'C',C,'Y',Y);
    end
    return
end

% ============================================================
% g != 0 : FIXED ze using NEW formula
%
%   Rbar2 = (∫ P cos^2θ R dΩ) / (∫ P cos^2θ dΩ),   P = 1/(<l> mu)
%   Y     = C * Rbar2
%   ze    = (C + Y)/(B - X) = C*(1+Rbar2)/(B - X)
% ============================================================

I2  = (1/lavg) * sum(sum( ((CHIh.^2) .* invmu_h) .* W_h ));                 % ∫ P cos^2θ dΩ
I2R = (1/lavg) * sum(sum( ((CHIh.^2) .* invmu_h .* R_h) .* W_h ));          % ∫ P cos^2θ R dΩ
if I2 <= 0
    error('BC_ADE:BadIntegral', 'Denominator integral for Rbar2 is non-positive (I2=%g).', I2);
end
Rbar2 = I2R / I2;

Y  = C * Rbar2;
ze = (C + Y) / (B - X);

% ---------- z0 accumulation (kept as in your code; depends on g via series) ----------
% We only need Hz (full-sphere projections). No Hez, no Ycorr.
sz_f = CHIf;
Bz_full = (sz_f .* invmu_f) .* W_f;                 % already includes weights

PHI_full   = ones(Nchi,1) * phi;                    % Nchi x Nphi
THETA_full = acos(CHIf);                            % Nchi x Nphi  (theta in [0,pi])

z0acc = 0;
z0_prev = NaN;
converged = false;
l_used = 1;

for l = 1:2:LmaxCap
    denom = 1 - g^l;

    % only m=0 needed for z0acc
    m = 0;
    Y_full = Ylm(l,m, THETA_full, PHI_full);        % Nchi x Nphi complex
    hz0 = sum(sum(Bz_full .* Y_full));              % projection H_{l,0}^z

    % pole value Y_l^0(theta=0) = sqrt((2l+1)/(4pi))
    Ypole0 = sqrt((2*l + 1) / (4*pi));
    z0acc = z0acc + ( hz0 * Ypole0 ) / denom;

    l_used = l;

    if l >= Lmax0
        z0_now = real(z0acc);

        if ~isnan(z0_prev)
            dz0 = abs(z0_now - z0_prev);
            if dz0 <= AbsTol + RelTol*abs(z0_now)
                z0 = z0_now;
                converged = true;
                break
            end
        end
        z0_prev = z0_now;
        z0 = z0_now;
    end
end

if ~converged
    % still return last estimate
    z0 = real(z0acc);
end

if nargout > 2
    info = struct('case','anisotropic_g', 'LmaxUsed',l_used, 'LmaxStart',Lmax0, ...
        'LmaxCap',LmaxCap,'converged',converged, ...
        'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol,'n',n, ...
        'v',v,'lavg',lavg,'Dz',Dz,'B',B,'X',X,'C',C, ...
        'I2',I2,'I2R',I2R,'Rbar2',Rbar2,'Y',Y,'ze',ze, ...
        'note','For g!=0: Y=C*Rbar2 with Rbar2=(∫P cos^2 R)/(∫P cos^2), P=1/(<l>mu). X,B unchanged; g enters ze only via Dz in C.');
end

end  % of main function


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

% ---- Accurate Gauss-Legendre nodes + weights via Golub-Welsch ----
function [x,w] = gauss_legendre(N)
if N==1
    x = 0; w = 2;
    return
end
beta = (1:N-1) ./ sqrt((2*(1:N-1)).^2 - 1);  % recurrence coeffs for Legendre
J = diag(beta,1) + diag(beta,-1);
[V,D] = eig(J);
[x, idx] = sort(diag(D));
V = V(:,idx);
w = 2 * (V(1,:).^2).';
x = x(:); w = w(:);
end
