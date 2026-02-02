function [ze, z0, info] = BC_ADE(n_in, n_ext, lx, ly, lz, g, varargin)
%BC_ADE  Boundary-condition lengths for ADE slab diffusion (fast quadrature).
%
% Computes:
%   ze : extrapolated boundary length  [same length units as lx,ly,lz]
%   z0 : equivalent isotropic source depth [same units]
%
% Model assumptions (see your project notes).
%
% Usage:
%   [ze,z0] = BC_ADE(n_in,n_ext,lx,ly,lz,g)
%   [ze,z0] = BC_ADE(..., Lmax)
%   [ze,z0] = BC_ADE(..., Lmax, RelTol)
%   [ze,z0] = BC_ADE(..., Lmax, RelTol, AbsTol)
%   [ze,z0,info] = BC_ADE(..., 'ApplyTailCorrection', false)
%
% Name/value parameters (optional):
%   'Nchi'               : Gauss-Legendre nodes in chi (default 200)
%   'Nphi'               : phi samples (default 512)
%   'LmaxCap'            : max odd-l allowed for convergence loop (default max(Lmax,51))
%   'ApplyTailCorrection': true (default) to apply additive z0 tail correction; set false to disable

% ---------- input parsing / validation ----------
p = inputParser();
p.FunctionName = mfilename();

p.addRequired('n_in', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('n_ext',@(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lx',   @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('ly',   @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lz',   @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('g',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','>',-1,'<',1}));

p.addOptional('Lmax',   15,    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive','odd'}));
p.addOptional('RelTol', 1e-5,  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addOptional('AbsTol', 1e-10, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','nonnegative'}));

p.addParameter('Nchi',    200, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('Nphi',    512, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addParameter('LmaxCap', 101,  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive','odd'}));
p.addParameter('ApplyTailCorrection', true, @(x) islogical(x) || (isnumeric(x) && ismember(x,[0,1])));

p.parse(n_in, n_ext, lx, ly, lz, g, varargin{:});

n_in   = p.Results.n_in;
n_ext  = p.Results.n_ext;
lx     = p.Results.lx;   ly = p.Results.ly;   lz = p.Results.lz;
g      = p.Results.g;
Lmax0  = p.Results.Lmax;
RelTol = p.Results.RelTol;
AbsTol = p.Results.AbsTol;
Nchi   = p.Results.Nchi;
Nphi   = p.Results.Nphi;
LmaxCap = max(Lmax0, p.Results.LmaxCap);

applyTail = logical(p.Results.ApplyTailCorrection);

% ---------- refractive index ratio ----------
n = n_in / n_ext;

% ---------- isotropic shortcut (exact) ----------
if lx == ly && lx == lz
    lt = lx / (1 - g);
    if n == 1
        ze = 2*lt/3;
    else
        % A = (1 + 3∫ chi^2 R dchi) / (1 - 2∫ chi R dchi)
        [x,w] = gauss_legendre(Nchi);      % on [-1,1]
        chi = (x + 1)/2;                  % map to [0,1]
        w   = w/2;
        Rchi = fresnel_R(chi, n);
        I1 = sum(w .* (chi .* Rchi));
        I2 = sum(w .* (chi.^2 .* Rchi));
        A = (1 + 3*I2) / (1 - 2*I1);
        ze = 2*A*lt/3;
    end
    z0 = lt;
    if nargout > 2
        info = struct('case','isotropic', 'LmaxUsed',0, 'converged',true, ...
                      'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol, ...
                      'n',n, 'Lxx',lt,'Lyy',lt,'Lzz',lt, 'z0_tail_correction',0, ...
                      'apply_tail_correction', applyTail);
    end
    return
end

% ---------- anisotropic mu parameters ----------
mux = 1/lx;  muy = 1/ly;  muz = 1/lz;

% ---------- quadrature grids ----------
% chi nodes/weights on [-1,1] (full sphere)
[chi_full, w_full] = gauss_legendre(Nchi);   % column
% reuse same nodes mapped to [0,1] (hemisphere)
chi_hemi = (chi_full + 1)/2;
w_hemi   = w_full/2;

phi  = (0:Nphi-1) * (2*pi/Nphi);            % row
wphi = 2*pi/Nphi;

cphi = cos(phi);  sphi = sin(phi);
COS2 = ones(Nchi,1) * (cphi.^2);
SIN2 = ones(Nchi,1) * (sphi.^2);

% ---------- hemisphere fields (for B,C,X,Y) ----------
CHIh = chi_hemi * ones(1,Nphi);
S_h  = sqrt(max(0, 1 - CHIh.^2));
sx_h = S_h .* (ones(Nchi,1) * cphi);
sy_h = S_h .* (ones(Nchi,1) * sphi);
sz_h = CHIh;

mu_h = mux.*(1-CHIh.^2).*COS2 + muy.*(1-CHIh.^2).*SIN2 + muz.*(CHIh.^2);
invmu_h = 1 ./ mu_h;

W_h = (w_hemi * ones(1,Nphi)) * wphi;

% Fresnel R depends only on chi
if n == 1
    Rchi = zeros(size(chi_hemi));
else
    Rchi = fresnel_R(chi_hemi, n);
end
R_h = Rchi * ones(1,Nphi);

% B and X are independent of lambda(s)
B = sum(sum( (CHIh .* invmu_h) .* W_h ));    % ∫ chi / mu_s dΩ (hemisphere)
X = sum(sum( (CHIh .* invmu_h .* R_h) .* W_h ));  % with R

% g == 0 branch: exact (lambda(s)=ell(s)=1/mu_s(s)), enforce z0=lz
if g == 0
    C = sum(sum( (CHIh.^2 .* (invmu_h.^2)) .* W_h ));
    Y = sum(sum( (CHIh.^2 .* (invmu_h.^2) .* R_h) .* W_h ));
    ze = (C + Y) / (B - X);
    z0 = lz;

    if nargout > 2
        info = struct('case','anisotropic_g0', 'LmaxUsed',0, 'converged',true, ...
                      'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol, ...
                      'n',n, 'z0_tail_correction',0, 'apply_tail_correction', applyTail);
    end
    return
end

% ---------- precompute hemisphere linear forms for C and Y under lambda(s)=sx^2 Lxx + sy^2 Lyy + sz^2 Lzz ----------
% C = ∫ chi^2 * lambda(s) / mu(s) dΩ = Lxx*Cx + Lyy*Cy + Lzz*Cz
Cx = sum(sum( (CHIh.^2 .* (sx_h.^2) .* invmu_h) .* W_h  ));
Cy = sum(sum( (CHIh.^2 .* (sy_h.^2) .* invmu_h) .* W_h  ));
Cz = sum(sum( (CHIh.^2 .* (sz_h.^2) .* invmu_h) .* W_h  ));

% Y = ∫ chi^2 * lambda(s) / mu(s) * R(chi) dΩ = Lxx*Yx + Lyy*Yy + Lzz*Yz
Yx = sum(sum( (CHIh.^2 .* (sx_h.^2) .* invmu_h .* R_h) .* W_h ));
Yy = sum(sum( (CHIh.^2 .* (sy_h.^2) .* invmu_h .* R_h) .* W_h ));
Yz = sum(sum( (CHIh.^2 .* (sz_h.^2) .* invmu_h .* R_h) .* W_h ));

denBC = (B - X);

% ---------- full-sphere fields (to compute Lxx,Lyy,Lzz) ----------
CHIf = chi_full * ones(1,Nphi);
S_f  = sqrt(max(0, 1 - CHIf.^2));
sx_f = S_f .* (ones(Nchi,1) * cphi);
sy_f = S_f .* (ones(Nchi,1) * sphi);
sz_f = CHIf;

mu_f = mux.*(1-CHIf.^2).*COS2 + muy.*(1-CHIf.^2).*SIN2 + muz.*(CHIf.^2);
invmu_f = 1 ./ mu_f;

W_f = (w_full * ones(1,Nphi)) * wphi;

% vectorize once for fast dot-products
tx = sx_f(:).';  ty = sy_f(:).';  tz = sz_f(:).';
wvec = W_f(:).';
invvec = invmu_f(:).';

% ---------- accumulate odd-l series for Lxx,Lyy,Lzz with convergence check on (ze,z0) ----------
Lxx = 0; Lyy = 0; Lzz = 0;
ze_prev = NaN; z0_prev = NaN;
converged = false;
l_used = 1;

% --- BOOKKEEPING ARRAYS: initialize before the loop so they always exist ---
Ilx_list = [];
Ily_list = [];
Ilz_list = [];
l_list   = [];

for l = 1:2:LmaxCap
    denom = 1 - g^l;  % safe because |g|<1

    Plx = legendre(l, tx); Plx = Plx(1,:);
    Ply = legendre(l, ty); Ply = Ply(1,:);
    Plz = legendre(l, tz); Plz = Plz(1,:);

    Ilx = sum( tx .* Plx .* invvec .* wvec );
    Ily = sum( ty .* Ply .* invvec .* wvec );
    Ilz = sum( tz .* Plz .* invvec .* wvec );

    alpha = ((2*l+1)/(4*pi)) / denom;

    Lxx = real(Lxx + alpha * Ilx);
    Lyy = real(Lyy + alpha * Ily);
    Lzz = real(Lzz + alpha * Ilz);

    % store for tail-correction computation (use base-alpha later with denom=1)
    Ilx_list(end+1) = Ilx; %#ok<AGROW>
    Ily_list(end+1) = Ily; %#ok<AGROW>
    Ilz_list(end+1) = Ilz; %#ok<AGROW>
    l_list(end+1)   = l;   %#ok<AGROW>

    l_used = l;

    % only start checking after reaching user-requested Lmax0
    if l >= Lmax0
        C = Lxx*Cx + Lyy*Cy + Lzz*Cz;
        Y = Lxx*Yx + Lyy*Yy + Lzz*Yz;
        ze_now = (C + Y) / denBC;
        z0_now = Lzz;

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

% ---------- additive tail correction for z0 (enforce correct g->0 limit) ----------
if applyTail && ~isempty(l_list)
    base_alphas = (2.*l_list + 1) / (4*pi);
    Lzz_g0 = real( sum( base_alphas .* Ilz_list ) );
    tail = lz - Lzz_g0;
    z0 = z0 + tail;
else
    if ~isempty(l_list)
        base_alphas = (2.*l_list + 1) / (4*pi);
        Lzz_g0 = real( sum( base_alphas .* Ilz_list ) ); % reported only
    else
        Lzz_g0 = 0;
    end
    tail = 0;
end

% ---------- output info ----------
if nargout > 2
    info = struct('case','anisotropic_g', 'LmaxUsed',l_used, 'LmaxStart',Lmax0, ...
                  'LmaxCap',LmaxCap, 'converged',converged, ...
                  'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol, ...
                  'n',n, 'Lxx',Lxx,'Lyy',Lyy,'Lzz',Lzz, ...
                  'Lzz_truncated_at_g0', Lzz_g0, ...
                  'z0_tail_correction', tail, ...
                  'apply_tail_correction', applyTail);
end

end

% ---------------- helper: Fresnel reflectance with TIR handling ----------------
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
