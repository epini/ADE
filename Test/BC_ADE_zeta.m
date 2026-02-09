function [ze, z0, info] = BC_ADE_zeta(n_in, n_ext, lx, ly, lz, g, varargin)
%BC_ADE  Boundary-condition lengths for ADE slab diffusion (fast quadrature).
%
% This variant computes:
%  - H^z_{lm} as integral over full solid angle 4pi (bulk projection)
%  - Htilde^z_{lm} as integral over upper hemisphere Omega_up (Fresnel-weighted)
% and uses the odd-l / even-m selection rules for the z component.
%
% Inputs/outputs and options same as previous versions.
% Dependencies: gauss_legendre(n)

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

% ---------- full sphere fields (for H bulk, Dz, lavg, z0 integrals) ----------
CHIf = chi_full * ones(1,Nphi);                     % Nchi x Nphi
mu_f = mux.*(1-CHIf.^2).*COS2 + muy.*(1-CHIf.^2).*SIN2 + muz.*(CHIf.^2);
invmu_f = 1 ./ mu_f;
W_f = (wchi * ones(1,Nphi)) * wphi;                 % dΩ weights on full sphere

% direction-averaged mean free path <ell>
lavg = (1/(4*pi)) * sum(sum(invmu_f .* W_f));

% ---------- hemisphere fields (for boundary integrals and Htilde) ----------
% map chi_full -> chi_hemi in [0,1] for hemisphere quadrature (same Nchi)
chi_hemi = (chi_full + 1)/2;
w_hemi   = wchi/2;
CHIh = chi_hemi * ones(1,Nphi);
mu_h = mux.*(1-CHIh.^2).*COS2 + muy.*(1-CHIh.^2).*SIN2 + muz.*(CHIh.^2);
invmu_h = 1 ./ mu_h;
W_h = (w_hemi * ones(1,Nphi)) * wphi;

% Fresnel reflectance on hemisphere
if n == 1
    Rchi = zeros(size(chi_hemi));
else
    Rchi = fresnel_R(chi_hemi, n);
end
R_h = Rchi * ones(1,Nphi);

% ---------- boundary integrals (internal normalization: multiply by 1/lavg when needed) ----------
B = (1/lavg) * sum(sum( (CHIh .* invmu_h) .* W_h ));                 % ∫_up P s_z dΩ
X = (1/lavg) * sum(sum( (CHIh .* invmu_h .* R_h) .* W_h ));          % ∫_up P s_z R dΩ

denBC = (B - X);
if ~(denBC > 0)
    error('BC_ADE:BadDenominator', 'B - X is non-positive (B=%g, X=%g).', B, X);
end

% I2 and I2R for g=0 (hemi integrals used as Y|g=0 and C internal)
I2  = (1/lavg) * sum(sum( (CHIh.^2 .* invmu_h.^2) .* W_h ));             % internal C
I2R = (1/lavg) * sum(sum( (CHIh.^2 .* invmu_h.^2 .* R_h) .* W_h ));      % Y|g=0 (emisfero)
Y0 = I2R;
Y  = Y0;

% ============================================================
% Dz(g): initial value from full-sphere integral
% ============================================================
Iz_full = sum(sum( (CHIf.^2) .* (invmu_f.^2) .* W_f ));
Dz0 = v/(4*pi*lavg) * Iz_full;
Dz = Dz0;

LmaxUsedDz = 0;
LmaxUsedY  = 0;
convergedDz = true;
convergedY  = true;

if g ~= 0
    convergedDz = false;
    convergedY  = false;
    corrZ = 0;
    corrY = 0;
    DzPrev = NaN;
    YPrev  = NaN;
    normfac = 1/sqrt(2*pi);

    % Preweighted arrays:
    % bulk (full-sphere) integrand for H: (s_z / mu) * dΩ on full grid
    Bz_f   = (CHIf .* invmu_f) .* W_f;          % Nchi x Nphi
    % hemi integrand for Htilde: (s_z / mu) * R(chi) * dΩ on hemi grid
    Bz_hR  = (CHIh .* invmu_h .* R_h) .* W_h;   % Nchi x Nphi

    for l = 1:2:LmaxCap
        denom = 1 - g^l;
        coeff = (g^l) / denom;

        mpos = 0:l;

        % Fourier modes in phi
        Eplus  = exp( 1i*(mpos(:)) * phi);   % (l+1) x Nphi
        Eminus = exp(-1i*(mpos(:)) * phi);   % (l+1) x Nphi (for Y* in Htilde)

        % phi-integrations
        Sz_f = Bz_f   * (Eplus.');    % Nchi x (l+1)  -- full sphere bulk projection
        Sz_h = Bz_hR  * (Eminus.');   % Nchi x (l+1)  -- hemisphere Htilde projection (Y^*)

        % Legendre polynomials:
        P_f = legendre(l, chi_full, 'norm'); % (l+1) x Nchi  (full-sphere chi)
        P_h = legendre(l, chi_hemi, 'norm'); % (l+1) x Nchi  (hemisphere chi)

        Hz = zeros(1, l+1);
        Ht = zeros(1, l+1);

        for k = 1:(l+1)
            m = mpos(k);

            % note: P_f and P_h return rows for m = 0..l
            af_full = P_f(m+1,:).';    % Nchi x 1 (full chi grid)
            af_hemi = P_h(m+1,:).';    % Nchi x 1 (hemi chi grid)

            % Condon–Shortley phase for consistency with spherical harmonics
            af_full = ((-1)^m) * af_full;
            af_hemi = ((-1)^m) * af_hemi;

            % H^z_{l,m} : integral over full 4pi of (s_z/mu) * Y_lm
            % Sz_f(:,k) contains the phi-integrated (s_z/mu)*e^{+im phi} part
            Hz(k) = normfac * (af_full.' * Sz_f(:,k));

            % Htilde^z_{l,m} : integral over upper hemisphere of (s_z/mu) * Y_lm^* * R
            % Sz_h(:,k) contains the phi-integrated (s_z/mu)*R * e^{-im phi}
            Ht(k) = normfac * (af_hemi.' * Sz_h(:,k));
        end

        % --- apply selection rules for z: only even m contribute ---
        mEven = 0:2:l;
        idxEven = mEven + 1;
        Hz_even = Hz(idxEven);
        Ht_even = Ht(idxEven);

        % Dz correction uses |H|^2 sum over m
        sumZ = abs(Hz_even(1))^2;
        if numel(Hz_even) > 1
            sumZ = sumZ + 2*sum(abs(Hz_even(2:end)).^2);
        end
        corrZ = corrZ + coeff * sumZ;
        DzNew = Dz0 + v/(4*pi*lavg) * corrZ;

        % Y correction uses H_lm * Htilde_lm summed over m (real part)
        sumY = real(Hz_even(1) * Ht_even(1));
        if numel(Hz_even) > 1
            sumY = sumY + 2*real(sum(Hz_even(2:end) .* Ht_even(2:end)));
        end
        corrY = corrY + coeff * sumY;
        YNew = Y0 + (1/lavg) * corrY;

        % convergence checks
        if l >= LmaxStart
            if ~isnan(DzPrev) && ~convergedDz
                if abs(DzNew - DzPrev) <= AbsTol + RelTol*abs(DzNew)
                    convergedDz = true;
                    LmaxUsedDz = l;
                end
            end
            if ~isnan(YPrev) && ~convergedY
                if abs(YNew - YPrev) <= AbsTol + RelTol*abs(YNew)
                    convergedY = true;
                    LmaxUsedY = l;
                end
            end
            if convergedDz && convergedY
                Dz = DzNew;
                Y  = YNew;
                break
            end
        end

        DzPrev = DzNew;
        YPrev  = YNew;
        Dz = DzNew;
        Y  = YNew;
        LmaxUsedDz = l;
        LmaxUsedY  = l;
    end

    if ~convergedDz
        warning('BC_ADE:NoConvergeDz', ...
            'Dz(g) did not meet tolerance by l=%d. Consider increasing LmaxCap (or Nchi/Nphi).', LmaxUsedDz);
    end
    if ~convergedY
        warning('BC_ADE:NoConvergeY', ...
            'Y(g) did not meet tolerance by l=%d. Consider increasing LmaxCap (or Nchi/Nphi).', LmaxUsedY);
    end
end

% C internal normalization (consistent with B,X,Y):
% In the paper C = D_z / v when Y is defined with 1/(2pi<ell>) prefactor;
% here B,X,Y are computed with the internal normalization (1/lavg) so we
% keep the C scaling consistent with the rest of the code:
C = (2*pi/v) * Dz;

% extrapolated length
ze = (C + Y) / denBC;

% ============================================================
% z0 = lambda(z-hat) : odd-l Legendre-kernel series (unchanged)
% ============================================================
if g == 0
    z0 = lz;                               % exact
    LmaxUsedZ0 = 0;
    convergedZ0 = true;
else
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
        'I2',I2,'I2R',I2R, ...
        'Y0',Y0,'Y',Y, ...
        'Dz',Dz,'Dz0',Dz0,'C',C, ...
        'ze',ze,'z0',z0, ...
        'LmaxStart',LmaxStart,'LmaxCap',LmaxCap, ...
        'LmaxUsedDz',LmaxUsedDz,'convergedDz',convergedDz, ...
        'LmaxUsedY',LmaxUsedY,'convergedY',convergedY, ...
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
