function [Dx, Dy, Dz, info] = D_Tensor_ADE(n_in, lx, ly, lz, g, varargin)
%D_TENSOR_ADE  Diffusion tensor (diagonal) from anisotropic mean free paths and scalar g.
%
% Computes Dx,Dy,Dz for a medium with scattering mean free paths lx,ly,lz
% (principal axes), refractive index n_in, and scalar HG asymmetry factor g.
%
% Numerical method:
%   - Sphere integrals via Gauss-Legendre quadrature in chi=cos(theta)
%     and trapezoidal quadrature in phi.
%   - g≠0 correction uses the odd-l spherical-harmonic series with
%     H_lm^i = ∫ (s_i Y_l^m)/mu_s dΩ, evaluated efficiently on the grid.
%
% Inputs (required):
%   n_in  : refractive index of the diffusive medium (positive scalar)
%   lx,ly,lz : scattering mean free paths (positive scalars)
%   g     : HG asymmetry factor, scalar with -1<g<1
%
% Optional positional inputs (in this order):
%   LmaxStart : starting max odd l for the g-correction series (default 15)
%   Nchi      : Gauss-Legendre nodes in chi (default 200)
%   Nphi      : phi samples (default 512)
%   RelTol    : relative tol for series convergence (default 1e-6)
%   AbsTol    : absolute tol for series convergence (default 1e-12)
%
% Outputs:
%   Dx,Dy,Dz : diffusion coefficients along x,y,z
%   info (optional): struct with fields:
%       .v, .lavg, .D0 (g=0 part),
%       .LmaxUsed, .converged, .Nchi, .Nphi, .RelTol, .AbsTol
%
% Usage:
%   [Dx,Dy,Dz] = D_Tensor_ADE(n_in,lx,ly,lz,g)
%   [Dx,Dy,Dz] = D_Tensor_ADE(n_in,lx,ly,lz,g,LmaxStart)
%   [Dx,Dy,Dz] = D_Tensor_ADE(n_in,lx,ly,lz,g,LmaxStart,Nchi,Nphi)
%   [Dx,Dy,Dz] = D_Tensor_ADE(n_in,lx,ly,lz,g,LmaxStart,Nchi,Nphi,RelTol,AbsTol)
%   [Dx,Dy,Dz,info] = D_Tensor_ADE(...)

% ---- parse/validate inputs (positional optionals) ----
p = inputParser();
p.FunctionName = mfilename();

p.addRequired('n_in', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lx',   @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('ly',   @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lz',   @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('g',    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','>',-1,'<',1}));

p.addOptional('LmaxStart', 15,  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive','odd'}));
p.addOptional('Nchi',      200, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addOptional('Nphi',      512, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addOptional('RelTol',    1e-5,@(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addOptional('AbsTol',    1e-10,@(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','nonnegative'}));

p.parse(n_in, lx, ly, lz, g, varargin{:});

n_in      = p.Results.n_in;
lx        = p.Results.lx;  ly = p.Results.ly;  lz = p.Results.lz;
g         = p.Results.g;
LmaxStart = p.Results.LmaxStart;
Nchi      = p.Results.Nchi;
Nphi      = p.Results.Nphi;
RelTol    = p.Results.RelTol;
AbsTol    = p.Results.AbsTol;

% speed of light in medium (same constant you used)
v = 299.7924589 / n_in;

% isotropic shortcut (exact similarity)
if lx == ly && lx == lz
    D = (v * lx) / (3*(1-g));
    Dx = D; Dy = D; Dz = D;
    if nargout > 3
        info = struct('v',v,'lavg',lx,'D0',[D D D], ...
                      'LmaxUsed',0,'converged',true, ...
                      'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol, ...
                      'note','Isotropic shortcut: D = v*lx/(3*(1-g)).');
    end
    return
end

mux = 1/lx; muy = 1/ly; muz = 1/lz;

% ---- quadrature grid ----
[chi, wchi] = gauss_legendre(Nchi);              % column
phi  = (0:Nphi-1) * (2*pi/Nphi);                 % row
wphi = 2*pi/Nphi;

[PHI, CHI] = meshgrid(phi, chi);
S = sqrt(max(0, 1 - CHI.^2));

cosP = cos(PHI);
sinP = sin(PHI);

% dΩ weights (Gauss–Legendre in chi, periodic trapezoid in phi)
W = (wchi(:) * ones(1,Nphi)) * wphi;

sx = S .* cosP;
sy = S .* sinP;
sz = CHI;

mu = mux.*(1-CHI.^2).*cosP.^2 + ...
     muy.*(1-CHI.^2).*sinP.^2 + ...
     muz.*(CHI.^2);

invmu = 1 ./ mu;

% ---- g=0 part ----
lavg = (1/(4*pi)) * sum(sum(invmu .* W));

Ix = sum(sum((sx.^2) .* (invmu.^2) .* W));
Iy = sum(sum((sy.^2) .* (invmu.^2) .* W));
Iz = sum(sum((sz.^2) .* (invmu.^2) .* W));

Dx0 = v/(4*pi*lavg) * Ix;
Dy0 = v/(4*pi*lavg) * Iy;
Dz0 = v/(4*pi*lavg) * Iz;

Dx = Dx0; Dy = Dy0; Dz = Dz0;

% ---- g≠0 correction via odd-l series ----
converged = true;
LmaxUsed = 0;

if g ~= 0
    % choose a reasonable cap; can be increased by user via LmaxStart
    LmaxCap = max(LmaxStart, 101);

    % precompute the B_i matrices used in H integrals:
    % H_lm^i ≈ sum( (s_i/mu) * Y_l^m * dΩ ) = sum( B_i .* Y )
    Bx = (sx .* invmu) .* W;
    By = (sy .* invmu) .* W;
    Bz = (sz .* invmu) .* W;

    corrX = 0; corrY = 0; corrZ = 0;

    DxPrev = NaN; DyPrev = NaN; DzPrev = NaN;
    converged = false;

    for l = 1:2:LmaxCap
        denom = 1 - g^l;
        coeff = (g^l) / denom;   % can be negative for g<0 (odd l), OK

        % The paper sum uses m = 2m' (even m) with m'=-n..n, l=2n+1.
        % That corresponds to m even in [-l+1, l-1]. Use nonnegative m and symmetry:
        % |H_{l,-m}|^2 = |H_{l,m}|^2 for real mu_s, so sum over ±m is 2*sum_{m>0}.
        mpos = 0:2:(l-1);                 % nonnegative even m
        M = numel(mpos);

        % exp(i m phi) for all needed m at once
        E = exp(1i*(mpos(:)) * phi);      % M x Nphi
        Et = E.';                         % Nphi x M

        % for each i: S_i = B_i * exp(i m phi)^T  -> Nchi x M
        Sx = Bx * Et;
        Sy = By * Et;
        Sz = Bz * Et;

        % legendre(l,chi,'norm') gives P_l^m(chi) for m=0..l, size (l+1) x Nchi
        P = legendre(l, chi, 'norm');

        Hx = zeros(1,M); Hy = zeros(1,M); Hz = zeros(1,M);
        normfac = 1/sqrt(2*pi);

        % build H for each m using your Ylm normalization convention:
        % Y_l^m(theta,phi) = (1/sqrt(2π)) * [(-1)^m P_l^m(cosθ)] * exp(i m phi), m>=0
        for k = 1:M
            m = mpos(k);
            a = P(m+1,:).';         % Nchi x 1
            a = ((-1)^m) * a;       % Condon-Shortley factor

            Hx(k) = normfac * (a.' * Sx(:,k));
            Hy(k) = normfac * (a.' * Sy(:,k));
            Hz(k) = normfac * (a.' * Sz(:,k));
        end

        % sum_{m even} |H_{lm}^i|^2 over negative+positive m:
        % m=0 counted once, m>0 counted twice
        sumX = abs(Hx(1))^2 + 2*sum(abs(Hx(2:end)).^2);
        sumY = abs(Hy(1))^2 + 2*sum(abs(Hy(2:end)).^2);
        sumZ = abs(Hz(1))^2 + 2*sum(abs(Hz(2:end)).^2);

        corrX = corrX + coeff * sumX;
        corrY = corrY + coeff * sumY;
        corrZ = corrZ + coeff * sumZ;

        DxNew = Dx0 + v/(4*pi*lavg) * corrX;
        DyNew = Dy0 + v/(4*pi*lavg) * corrY;
        DzNew = Dz0 + v/(4*pi*lavg) * corrZ;

        if l >= LmaxStart && ~isnan(DxPrev)
            okx = abs(DxNew - DxPrev) <= AbsTol + RelTol*abs(DxNew);
            oky = abs(DyNew - DyPrev) <= AbsTol + RelTol*abs(DyNew);
            okz = abs(DzNew - DzPrev) <= AbsTol + RelTol*abs(DzNew);
            if okx && oky && okz
                Dx = DxNew; Dy = DyNew; Dz = DzNew;
                converged = true;
                LmaxUsed = l;
                break
            end
        end

        DxPrev = DxNew; DyPrev = DyNew; DzPrev = DzNew;

        % keep updating outputs (best-so-far)
        Dx = DxNew; Dy = DyNew; Dz = DzNew;
        LmaxUsed = l;
    end

    if ~converged
        warning('D_Tensor_ADE:NoConverge', ...
            'g-correction did not meet tolerance by l=%d. Consider increasing LmaxStart (or, secondarily, Nchi/Nphi).', LmaxUsed);
    end
end

if nargout > 3
    info = struct('v',v,'lavg',lavg,'D0',[Dx0 Dy0 Dz0], ...
                  'LmaxUsed',LmaxUsed,'converged',converged, ...
                  'Nchi',Nchi,'Nphi',Nphi,'RelTol',RelTol,'AbsTol',AbsTol, ...
                  'LmaxStart',LmaxStart);
end

end


function [x, w] = gauss_legendre(n)
% n-point Gauss-Legendre nodes/weights on [-1,1] (Golub-Welsch)
% TODO: split into standalone file
i = (1:n-1)';
beta = 0.5 ./ sqrt(1 - (2*i).^(-2));
T = diag(beta,1) + diag(beta,-1);
[V,D] = eig(T);
x = diag(D);
[x, idx] = sort(x);
V = V(:, idx);
w = 2 * (V(1,:).^2).';
end

