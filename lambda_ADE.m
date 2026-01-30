function [Lxx, Lyy, Lzz, info] = lambda_ADE(lx, ly, lz, g, varargin)
%LAMBDA_ADE  Fast evaluation of Lxx,Lyy,Lzz via odd-l Legendre-kernel formula.
%
% Uses Gauss-Legendre quadrature in chi=cos(theta) and trapezoidal in phi.
%
% Inputs:
%   lx,ly,lz : microscopic mean free paths (positive scalars)
%   g        : scalar HG asymmetry factor (real scalar, -1<g<1)
%
% Optional positional inputs (in this order):
%   Lmax     : starting maximum l (odd, default 9)
%   Nchi     : number of Gauss-Legendre nodes in chi (default 200)
%   Nphi     : number of phi samples (default 512)
%   RelTol   : relative tolerance for auto-increasing l (default 1e-5)
%   AbsTol   : absolute tolerance for auto-increasing l (default 1e-10)
%
% Output:
%   Lxx = lambda(x-hat), Lyy = lambda(y-hat), Lzz = lambda(z-hat)
%   info (optional) contains convergence metadata:
%       info.LmaxUsed = [lUsedX lUsedY lUsedZ]
%       info.converged = [cX cY cZ]
%       info.RelTol, info.AbsTol, info.Nchi, info.Nphi, info.LmaxStart, info.LmaxCap
%
% Usage:
%   [Lxx,Lyy,Lzz] = lambda_ADE(lx,ly,lz,g)
%   [Lxx,Lyy,Lzz] = lambda_ADE(lx,ly,lz,g,Lmax)
%   [Lxx,Lyy,Lzz] = lambda_ADE(lx,ly,lz,g,Lmax,Nchi)
%   [Lxx,Lyy,Lzz] = lambda_ADE(lx,ly,lz,g,Lmax,Nchi,Nphi)
%   [Lxx,Lyy,Lzz] = lambda_ADE(lx,ly,lz,g,Lmax,Nchi,Nphi,RelTol,AbsTol)
%   [Lxx,Lyy,Lzz,info] = lambda_ADE(...)

% use inputParser to assign defaults for optional arguments
p = inputParser();
p.FunctionName = mfilename();

p.addRequired('lx', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('ly', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('lz', @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addRequired('g',  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','>',-1,'<',1}));

p.addOptional('Lmax',   9,    @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive','odd'}));
p.addOptional('Nchi',   200,  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addOptional('Nphi',   512,  @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','integer','positive'}));
p.addOptional('RelTol', 1e-5, @(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','positive'}));
p.addOptional('AbsTol', 1e-10,@(x) validateattributes(x, {'numeric'}, {'real','finite','scalar','nonnegative'}));

p.parse(lx, ly, lz, g, varargin{:});

lx = p.Results.lx; ly = p.Results.ly; lz = p.Results.lz; g = p.Results.g;
LmaxStart = p.Results.Lmax; Nchi = p.Results.Nchi; Nphi = p.Results.Nphi;
RelTol = p.Results.RelTol; AbsTol = p.Results.AbsTol;

% Choose an internal cap for l. (Odd; large enough for most cases but finite.)
LmaxCap = max(LmaxStart, 101);

% isotropic shortcut
if lx == ly && lx == lz
    lt  = lx/(1-g);
    Lxx = lt; Lyy = lt; Lzz = lt;

    if nargout > 3
        info = struct('LmaxUsed',[0 0 0], ...
                      'converged',[true true true], ...
                      'RelTol',RelTol, 'AbsTol',AbsTol, ...
                      'Nchi',Nchi, 'Nphi',Nphi, ...
                      'LmaxStart',LmaxStart, 'LmaxCap',LmaxCap, ...
                      'note','Isotropic shortcut: lambda = lx/(1-g).');
    end
    return
end

mux = 1/lx; muy = 1/ly; muz = 1/lz;

% Gauss-Legendre in chi, trapezoid in phi
[chi, wchi] = gauss_legendre(Nchi);      % column
phi  = (0:Nphi-1) * (2*pi/Nphi);         % row
wphi = 2*pi/Nphi;

[PHI, CHI] = meshgrid(phi, chi);

S  = sqrt(max(0, 1 - CHI.^2));
sx = S .* cos(PHI);
sy = S .* sin(PHI);
sz = CHI;

mu = mux.*(1-CHI.^2).*cos(PHI).^2 + ...
     muy.*(1-CHI.^2).*sin(PHI).^2 + ...
     muz.*(CHI.^2);

invmu = 1 ./ mu;
W = (wchi(:) * ones(1,Nphi)) * wphi;  % dOmega weights

    function [L, lUsed, converged] = lambda_axis(dotmat, RelTol_, AbsTol_, Lmax0_, LmaxCap_)
        t = dotmat(:).';
        L = 0;
        Lprev = NaN;
        lUsed = 1;
        converged = false;

        for l = 1:2:LmaxCap_
            Pl = legendre(l, t);
            Pl = reshape(Pl(1,:), size(dotmat));
            Il = sum(sum((dotmat .* Pl .* invmu) .* W));
            term = ((2*l+1)/(4*pi)) * (Il / (1 - g^l));
            L = real(L + term);
            lUsed = l;

            if l >= Lmax0_
                if ~isnan(Lprev)
                    d = abs(L - Lprev);
                    if d <= AbsTol_ + RelTol_*abs(L)
                        converged = true;
                        return
                    end
                end
                Lprev = L;
            end
        end
    end

[Lxx, lxxUsed, cx] = lambda_axis(sx, RelTol, AbsTol, LmaxStart, LmaxCap);
[Lyy, lyyUsed, cy] = lambda_axis(sy, RelTol, AbsTol, LmaxStart, LmaxCap);
[Lzz, lzzUsed, cz] = lambda_axis(sz, RelTol, AbsTol, LmaxStart, LmaxCap);

if nargout > 3
    info = struct('LmaxUsed',[lxxUsed, lyyUsed, lzzUsed], ...
                  'converged',[cx cy cz], ...
                  'RelTol',RelTol, 'AbsTol',AbsTol, ...
                  'Nchi',Nchi, 'Nphi',Nphi, ...
                  'LmaxStart',LmaxStart, 'LmaxCap',LmaxCap);
    if ~(cx && cy && cz)
        info.note = 'Did not reach tolerance for all components. Consider increasing LmaxStart or Nchi/Nphi.';
    end
end

end

