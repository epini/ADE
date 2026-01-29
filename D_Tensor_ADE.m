function [Dx, Dy, Dz] = D_Tensor_ADE(n_in, lx, ly, lz, g)
% D_TENSOR_ADE Diffusive rate tensor from scattering mean free paths
%
% Brief: This function returns the (diagonal) diffusive rate tensor
% elements Dx, Dy, Dz starting from the scattering mean free paths
% lx, ly, lz and a single asymmetry factor g.
% Quantities can be expressed either in μm and ps or in mm and ns.
%
% Inputs:
%    lx - scattering mean free path along x [μm] or [mm]
%    ly - scattering mean free path along y [μm] or [mm]
%    lz - scattering mean free path along z [μm] or [mm]
%    g - asymmetry factor (scalar), assuming Heyney-Greenstein phase-function
%    n_in - refractive index of the diffusive medium
%
% Outputs:
%    Dx - diffusive rate along x [μm^2/ps] or [mm^2/ns]
%    Dy - diffusive rate along y [μm^2/ps] or [mm^2/ns]
%    Dz - diffusive rate along z [μm^2/ps] or [mm^2/ns]
%
% See also: example_all_functions.m

% Author:       Ernesto Pini
% Affiliation:  Department of Physics and Astronomy, Università di Firenze
% Email:        pinie@lens.unifi.it

v = 299.7924589/n_in;

if lx == lz && lx == ly % the isotropic case is treated separately

    Dx = lx*v/3/(1-g); % use standard similarity relation
    Dy = Dx;
    Dz = Dx;

else % anisotropic case
    mux = 1/lx;
    muy = 1/ly;
    muz = 1/lz;

    lavgfun = @(chi,phi) 1./(mux.*(1 - chi.^2).*((cos(phi)).^2) + muy.*(1 - chi.^2).*((sin(phi)).^2) + muz.*(chi.^2));
    lavg = (1/4/pi)*integral2(lavgfun, -1, 1, 0, 2*pi);

    Dxfun = @(chi,phi) (1-chi.^2).*((cos(phi)).^2)./(mux.*(1 - chi.^2).*((cos(phi)).^2) + muy.*(1 - chi.^2).*((sin(phi)).^2) + muz.*(chi.^2)).^2;
    Dyfun = @(chi,phi) (1-chi.^2).*((sin(phi)).^2)./(mux.*(1 - chi.^2).*((cos(phi)).^2) + muy.*(1 - chi.^2).*((sin(phi)).^2) + muz.*(chi.^2)).^2;
    Dzfun = @(chi,phi) (chi.^2)./(mux.*(1 - chi.^2).*((cos(phi)).^2) + muy.*(1 - chi.^2).*((sin(phi)).^2) + muz.*(chi.^2)).^2;

    Dx = v/(4*pi*lavg)*integral2(Dxfun, -1, 1, 0, 2*pi);
    Dy = v/(4*pi*lavg)*integral2(Dyfun, -1, 1, 0, 2*pi);
    Dz = v/(4*pi*lavg)*integral2(Dzfun, -1, 1, 0, 2*pi);

    if g ~= 0 % add correction terms for anisotropic phase-function
        
        % cutoff for the infinite sum
        Nmax = 4;
        corrX = 0; corrY = 0; corrZ = 0;
        
        % loop only over odd terms (2n+1)
        for n = 0:Nmax
            k = 2*n + 1;  % odd index
            coeff = g^k / (1 - g^k);   % since Ps = 1
            
            % Sum over m = -n..n of (H_{k,2m}^z)^2
            for m = -n:n
                HfunX = @(chi,phi) arrayfun(@(c,p) ...
                    (c .* conj(Ylm(k,2*m,acos(c),p))) ./ ...
                    (muy.*(1 - c.^2).*(cos(p).^2) + ...
                     muz.*(1 - c.^2).*(sin(p).^2) + ...
                     mux.*(c.^2)), ...
                    chi, phi);
                 
                HvalX = integral2(HfunX,-1,1,0,2*pi,'RelTol',1e-3,'AbsTol',1e-6);
                corrX = corrX + coeff * abs(HvalX)^2;

                HfunY = @(chi,phi) arrayfun(@(c,p) ...
                    (c .* conj(Ylm(k,2*m,acos(c),p))) ./ ...
                    (mux.*(1 - c.^2).*(cos(p).^2) + ...
                     muz.*(1 - c.^2).*(sin(p).^2) + ...
                     muy.*(c.^2)), ...
                    chi, phi);
                 
                HvalY = integral2(HfunY,-1,1,0,2*pi,'RelTol',1e-3,'AbsTol',1e-6);
                corrY = corrY + coeff * abs(HvalY)^2;

                HfunZ = @(chi,phi) arrayfun(@(c,p) ...
                    (c .* conj(Ylm(k,2*m,acos(c),p))) ./ ...
                    (mux.*(1 - c.^2).*(cos(p).^2) + ...
                     muy.*(1 - c.^2).*(sin(p).^2) + ...
                     muz.*(c.^2)), ...
                    chi, phi);
                 
                HvalZ = integral2(HfunZ,-1,1,0,2*pi,'RelTol',1e-3,'AbsTol',1e-6);
                corrZ = corrZ + coeff * abs(HvalZ)^2;
            end
        end
        Dx = Dx + v/(4*pi*lavg) * corrX;
        Dy = Dy + v/(4*pi*lavg) * corrY;
        Dz = Dz + v/(4*pi*lavg) * corrZ;
    end

end