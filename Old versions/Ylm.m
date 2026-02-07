function Y = Ylm(l, m, theta, phi)
% Ylm  Evaluate complex spherical harmonic Y_l^m on elementwise grids.
%   Y = Ylm(l,m,theta,phi)
%   - theta and phi must be same size (scalar, vector or matrix)
%   - returns complex array same size as theta/phi
%   - uses normalization N_{l,m} = sqrt((2l+1)/(4*pi) * (l-|m|)!/(l+|m|)! )
%
% Note: this implementation flattens the grid, calls legendre once, then
% reshapes back. No large matrix multiplications.

    if ~isequal(size(theta), size(phi))
        error('theta and phi must have the same size');
    end

    % flatten
    sz = size(theta);
    theta_flat = theta(:);
    phi_flat   = phi(:);
    Npts = numel(theta_flat);

    % compute associated Legendre P_l^{|m|}(cos(theta)) for all points
    cos_t = cos(theta_flat);                % column vector
    P_all = legendre(l, cos_t.');           % returns (l+1) x Npts  (note legendre expects row vector)
    mm = abs(m);
    P_lm_vec = squeeze(P_all(mm+1, :)).';   % Npts x 1  (P_l^{|m|}(cos(theta)) )

    % normalization factor (standard complex spherical harmonic)
    Nlm = sqrt( (2*l + 1) / (4*pi) * factorial(l - mm) / factorial(l + mm) );

    % form spherical harmonic elementwise. m may be negative; exp handles it.
    Y_flat = Nlm .* P_lm_vec .* exp(1i * m * phi_flat);

    % reshape back to original shape
    Y = reshape(Y_flat, sz);
end
