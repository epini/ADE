function [x, w] = gauss_legendre(n)
%GAUSS_LEGENDRE Gauss-Legendre quadrature nodes and weights on [-1, 1].
%
%   [x, w] = GAUSS_LEGENDRE(n)
%
%   Computes the n-point Gauss-Legendre quadrature rule on the interval
%   [-1, 1].
%
%   Input
%   -----
%   n   - Number of quadrature points [positive integer].
%
%   Output
%   ------
%   x   - Quadrature nodes on [-1, 1] [n x 1 double].
%   w   - Quadrature weights associated with x [n x 1 double].
%
%   Notes
%   -----
%   The quadrature satisfies
%
%       integral_{-1}^1 f(t) dt  ~=  sum_{k=1}^n w(k) * f(x(k)).
%
%   The nodes and weights are computed using the Golub-Welsch eigenvalue
%   method applied to the symmetric tridiagonal Jacobi matrix.
%
%   Author:       Ernesto Pini
%   Affiliation:  Istituto Nazionale di Ricerca Metrologica (INRiM)
%   Email:        pinie@lens.unifi.it

validateattributes(n, {'numeric'}, ...
    {'real','finite','scalar','integer','positive'});

i    = (1:n-1)';
beta = 0.5 ./ sqrt(1 - (2*i).^(-2));
T    = diag(beta,1) + diag(beta,-1);

[V, D] = eig(T);

x = diag(D);
[x, idx] = sort(x);
V = V(:, idx);

w = 2 * (V(1,:).^2).';
end