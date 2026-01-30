function [x, w] = gauss_legendre(n)
%GAUSS_LEGENDRE  n-point Gauss–Legendre nodes and weights on [-1, 1].
%
%   [x, w] = gauss_legendre(n)
%
% Returns column vectors x (nodes) and w (weights) such that
%   integral_{-1}^1 f(t) dt  ≈  sum_{k=1}^n w(k) * f(x(k)).
%
% Implementation: Golub–Welsch eigenvalue method (symmetric tridiagonal Jacobi matrix).
%
% Notes:
%   - O(n^2) time, O(n^2) memory because of eig().

validateattributes(n, {'numeric'}, {'real','finite','scalar','integer','positive'});

i    = (1:n-1)';
beta = 0.5 ./ sqrt(1 - (2*i).^(-2));
T    = diag(beta,1) + diag(beta,-1);

[V, D] = eig(T);

x = diag(D);
[x, idx] = sort(x);
V = V(:, idx);

w = 2 * (V(1,:).^2).';
end

