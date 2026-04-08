from __future__ import annotations

import numpy as np


def gauss_legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre quadrature nodes and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of quadrature points.

    Returns
    -------
    x, w : ndarray
        Quadrature nodes and weights on [-1, 1].
    """
    if not isinstance(n, (int, np.integer)) or n <= 0:
        raise ValueError("n must be a positive integer.")

    x, w = np.polynomial.legendre.leggauss(int(n))
    return x.astype(float), w.astype(float)
