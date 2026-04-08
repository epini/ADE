"""generalized_ade: generalized anisotropic diffusion equation tools."""

from .quadrature import gauss_legendre
from .diffusion import d_tensor_ade
from .boundary import bc_ade
from .reflectance import r_ade, rt_ade, rxy_ade, rxyt_ade
from .transmittance import t_ade, tt_ade, txy_ade, txyt_ade

__all__ = [
    "gauss_legendre",
    "d_tensor_ade",
    "bc_ade",
    "r_ade",
    "rt_ade",
    "rxy_ade",
    "rxyt_ade",
    "t_ade",
    "tt_ade",
    "txy_ade",
    "txyt_ade",
]

__version__ = "0.1.0"
