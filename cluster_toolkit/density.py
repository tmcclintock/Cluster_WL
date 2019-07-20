"""Galaxy cluster density profiles.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def rho_nfw_at_r(r, M, c, Omega_m, delta=200):
    """NFW halo density profile.

    Args:
        r (float or array like): 3d distances from halo center in Mpc/h comoving.
        M (float): Mass in Msun/h.
        c (float): Concentration.
        Omega_m (float): Omega_matter, matter fraction of the density.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: NFW halo density profile in Msun h^2/Mpc^3 comoving.

    """
    r = _ArrayWrapper(r, 'r')

    rho = _ArrayWrapper.zeros_like(r)
    cluster_toolkit._lib.calc_rho_nfw(r.cast(), len(r), M, c, delta,
                                      Omega_m, rho.cast())
    return rho.finish()


def rho_einasto_at_r(r, M, rs, alpha, Omega_m, delta=200, rhos=-1.):
    """Einasto halo density profile. Distances are Mpc/h comoving.

    Args:
        r (float or array like): 3d distances from halo center.
        M (float): Mass in Msun/h; not used if rhos is specified.
        rhos (float): Scale density in Msun h^2/Mpc^3 comoving; optional.
        rs (float): Scale radius.
        alpha (float): Profile exponent.
        Omega_m (float): Omega_matter, matter fraction of the density.
        delta (int): Overdensity, default is 200.

    Returns:
        float or array like: Einasto halo density profile in Msun h^2/Mpc^3 comoving.

    """
    r = _ArrayWrapper(r, 'r')

    rho = _ArrayWrapper.zeros_like(r)
    cluster_toolkit._lib.calc_rho_einasto(r.cast(), len(r), M, rhos, rs,
                                          alpha, delta, Omega_m, rho.cast())
    return rho.finish()

