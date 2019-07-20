"""Halo bias.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np
# from .peak_height import *

def bias_at_M(M, k, P, Omega_m, delta=200):
    """Tinker et al. 2010 bais at mass M [Msun/h].

    Args:
        M (float or array like): Mass in Msun/h.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.
        Omega_m (float): Matter density fraction.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Halo bias.

    """
    M = _ArrayWrapper(M, 'M')
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    if k.shape != P.shape:
        raise ValueError('k and P must have the same shape')

    bias = _ArrayWrapper.zeros_like(M)
    cluster_toolkit._lib.bias_at_M_arr(M.cast(), len(M), delta,
                                       k.cast(), P.cast(), len(k),
                                       Omega_m, bias.cast())
    return bias.finish()

def bias_at_R(R, k, P, delta=200):
    """Tinker 2010 bais at mass M [Msun/h] corresponding to radius R [Mpc/h comoving].

    Args:
        R (float or array like): Lagrangian radius in Mpc/h comoving.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Halo bias.

    """
    R = _ArrayWrapper(R, 'R')
    k = _ArrayWrapper(k)
    P = _ArrayWrapper(P)

    bias = _ArrayWrapper.zeros_like(R)
    cluster_toolkit._lib.bias_at_R_arr(R.cast(), len(R), delta,
                                       k.cast(), P.cast(), len(k),
                                       bias.cast())
    return bias.finish()

def bias_at_nu(nu, delta=200):
    """Tinker 2010 bais at peak height nu.

    Args:
        nu (float or array like): Peak height.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Halo bias.

    """
    nu = _ArrayWrapper(nu, 'nu')

    bias = _ArrayWrapper.zeros_like(nu)
    cluster_toolkit._lib.bias_at_nu_arr(nu.cast(), len(nu), delta,
                                        bias.cast())
    return bias.finish()

def dbiasdM_at_M(M, k, P, Omega_m, delta=200):
    """d/dM of Tinker et al. 2010 bais at mass M [Msun/h].

    Args:
        M (float or array like): Mass in Msun/h.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.
        Omega_m (float): Matter density fraction.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Derivative of the halo bias.

    """
    M = _ArrayWrapper(M, 'M')
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)

    deriv = _ArrayWrapper.zeros_like(M)
    cluster_toolkit._lib.dbiasdM_at_M_arr(M.cast(), len(M), delta, k.cast(),
                                          P.cast(), len(k), Omega_m,
                                          deriv.cast())
    return deriv.finish()

def _bias_at_nu_FREEPARAMS(nu, A, a, B, b, C, c, delta=200):
    """A special function used only for quickly computing best fit parameters
    for the halo bias models.
    """
    nu = _ArrayWrapper(nu, allow_multidim=True)
    bias = _ArrayWrapper.zeros_like(nu)
    cluster_toolkit._lib.bias_at_nu_arr_FREEPARAMS(nu.cast(), len(nu), delta,
                                                   A, a, B, b, C, c,
                                                   bias.cast())
    return bias.finish()
