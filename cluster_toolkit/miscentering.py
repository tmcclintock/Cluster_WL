"""Miscentering effects for projected profiles.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def Sigma_mis_single_at_R(R, Rsigma, Sigma, M, conc, Omega_m, Rmis, delta=200):
    """Miscentered surface mass density [Msun h/pc^2 comoving] of a profile miscentered by an
    amount Rmis Mpc/h comoving. Units are Msun h/pc^2 comoving.

    Args:
        R (float or array like): Projected radii Mpc/h comoving.
        Rsigma (array like): Projected radii of the centered surface mass density profile.
        Sigma (float or array like): Surface mass density Msun h/pc^2 comoving.
        M (float): Halo mass Msun/h.
        conc (float): concentration.
        Omega_m (float): Matter density fraction.
        Rmis (float): Miscentered distance in Mpc/h comoving.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Miscentered projected surface mass density.

    """
    R = _ArrayWrapper(R, 'R')

    if np.min(R.arr) < np.min(Rsigma):
        raise Exception("Minimum R must be >= min(R_Sigma)")
    if np.max(R.arr) > np.max(Rsigma):
        raise Exception("Maximum R must be <= max(R_Sigma)")

    Rsigma = _ArrayWrapper(Rsigma, allow_multidim=True)
    Sigma = _ArrayWrapper(Sigma, allow_multidim=True)

    if Rsigma.shape != Sigma.shape:
        raise ValueError('Rsigma and Sigma must have the same shape')

    Sigma_mis = _ArrayWrapper.zeros_like(R)
    cluster_toolkit._lib.Sigma_mis_single_at_R_arr(R.cast(), len(R),
                                                   Rsigma.cast(), Sigma.cast(),
                                                   len(Rsigma), M, conc, delta,
                                                   Omega_m, Rmis,
                                                   Sigma_mis.cast())
    return Sigma_mis.finish()

def Sigma_mis_at_R(R, Rsigma, Sigma, M, conc, Omega_m, Rmis, delta=200, kernel="rayleigh"):
    """Miscentered surface mass density [Msun h/pc^2 comoving]
    convolved with a distribution for Rmis. Units are Msun h/pc^2 comoving.

    Args:
        R (float or array like): Projected radii Mpc/h comoving.
        Rsigma (array like): Projected radii of the centered surface mass density profile.
        Sigma (float or array like): Surface mass density Msun h/pc^2 comoving.
        M (float): Halo mass Msun/h.
        conc (float): concentration.
        Omega_m (float): Matter density fraction.
        Rmis (float): Miscentered distance in Mpc/h comoving.
        delta (int; optional): Overdensity, default is 200.
        kernel (string; optional): Kernal for convolution. Options: rayleigh or gamma.

    Returns:
        float or array like: Miscentered projected surface mass density.

    """
    R = _ArrayWrapper(R, 'R')

    # Exception checking
    if np.min(R.arr) < np.min(Rsigma):
        raise Exception("Minimum R must be >= min(R_Sigma)")
    if np.max(R.arr) > np.max(Rsigma):
        raise Exception("Maximum R must be <= max(R_Sigma)")
    if kernel == "rayleigh":
        integrand_switch = 0
    elif kernel == "gamma":
        integrand_switch = 1
    else:
        raise Exception("Miscentering kernel must be either "+
                        "'rayleigh' or 'gamma'")

    Rsigma = _ArrayWrapper(Rsigma, allow_multidim=True)
    Sigma = _ArrayWrapper(Sigma, allow_multidim=True)

    if Rsigma.shape != Sigma.shape:
        raise ValueError('Rsigma and Sigma must have the same shape')

    Sigma_mis = _ArrayWrapper.zeros_like(R)
    cluster_toolkit._lib.Sigma_mis_at_R_arr(R.cast(), len(R), Rsigma.cast(),
                                            Sigma.cast(), len(Rsigma),
                                            M, conc, delta, Omega_m, Rmis,
                                            integrand_switch, Sigma_mis.cast())
    return Sigma_mis.finish()

def DeltaSigma_mis_at_R(R, Rsigma, Sigma_mis):
    """Miscentered excess surface mass density profile at R. Units are Msun h/pc^2 comoving.

    Args:
        R (float or array like): Projected radii to evaluate profile.
        Rsigma (array like): Projected radii of miscentered Sigma profile.
        Sigma_mis (array like): Miscentered Sigma profile.

    Returns:
        float array like: Miscentered excess surface mass density profile.

    """
    R = _ArrayWrapper(R, 'R')
    if np.min(R.arr) < np.min(Rsigma):
        raise Exception("Minimum R must be >= min(R_Sigma)")
    if np.max(R.arr) > np.max(Rsigma):
        raise Exception("Maximum R must be <= max(R_Sigma)")

    Rsigma = _ArrayWrapper(Rsigma, allow_multidim=True)
    Sigma_mis = _ArrayWrapper(Sigma_mis, allow_multidim=True)

    if Rsigma.shape != Sigma_mis.shape:
        raise ValueError('Rsigma and Sigma must have the same shape')

    DeltaSigma_mis = _ArrayWrapper.zeros_like(R)
    cluster_toolkit._lib.DeltaSigma_mis_at_R_arr(R.cast(), len(R),
                                                 Rsigma.cast(),
                                                 Sigma_mis.cast(),
                                                 len(Rsigma),
                                                 DeltaSigma_mis.cast())
    return DeltaSigma_mis.finish()
