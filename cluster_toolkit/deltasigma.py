"""Galaxy cluster shear and magnification profiles also known as DeltaSigma and Sigma, respectively.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def Sigma_nfw_at_R(R, mass, concentration, Omega_m, delta=200):
    """Surface mass density of an NFW profile [Msun h/pc^2 comoving].

    Args:
        R (float or array like): Projected radii Mpc/h comoving.
        mass (float): Halo mass Msun/h.
        concentration (float): concentration.
        Omega_m (float): Matter density fraction.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Surface mass density Msun h/pc^2 comoving.

    """
    R = _ArrayWrapper(R, 'R')

    Sigma = _ArrayWrapper.zeros_like(R)
    cluster_toolkit._lib.Sigma_nfw_at_R_arr(R.cast(), len(R), mass,
                                            concentration, delta,
                                            Omega_m, Sigma.cast())
    return Sigma.finish()

def Sigma_at_R(R, Rxi, xi, mass, concentration, Omega_m, delta=200):
    """Surface mass density given some 3d profile [Msun h/pc^2 comoving].

    Args:
        R (float or array like): Projected radii Mpc/h comoving.
        Rxi (array like): 3D radii of xi_hm Mpc/h comoving.
        xi_hm (array like): Halo matter correlation function.
        mass (float): Halo mass Msun/h.
        concentration (float): concentration.
        Omega_m (float): Matter density fraction.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Surface mass density Msun h/pc^2 comoving.

    """
    R = _ArrayWrapper(R, 'R')
    Rxi = _ArrayWrapper(Rxi, allow_multidim=True)
    xi = _ArrayWrapper(xi, allow_multidim=True)

    if np.min(R.arr) < np.min(Rxi.arr):
        raise Exception("Minimum R for Sigma(R) must be >= than min(r) of xi(r).")
    if np.max(R.arr) > np.max(Rxi.arr):
        raise Exception("Maximum R for Sigma(R) must be <= than max(r) of xi(r).")

    Sigma = _ArrayWrapper.zeros_like(R)
    cluster_toolkit._lib.Sigma_at_R_full_arr(R.cast(), len(R), Rxi.cast(),
                                             xi.cast(), len(Rxi), mass, concentration,
                                             delta, Omega_m, Sigma.cast())
    return Sigma.finish()

def DeltaSigma_at_R(R, Rs, Sigma, mass, concentration, Omega_m, delta=200):
    """Excess surface mass density given Sigma [Msun h/pc^2 comoving].

    Args:
        R (float or array like): Projected radii Mpc/h comoving.
        Rs (array like): Projected radii of Sigma, the surface mass density.
        Sigma (array like): Surface mass density.
        mass (float): Halo mass Msun/h.
        concentration (float): concentration.
        Omega_m (float): Matter density fraction.
        delta (int; optional): Overdensity, default is 200.

    Returns:
        float or array like: Excess surface mass density Msun h/pc^2 comoving.

    """
    R = _ArrayWrapper(R, 'R')
    Rs = _ArrayWrapper(Rs, allow_multidim=True)
    Sigma = _ArrayWrapper(Sigma, allow_multidim=True)

    if np.min(R.arr) < np.min(Rs.arr):
        raise Exception("Minimum R for DeltaSigma(R) must be "+
                        ">= than min(R) of Sigma(R).")
    if np.max(R.arr) > np.max(Rs.arr):
        raise Exception("Maximum R for DeltaSigma(R) must be "+
                        "<= than max(R) of Sigma(R).")

    DeltaSigma = _ArrayWrapper.zeros_like(R)
    cluster_toolkit._lib.DeltaSigma_at_R_arr(R.cast(), len(R), Rs.cast(),
                                             Sigma.cast(), len(Rs), mass,
                                             concentration, delta, Omega_m,
                                             DeltaSigma.cast())
    return DeltaSigma.finish()
