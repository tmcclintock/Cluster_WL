"""
Derivatives of halo profiles. Used to plot splashback results.
"""
import cluster_toolkit as ct
from cluster_toolkit import _ArrayWrapper
import numpy as np


def drho_nfw_dr_at_R(Radii, Mass, conc, Omega_m, delta=200):
    """Derivative of the NFW halo density profile.

    Args:
        Radii (float or array like): 3d distances from halo center in Mpc/h comoving
        Mass (float): Mass in Msun/h
        conc (float): Concentration
        Omega_m (float): Matter fraction of the density
        delta (int; optional): Overdensity, default is 200

    Returns:
        float or array like: derivative of the NFW profile.

    """
    Radii = _ArrayWrapper(Radii, allow_multidim=True)
    if isinstance(Radii, list) or isinstance(Radii, np.ndarray):
        drhodr = _ArrayWrapper.zeros_like(Radii)
        ct._lib.drho_nfw_dr_at_R_arr(Radii.cast(), len(Radii), Mass, conc,
                                     delta, Omega_m, drhodr.cast())
        return drhodr.finish()
    else:
        return ct._lib.drho_nfw_dr_at_R(Radii, Mass, conc, delta, Omega_m)
