"""Halo concentration.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def concentration_at_M(Mass, k, P, n_s, Omega_b, Omega_m, h, T_CMB=2.7255, delta=200, Mass_type="crit"):
    """Concentration of the NFW profile at mass M [Msun/h].
    Only implemented relation at the moment is Diemer & Kravtsov (2015).

    Note: only single concentrations at a time are allowed at the moment.

    Args:
        Mass (float): Mass in Msun/h.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Linear matter power spectrum in (Mpc/h)^3 comoving.
        n_s (float): Power spectrum tilt.
        Omega_b (float): Baryonic matter density fraction.
        Omega_m (float): Matter density fraction.
        h (float): Reduced Hubble constant.
        T_CMB (float): CMB temperature in Kelvin, default is 2.7.
        delta (int; optional): Overdensity, default is 200.
        Mass_type(string; optional); Defines either Mcrit or Mmean. Default is mean. Choose "crit" for Mcrit. Other values will raise an exception.

    Returns:
        float: NFW concentration.

    """
    if delta != 200:
        raise Exception("ConcentrationError: delta=%d. Currently only delta=200 supported"%delta)

    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)

    if Mass_type is "mean":
        return cluster_toolkit._lib.DK15_concentration_at_Mmean(Mass, k.cast(), P.cast(), len(k), delta, n_s, Omega_b, Omega_m, h, T_CMB)
    elif Mass_type is "crit":
        return cluster_toolkit._lib.DK15_concentration_at_Mcrit(Mass, k.cast(), P.cast(), len(k), delta, n_s, Omega_b, Omega_m, h, T_CMB)
    else:
        raise Exception("ConcentrationError: must choose either 'mean' or 'crit', %s is not supported"%Mass_type)
