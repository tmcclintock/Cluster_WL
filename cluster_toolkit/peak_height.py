"""Integrals of the power spectrum. This includes RMS variance of the density field, sigma2, as well as peak neight, nu. These were previously implemented in the bias module, but have been migrated to here.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def sigma2_at_M(M, k, P, Omega_m):
    """RMS variance in top hat sphere of lagrangian radius R [Mpc/h comoving] corresponding to a mass M [Msun/h] of linear power spectrum.

    Args:
        M (float or array like): Mass in Msun/h.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.
        Omega_m (float): Omega_matter, matter density fraction.

    Returns:
        float or array like: RMS variance of top hat sphere.

    """
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    if isinstance(M, list) or isinstance(M, np.ndarray):
        M = _ArrayWrapper(M, allow_multidim=True)
        s2 = _ArrayWrapper.zeros_like(M)
        cluster_toolkit._lib.sigma2_at_M_arr(M.cast(), len(M), k.cast(), P.cast(), len(k), Omega_m, s2.cast())
        return s2.finish()
    else:
        return cluster_toolkit._lib.sigma2_at_M(M, k.cast(), P.cast(), len(k), Omega_m)

def sigma2_at_R(R, k, P):
    """RMS variance in top hat sphere of radius R [Mpc/h comoving] of linear power spectrum.

    Args:
        R (float or array like): Radius in Mpc/h comoving.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.

    Returns:
        float or array like: RMS variance of a top hat sphere.

    """
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    if isinstance(R, list) or isinstance(R, np.ndarray):
        R = _ArrayWrapper(R)
        s2 = _ArrayWrapper.zeros_like(R)
        cluster_toolkit._lib.sigma2_at_R_arr(R.cast(), len(R), k.cast(), P.cast(), len(k), s2.cast())
        return s2.finish()
    else:
        return cluster_toolkit._lib.sigma2_at_R(R, k.cast(), P.cast(), len(k))

def nu_at_M(M, k, P, Omega_m):
    """Peak height of top hat sphere of lagrangian radius R [Mpc/h comoving] corresponding to a mass M [Msun/h] of linear power spectrum.

    Args:
        M (float or array like): Mass in Msun/h.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.
        Omega_m (float): Omega_matter, matter density fraction.

    Returns:
        nu (float or array like): Peak height.

    """
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    if isinstance(M, list) or isinstance(M, np.ndarray):
        M = _ArrayWrapper(M)
        nu = _ArrayWrapper.zeros_like(M)
        cluster_toolkit._lib.nu_at_M_arr(M.cast(), len(M), k.cast(), P.cast(), len(k), Omega_m, nu.cast())
        return nu.finish()
    else:
        return cluster_toolkit._lib.nu_at_M(M, k.cast(), P.cast(), len(k), Omega_m)

def nu_at_R(R, k, P):
    """Peak height of top hat sphere of radius R [Mpc/h comoving] of linear power spectrum.

    Args:
        R (float or array like): Radius in Mpc/h comoving.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.

    Returns:
        float or array like: Peak height.

    """
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    if isinstance(R, list) or isinstance(R, np.ndarray):
        R = _ArrayWrapper(R)
        nu = _ArrayWrapper.zeros_like(R)
        cluster_toolkit._lib.nu_at_R_arr(R.cast(), len(R), k.cast(), P.cast(), len(k), nu.cast())
        return nu.finish()
    else:
        return cluster_toolkit._lib.nu_at_R(R, k.cast(), P.cast(), len(k))

def dsigma2dM_at_M(M, k, P, Omega_m):
    """Derivative w.r.t. mass of RMS variance in top hat sphere of
    lagrangian radius R [Mpc/h comoving] corresponding to a mass
    M [Msun/h] of linear power spectrum.

    Args:
        M (float or array like): Mass in Msun/h.
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving.
        P (array like): Power spectrum in (Mpc/h)^3 comoving.
        Omega_m (float): Omega_matter, matter density fraction.

    Returns:
        float or array like: d/dM of RMS variance of top hat sphere.

    """
    P = _ArrayWrapper(P, allow_multidim=True)
    k = _ArrayWrapper(k, allow_multidim=True)
    if isinstance(M, list) or isinstance(M, np.ndarray):
        M = _ArrayWrapper(M, allow_multidim=True)
        ds2dM = _ArrayWrapper.zeros_like(M)
        cluster_toolkit._lib.dsigma2dM_at_M_arr(M.cast(), len(M), k.cast(),
                                                P.cast(), len(k), Omega_m,
                                                ds2dM.cast())
        return ds2dM.finish()
    else:
        return cluster_toolkit._lib.dsigma2dM_at_M(M, k.cast(), P.cast(),
                                                   len(k), Omega_m)


def _calc_sigma2_at_R(R, k, P, s2):
    """Direct call to vectorized version of RMS variance in top hat
    sphere of radius R [Mpc/h comoving] of linear power spectrum.

    """
    R = _ArrayWrapper(R, allow_multidim=True)
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    s2 = _ArrayWrapper(s2, allow_multidim=True)
    cluster_toolkit._lib.sigma2_at_R_arr(R.cast(), len(R), k.cast(), P.cast(), len(k), s2.cast())

def _calc_sigma2_at_M(M, k, P, Omega_m, s2):
    """Direct call to vectorized version of RMS variance in top hat sphere of lagrangian radius R [Mpc/h comoving] corresponding to a mass M [Msun/h] of linear power spectrum.

    """
    M = _ArrayWrapper(M, allow_multidim=True)
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    s2 = _ArrayWrapper(s2, allow_multidim=True)
    cluster_toolkit._lib.sigma2_at_M_arr(M.cast(), len(M), k.cast(), P.cast(), len(k), Omega_m, s2.cast())

def _calc_nu_at_R(R, k, P, nu):
    """Direct call to vectorized version of peak height of R.

    """
    R = _ArrayWrapper(R, allow_multidim=True)
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    nu = _ArrayWrapper(nu, allow_multidim=True)
    cluster_toolkit._lib.nu_at_R_arr(R.cast(), len(R), k.cast(), P.cast(), len(k), nu.cast())

def _calc_nu_at_M(M, k, P, Omega_m, nu):
    """Direct call to vectorized version of peak height of M.

    """
    M = _ArrayWrapper(M, allow_multidim=True)
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    nu = _ArrayWrapper(nu, allow_multidim=True)
    cluster_toolkit._lib.nu_at_M_arr(M.cast(), len(M), k.cast(), P.cast(), len(k), Omega_m, nu.cast())
