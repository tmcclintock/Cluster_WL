"""Halo mass function.
"""

import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def dndM_at_M(M, k, P, Omega_m, d=1.97, e=1.0, f=0.51, g=1.228):
    """Tinker et al. 2008 appendix C mass function at a given mass.
    Default behavior is for :math:`M_{200m}` mass definition.

    NOTE: by default, this function is only valid at :math:`z=0`. For use
    at higher redshifts either recompute the parameters yourself, or
    wait for this behavior to be patched.

    Args:
        M (float or array like): Mass in Msun/h.
        k (array like): Wavenumbers of the matter power spectrum in h/Mpc comoving.
        P_lin (array like): Linear matter power spectrum in (Mpc/h)^3 comoving.
        Omega_m (float): Matter density fraction.
        d (float; optional): First Tinker parameter. Default is 1.97.
        e (float; optional): Second Tinker parameter. Default is 1.
        f (float; optional): Third Tinker parameter. Default is 0.51.
        g (float; optional): Fourth Tinker parameter. Default is 1.228.

    Returns:
        float or array like: Mass function :math:`dn/dM`.

    """
    M = _ArrayWrapper(M, 'M')
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)

    dndM = _ArrayWrapper.zeros_like(M)
    cluster_toolkit._lib.dndM_at_M_arr(M.cast(), len(M), k.cast(),
                                       P.cast(), len(k), Omega_m,
                                       d, e, f, g, dndM.cast())
    return dndM.finish()

def G_at_M(M, k, P, Omega_m, d=1.97, e=1.0, f=0.51, g=1.228):
    """Tinker et al. 2008 appendix C multiplicity funciton G(M) as
    a function of mass. Default behavior is for :math:`M_{200m}` mass
    definition.

    Args:
        M (float or array like): Mass in Msun/h.
        k (array like): Wavenumbers of the matter power spectrum in h/Mpc comoving.
        P_lin (array like): Linear matter power spectrum in (Mpc/h)^3 comoving.
        Omega_m (float): Matter density fraction.
        d (float; optional): First Tinker parameter. Default is 1.97.
        e (float; optional): Second Tinker parameter. Default is 1.
        f (float; optional): Third Tinker parameter. Default is 0.51.
        g (float; optional): Fourth Tinker parameter. Default is 1.228.

    Returns:
        float or array like: Halo multiplicity :math:`G(M)`.
    """
    M = _ArrayWrapper(M, 'M')
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)

    G = _ArrayWrapper.zeros_like(M)
    cluster_toolkit._lib.G_at_M_arr(M.cast(), len(M),
                                    k.cast(), P.cast(), len(k),
                                    Omega_m, d, e, f, g, G.cast())
    return G.finish()

def G_at_sigma(sigma, d=1.97, e=1.0, f=0.51, g=1.228):
    """Tinker et al. 2008 appendix C multiplicity funciton G(sigma) as
    a function of sigma.

    NOTE: by default, this function is only valid at :math:`z=0`. For use
    at higher redshifts either recompute the parameters yourself, or
    wait for this behavior to be patched.

    Args:
        sigma (float or array like): RMS variance of the matter density field.
        d (float; optional): First Tinker parameter. Default is 1.97.
        e (float; optional): Second Tinker parameter. Default is 1.
        f (float; optional): Third Tinker parameter. Default is 0.51.
        g (float; optional): Fourth Tinker parameter. Default is 1.228.

    Returns:
        float or array like: Halo multiplicity G(sigma).
    """
    sigma = _ArrayWrapper(sigma, 'sigma')

    G = _ArrayWrapper.zeros_like(sigma)
    cluster_toolkit._lib.G_at_sigma_arr(sigma.cast(), len(sigma),
                                        d, e, f, g, G.cast())
    return G.finish()

def n_in_bins(edges, Marr, dndM):
    """Tinker et al. 2008 appendix C binned mass function.

    Args:
        edges (array like): Edges of the mass bins.
        Marr (array like): Array of locations that dndM has been evaluated at.
        dndM (array like): Array of dndM.

    Returns:
       numpy.ndarray: number density of halos in the mass bins. Length is :code:`len(edges)-1`.

    """
    edges = _ArrayWrapper(edges, 'edges')

    n = _ArrayWrapper.zeros(len(edges)-1)
    Marr = _ArrayWrapper(Marr, 'Marr')
    dndM = _ArrayWrapper(dndM, 'dndM')
    cluster_toolkit._lib.n_in_bins(edges.cast(), len(edges),
                                   Marr.cast(), dndM.cast(), len(Marr),
                                   n.cast())
    return n.finish()

def n_in_bin(Mlo, Mhi, Marr, dndM):
    """Tinker et al. 2008 appendix C binned mass function.

    Args:
        Mlo (float): Lower mass edge.
        Mhi (float): Upper mass edge.
        Marr (array like): Array of locations that dndM has been evaluated at.
        dndM (array like): Array of dndM.

    Returns:
       float: number density of halos in the mass bin.

    """
    return np.squeeze(n_in_bins([Mlo, Mhi], Marr, dndM))

def _dndM_sigma2_precomputed(M, sigma2, dsigma2dM, Omega_m, d=1.97, e=1.0, f=0.51, g=1.228):
    M = _ArrayWrapper(M, allow_multidim=True)
    sigma2 = _ArrayWrapper(sigma2, allow_multidim=True)
    dsigma2dM = _ArrayWrapper(dsigma2dM, allow_multidim=True)
    dndM = _ArrayWrapper.zeros_like(M)
    cluster_toolkit._lib.dndM_sigma2_precomputed(M.cast(), sigma2.cast(),
                                                 dsigma2dM.cast(), len(M),
                                                 Omega_m, d, e, f, g,
                                                 dndM.cast())
    return dndM.finish()
