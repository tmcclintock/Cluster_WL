"""Correlation functions for matter and halos.

"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper, _handle_gsl_error
import numpy as np

def xi_nfw_at_r(r, M, c, Omega_m, delta=200):
    """NFW halo profile correlation function.

    Args:
        r (float or array like): 3d distances from halo center in Mpc/h comoving
        M (float): Mass in Msun/h
        c (float): Concentration
        Omega_m (float): Omega_matter, matter fraction of the density
        delta (int; optional): Overdensity, default is 200

    Returns:
        float or array like: NFW halo profile.

    """
    r = _ArrayWrapper(r, 'r')

    xi = _ArrayWrapper.zeros_like(r)
    cluster_toolkit._lib.calc_xi_nfw(r.cast(), len(r), M, c, delta,
                                     Omega_m, xi.cast())
    return xi.finish()

def xi_einasto_at_r(r, M, conc, alpha, om, delta=200, rhos=-1.):
    """Einasto halo profile.

    Args:
        r (float or array like): 3d distances from halo center in Mpc/h comoving
        M (float): Mass in Msun/h; not used if rhos is specified
        conc (float): Concentration
        alpha (float): Profile exponent
        om (float): Omega_matter, matter fraction of the density
        delta (int): Overdensity, default is 200
        rhos (float): Scale density in Msun h^2/Mpc^3 comoving; optional

    Returns:
        float or array like: Einasto halo profile.

    """
    r = _ArrayWrapper(r, 'r')

    xi = _ArrayWrapper.zeros_like(r)
    cluster_toolkit._lib.calc_xi_einasto(r.cast(), len(r), M, rhos,
                                         conc, alpha, delta, om, xi.cast())
    return xi.finish()

def xi_mm_at_r(r, k, P, N=500, step=0.005, exact=False):
    """Matter-matter correlation function.

    Args:
        r (float or array like): 3d distances from halo center in Mpc/h comoving
        k (array like): Wavenumbers of power spectrum in h/Mpc comoving
        P (array like): Matter power spectrum in (Mpc/h)^3 comoving
        N (int; optional): Quadrature step count, default is 500
        step (float; optional): Quadrature step size, default is 5e-3
        exact (boolean): Use the slow, exact calculation; default is False

    Returns:
        float or array like: Matter-matter correlation function

    """
    r = _ArrayWrapper(r, 'r')
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)

    xi = _ArrayWrapper.zeros_like(r)
    if not exact:
        rc = cluster_toolkit._lib.calc_xi_mm(r.cast(), len(r), k.cast(),
                                             P.cast(), len(k), xi.cast(),
                                             N, step)
        _handle_gsl_error(rc, xi_mm_at_r)
    else:
        if r.arr.max() > 1e3:
            raise Exception("max(r) cannot be >1e3 for numerical stability.")
        rc = cluster_toolkit._lib.calc_xi_mm_exact(r.cast(), len(r),
                                                   k.cast(), P.cast(),
                                                   len(k), xi.cast())
        _handle_gsl_error(rc, xi_mm_at_r)

    return xi.finish()

def xi_2halo(bias, xi_mm):
    """2-halo term in halo-matter correlation function

    Args:
        bias (float): Halo bias
        xi_mm (float or array like): Matter-matter correlation function

    Returns:
        float or array like: 2-halo term in halo-matter correlation function

    """
    xi_mm = _ArrayWrapper(xi_mm, allow_multidim=True)
    xi = _ArrayWrapper.zeros_like(xi_mm)
    cluster_toolkit._lib.calc_xi_2halo(len(xi_mm), bias, xi_mm.cast(),
                                       xi.cast())
    return xi.finish()

def xi_hm(xi_1halo, xi_2halo, combination="max"):
    """Halo-matter correlation function

    Note: at the moment you can combine the 1-halo and 2-halo terms by either taking the max of the two or the sum of the two. The 'combination' field must be set to either 'max' (default) or 'sum'.

    Args:
        xi_1halo (float or array like): 1-halo term
        xi_2halo (float or array like, same size as xi_1halo): 2-halo term
        combination (string; optional): specifies how the 1-halo and 2-halo terms are combined, default is 'max' which takes the max of the two

    Returns:
        float or array like: Halo-matter correlation function

    """
    if combination == "max":
        switch = 0
    elif combination == 'sum':
        switch = 1
    else:
        raise Exception("Combinations other than maximum not implemented yet")

    xi_1halo = _ArrayWrapper(xi_1halo, allow_multidim=True)
    xi_2halo = _ArrayWrapper(xi_2halo, allow_multidim=True)
    xi = _ArrayWrapper.zeros_like(xi_1halo)
    cluster_toolkit._lib.calc_xi_hm(len(xi_1halo), xi_1halo.cast(),
                                    xi_2halo.cast(), xi.cast(), switch)
    return xi.finish()

def xi_DK(r, M, conc, be, se, k, P, om, delta=200, rhos=-1., alpha=-1., beta=-1., gamma=-1.):
    """Diemer-Kravtsov 2014 profile.

    Args:
        r (float or array like): radii in Mpc/h comoving
        M (float): mass in Msun/h
        conc (float): Einasto concentration
        be (float): DK transition parameter
        se (float): DK transition parameter
        k (array like): wavenumbers in h/Mpc
        P (array like): matter power spectrum in [Mpc/h]^3
        Omega_m (float): matter density fraction
        delta (float): overdensity of matter. Optional, default is 200
        rhos (float): Einasto density. Optional, default is compute from the mass
        alpha (float): Einasto parameter. Optional, default is computed from peak height
        beta (float): DK 2-halo parameter. Optional, default is 4
        gamma (float): DK 2-halo parameter. Optional, default is 8

    Returns:
        float or array like: DK profile evaluated at the input radii

    """
    r = _ArrayWrapper(r, 'r')
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)

    xi = _ArrayWrapper.zeros_like(r)
    cluster_toolkit._lib.calc_xi_DK(r.cast(), len(r), M, rhos, conc, be, se, alpha, beta, gamma, delta, k.cast(), P.cast(), len(k), om, xi.cast())

    return xi.finish()

def xi_DK_appendix1(r, M, conc, be, se, k, P, om, bias, xi_mm, delta=200, rhos=-1., alpha=-1., beta=-1., gamma=-1.):
    """Diemer-Kravtsov 2014 profile, first form from the appendix, eq. A3.

    Args:
        r (float or array like): radii in Mpc/h comoving
        M (float): mass in Msun/h
        conc (float): Einasto concentration
        be (float): DK transition parameter
        se (float): DK transition parameter
        k (array like): wavenumbers in h/Mpc
        P (array like): matter power spectrum in [Mpc/h]^3
        Omega_m (float): matter density fraction
        bias (float): halo bias
        xi_mm (float or array like): matter correlation function at r
        delta (float): overdensity of matter. Optional, default is 200
        rhos (float): Einasto density. Optional, default is compute from the mass
        alpha (float): Einasto parameter. Optional, default is computed from peak height
        beta (float): DK 2-halo parameter. Optional, default is 4
        gamma (float): DK 2-halo parameter. Optional, default is 8

    Returns:
        float or array like: DK profile evaluated at the input radii

    """
    r = _ArrayWrapper(r, 'r')
    k = _ArrayWrapper(k, allow_multidim=True)
    P = _ArrayWrapper(P, allow_multidim=True)
    xi_mm = _ArrayWrapper(xi_mm, allow_multidim=True)

    xi = np.zeros_like(r)
    cluster_toolkit._lib.calc_xi_DK_app1(r.cast(), len(r), M, rhos, conc, be, se, alpha, beta, gamma, delta, k.cast(), P.cast(), len(k), om, bias, xi_mm.cast(), xi.cast())

    return xi.finish()

def xi_DK_appendix2(r, M, conc, be, se, k, P, om, bias, xi_mm, delta=200, rhos=-1., alpha=-1., beta=-1., gamma=-1.):
    """Diemer-Kravtsov 2014 profile, second form from the appendix, eq. A4.

    Args:
        r (float or array like): radii in Mpc/h comoving
        M (float): mass in Msun/h
        conc (float): Einasto concentration
        be (float): DK transition parameter
        se (float): DK transition parameter
        k (array like): wavenumbers in h/Mpc
        P (array like): matter power spectrum in [Mpc/h]^3
        Omega_m (float): matter density fraction
        bias (float): halo bias
        xi_mm (float or array like): matter correlation function at r
        delta (float): overdensity of matter. Optional, default is 200
        rhos (float): Einasto density. Optional, default is compute from the mass
        alpha (float): Einasto parameter. Optional, default is computed from peak height
        beta (float): DK 2-halo parameter. Optional, default is 4
        gamma (float): DK 2-halo parameter. Optional, default is 8

    Returns:
        float or array like: DK profile evaluated at the input radii
    """
    r = _ArrayWrapper(r, 'r')
    k = _ArrayWrapper(k)
    P = _ArrayWrapper(P)
    xi_mm = _ArrayWrapper(xi_mm)

    xi = _ArrayWrapper.zeros_like(r)
    cluster_toolkit._lib.calc_xi_DK_app2(r.cast(), len(r), M, rhos, conc, be,
                                         se, alpha, beta, gamma, delta,
                                         k.cast(), P.cast(), len(k), om, bias,
                                         xi_mm.cast(), xi.cast())
    return xi.finish()
