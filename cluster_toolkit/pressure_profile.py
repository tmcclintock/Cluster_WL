#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com
'''
Galaxy cluster pressure profiles.

This module implements pressure profiles presented by Battaglia et al. 2012
(https://ui.adsabs.harvard.edu/abs/2012ApJ...758...75B/abstract), referred to
as BBPS.

Their best-fit pressure profile is implemented in the function
`P_BBPS`, and projected profiles are implemented in `projected_P_BBPS` and
`projected_P_BBPS_real`. The difference in the latter is that `projected_P_BBPS`
makes an approximation that reduces cosmology dependence, and
`projected_P_BBPS_real` interpolates over a table of comoving distances to
obtain a more precise answer.
'''
import numpy as np
from scipy.integrate import quad


def _rho_crit(z, omega_m):
    '''
    The critical density of the universe :math:`\\rho_{crit}`, in units of
    :math:`Msun*Mpc^{-3}*h^2`.
    '''
    # The below formula assumes a flat univers, i.e. omega_m + omega_lambda = 1
    omega_lambda = 1 - omega_m
    # The constant is 3 * (100 km / s / Mpc)**2 / (8 * pi * G)
    # in units of Msun h^2 Mpc^{-3}
    # (source: astropy's constants module and unit conversions)
    return 2.77536627e+11 * (omega_m * (1 + z)**3 + omega_lambda)


def P_delta(M, z, omega_b, omega_m, delta=200):
    '''
    The pressure amplitude of a halo:

    :math:`P_{\\Delta} = G * M_{\\Delta} * \\Delta * \\rho_{crit}(z) \
                * \\Omega_b / \\Omega_m / (2R_{\\Delta})`

    See BBPS, section 4.1 for details.

    Args:
        M (float): Halo mass :math:`M_{\\Delta}`, in units of Msun.
        omega_b (float): The baryon fraction :math:`\\Omega_b`.
        omega_m (float): The matter fraction :math:`\\Omega_m`.
        delta (float): The halo overdensity :math:`\\Delta`.

    Returns:
        float: Pressure amplitude, in units of Msun h^{8/3} s^{-2} Mpc^{-1}.
    '''
    # G = 4.51710305e-48 Mpc^3 Msun^{-1} s^{-2}
    # (source: astropy's constants module and unit conversions)
    return 4.51710305e-48 * M * delta * _rho_crit(z, omega_m) * \
        omega_b / omega_m / 2 / R_delta(M, omega_m, z, delta)


def R_delta(M, z, omega_m, delta=200):
    '''
    The radius of a sphere of mass M (in Msun), which has a density `delta`
    times the critical density of the universe.

    :math:`R_{\\Delta} = \\Big(\\frac{3 M_{\\Delta}} \
                                     {8 \\pi \\Delta \\rho_{crit}}\Big)^{1/3}`

    Args:
        M (float): Halo mass :math:`M_{\\Delta}`, in units of Msun.
        omega_b (float): The baryon fraction :math:`\\Omega_b`.
        omega_m (float): The matter fraction :math:`\\Omega_m`.
        delta (float): The halo overdensity :math:`\\Delta`.

    Returns:
        float: Radius, in :math:`\\text{Mpc} h^\\frac{-2}{3}`.
    '''
    volume = M / (delta * _rho_crit(z, omega_m))
    return (3 * volume / (4 * np.pi))**(1./3)


def P_simple_BBPS_generalized(x, M, z, P_0, x_c, beta,
                              alpha=1, gamma=-0.3, delta=200):
    '''
    The generalized dimensionless BBPS pressure profile. Input x should be
    :math:`r / R_{\\Delta}`.
    '''
    return P_0 * (x / x_c)**gamma * (1 + (x / x_c)**alpha)**(-beta)


def P_BBPS_generalized(r, M, z, omega_b, omega_m,
                       P_0, x_c, beta, alpha=1, gamma=-0.3, delta=200):
    '''
    The generalized NFW form of the Battaglia profile, presented in BBPS2
    equation 10 as:

    :math:`P = P_{\\Delta} P_0 (x / x_c)^\\gamma \
                [1 + (x / x_c)^\\alpha]^{-\\beta}`

    Where :math:`x = r/R_{\\Delta}`.

    Args:
        r (float): Radius from cluster center, in Mpc.
        M (float): Halo mass M_{200}, in Msun.
        z (float): Redshift to cluster.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        P_0 (float): Pressure scale parameter, see BBPS2 Eq. 10.
        x_c (float): Core scale parameter, see BBPS2 Eq. 10.
        beta (float): Asymptotic dropoff parameter, see BBPS2 Eq. 10.

    Returns:
        float: Pressure at a distance `r` from a cluster center, in units of \
               Msun s^{-2} Mpc^{-1}.
    '''
    x = r / R_delta(M, z, omega_m, delta=delta)
    Pd = P_delta(M, z, omega_b, omega_m, delta=delta)
    return Pd * P_simple_BBPS_generalized(x, M, z, P_0, x_c, beta,
                                          alpha=alpha, gamma=gamma, delta=delta)


def _A_BBPS(M, z, A_0, alpha_m, alpha_z):
    '''
    Mass-Redshift dependency model for the generalized BBPS profile parameters,
    fit to simulated halos in that data. The best-fit parameters are presented
    in Table 1. of BBPS2
    '''
    return A_0 * (M / 10**14)**alpha_m * (1 + z)**alpha_z


def P_simple_BBPS(x, M, z):
    '''
    The best-fit pressure profile presented in BBPS2.
    '''
    params_P = (18.1, 0.154, -0.758)
    params_x_c = (0.497, -0.00865, 0.731)
    params_beta = (4.35, 0.0393, 0.415)
    P_0 = _A_BBPS(M, z, *params_P)
    x_c = _A_BBPS(M, z, *params_x_c)
    beta = _A_BBPS(M, z, *params_beta)
    return P_simple_BBPS_generalized(x, M, z, P_0, x_c, beta)


def P_BBPS(r, M, z, omega_b, omega_m):
    '''
    The best-fit pressure profile presented in BBPS2.

    Args:
        r (float): Radius from the cluster center, in Mpc.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.

    Returns:
        float: Pressure at distance `r` from the cluster, in units of \
               Msun s^{-2} Mpc^{-1}.
    '''
    # These are the best-fit parameters from BBPS2 Table 1, under AGN Feedback
    # \Delta = 200
    params_P = (18.1, 0.154, -0.758)
    params_x_c = (0.497, -0.00865, 0.731)
    params_beta = (4.35, 0.0393, 0.415)

    P_0 = _A_BBPS(M, z, *params_P)
    x_c = _A_BBPS(M, z, *params_x_c)
    beta = _A_BBPS(M, z, *params_beta)
    return P_BBPS_generalized(r, M, z, omega_b, omega_m,
                              P_0, x_c, beta)


def projected_P_BBPS(r, M, z, omega_b, omega_m,
                     dist=8, epsrel=1e-3):
    '''
    Computes the projected line-of-sight density of a cluster at a radius r
    from the cluster center.

    Args:
        r (float): Radius from the cluster center, in Mpc.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.

    Returns:
        float: Integrated line-of-sight pressure at distance `r` from the \
               cluster, in units of Msun s^{-2}.
    '''
    R_del = R_delta(M, z, omega_m)
    return quad(lambda x: P_BBPS(np.sqrt(x*x + r*r), M, z,
                                 omega_b, omega_m),
                -dist * R_del, dist * R_del,
                epsrel=epsrel)[0] / (1 + z)


def projected_y_BBPS(r, M, z, omega_b, omega_m,
                     dist=8, epsrel=1e-3):
    '''
    Projected Compton-y parameter along the line of sight, at a perpendicular
    distance `r` from a cluster of mass `M` at redshift `z`. All arguments have
    the same meaning as `projected_P_BBPS`.

    Args:
        r (float): Radius from the cluster center, in Mpc.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.

    Returns:
        float: Compton y parameter. Unitless.
    '''
    # The constant is \sigma_T / (m_e * c^2), i.e. the Thompson cross-section
    # divided by the mass-energy of the electron, in units of s^2 Msun^{-1}.
    # Source: Astropy constants and unit conversions.
    cy = 1.61574202e+15
    return cy * projected_P_BBPS(r, M, z, omega_b, omega_m,
                                 dist=dist, epsrel=epsrel)


def projected_P_BBPS_real(r, M, z, omega_b, omega_m, chis, zs,
                          dist=8, epsrel=1e-3):
    '''
    Computes the projected line-of-sight density of a cluster at a radius r
    from the cluster center.

    Args:
        r (float): Radius from the cluster center, in Mpc.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        chis (1d array of floats): The comoving line-of-sight distance, in Mpc.
        zs (1d array of floats): The redshifts corresponding to `chis`.

    Returns:
        float: Integrated line-of-sight pressure at distance `r` from the \
               cluster, in units of Msun s^{-2}.
    '''
    R_del = R_delta(M, z, omega_m)
    chi_cluster = np.interp(z, zs, chis)
    return quad(lambda x: P_BBPS(np.sqrt((x - chi_cluster)**2 + r*r),
                                 M, z,
                                 omega_b, omega_m) / (1 + np.interp(x, chis, zs)),
                chi_cluster - dist * R_del,
                chi_cluster + dist * R_del,
                epsrel=epsrel)[0]
