"""Correlation functions with halo exclusion.
"""
import cluster_toolkit
from cluster_toolkit import _ArrayWrapper
import numpy as np

def xi_hm_exclusion_at_r(radii, Mass, conc, alpha,
                         rt, beta, r_eff, beta_eff,
                         r_A, r_B, beta_ex,
                         bias, xi_mm, Omega_m, delta=200):
    """Halo-matter correlation function with halo exclusion incorporated.

    Args:
        radii (float or array-like): Radii of the profile in Mpc/h
        Mass (float): Mass in Msun/h
        conc (float): concentration of the 1-halo profile
        alpha (float): Einasto parameter
        rt (float): truncation radius in Mpc/h
        beta (float): width of the truncation distribution (erfc) in Mpc/h
        r_eff (float): effective radius for 2-halo subtraction in Mpc/h
        beta_eff (float): width for effective radius truncation
        r_A (float): radius of first correction term in Mpc/h
        r_B (float): radius of second correction term in Mpc/h
        beta_ex (float): width parameter for exclusion terms
        bias (float): linear halo bias
        xi_mm (float or array-like): matter correlation function.
            same shape as radii
        Omega_m (float): matter density fraction
        delta (int): halo overdensity. Default is 200

    Returns:
        float or array-like: exclusion profile at each radii

    """
    radii = np.asarray(radii)
    xi_mm = np.asarray(xi_mm)
    scalar_input = False
    if radii.ndim == 0:
        radii = radii[None] #makes r 1D
        scalar_input = True
    if radii.ndim > 1:
        raise Exception("radii cannot be a >1D array.")

    if len(radii) != len(xi_mm):
        raise Exception("len(r) must equal len(xi_mm)")

    lib =  cluster_toolkit._lib
    xi_hm = np.zeros_like(radii)
    lib.xi_hm_exclusion_at_r_arr(dc(radii), len(radii),
                                 Mass, conc, alpha,
                                 rt, beta,
                                 r_eff, beta_eff,
                                 r_A, r_B, beta_ex,
                                 bias, dc(xi_mm), delta,
                                 Omega_m, dc(xi_hm))
    if scalar_input:
        return np.squeeze(xi_hm)
    return xi_hm

def xi_1h_exclusion_at_r(radii, Mass, conc, alpha,
                         rt, beta, Omega_m, delta=200):
    """Halo-matter correlation function with halo exclusion incorporated,
    but just the 1-halo term.

    Args:
        radii (float or array-like): Radii of the profile in Mpc/h
        Mass (float): Mass in Msun/h
        conc (float): concentration of the 1-halo profile
        alpha (float): Einasto parameter
        rt (float): truncation radius in Mpc/h
        beta (float): width of the truncation distribution (erfc) in Mpc/h
        Omega_m (float): Matter density fraction
        delta (float): halo overdensity; default is 200

    Returns:
        float or array-like: 1-halo of the exclusion profile at each radii

    """
    radii = np.asarray(radii)
    scalar_input = False
    if radii.ndim == 0:
        radii = radii[None] #makes r 1D
        scalar_input = True
    if radii.ndim > 1:
        raise Exception("radii cannot be a >1D array.")

    lib =  cluster_toolkit._lib
    xi_1h = np.zeros_like(radii)
    lib.xi_1h_at_r_arr(dc(radii), len(radii),
                       Mass, conc, alpha, rt, beta, delta,
                       Omega_m, dc(xi_1h))
    if scalar_input:
        return np.squeeze(xi_1h)
    return xi_1h

def xi_2h_exclusion_at_r(radii, r_eff, beta_eff, bias, xi_mm):
    """2-halo term in the halo-matter correlation function
    using halo exclusion theory.

    Args:
        radii (float or array-like): Radii of the profile in Mpc/h
        r_eff (float): effective radius for 2-halo subtraction in Mpc/h
        beta_eff (float): width for effective radius truncation
        bias (float): halo bias at large scales
        xi_mm (float or array-like): matter correlation function.
            Must have same shape as the radii.

    Returns:
        float or array-like: 2-halo of the exclusion profile at each radii

    """
    radii = np.asarray(radii)
    xi_mm = np.asarray(xi_mm)
    scalar_input = False
    if radii.ndim == 0:
        radii = radii[None] #makes r 1D
        scalar_input = True
    if radii.ndim > 1:
        raise Exception("radii cannot be a >1D array.")

    if len(radii) != len(xi_mm):
        raise Exception("len(r) must equal len(xi_mm)")

    lib =  cluster_toolkit._lib
    xi_2h = np.zeros_like(radii)
    lib.xi_2h_at_r_arr(dc(radii), len(radii), r_eff, beta_eff,
                       bias, dc(xi_mm), dc(xi_2h))
    if scalar_input:
        return np.squeeze(xi_2h)
    return xi_2h

def xi_C_at_r(radii, r_A, r_B, beta_ex, xi_2h):
    """Halo-matter correlation function with halo exclusion incorporated.

    Args:
        radii (float or array-like): Radii of the profile in Mpc/h
        r_A (float): radius of first correction term in Mpc/h
        r_B (float): radius of second correction term in Mpc/h
        beta_ex (float): width parameter for exclusion terms
        xi_2h (float or array-like): 2-halo term of the exclusion profile

    Returns:
        float or array-like: correction term for the exclusion profile

    """
    radii = np.asarray(radii)
    xi_2h = np.asarray(xi_2h)
    scalar_input = False
    if radii.ndim == 0:
        radii = radii[None] #makes r 1D
        scalar_input = True
    if radii.ndim > 1:
        raise Exception("radii cannot be a >1D array.")

    if len(radii) != len(xi_2h):
        raise Exception("len(r) must equal len(xi_2h)")

    lib =  cluster_toolkit._lib
    xi_C = np.zeros_like(radii)
    lib.xi_C_at_r_arr(dc(radii), len(radii), r_A, r_B, beta_ex,
                      dc(xi_2h), dc(xi_C))
    if scalar_input:
        return np.squeeze(xi_C)
    return xi_C

def theta_at_r(radii, rt, beta):
    """Truncation function.

    Args:
        radii (float or array-like): Radii of the profile in Mpc/h
        rt (float): truncation radius in Mpc/h
        beta (float): width of the truncation distribution (erfc) in Mpc/h

    Returns:
        float or array-like: Truncation function

    """
    radii = np.asarray(radii)
    scalar_input = False
    if radii.ndim == 0:
        radii = radii[None] #makes r 1D
        scalar_input = True
    if radii.ndim > 1:
        raise Exception("radii cannot be a >1D array.")

    theta = np.zeros_like(radii)
    cluster_toolkit._lib.theta_erfc_at_r_arr(dc(radii), len(radii),
                                             rt, beta, dc(theta))
    if scalar_input:
        return np.squeeze(theta)
    return theta
