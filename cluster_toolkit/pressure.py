#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com
'''
Galaxy cluster pressure profiles. Useful for modeling Sunyaev-Zeldovich cluster
observables.

This module implements pressure profiles presented by Battaglia et al. 2012
(https://ui.adsabs.harvard.edu/abs/2012ApJ...758...75B/abstract), referred to
as BBPS.

Their best-fit 3D pressure profile is implemented in the function
:meth:`P_BBPS`, and projected profiles are implemented in
:meth:`projected_P_BBPS` and :meth:`projected_y_BBPS`.

The most important functions are:

* :meth:`P_BBPS` computes the 3D pressure profile.
* :meth:`projected_y_BBPS` computes the projected Compton-y parameter.
* :meth:`convolved_y_BBPS` computes the observed Compton-y parameter, i.e. the \
  projected Compton-y convolved with a Gaussian beam function.
'''

from astropy.convolution import Gaussian2DKernel, convolve_fft
from cluster_toolkit import _dcast, _lib
import numpy as np
from scipy.integrate import quad


__BBPS_params_P_0 = (18.1, 0.154, -0.758)
__BBPS_params_x_c = (0.497, -0.00865, 0.731)
__BBPS_params_beta = (4.35, 0.0393, 0.415)


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


def R_delta(M, z, omega_m, delta=200):
    '''
    The radius of a sphere of mass M (in Msun), which has a density `delta`
    times the critical density of the universe.

    :math:`R_{\\Delta} = \\Big(\\frac{3 M_{\\Delta}} \
                                     {8 \\pi \\Delta \\rho_{crit}}\Big)^{1/3}`

    Args:
        M (float or array): Halo mass :math:`M_{\\Delta}`, in units of Msun.
        z (float or array): Redshift to the cluster center.
        omega_m (float or array): The matter fraction :math:`\\Omega_m`.
        delta (float or array): The halo overdensity :math:`\\Delta`.

    Returns:
        float or array: Radius, in :math:`\\text{Mpc} h^\\frac{-2}{3}`.
    '''
    volume = M / (delta * _rho_crit(z, omega_m))
    return (3 * volume / (4 * np.pi))**(1./3)


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
        (omega_b / omega_m) / (2 * R_delta(M, z, omega_m, delta))


def P_BBPS(r, M, z, omega_b, omega_m,
           params_P_0=__BBPS_params_P_0,
           params_x_c=__BBPS_params_x_c,
           params_beta=__BBPS_params_beta,
           alpha=1, gamma=-0.3,
           delta=200):
    '''
    The best-fit pressure profile presented in BBPS2.

    Args:
        r (float or array): Radii from the cluster center, \
                            in Mpc :math:`h^{-2/3}`. If an array, an array \
                            is returned, if a scalar, a scalar is returned.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        params_P_0 (tuple): 3-tuple of :math:`P_0` mass, redshift dependence \
                parameters A, :math:`\\alpha_m`, :math:`\\alpha_z`, \
                respectively. See BBPS2 Equation 11. Default is BBPS2's \
                best-fit.
        params_x_c (tuple): 3-tuple of :math:`x_c` mass, redshift dependence, \
                same as `params_P_0`. Default is BBPS2's \
                best-fit.
        params_beta (tuple): 3-tuple of :math:`\\beta` mass, redshift \
                dependence, same as `params_P_0`. Default is BBPS2's \
                best-fit.

    Returns:
        float: Pressure at distance `r` from the cluster, in units of \
                :math:`h^{8/3} Msun s^{-2} Mpc^{-1}`.
    '''
    r = np.asarray(r, dtype=np.double)

    scalar_input = False
    if r.ndim == 0:
        scalar_input = True
        # Convert r to 2d
        r = r[None]
    if r.ndim > 1:
        raise Exception('r cannot be a >1D array.')

    P_out = np.zeros_like(r, dtype=np.double)

    # Set parameters
    P_0 = _A_BBPS(M, z, *params_P_0)
    x_c = _A_BBPS(M, z, *params_x_c)
    beta = _A_BBPS(M, z, *params_beta)

    _lib.P_BBPS(_dcast(P_out),
                _dcast(r), len(r),
                M, z,
                omega_b, omega_m,
                P_0, x_c, beta,
                float(alpha), gamma,
                delta)

    if scalar_input:
        return np.squeeze(P_out)
    return P_out


def projected_P_BBPS(r, M, z, omega_b, omega_m,
                     params_P_0=__BBPS_params_P_0,
                     params_x_c=__BBPS_params_x_c,
                     params_beta=__BBPS_params_beta,
                     alpha=1, gamma=-0.3,
                     delta=200,
                     limit=1000,
                     epsabs=1e-15, epsrel=1e-3,
                     return_errs=False):
    '''
    Computes the projected line-of-sight pressure of a cluster at a radius r
    from the cluster center.

    Args:
        r (float or array): Radius from the cluster center, in Mpc.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        params_P_0 (tuple): 3-tuple of :math:`P_0` mass, redshift dependence \
                parameters A, :math:`\\alpha_m`, :math:`\\alpha_z`, \
                respectively. See BBPS2 Equation 11. Default is BBPS2's \
                best-fit.
        params_x_c (tuple): 3-tuple of :math:`x_c` mass, redshift dependence, \
                same as `params_P_0`. Default is BBPS2's \
                best-fit.
        params_beta (tuple): 3-tuple of :math:`\\beta` mass, redshift \
                dependence, same as `params_P_0`. Default is BBPS2's \
                best-fit.

    Returns:
        float or array: Integrated line-of-sight pressure at distance `r` from \
                        the cluster, in units of Msun s^{-2} h^{8/3}.
    '''
    r = np.asarray(r, dtype=np.double)

    scalar_input = False
    if r.ndim == 0:
        scalar_input = True
        # Convert r to 2d
        r = r[None]
    if r.ndim > 1:
        raise Exception('r cannot be a >1D array.')

    P_out = np.zeros_like(r, dtype=np.double)
    P_err_out = np.zeros_like(r, dtype=np.double)

    # Set parameters
    P_0 = _A_BBPS(M, z, *params_P_0)
    x_c = _A_BBPS(M, z, *params_x_c)
    beta = _A_BBPS(M, z, *params_beta)

    rc = _lib.projected_P_BBPS(_dcast(P_out), _dcast(P_err_out),
                               _dcast(r), len(r),
                               M, z,
                               omega_b, omega_m,
                               P_0, x_c, beta,
                               alpha, gamma,
                               delta,
                               limit,
                               epsabs, epsrel)

    if rc != 0:
        msg = 'C_projected_P_BBPS returned error code: {}'.format(rc)
        raise RuntimeError(msg)

    if scalar_input:
        if return_errs:
            return np.squeeze(P_out), np.squeeze(P_err_out)
        return np.squeeze(P_out)
    if return_errs:
        return P_out, P_err_out
    return P_out


def fourier_P_BBPS(rmax, nr, M, z, omega_b, omega_m,
                   params_P_0=__BBPS_params_P_0,
                   params_x_c=__BBPS_params_x_c,
                   params_beta=__BBPS_params_beta,
                   alpha=1, gamma=-0.3,
                   delta=200):
    '''
    Computes the 3D fourier transform of the BBPS pressure profile. Necessary
    for computing the 2-halo term. Computed by evaluating the pressure profile
    at a discrete set of radii, and applying a fast Fourier transform (FFT).

    Args:
        rmax (float): The maximum R to evaluate the pressure profile at. \
                      (r = 0..maxR, in nr steps, is used).
        nr (int): Number of r samples to use in the FFT.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        params_P_0 (tuple): 3-tuple of :math:`P_0` mass, redshift dependence \
                parameters A, :math:`\\alpha_m`, :math:`\\alpha_z`, \
                respectively. See BBPS2 Equation 11. Default is BBPS2's \
                best-fit.
        params_x_c (tuple): 3-tuple of :math:`x_c` mass, redshift dependence, \
                same as `params_P_0`. Default is BBPS2's \
                best-fit.
        params_beta (tuple): 3-tuple of :math:`\\beta` mass, redshift \
                dependence, same as `params_P_0`. Default is BBPS2's \
                best-fit.

    Returns:
        float or array: The FFT for each `k`.
    '''
    # the FFT needs an even grid spacing
    rs = np.linspace(0.0, rmax, nr)
    Ps = P_BBPS(rs, M, z, omega_b, omega_m,
                params_P_0=params_P_0,
                params_x_c=params_x_c,
                params_beta=params_beta,
                alpha=alpha, gamma=gamma,
                delta=delta)
    # The pressure profile is singular - but since we are multiplying by R,
    # the P(r = 0) * r should be 0.
    Ps[0] = 0.0

    fftd = np.fft.fft(rs * Ps)
    ks = np.fft.fftfreq(nr, rmax / (nr - 1))

    fftd, ks = np.abs(fftd[ks > 0].imag), ks[ks > 0]

    return 2*np.pi*ks, (fftd / ks) * 2 * (rmax / (nr - 1))


def _C_fourier_P_BBPS(k, M, z, omega_b, omega_m,
                      params_P_0=__BBPS_params_P_0,
                      params_x_c=__BBPS_params_x_c,
                      params_beta=__BBPS_params_beta,
                      alpha=1, gamma=-0.3,
                      delta=200,
                      limit=1000,
                      epsabs=1e-23,
                      return_errs=False):
    '''
    Computes the 3D fourier transform of the BBPS pressure profile. Necessary
    for computing the 2-halo term.

    Args:
        k (float or array): Frequencies to compute FFT at, :math:`1/\\text{Mpc}`
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        params_P_0 (tuple): 3-tuple of :math:`P_0` mass, redshift dependence \
                parameters A, :math:`\\alpha_m`, :math:`\\alpha_z`, \
                respectively. See BBPS2 Equation 11. Default is BBPS2's \
                best-fit.
        params_x_c (tuple): 3-tuple of :math:`x_c` mass, redshift dependence, \
                same as `params_P_0`. Default is BBPS2's \
                best-fit.
        params_beta (tuple): 3-tuple of :math:`\\beta` mass, redshift \
                dependence, same as `params_P_0`. Default is BBPS2's \
                best-fit.

    Returns:
        float or array: The FFT for each `k`.
    '''
    k = np.asarray(k, dtype=np.double)

    scalar_input = False
    if k.ndim == 0:
        scalar_input = True
        # Convert r to 2d
        k = k[None]
    if k.ndim > 1:
        raise Exception('r cannot be a >1D array.')

    up_out = np.zeros_like(k, dtype=np.double)
    up_err_out = np.zeros_like(k, dtype=np.double)

    # Set parameters
    P_0 = _A_BBPS(M, z, *params_P_0)
    x_c = _A_BBPS(M, z, *params_x_c)
    beta = _A_BBPS(M, z, *params_beta)

    rc = _lib.fourier_P_BBPS(_dcast(up_out), _dcast(up_err_out),
                             _dcast(k), len(k),
                             M, z,
                             omega_b, omega_m,
                             P_0, x_c, beta,
                             alpha, gamma,
                             delta,
                             limit,
                             epsabs)

    if rc != 0:
        msg = 'C_projected_P_BBPS returned error code: {}'.format(rc)
        raise RuntimeError(msg)

    if scalar_input:
        if return_errs:
            return np.squeeze(up_out), np.squeeze(up_err_out)
        return np.squeeze(up_out)
    if return_errs:
        return up_out, up_err_out
    return up_out


def _A_BBPS(M, z, A_0, alpha_m, alpha_z):
    '''
    Mass-Redshift dependency model for the generalized BBPS profile parameters,
    fit to simulated halos in that data. The best-fit parameters are presented
    in Table 1. of BBPS2
    '''
    return A_0 * (M / 10**14)**alpha_m * (1 + z)**alpha_z


def projected_y_BBPS(r, M, z, omega_b, omega_m,
                     params_P_0=__BBPS_params_P_0,
                     params_x_c=__BBPS_params_x_c,
                     params_beta=__BBPS_params_beta,
                     alpha=1, gamma=-0.3,
                     delta=200,
                     Xh=0.76, epsrel=1e-3):
    '''
    Projected Compton-y parameter along the line of sight, at a perpendicular
    distance `r` from a cluster of mass `M` at redshift `z`. All arguments have
    the same meaning as `projected_P_BBPS`.

    Args:
        r (float or array): Radius from the cluster center, in Mpc.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        Xh (float): Primordial hydrogen mass fraction.

    Returns:
        float or array: Compton y parameter. Units are :math:`h^{8/3}`, so \
                        multiply by :math:`h^{8/3}` to obtain the true value.
    '''
    # The constant is \sigma_T / (m_e * c^2), i.e. the Thompson cross-section
    # divided by the mass-energy of the electron, in units of s^2 Msun^{-1}.
    # Source: Astropy constants and unit conversions.
    cy = 1.61574202e+15
    # We need to convert from GAS pressure to ELECTRON pressure. This is the
    # equation to do so, see BBPS2 p. 3.
    ch = (2 * Xh + 2) / (5 * Xh + 3)
    return ch * cy * projected_P_BBPS(r, M, z, omega_b, omega_m,
                                      params_P_0=params_P_0,
                                      params_x_c=params_x_c,
                                      params_beta=params_beta,
                                      alpha=alpha, gamma=gamma,
                                      delta=delta,
                                      epsrel=epsrel)


###############################################
# Functions for performing Image Convolutions #
###############################################


def create_image(fn, theta=15, n=200):
    xs, ys = np.meshgrid(range(n), range(n))
    midpt = (n - 1) / 2
    rs = (theta / midpt) * np.sqrt((xs - midpt)**2 + (ys - midpt)**2)
    y = fn(rs.flatten()).reshape(rs.shape)
    return rs, y


def create_convolved_profile(fn, theta=15, n=200,
                             sigma=5 / np.sqrt(2 * np.log(2))):
    '''
    Convolves the profile `fn` with a gaussian with std == :math`\sigma`, and
    returns the new 1D profile. We don't recommend using this directly.

    Args:
        fn (function): The function to be convolved. Should accept an array. \
                       Should take a single variable with units of `theta`.
        theta (float): Half-width of the image. i.e., for a 30 x 30 arcmin \
                       image centered on radius = 0, use theta = 15. \
                       (The units are not necessarily arcmin, but whatever the \
                       argument to `fn` uses.)
        n (int): The side length of the image, in pixels. The convolution is \
                 performed on an n x n image.
        sigma (float): The standard deviation of the Gaussian beam, in the \
                       same units as `theta`. The default is the Planck beam, \
                       in arcmin.

    Returns:
        (array, array): Radii and convolved profile. Each array is size n // 2.
    '''
    rs, img = create_image(fn, theta, n)
    kernel = Gaussian2DKernel(sigma * (n / 2) / theta)
    convolved = convolve_fft(img, kernel)
    return np.diag(rs)[n//2:], np.diag(convolved)[n//2:]


def convolved_y_BBPS(M, z, omega_b, omega_m, da,
                     theta=15, n=200,
                     sigma=5 / np.sqrt(2 * np.log(2)),
                     params_P_0=__BBPS_params_P_0,
                     params_x_c=__BBPS_params_x_c,
                     params_beta=__BBPS_params_beta,
                     alpha=1, gamma=-0.3,
                     delta=200,
                     Xh=0.76, epsrel=1e-3):
    '''
    Create an observed Compton-y profile of a halo by convolving it with
    a Gaussian beam function.

    Args:
        r (float or array): Radius from the cluster center, in Mpc.
        M (float): Cluster :math:`M_{\\Delta}`, in Msun.
        z (float): Cluster redshift.
        omega_b (float): Baryon fraction.
        omega_m (float): Matter fraction.
        theta (float): Half-width of the convolved image, in arcmin.
        n (int): The side length of the image, in pixels. The convolution is \
                 performed on an n x n image.
        sigma (float): The standard deviation of the Gaussian beam, in the \
                       same units as `theta`. The default is the Planck beam, \
                       in arcmin.
        Xh (float): Primordial hydrogen mass fraction.

    Returns:
        (2-tuple of array): Pair of (rs, smoothed ys).
    '''
    def image_func(thetas):
        return projected_y_BBPS(thetas * da / 60 * np.pi / 180,
                                M, z,
                                omega_b, omega_m,
                                params_P_0=params_P_0,
                                params_x_c=params_x_c,
                                params_beta=params_beta,
                                alpha=alpha, gamma=gamma,
                                delta=delta,
                                Xh=Xh, epsrel=epsrel)
    return create_convolved_profile(image_func, theta=theta, n=n, sigma=sigma)


##################################################
# The following functions are for testing only!! #
##################################################

def _py_projected_P_BBPS(r, M, z, omega_b, omega_m,
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


def _projected_P_BBPS_real(r, M, z, omega_b, omega_m, chis, zs,
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
                                 omega_b, omega_m)
                / (1 + np.interp(x, chis, zs)),
                chi_cluster - dist * R_del,
                chi_cluster + dist * R_del,
                epsrel=epsrel)[0]
