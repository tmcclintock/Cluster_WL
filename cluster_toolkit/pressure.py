#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jackson O'Donnell
#   jacksonhodonnell@gmail.com
'''
Galaxy cluster pressure profiles. Useful for modeling Sunyaev-Zeldovich cluster
observables.

This module implements pressure profiles presented by `Battaglia et al. 2012
<https://ui.adsabs.harvard.edu/abs/2012ApJ...758...75B/abstract>`_, referred to
here as BBPS. Calculations related to it are in the class :class:`BBPSProfile`.

The best-fit 3D one-halo profile, :math:`\\xi_{h, P}^{1h}(r | M, z)`,
from BBPS is implemented in :meth:`BBPSProfile.pressure`. Its 3D
projection and the projected Compton-y parameter are computed by
:meth:`BBPSProfile.projected_pressure` and :meth:`BBPSProfile.compton_y`,
respectively.

The two-halo term :math:`\\xi_{h, P}^{2h}(r | M, z)` is a result of correlated
structure near a halo. (I.e. halos tend to be near each other, which means the
one halo :math:`\\xi_{h, P}^{1h}(r | M, z)` alone is inadequate). For an
overview of the two-halo term and how to compute it, see
`Vikram et al. 2017
<https://ui.adsabs.harvard.edu/abs/2017MNRAS.467.2315V/abstract>`_
or
`Hill et al. 2018
<https://ui.adsabs.harvard.edu/abs/2018PhRvD..97h3501H/abstract>`_.

A general class for computing the 2-halo correlation is given in
:class:`TwoHaloProfile`, which uses the BBPS profile by default.

TODO: Document expected 1-halo arguments used by the 2-halo class.
'''

from __future__ import division
import abc
from astropy.convolution import Gaussian2DKernel, convolve_fft
from cluster_toolkit import _dcast, _lib
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d, interp2d
import scipy.special as sp
import sys

# Battaglia best fit parameters
_BBPS_params_P_0 = (18.1, 0.154, -0.758)
_BBPS_params_x_c = (0.497, -0.00865, 0.731)
_BBPS_params_beta = (4.35, 0.0393, 0.415)
_pressure_to_y = 1.61574202e+15

# The Planck PSF beam is (modeled as) a gaussian with this standard
# deviation (in arcmin)
# (The FWHM is 10 arcmin, hence the math)
PLANCK_SIGMA_PSF = 5 / np.sqrt(2 * np.log(2))


def _rho_crit(z, omega_m, h):
    '''
    The critical density of the universe :math:`\\rho_{crit}`, in units of
    :math:`Msun*Mpc^{-3}`.
    '''
    # The below formula assumes a flat univers, i.e. omega_m + omega_lambda = 1
    omega_lambda = 1 - omega_m
    # The constant is 3 * (100 km / s / Mpc)**2 / (8 * pi * G)
    # in units of Msun h^2 Mpc^{-3}
    # (source: astropy's constants module and unit conversions)
    return 2.77536627e+11 * h*h * (omega_m * (1 + z)**3 + omega_lambda)


def R_delta(M, z, omega_m, h, delta=200):
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
        float or array: Radius, in :math:`\\text{Mpc}`.
    '''
    volume = M / (delta * _rho_crit(z, omega_m, h))
    return (3 * volume / (4 * np.pi))**(1./3)


def P_delta(M, z, omega_b, omega_m, h, delta=200):
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
        float: Pressure amplitude, in units of Msun s^{-2} Mpc^{-1}.
    '''
    # G = 4.51710305e-48 Mpc^3 Msun^{-1} s^{-2}
    # (source: astropy's constants module and unit conversions)
    return 4.51710305e-48 * M * delta * _rho_crit(z, omega_m, h) * \
        (omega_b / omega_m) / (2 * R_delta(M, z, omega_m, h, delta))


def inverse_spherical_fourier_transform(rs, ks, Fs,
                                        limit=1000, epsabs=1e-21, epsrel=1e-3,
                                        return_errs=False):
    '''
    Inverse spherical fourier transform of a spectrum F(k), evaluated at a grid
    of radius values r. The spectrum F(k) is given at discrete points and
    interpolated.

    :math:`f(r) = \\frac{1}{2 \\pi^2 r} \\int_0^\\infty dk sin(kr) k F(k)`

    Note:
        The above integral over :math:`k` is done using a GSL integration
        routine. This means, however, that if the integrand is not
        well-behaved (i.e. Fs = 0 except at a single `k` value, or if `F` is
        singular) the integration routine may not locate the singularity, or it
        may not converge.

    Args:
        rs (array): The radii at which to compute the inverse Fourier transform.
        ks (array): Grid of angular frequencies `k` that the spectrum `F(k)` is
                    evaluated at.
        Fs (array): The spectrum `F(k)`, used for interpolation. Must be of
                    the same size as `ks`.
        limit (int): Number of subdivisions to use for integration
                     algorithm.
        epsabs (float): Absolute allowable error for integration.

    Returns:
        (array): The inverse-Fourier transformed profile `f(r)`. The same \
                 shape as `rs`.
    '''
    rs = np.asarray(rs, dtype=np.double)
    ks = np.asarray(ks, dtype=np.double)
    Fs = np.asarray(Fs, dtype=np.double)

    if ks.shape != Fs.shape:
        raise ValueError('ks and Fs must be the same shape')

    scalar_input = False
    if rs.ndim == 0:
        scalar_input = True
        # Convert r to 2d
        rs = rs[None]
    if rs.ndim > 1:
        raise Exception('rs cannot be a >1D array.')

    f_out = np.zeros_like(rs, dtype=np.double)
    f_err_out = np.zeros_like(rs, dtype=np.double)

    rc = _lib.inverse_spherical_fourier_transform(_dcast(f_out),
                                                  _dcast(f_err_out),
                                                  _dcast(rs), len(rs),
                                                  _dcast(ks), _dcast(Fs),
                                                  len(Fs),
                                                  limit, epsabs, epsrel)

    if rc != 0:
        msg = 'inverse_spherical_fourier_transform returned error code: {}'
        raise RuntimeError(msg.format(rc))

    if scalar_input:
        if return_errs:
            return np.squeeze(f_out), np.squeeze(f_err_out)
        return np.squeeze(f_out)
    if return_errs:
        return f_out, f_err_out
    return f_out


def forward_spherical_fourier_transform(ks, rs, fs, limit=1000,
                                        epsabs=1e-21, epsrel=1e-3,
                                        return_errs=False):
    '''
    Forward spherical fourier transform of a spectrum f(r), evaluated at a grid
    of radius values k. The profile f(r) is given at discrete points and
    interpolated.

    :math:`f(r) = 4\\pi^2 \\int_0^\\infty dr sin(kr)/(kr) r^2 f(r)`

    Note:
        The above integral over :math:`k` is done using a GSL integration
        routine. This means, however, that if the integrand is not
        well-behaved (i.e. Fs = 0 except at a single `k` value, or if `F` is
        singular) the integration routine may not locate the singularity, or it
        may not converge.

    Args:
        ks (array): The wavenumbers at which to compute the forward Fourier
                    transform.
        rs (array): Grid of radii `r` that the spectrum `f(r)` is
                    evaluated at.
        fs (array): The spectrum `f(r)`, used for interpolation. Must be of
                    the same size as `rs`.
        limit (int): Number of subdivisions to use for integration
                     algorithm.
        epsabs (float): Absolute allowable error for integration.

    Returns:
        (array): The forward-Fourier transformed profile `F(k)`. The same \
                 shape as `ks`.
    '''
    rs = np.asarray(rs, dtype=np.double)
    ks = np.asarray(ks, dtype=np.double)
    fs = np.asarray(fs, dtype=np.double)

    if rs.shape != fs.shape:
        raise ValueError('rs and Fs must be the same shape')

    scalar_input = False
    if rs.ndim == 0:
        scalar_input = True
        # Convert r to 2d
        rs = rs[None]
    if rs.ndim > 1:
        raise Exception('rs cannot be a >1D array.')

    f_out = np.zeros_like(ks, dtype=np.double)
    f_err_out = np.zeros_like(ks, dtype=np.double)

    rc = _lib.forward_spherical_fourier_transform(_dcast(f_out),
                                                  _dcast(f_err_out),
                                                  _dcast(ks), len(ks),
                                                  _dcast(rs), _dcast(fs),
                                                  len(fs),
                                                  limit, epsabs, epsrel)

    if rc != 0:
        msg = 'forward_spherical_fourier_transform returned error code: {}'
        raise RuntimeError(msg.format(rc))

    if scalar_input:
        if return_errs:
            return np.squeeze(f_out), np.squeeze(f_err_out)
        return np.squeeze(f_out)
    if return_errs:
        return f_out, f_err_out
    return f_out


def inverse_circular_fourier_transform(rs, ks, Fs, limit=1000,
                                       epsabs=1e-21, epsrel=1e-3,
                                       return_errs=False):
    '''
    Inverse circular fourier transform of a spectrum F(k), evaluated at a grid
    of radius values r. The spectrum F(k) is given at discrete points and
    interpolated.

    :math:`f(r) = \\frac{1}{2 \\pi} \\int_0^\\infty dk J_0(kr) k F(k)`

    Note:
        The above integral over :math:`k` is done using a GSL integration
        routine. This means, however, that if the integrand is not
        well-behaved (i.e. Fs = 0 except at a single `k` value, or if `F` is
        singular) the integration routine may not locate the singularity, or it
        may not converge.

    Args:
        rs (array): The radii at which to compute the inverse Fourier transform.
        ks (array): Grid of angular frequencies `k` that the spectrum `F(k)` is
                    evaluated at.
        Fs (array): The spectrum `F(k)`, used for interpolation. Must be of
                    the same size as `ks`.
        limit (int): Number of subdivisions to use for integration
                     algorithm.
        epsabs (float): Absolute allowable error for integration.
        epsrel (float): Relative allowable error for integration.

    Returns:
        (array): The inverse-Fourier transformed profile `f(r)`. The same \
                 shape as `rs`.
    '''
    rs = np.asarray(rs, dtype=np.double)
    ks = np.asarray(ks, dtype=np.double)
    Fs = np.asarray(Fs, dtype=np.double)

    if ks.shape != Fs.shape:
        raise ValueError('ks and Fs must be the same shape')

    scalar_input = False
    if rs.ndim == 0:
        scalar_input = True
        # Convert r to 2d
        rs = rs[None]
    if rs.ndim > 1:
        raise Exception('rs cannot be a >1D array.')

    f_out = np.zeros_like(rs, dtype=np.double)
    f_err_out = np.zeros_like(rs, dtype=np.double)

    rc = _lib.inverse_circular_fourier_transform(_dcast(f_out),
                                                 _dcast(f_err_out),
                                                 _dcast(rs), len(rs),
                                                 _dcast(ks), _dcast(Fs),
                                                 len(Fs),
                                                 limit, epsabs, epsrel)

    if rc != 0:
        msg = 'inverse_circular_fourier_transform returned error code: {}'
        raise RuntimeError(msg.format(rc))

    if scalar_input:
        if return_errs:
            return np.squeeze(f_out), np.squeeze(f_err_out)
        return np.squeeze(f_out)
    if return_errs:
        return f_out, f_err_out
    return f_out


def forward_circular_fourier_transform(ks, rs, fs, limit=1000,
                                       epsabs=1e-21, epsrel=1e-3,
                                       return_errs=False):
    '''
    Forward circular fourier transform of a spectrum f(r), evaluated at a grid
    of radius values k. The profile f(r) is given at discrete points and
    interpolated.

    :math:`f(k) = 2\\pi \\int_0^\\infty dr J_0(kr) r f(r)`

    Note:
        The above integral over :math:`k` is done using a GSL integration
        routine. This means, however, that if the integrand is not
        well-behaved (i.e. Fs = 0 except at a single `k` value, or if `F` is
        singular) the integration routine may not locate the singularity, or it
        may not converge.

    Args:
        ks (array): The wavenumbers at which to compute the forward Fourier
                    transform.
        rs (array): Grid of radii `r` that the spectrum `f(r)` is
                    evaluated at.
        fs (array): The spectrum `f(r)`, used for interpolation. Must be of
                    the same size as `rs`.
        limit (int): Number of subdivisions to use for integration
                     algorithm.
        epsabs (float): Absolute allowable error for integration.
        epsrel (float): Relative allowable error for integration.

    Returns:
        (array): The forward-Fourier transformed profile `F(k)`. The same \
                 shape as `ks`.
    '''
    rs = np.asarray(rs, dtype=np.double)
    ks = np.asarray(ks, dtype=np.double)
    fs = np.asarray(fs, dtype=np.double)

    if rs.shape != fs.shape:
        raise ValueError('rs and Fs must be the same shape')

    scalar_input = False
    if rs.ndim == 0:
        scalar_input = True
        # Convert r to 2d
        rs = rs[None]
    if rs.ndim > 1:
        raise Exception('rs cannot be a >1D array.')

    f_out = np.zeros_like(ks, dtype=np.double)
    f_err_out = np.zeros_like(ks, dtype=np.double)

    rc = _lib.forward_circular_fourier_transform(_dcast(f_out),
                                                 _dcast(f_err_out),
                                                 _dcast(ks), len(ks),
                                                 _dcast(rs), _dcast(fs),
                                                 len(fs),
                                                 limit, epsabs, epsrel)

    if rc != 0:
        msg = 'forward_circular_fourier_transform returned error code: {}'
        raise RuntimeError(msg.format(rc))

    if scalar_input:
        if return_errs:
            return np.squeeze(f_out), np.squeeze(f_err_out)
        return np.squeeze(f_out)
    if return_errs:
        return f_out, f_err_out
    return f_out


def abel_transform(xs, ys, rs, limit=1000, epsabs=1e-29, epsrel=1e-3):
    '''
    Perform the Abel transform (line-of-sight projection) for a spherically
    symmetric function defined by (xs, ys), on the grid of transverse radii
    `rs`.

    TODO: fully document.
    '''
    xs = np.ascontiguousarray(xs, dtype=np.double)
    ys = np.ascontiguousarray(ys, dtype=np.double)
    rs = np.ascontiguousarray(rs, dtype=np.double)

    if xs.shape != ys.shape:
        raise ValueError('abel_transform: xs and ys must be same shape')

    f_out = np.zeros_like(rs)
    f_out_err = np.zeros_like(rs)

    rc = _lib.abel_transform_interp(_dcast(f_out), _dcast(f_out_err),
                                    _dcast(xs), _dcast(ys), ys.size,
                                    _dcast(rs), rs.size,
                                    limit, epsabs, epsrel)

    if rc != 0:
        msg = 'abel_transform returned error code: {}'
        raise RuntimeError(msg.format(rc))
    return f_out


def integrate_spline(xs, ys, a, b):
    '''
    Integrate a cubic spline of the function `ys = f(xs)` from x=a to x=b.

    TODO: fully document.
    '''
    xs = np.ascontiguousarray(xs, dtype=np.double)
    ys = np.ascontiguousarray(ys, dtype=np.double)

    if xs.shape != ys.shape:
        raise ValueError('integrate_spline: xs and ys must be same shape')

    # Create a one-element array for the result
    f_out = np.array(0.0, dtype=np.double)

    rc = _lib.integrate_spline(_dcast(xs), _dcast(ys), len(xs),
                               a, b, _dcast(f_out))

    if rc != 0:
        msg = 'integrate_spline returned error code: {}'
        raise RuntimeError(msg.format(rc))
    return f_out[()]


###############################################
# Functions for performing Image Convolutions #
###############################################

def create_image(fn, theta=15, n=200):
    xs, ys = np.meshgrid(range(n), range(n))
    midpt = (n - 1) / 2
    rs = (theta / midpt) * np.sqrt((xs - midpt)**2 + (ys - midpt)**2)
    y = fn(rs.flatten()).reshape(rs.shape)
    return rs, y


# TODO: should we allow a custom kernel?
def create_convolved_profile(fn, theta=15, n=200,
                             sigma=PLANCK_SIGMA_PSF):
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


# Some tricky stuff with abstract base class and py2/3 compatibility
if sys.version_info.major < 3:
    __pprofile_supclass = object
else:
    __pprofile_supclass = abc.ABC


class PressureProfile(__pprofile_supclass):
    if sys.version_info.major < 3:
        __metaclass__ = abc.ABCMeta
    '''
    Abstract base class for cluster pressure profiles. Subclasses need only
    define the parameters and the method `pressure`, and the projection, fourier
    transforms, and convolution will automatically be available. They can
    overload more functions if they wish (if e.g. there is an analytical
    solution to the projection or FT, or to adjust between a
    :math:`M_{\Delta c}` and :math:`M_{\Delta m}` mass definition).

    Args:
        M (float): Mass of the cluster, default :math:`M_{200 c}` in
                   :math:`M_{sun}`. Subclasses may use a different mass
                   definition.
        z (float): Redshift of the cluster center.
        cosmo (dictionary):
            A dictionary of cosmological parameters. Expected to
            contain the following (subclasses may require more or less):

            * **omega_b:** (float) Baryon fraction.
            * **omega_m:** (float) Mass fraction.
            * **h0:** (float) Reduced hubble constant,
              :math:`H_0 / (100 km s^{-1} Mpc^{-1})`.
        delta (float): The matter overdensity of the mass definition, the
                       :math:`\Delta` in :math:`M_{\Delta c}`. Default of 200
    '''
    def __init__(self, M, z, cosmo, delta=200):
        self.__M = M
        self.__z = z
        self.__delta = delta

        self.__omega_b = cosmo['omega_b']
        self.__omega_m = cosmo['omega_m']
        self.__h = cosmo['h0']

    def _update_halo(self):
        self.__R_delta = R_delta(self.M, self.z, self.omega_m, self.h,
                                 delta=self.delta)
        self.__P_delta = P_delta(self.M, self.z,
                                 self.omega_b, self.omega_m, self.h,
                                 delta=self.delta)

    def update_halo(self, M, z, delta=200):
        '''
        Update the mass, redshift, and optionally overdensity of the halo.

        Args:
            M (float): Halo mass, in :math:`M_{sun}`.
            z (float): redshift.
            delta (number): Halo overdensity :math:`\\Delta`.
        '''
        self.__M = M
        self.__z = z
        self.__delta = delta
        self._update_halo()
        return self

    def update_cosmology(self, cosmo):
        '''
        Update cosmological parameters.

        Args:
            cosmo (dictionary):
                A dictionary of cosmological parameters. Expected to
                contain the following:

                * **omega_b:** (float) Baryon fraction.
                * **omega_m:** (float) Mass fraction.
                * **h0:** (float) Reduced hubble constant.
        '''
        self.__omega_b = cosmo['omega_b']
        self.__omega_m = cosmo['omega_m']
        self.__h = cosmo['h0']
        self._update_halo()
        return self

    @property
    def M(self):
        '''
        Halo mass, in units of :math:`M_{sun}`.
        '''
        return self.__M

    @M.setter
    def set_M(self, M):
        self.__M = M
        self._update_halo()

    @property
    def z(self):
        '''
        Halo redshift.
        '''
        return self.__z

    @z.setter
    def set_z(self, z):
        self.__z = z
        self._update_halo()

    @property
    def delta(self):
        '''
        Halo overdensity :math:`\\Delta`. Default 200.
        '''
        return self.__delta

    @delta.setter
    def set_delta(self, delta):
        self.__delta = delta
        self._update_halo()

    @property
    def omega_b(self):
        '''
        Baryon fraction.
        '''
        return self.__omega_b

    @omega_b.setter
    def set_omega_b(self, omega_b):
        self.__omega_b = omega_b
        self._update_halo()

    @property
    def omega_m(self):
        '''
        Matter fraction.
        '''
        return self.__omega_m

    @omega_m.setter
    def set_omega_m(self, omega_m):
        self.__omega_m = omega_m
        self._update_halo()

    @property
    def h(self):
        '''
        Reduced Hubble constant.
        '''
        return self.__h

    @h.setter
    def set_h(self, h):
        self.__h = h
        self._update_halo()

    @property
    def R_delta(self):
        '''
        Virial radius :math:`R_{\\Delta}`, cosmology-dependent.
        See BBPS equation 6.

        :math:`R_{\\Delta}` cannot be set directly, and is automatically updated
        whenever the parameters it depends on are.

        Units:
            :math:`\\text{Mpc}`.
        '''
        return self.__R_delta

    @property
    def P_delta(self):
        '''
        Virial pressure :math:`P_{\\Delta}`, cosmology-dependent.
        See BBPS section 4.1.

        :math:`P_{\\Delta}` cannot be set directly, and is automatically updated
        whenever the parameters it depends on are.

        Units:
            :math:`M_{sun} s^{-2} \\text{Mpc}^{-1}`
        '''
        return self.__P_delta

    @abc.abstractmethod
    def pressure(self):
        pass


class BBPSProfile(PressureProfile):
    '''
    The best-fit pressure profile presented in BBPS.

    The 3D pressure profile is computed in :meth:`pressure`, and
    the projected pressure and Compton y are computed in
    :meth:`projected_pressure` and :meth:`compton_y`.

    >>> halo = BBPSProfile(3e14, 0.2, 0.04, 0.28)
    >>> # Let's compute the pressure profile over a small radial range
    >>> halo.pressure(np.linspace(0.1, 5, 10))
    array([9.10170657e-20, 2.87696683e-21, 3.92779676e-22, 9.39464388e-23,
           3.06798145e-23, 1.22290395e-23, 5.60028834e-24, 2.84086857e-24,
           1.55884172e-24, 9.10278668e-25])
    >>> # Now let's do it in absolute units
    >>> h0 = 0.8
    >>> h0**(8/3) * halo.pressure(np.linspace(0.1, 5, 10) * h0**(2/3))
    array([4.41304440e-20, 2.13323485e-21, 3.46223667e-22, 9.09642709e-23,
           3.15108279e-23, 1.30793006e-23, 6.16910021e-24, 3.20053361e-24,
           1.78752532e-24, 1.05882458e-24])

    Args:
        M (float):
            Cluster :math:`M_{\\Delta c}`, in Msun.  **NB** this is the
            **CRITICAL MASS OVERDENSITY** mass definition.
        z (float): Cluster redshift.
        cosmo (dictionary):
            A dictionary of cosmological parameters. Expected to
            contain the following:

            * **omega_b:** (float) Baryon fraction.
            * **omega_m:** (float) Mass fraction.
            * **h0:** (float) Reduced hubble constant.
        params_P_0 (tuple):
            3-tuple of :math:`P_0` mass, redshift dependence parameters A,
            :math:`\\alpha_m`, :math:`\\alpha_z`, respectively. See BBPS
            Equation 11. Default is BBPS's best-fit.
        params_x_c (tuple):
            3-tuple of :math:`x_c` mass, redshift dependence, same as
            `params_P_0`. Default is BBPS's best-fit.
        params_beta (tuple):
            3-tuple of :math:`\\beta` mass, redshift dependence, same as
            `params_P_0`. Default is BBPS's best-fit.
        alpha (float): Profile parameter. See BBPS Eq. 10.
        gamma (float): Profile parameter. See BBPS Eq. 10.
        delta (float): Halo overdensity :math:`\\Delta`.
    '''
    def __init__(self, M, z,
                 cosmo,
                 params_P_0=_BBPS_params_P_0,
                 params_x_c=_BBPS_params_x_c,
                 params_beta=_BBPS_params_beta,
                 M_pivot=1e14,
                 alpha=1, gamma=-0.3,
                 delta=200):
        # Halo definition
        # (we use this instead of just super() for py2 compatibility)
        super(BBPSProfile, self).__init__(M, z, cosmo, delta=delta)

        # Profile fit parameters
        self.__params_P_0 = params_P_0
        self.__params_x_c = params_x_c
        self.__params_beta = params_beta
        self.__M_pivot = M_pivot
        self.alpha = alpha
        self.gamma = gamma

        # Set parameters
        self._update_halo()

    def _update_halo(self):
        # (we use this instead of just super() for py2 compatibility)
        super(BBPSProfile, self)._update_halo()
        self.__P_0 = self._A(self.M, self.z, *self.__params_P_0,
                             **{'M_pivot': self.__M_pivot})
        self.__x_c = self._A(self.M, self.z, *self.__params_x_c,
                             **{'M_pivot': self.__M_pivot})
        self.__beta = self._A(self.M, self.z, *self.__params_beta,
                              **{'M_pivot': self.__M_pivot})

    @staticmethod
    def _A(M, z, A_0, alpha_m, alpha_z, M_pivot=1e14):
        '''
        Mass-Redshift dependency model for the generalized BBPS profile
        parameters, fit to simulated halos in that data. The best-fit
        parameters are presented in Table 1. of BBPS
        '''
        return A_0 * (M / M_pivot)**alpha_m * (1 + z)**alpha_z

    def pressure(self, r):
        '''
        The best-fit 3D pressure profile.

        Args:
            r (float or array): Radii from the cluster center,
                                in :math:`Mpc`. If an array, an array
                                is returned, if a scalar, a scalar is returned.

        Returns:
            (float or array): Pressures corresponding to `r`, in units of
                              :math:`Msun s^{-2} Mpc^{-1}`.
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

        _lib.P_BBPS(_dcast(P_out),
                    _dcast(r), len(r),
                    self.M, self.z,
                    self.omega_b, self.omega_m, self.h,
                    self.__P_0, self.__x_c, self.__beta,
                    float(self.alpha), self.gamma,
                    self.delta)

        if scalar_input:
            return np.squeeze(P_out)
        return P_out

    def projected_pressure(self, r, return_errs=False, limit=1000,
                           epsabs=1e-15, epsrel=1e-3):
        '''
        Computes the projected line-of-sight pressure of a cluster at a radius r
        from the cluster center.

        Args:
            r (float or array): Radius from the cluster center, in Mpc.
            return_errs (bool): Whether to return integration errors.
            limit (int): Number of subdivisions to use for integration
                         algorithm.
            epsabs (float): Absolute allowable error for integration.
            epsrel (float): Relative allowable error for integration.

        Returns:
            tuple or array: Integrated line-of-sight pressure at distance `r`
                            from the cluster, in units of :math:`Msun s^{-2}`.
                            If `return_errs` is set, returns a 2-tuple of
                            (values, errors).
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

        rc = _lib.projected_P_BBPS(_dcast(P_out), _dcast(P_err_out),
                                   _dcast(r), len(r),
                                   self.M, self.z,
                                   self.omega_b, self.omega_m, self.h,
                                   self.__P_0, self.__x_c, self.__beta,
                                   self.alpha, self.gamma,
                                   self.delta,
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

    def compton_y(self, r, Xh=0.76, limit=1000, epsabs=1e-15, epsrel=1e-3):
        '''
        Projected Compton-y parameter along the line of sight, at a set of
        perpendicular distances `r` from the halo.
        All arguments have the same meaning as `projected_pressure`.

        Args:
            r (float or array): Radius from the cluster center, in Mpc.

        Returns:
            float or array: Compton y parameter. Unitless.
        '''
        # The constant is \sigma_T / (m_e * c^2), the Thompson cross-section
        # divided by the mass-energy of the electron, in units of s^2 Msun^{-1}.
        # Source: Astropy constants and unit conversions.
        # We need to convert from GAS pressure to ELECTRON pressure. This is the
        # equation to do so, see BBPS p. 3.
        ch = (2 * Xh + 2) / (5 * Xh + 3) * _pressure_to_y
        return ch * self.projected_pressure(r, limit=limit,
                                            epsabs=epsabs, epsrel=epsrel)

    def convolved_y_FT(self, thetas, ks, da,
                       Xh=0.76,
                       miscent_offset=None,
                       sigma_beam=PLANCK_SIGMA_PSF,
                       nr=1000, nM=1000,
                       limit=1000,
                       epsabs=1e-25,
                       epsrel=1e-3):

        ft_pressure = self._C_fourier_pressure(ks)

        # Due to the projection-slice thm, the 3D FT is the same as the
        # 2D FT of a projection of the 3D distribution.
        # Basically, we can skip a step by using the 3D FT of the 2halo as the
        # 2D FT of the projected 2halo
        y_conversion = _pressure_to_y * (2*Xh + 2) / (5*Xh + 3)
        fourier_2h = y_conversion * ft_pressure

        # Transverse radii - convert theta (in arcmin) to physical radii
        rs = da * (np.pi / 180) * (thetas / 60)

        # If a miscentering distance has been specified, apply it
        if miscent_offset is not None:
            fourier_2h *= sp.j0(miscent_offset * ks)

        # If a beam width has been specified, apply it
        if sigma_beam is not None:
            # Convert sigma from arcmin to physical units
            sigma_phys = da * (np.pi / 180) * (sigma_beam / 60)
            fourier_2h *= np.exp(-ks*ks*sigma_phys*sigma_phys/2)

        return inverse_circular_fourier_transform(rs, ks, fourier_2h,
                                                  limit=limit,
                                                  epsabs=epsabs,
                                                  epsrel=epsrel) / (1 + self.z)

    def convolved_y(self, da, theta=15, n=200,
                    sigma_beam=PLANCK_SIGMA_PSF,
                    Xh=0.76, limit=1000,
                    epsabs=1e-14, epsrel=1e-3):
        '''
        Create an observed Compton-y profile of a halo by convolving it with
        a Gaussian beam function.

        Args:
            da (float): Angular diameter distance at cluster redshift.
            theta (float): Half-width of the convolved image, in arcmin.
            n (int): The side length of the image, in pixels. The convolution \
                     is performed on an n x n image.
            sigma (float): The standard deviation of the Gaussian beam, in the \
                           same units as `theta`. The default is the Planck \
                           beam, in arcmin.
            Xh (float): Primordial hydrogen mass fraction.

        Returns:
            (2-tuple of array): Pair of (rs, smoothed ys). `rs` runs from \
                                :math:`(theta / (n // 2)) / 2` to \
                                :math:`\\sqrt(2) theta`, and contains `n` \
                                points.
        '''
        def image_func(thetas):
            return self.compton_y(thetas * da / 60 * np.pi / 180,
                                  Xh=Xh, limit=limit,
                                  epsabs=epsabs, epsrel=epsrel)
        return create_convolved_profile(image_func,
                                        theta=theta, n=n, sigma=sigma_beam)

    def fourier_pressure(self, rmax, nr):
        '''
        Computes the 3D fourier transform of the BBPS pressure profile.
        Necessary for computing the 2-halo term. Computed by evaluating the
        pressure profile at a discrete set of radii, and applying a fast
        Fourier transform (FFT).

        Args:
            rmax (float): The maximum R to evaluate the pressure profile at. \
                          (r = 0..maxR, in nr steps, is used).
            nr (int): Number of r samples to use in the FFT.

        Returns:
            (array, array): Two arrays, the `ks` and the Fourier transform \
                            `P(k)`.
        '''
        # the FFT needs an even grid spacing
        rs = np.linspace(0.0, rmax, nr)
        Ps = self.pressure(rs)

        # The pressure profile is singular - but since we are multiplying by R,
        # the P(r = 0) * r should be 0.
        Ps[0] = 0.0

        fftd = np.fft.rfft(rs * Ps)
        ks = np.fft.fftfreq(nr, rmax / (nr - 1))

        # We only need the `sin()` (i.e. imaginary) terms
        fftd, ks = np.abs(fftd[1:-1].imag), ks[ks > 0]

        return 2*np.pi*ks, (fftd / ks) * 2 * (rmax / (nr - 1))

    def _projected_pressure(self, r, dist=8, epsrel=1e-3):
        '''
        THIS FUNCTION IS FOR TESTING ONLY.

        Computes the projected line-of-sight density of a cluster at a radius r
        from the cluster center.

        Args:
            r (float): Radius from the cluster center, in Mpc.

        Returns:
            float: Integrated line-of-sight pressure at distance `r` from the \
                   cluster, in units of Msun s^{-2}.
        '''
        return quad(lambda x: self.pressure(np.sqrt(x*x + r*r)),
                    -dist * self.R_delta, dist * self.R_delta,
                    epsrel=epsrel)[0] / (1 + self.z)

    def _projected_pressure_real(self, r, chis, zs,
                                 dist=8, epsrel=1e-3):
        '''
        THIS FUNCTION IS FOR TESTING ONLY.

        Computes the projected line-of-sight density of a cluster at a radius r
        from the cluster center.

        Args:
            r (float): Radius from the cluster center, in Mpc.
            chis (1d array of floats): The comoving line-of-sight distance, \
                                       in Mpc.
            zs (1d array of floats): The redshifts corresponding to `chis`.

        Returns:
            float: Integrated line-of-sight pressure at distance `r` from the \
                   cluster, in units of Msun s^{-2}.
        '''
        chi_cluster = np.interp(self.z, zs, chis)
        return quad(lambda x: self.pressure(np.sqrt((x - chi_cluster)**2 + r*r))
                    / (1 + np.interp(x, chis, zs)),
                    chi_cluster - dist * self.R_delta,
                    chi_cluster + dist * self.R_delta,
                    epsrel=epsrel)[0]

    def _C_fourier_pressure(self, k,
                            limit=1000,
                            epsabs=1e-23, epsrel=1e-3,
                            return_errs=False):
        '''
        Computes the 3D fourier transform of the BBPS pressure profile.
        Necessary for computing the 2-halo term.

        Args:
            k (float or array): Frequencies to compute FFT at,
                                :math:`1/\\text{Mpc}`
            M (float): Cluster :math:`M_{\\Delta}`, in Msun.
            z (float): Cluster redshift.
            omega_b (float): Baryon fraction.
            omega_m (float): Matter fraction.
            params_P_0 (tuple): 3-tuple of :math:`P_0` mass, redshift
                                dependence parameters A, :math:`\\alpha_m`,
                                :math:`\\alpha_z`, respectively. See BBPS2
                                Equation 11.  Default is BBPS2's best-fit.
            params_x_c (tuple): 3-tuple of :math:`x_c` mass, redshift
                                dependence, same as `params_P_0`. Default is
                                BBPS2's best-fit.
            params_beta (tuple): 3-tuple of :math:`\\beta` mass, redshift
                                 dependence, same as `params_P_0`. Default is
                                 BBPS2's best-fit.

        Returns:
            float or array: The FFT for each `k`.
        '''
        k = np.ascontiguousarray(k, dtype=np.double)

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

        rc = _lib.fourier_P_BBPS(_dcast(up_out), _dcast(up_err_out),
                                 _dcast(k), len(k),
                                 self.M, self.z,
                                 self.omega_b, self.omega_m, self.h,
                                 self.__P_0, self.__x_c, self.__beta,
                                 self.alpha, self.gamma,
                                 self.delta,
                                 limit,
                                 epsabs, epsrel)

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


class TwoHaloProfile:
    '''
    A general interface for computing the 2-halo halo-pressure correlation
    function, :math:`\\xi_{h, P}^{2h}(r | M, z)`.

    By default, the halo mass functions (`hmb_[mzb]`, `hmf_[mzf]`) are expected
    to use a **mean mass overdensity** halo definition. The one halo model is
    expected to use a **critical mass overdensity** halo definition.
    The `mdelta_m` and `mdelta_c` arguments are used in an interpolation table
    to convert between them.

    If these assumptions are false, use the `halo_mass_def` and `one_halo_def`
    args to adjust accordingly.

    Args:
        cosmo (dictionary):
            A dictionary of cosmological parameters. Expected to
            contain the following:

            * **omega_b:** (float) Baryon fraction.
            * **omega_m:** (float) Mass fraction.
            * **h0:** (float) Reduced hubble constant,
              :math:`H_0 / (100 km s^{-1} Mpc^{-1})`.
            * **hmb_m:** (1d array) The mass points at which the halo mass bias
              is evaluated. Units: :math:`M_{sun}`.  Mass definition:
              :math:`M_{200m}`.
            * **hmb_z:** (1d array) The redshifts at which the halo mass bias is
              evaluated.
            * **hmb_b:** (2d array) The evaluated halo mass bias, for each
              (m, z) combination.
            * **hmf_m:** (1d array) The mass points at which the halo mass
              function is evaluated.  Units: :math:`M_{sun}`.  Mass definition:
              :math:`M_{200m}`.
            * **hmf_z:** (1d array) The redshifts at which the halo mass
              function is evaluated.
            * **hmf_f:** (2d array) The evaluated halo mass function, for each
              (m, z) combination.
            * **P_lin_k:** (1d array) The wavenumbers at which the linear matter
              power spectrum is evaluated. Expected to be defined in terms of
              *comoving* distance.
            * **P_lin_z:** (1d array) The redshifts at which the linear matter
              power spectrum is evaluated.
            * **P_lin:** (2d array) The evaluated linear matter power spectrum,
              for each (k, z) combination.
        mdelta_m (1d array): A set of *mean overdensity* halo masses.
        mdelta_c (1d array): A set of *critical overdensity* halo masses
                             corresponding to `mdelta_m`. Needed to convert
                             between the two halo definitions.
        halo_mass_def (string): One of "mean", "critical". The mass
                                definiton used in `hmf_m` and `hmb_m`.
        one_halo_def (string): One of "mean", "critical". The mass definiton
                               used by `one_halo`.
        one_halo (class): The 1-halo pressure model to use, default is
                          :class:`BBPSProfile`. TODO: Document expected
                          arguments.
        one_halo_args (iterable): Any extra arguments to give the `one_halo`
                                  class, e.g. profile parameters.
        one_halo_kwargs (dictionary): Any extra keyword arguments to give
                                      the `one_halo` class, e.g. profile
                                      parameters.
    '''
    def __init__(self, cosmo,
                 mdelta_m, mdelta_c,
                 halo_mass_def='mean',
                 one_halo_def='critical',
                 one_halo=BBPSProfile, one_halo_args=(), one_halo_kwargs={},
                 allow_weird_h=False):
        self.cosmo = cosmo.copy()
        omega_m, omega_b, h = cosmo['omega_m'], cosmo['omega_b'], cosmo['h0']
        hmb_m, hmb_z, hmb_b = cosmo['hmb_m'], cosmo['hmb_z'], cosmo['hmb_b']
        hmf_m, hmf_z, hmf_f = cosmo['hmf_m'], cosmo['hmf_z'], cosmo['hmf_f']
        P_lin_k, P_lin_z, P_lin = cosmo['P_lin_k'], cosmo['P_lin_z'], \
            cosmo['P_lin']

        # To help user get the right units
        if not allow_weird_h:
            if h > 2 or h < 0.1:
                raise ValueError('The **reduced** Hubble constant is needed')

        self.omega_b = omega_b
        self.omega_m = omega_m
        self.h = h

        self.hmb = interp2d(hmb_m, hmb_z, hmb_b)
        self.hmf = interp2d(hmf_m, hmf_z, hmf_f)
        self.P_lin = interp2d(P_lin_k, P_lin_z, P_lin)

        # TODO: these should be unnecessary if halo_mass_def == one_halo_def
        self.mean_to_critical = interp1d(mdelta_m, mdelta_c)
        self.critical_to_mean = interp1d(mdelta_c, mdelta_m)

        if halo_mass_def not in ('mean', 'critical'):
            raise ValueError('Invalid mass def: {}'.format(halo_mass_def))
        if one_halo_def not in ('mean', 'critical'):
            raise ValueError('Invalid mass def: {}'.format(one_halo_def))

        self.halo_mass_def = halo_mass_def
        self.one_halo_def = one_halo_def

        self._profile_model = one_halo
        self._one_halo_args = one_halo_args
        self._one_halo_kwargs = one_halo_kwargs

    def projected(self, rs_proj, rs_2h, ks, z,
                  nM=1000, limit=1000,
                  epsabs_2h=1e-21,
                  epsabs_abel=1e-23,
                  epsrel_abel=1e-3):
        '''
        Computes the projected 2-halo term.

        Args:
            rs_proj (array): The transverse distances at which to project.
                             Units: :math:`Mpc`
            rs_2h (array): The radii at which to compute the 3D 2-halo term.
                           (The projection is performed on an interpolation
                           table, this is the spacing of that interpolation
                           table.)
                           Units: :math:`Mpc`
            ks (array): Wavenumbers to use for computing 3D 2h term.
                        Units: :math:`Mpc^{-1}`.
            z (float): Redshift to use.

        Returns:
            (array): The projected 2-halo term, corresponding to `rs_proj`. \
                    Units: :math:`M_{sun} s^{-2}`.
        '''
        two_halo_3d = self.two_halo(rs_2h, ks, z,
                                    nM=nM, limit=limit,
                                    epsabs=epsabs_2h)

        return abel_transform(rs_2h, two_halo_3d, rs_proj,
                              limit=limit,
                              epsabs=epsabs_abel,
                              epsrel=epsrel_abel) / (1 + z)

    def compton_y(self, rs_proj, rs_2h, ks, z,
                  Xh=0.76,
                  nM=1000, limit=1000,
                  epsabs_2h=1e-21,
                  epsabs_abel=1e-23,
                  epsrel_abel=1e-3):
        '''
        Computes the Compton-y 2-halo term, i.e. the projected line-of-sight
        CMB distortion. It is dimensionless.

        Args:
            rs_proj (array): The transverse distances at which to project.
                             Units: Physical (non-comoving) :math:`Mpc`
            rs_2h (array): The radii at which to compute the 3D 2-halo term.
                           (The projection is performed on an interpolation
                           table, this is the spacing of that interpolation
                           table.)
                           Units: Physical (non-comoving) :math:`Mpc`
            ks (array): Wavenumbers to use for computing 3D 2h term.
                        Units: :math:`Mpc^{-1}`.
            z (float): Redshift to use.
            Xh (float): Primordial hydrogen mass fraction. Necessary to compute
                        the *electron pressure* from the *gas pressure*.

        Returns:
            (array): The unitless projected 2-halo term, at `rs_proj`.
        '''
        const = (2 * Xh + 2) / (5 * Xh + 3) * _pressure_to_y
        return const * self.projected(rs_proj, rs_2h, ks, z,
                                      nM=1000, limit=1000,
                                      epsabs_2h=1e-21,
                                      epsabs_abel=1e-23,
                                      epsrel_abel=1e-3)

    def two_halo(self, rs, ks, z,
                 nM=1000, limit=1000,
                 epsabs=1e-25, epsrel=1e-3):
        '''
        Computes the 3D 2-halo halo-pressure correlation
        :math:`\\xi_{h, P}(r | M, z)`:

        :math:`\\xi_{h,P}(r | M, z) = \\int_0^\infty dk \\frac{k^2}{2 \\pi^2}\
               sin(kr) / (kr) P_{h, P}(k | M, z)`
        :math:`P_{h, P}(k | M, z) = b(M, z) P_{lin}(k, z) \\int_0^\infty\
               dM^\\prime \\frac{dn}{dM^\\prime} b(M^\\prime, z)\
               u_P(k | M^\\prime, z)`
        :math:`u_P(k | M, z) = \int_0^\\infty dr 4\\pi r^2 sin(kr)/(kr)\
               P_e(r | M^\\prime, z)`

        Args:
            rs (array of float): The radii at which to compute :math:`\\xi`. NB
                                 this is physical (NOT comoving) distance.
            ks (array of float): The wavenumbers to compute the Fourier
                                 transform.
            z (float): Redshift to use.

        Returns:
            (array): The 3D halo-pressure correlation at the radii `rs`. Same \
                     shape as `rs`.
        '''
        igrnds = self.two_halo_mass_integrand(ks, z,
                                              nM=nM,
                                              epsabs=epsabs,
                                              epsrel=epsrel)

        # The matter power spectrum uses *comoving* units, we want *physical*
        # units, hence `/ (1 + z)`
        # (We would also need a (1 + z)**3 in front of P_lin, but it actually
        #  cancels with the HMF 1/volume)
        Ps = self.P_lin(ks / (1 + z), z)
        radial_term = inverse_spherical_fourier_transform(rs, ks, Ps * igrnds,
                                                          limit=limit,
                                                          epsabs=epsabs,
                                                          epsrel=epsrel)

        # TODO - mass bias
        return radial_term

    def two_halo_mass_integrand(self, ks, z,
                                nr=1000, nM=1000,
                                limit=1000, epsabs=1e-25, epsrel=1e-3):
        '''
        A mass-weighted pressure profile, in Fourier space.

        Args:
            ks (array): The wavenumbers at which to compute the FT.
                        Units: :math:`Mpc^{-1}`
            z (float): Redshift to use.

        Returns:
            (array): Spherical FT of distribution corresponding to `ks`.
        '''
        rs = np.geomspace(1 / ks.max(), 2 * np.pi / ks.min(), nr)
        mass_weighted = self.mass_weighted_profile(rs, z,
                                                   nM=nM)

        return forward_spherical_fourier_transform(ks, rs, mass_weighted,
                                                   limit=limit, epsabs=epsabs,
                                                   epsrel=epsrel)

    def mass_weighted_profile(self, rs, z, nM=1000):
        '''
        Computes the mass weighted pressure profile:

        :math:`P_{mean}(r | z) = \int dM dn/dM b(M) P(r | M, z)`

        Args:
            rs (array): The radii at which to compute the mean pressure.
                        Units: :math:`Mpc`.
            z (float): Redshift to use.

        Returns:
            (array): Mean pressure for each r.
        '''

        Mmin = max(self.hmb.x.min(), self.hmf.x.min())
        Mmax = min(self.hmb.x.max(), self.hmf.x.max())

        Ms_halo_mass = np.geomspace(Mmin, Mmax, nM)
        lnMs_halo_mass = np.log(Ms_halo_mass)

        # If the one-halo and halo-mass tables are in different mass
        # definitions, we need to convert them.
        if self.halo_mass_def != self.one_halo_def:
            if self.one_halo_def == 'critical':
                Ms_one_halo = self.mean_to_critical(Ms_halo_mass)
            if self.one_halo_def == 'mean':
                Ms_one_halo = self.critical_to_mean(Ms_halo_mass)

        # First compute a grid of the profile in mass-radius
        profiles = np.zeros((nM, rs.size), dtype=np.double)
        for Mi, M in enumerate(Ms_one_halo):
            pmodel = self._profile_model(M, z, self.cosmo,
                                         *self._one_halo_args,
                                         **self._one_halo_kwargs)
            profiles[Mi] = pmodel.pressure(rs)

        # Perform the weighted integral
        # \int dx f(x) = \int d(lnx) x f(x)
        weighted_profiles = np.zeros_like(rs, dtype=np.double)
        for ri, r in enumerate(rs):
            igrnd = (Ms_halo_mass * self.hmf(Ms_halo_mass, z)
                                  * self.hmb(Ms_halo_mass, z)
                                  * profiles[:, ri])
            weighted_profiles[ri] = integrate_spline(lnMs_halo_mass, igrnd,
                                                     lnMs_halo_mass[0],
                                                     lnMs_halo_mass[-1])

        return weighted_profiles

    def convolved_y_FT(self, thetas, ks, z, da,
                       Xh=0.76,
                       miscent_offset=None,
                       sigma_beam=PLANCK_SIGMA_PSF,
                       nr=1000, nM=1000,
                       limit=1000,
                       epsabs_2h=1e-25,
                       epsabs_re=1e-9,
                       epsrel=1e-3):
        igrnds = self.two_halo_mass_integrand(ks, z, nr=nr, nM=nM,
                                              limit=limit,
                                              epsabs=epsabs_2h,
                                              epsrel=epsrel)
        Ps = self.P_lin(ks, z)

        # Due to the projection-slice thm, the 3D FT is the same as the
        # 2D FT of a projection of the 3D distribution.
        # Basically, we can skip a step by using the 3D FT of the 2halo as the
        # 2D FT of the projected 2halo
        y_conversion = _pressure_to_y * (2*Xh + 2) / (5*Xh + 3)
        fourier_2h = y_conversion * igrnds * Ps

        # Transverse radii - convert theta (in arcmin) to physical radii
        rs = da * (np.pi / 180) * (thetas / 60)

        # If a miscentering distance has been specified, apply it
        if miscent_offset is not None:
            fourier_2h *= sp.j0(miscent_offset * ks)

        # If a beam width has been specified, apply it
        if sigma_beam is not None:
            # Convert sigma from arcmin to physical units
            sigma_phys = da * (np.pi / 180) * (sigma_beam / 60)
            fourier_2h *= np.exp(-ks*ks*sigma_phys*sigma_phys/2)

        return inverse_circular_fourier_transform(rs, ks, fourier_2h,
                                                  limit=limit,
                                                  epsabs=epsabs_re,
                                                  epsrel=epsrel) / (1 + z)

    def convolved_y(self, da, z, rs_proj, rs_2h, ks,
                    Xh=0.76,
                    theta=15, n=200,
                    psf_sigma=PLANCK_SIGMA_PSF,
                    nM=1000, limit=1000,
                    epsabs_2h=1e-25,
                    epsabs_abel=1e-23,
                    epsrel_abel=1e-3):
        '''
        Computes the 2halo term as the projected-and-convolved Compton-y
        parameter.

        Args:
            da (float): Angular diameter distance at `z`.
            z (float): Redshift to use.
            rs_proj (array): The transverse distances at which to project.
                             Units: :math:`Mpc`
            rs_2h (array): The radii at which to compute the 3D 2-halo term.
                           (The projection is performed on an interpolation
                           table, this is the spacing of that interpolation
                           table.)
                           Units: :math:`Mpc`
            ks (array): Wavenumbers to use for computing 3D 2h term.
                        Units: :math:`Mpc^{-1}`.
            theta (float): Half-width of the convolved image, in arcmin.
            n (int): The side length of the image, in pixels. The convolution
                     is performed on an n x n image.
            psf_sigma (float): The standard deviation of the Gaussian beam. The
                               default is the Planck beam, in arcmin.

        Returns:
            (2-tuple of array): Pair of (rs, smoothed ys). `rs` runs from \
                                :math:`(theta / (n // 2)) / 2` to \
                                :math:`\\sqrt(2) theta`, and contains `n` \
                                points.
        '''
        projection = self.projected(rs_proj, rs_2h, ks, z,
                                    nM=nM, limit=limit,
                                    epsabs_2h=epsabs_2h,
                                    epsabs_abel=epsabs_abel,
                                    epsrel_abel=epsrel_abel)
        y_conversion = _pressure_to_y * (2*Xh + 2) / (5*Xh + 3)
        interp = interp1d(rs_proj, projection * y_conversion)

        def image_func(thetas):
            # Convert arcmin to physical transverse distance
            return interp(thetas * da / 60 * np.pi / 180)
        return create_convolved_profile(image_func,
                                        theta=theta, n=n, sigma=psf_sigma)
