from __future__ import division
from cluster_toolkit import pressure as pp
from cosmology import get_cosmology, convert_mass
from itertools import product
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytest
import random
from scipy.interpolate import interp1d


def sample_rMz():
    # Log-space masses between 1e14 and 5e15 Msun
    lnMlow, lnMhigh = math.log(1e14), math.log(5e15)
    M = math.exp(lnMlow + (lnMhigh - lnMlow) * random.random())
    # Evenly space radii between 0.01 and 3 Mpc
    r = 0.01 + 2.99*random.random()
    # Evenly space redshift between 0.01 and 1
    z = 0.01 + 0.99*random.random()
    return r, M, z


def do_test_projection_approximation(n, epsrel=1e-4):
    cosmo = get_cosmology(n)
    r, M, z = sample_rMz()

    bbps = pp.BBPSProfile(M, z, cosmo)

    # Compute the 'true' value
    expected = bbps._projected_pressure_real(r, cosmo['chi'], cosmo['z_chi'],
                                             epsrel=epsrel*0.01)

    # Compute the approximate value
    actual = bbps._projected_pressure(r, epsrel=epsrel*0.01)

    # Check that the relative difference is acceptable
    assert abs((expected - actual) / expected) < epsrel


def test_projection_approximation_0():
    for i in range(8):
        do_test_projection_approximation(0)


def test_projection_approximation_1():
    for i in range(8):
        do_test_projection_approximation(1)


def test_projection_approximation_2():
    for i in range(8):
        do_test_projection_approximation(2)


def test_projection_approximation_3():
    for i in range(8):
        do_test_projection_approximation(3)


def test_projection_approximation_4():
    for i in range(8):
        do_test_projection_approximation(4)


def test_projection_approximation_5():
    for i in range(8):
        do_test_projection_approximation(5)


def test_projection_approximation_6():
    for i in range(8):
        do_test_projection_approximation(6)


# TODO: change this to remove pandas dependency
def test_pressure():
    cosmo = get_cosmology(0)
    profiles = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                        'data_for_testing/y_profiles.csv'))
    for ibin in profiles.ibin.unique():
        bin_ = profiles[profiles.ibin == ibin]
        Xh = 0.76
        # TODO better way to get this?
        row1 = next(bin_.iterrows())[1]
        M200, z = row1.M200, row1.z
        bbps = pp.BBPSProfile(M200, z, cosmo)
        for r, P in zip(bin_.r, bin_.P):
            ourP = bbps.pressure(r)
            # Convert to dimensionless `y` (see pp.projected_y_BBPS)
            ourP *= 1.61574202e+15
            # Make unitful
            # Adjust P_{gas} to P_{electron}
            ourP *= (2 * Xh + 2) / (5 * Xh + 3)
            assert abs((ourP - P) / P) < 5e-3


# TODO: change this to remove pandas dependency
@pytest.mark.skip(reason='fiducial table used different integration method')
def test_y_projection():
    cosmo = get_cosmology(0)
    profiles = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                        'data_for_testing/y_profiles.csv'))
    for ibin in profiles.ibin.unique():
        bin_ = profiles[profiles.ibin == ibin]
        # TODO better way to get this?
        row1 = next(bin_.iterrows())[1]
        M200, z = row1.M200, row1.z
        bbps = pp.BBPSProfile(M200, z, cosmo)
        for r, y in zip(bin_.r, bin_.y):
            oury = bbps.compton_y(r)
            # Convert to dimensionless `y` (see pp.projected_y_BBPS)
            assert abs((oury - y) / y) < 1e-2


def test_pressure_fourier():
    cosmo = get_cosmology(0)
    Ms = [2e13, 5e14, 5e14, 2e15]
    zs = [0.1, 0.2, 0.3]
    # Units: Mpc
    rmin, nr = 20, 500
    for m, z in product(Ms, zs):
        halo = pp.BBPSProfile(m, z, cosmo)
        # Compare both Python and C versions of the Fourier transform
        ks, ps = halo._fft_fourier_pressure(rmin, nr)
        ps_C = halo.fourier_pressure(ks)
        msk = ks < 10
        assert np.all(np.abs((ps[msk] - ps_C[msk]) / ps[msk]) < 1e-2)


def test_inverse_3d_fourier():
    # Create our test halo
    M, z = 1e14, 0.2
    omega_b, omega_m = 0.04, 0.28
    h0 = 0.7
    halo = pp.BBPSProfile(M, z,
                          {'omega_b': omega_b, 'omega_m': omega_m, 'h0': h0})

    # Create our real and Fourier space evaluation grids
    rs = np.geomspace(0.01, 15, 100)
    ks = np.geomspace(0.01, 1000, 1000)

    true_pressure = halo.pressure(rs)

    # Convert the pressure profile to Fourier space, and back again
    Fs = halo.fourier_pressure(ks)
    epsabs = halo.pressure(halo.R_delta * 2)
    ifft_pressure = pp.inverse_spherical_fourier_transform(rs, ks, Fs,
                                                           epsabs=epsabs)

    # Check that the answer is right
    diff = np.abs(true_pressure - ifft_pressure)
    passes = ((diff / true_pressure) < 1e-1) | (diff < epsabs)
    assert np.all(passes)


def test_two_way_3d_fourier():
    '''
    Ensure that forward and backward 3d symmetric FTs are properly
    normalized inverses.
    '''
    rs = np.geomspace(0.1, 5, 75)
    real = np.exp(-rs*rs / 2)
    ks = np.geomspace(1 / (5 * 2 * np.pi), 20, 75)

    epsabs = 1e-3
    ftd = pp.forward_spherical_fourier_transform(ks, rs, real, epsabs=epsabs)
    back = pp.inverse_spherical_fourier_transform(rs, ks, ftd, epsabs=epsabs)

    diff = np.abs(real - back)
    passes = ((diff / real) < 1e-2) | (diff < epsabs)
    assert np.all(passes)


def test_2d_fourier():
    '''
    Ensure that forward and backward 2d symmetric FTs are properly
    normalized inverses.
    '''
    def gaussian_2d(r, sigma):
        const = 1 / (2 * np.pi * sigma * sigma)
        return const * np.exp(-r*r / (2 * sigma * sigma))

    rs = np.geomspace(0.01, 10, 2000)
    ks = np.copy(rs)

    epsabs = 1e-5
    epsrel = 1e-3

    for sigma in [0.5, 1, 2]:
        real = gaussian_2d(rs, sigma)
        ftd = pp.forward_circular_fourier_transform(ks, rs, real,
                                                    epsabs=epsabs,
                                                    epsrel=epsrel)

        # Inverse should be e^(-k^2 \sigma^2 / 2)
        expected_fsp = np.exp(-ks*ks * sigma*sigma / 2)
        diff_fsp = np.abs(expected_fsp - ftd)
        passes_fsp = ((diff_fsp / expected_fsp) < epsrel) | (diff_fsp < epsabs)
        assert np.all(passes_fsp)

        back = pp.inverse_circular_fourier_transform(rs, ks, ftd,
                                                     epsabs=epsabs,
                                                     epsrel=epsrel)

        # Make sure we reconstruct original
        diff = np.abs(real - back)
        passes = ((diff / real) < epsrel) | (diff < epsabs)
        assert np.all(passes)


def test_proj_slice():
    '''
    According to the projection-slice theorem, the 2d fourier transform of an
    Abel transform (LOS projection) should be equivalent to a slice of the
    3D fourier transform.

    In the language of the functions here, this means that:

        forward_circular_fourier_transform(abel_transform(f)) ==
            forward_spherical_fourier_transform(f)

    The proj.-sl. thm is a mathematical fact, so this test is a sanity check
    that our functions are properly normalized, and work well enough to use it.
    '''
    def gaussian(r, sigma):
        '''
        A 3d gaussian
        '''
        const = (2 * np.pi)**(1.5) * sigma**3
        return np.exp(-r*r / 2 * (sigma * sigma)) / const

    rs = np.geomspace(0.01, 20, 1000)
    r_trans = np.geomspace(0.01, 10, 1000)
    ks = np.geomspace(0.1, 10, 100)

    epsabs = 1e-3
    epsrel = 1e-3
    for sigma in [0.5, 1, 2]:
        g = gaussian(rs, sigma)
        abel_transformed = pp.abel_transform(rs, g, r_trans)
        ft_2d = pp.forward_circular_fourier_transform(ks, r_trans, abel_transformed,
                                                      epsabs=epsabs, epsrel=epsrel)

        ft_3d = pp.forward_spherical_fourier_transform(ks, rs, g,
                                                       epsabs=epsabs)

        diff = np.abs(ft_2d - ft_3d)
        assert np.all((diff < 10*epsabs) | ((diff / np.abs(ft_3d)) < 5*epsrel))


def test_spline_integration():
    xs = np.arange(0.5, 10, 0.5)

    # Test a simple quadratic
    ys = xs**2 - 3 * xs
    for a, b in [(1, 2), (2, 3), (3, 5), (2, 9.3), (0.7, 8.5)]:
        res = pp.integrate_spline(xs, ys, a, b)
        truth = (b**3 / 3 - 3 * b**2 / 2) - (a**3 / 3 - 3 * a**2 / 2)
        assert (abs(res - truth) / truth) < 1e-4

        # Test integrating over log
        res = pp.integrate_spline(np.log(xs), xs*ys, np.log(a), np.log(b))
        assert (abs(res - truth) / truth) < 1e-4

    # Test a simple cubic
    ys = xs**3 - xs**2 + xs
    for a, b in [(1, 2), (2, 3), (3, 5), (2, 9.3), (0.7, 8.5)]:
        res = pp.integrate_spline(xs, ys, a, b)
        end = b**4 / 4 - b**3 / 3 + b**2 / 2
        start = a**4 / 4 - a**3 / 3 + a**2 / 2
        truth = end - start
        assert (abs(res - truth) / truth) < 1e-3

        # Test integrating over log
        res = pp.integrate_spline(np.log(xs), xs*ys, np.log(a), np.log(b))
        assert (abs(res - truth) / truth) < 1e-2


def test_abel_transform():
    # Abel(tophat from 0 to a) = 2 sqrt(a*a - x*x)
    xs = np.array([0, 0.5, 1])
    ys = np.array([1, 1, 1])

    rs = np.linspace(0, 2, 100)

    for a in np.geomspace(0.1, 10, 10):
        result = pp.abel_transform(a*xs, ys, rs)
        truth = 2 * np.sqrt(a*a - rs[rs <= a]**2)

        assert (np.abs(truth - result[rs <= a]) / truth <= 1e-2).all()
        assert (result[rs > a] == 0).all()

    # Abel tophat of Gaussian
    epsabs = 1e-18
    xs = np.linspace(0.001, 5, 1000)
    rs = np.geomspace(0.1, 2, 50)

    for sigma in np.geomspace(0.1, 1, 10):
        ys = np.exp(-xs*xs/sigma/sigma)
        result = pp.abel_transform(xs, ys, rs, epsabs=epsabs)
        truth = sigma * np.sqrt(np.pi) * np.exp(-rs*rs/sigma/sigma)

        diff = (np.abs(truth - result) / truth)
        cond = (diff < 1e-3) | (np.abs(result) < epsabs)
        assert cond.all()


def test_convolution_methods():
    '''
    There are two convolution methods - one is an analytical method, performing
    the convolution in fourier space and using GSL integration methods to
    convert to and from fourier space. The other creates a "fake image" on a
    grid, and uses astropy methods to perform the convolution. We want to test
    that they are equivalent.
    '''

    epsabs = 1e-8
    epsrel = 1e-3
    Ms = [1e14, 5e14, 1e15]
    zs = [0.1, 0.2, 0.3]

    cosmo = get_cosmology(0)
    thetas = np.geomspace(0.1, 30, 100)
    for M in Ms:
        for z in zs:
            da = cosmo['d_a_i'](z)
            halo = pp.BBPSProfile(M, z, cosmo)

            integration_method = halo.convolved_y(thetas, da,
                                                  epsabs=epsabs,
                                                  epsrel=epsrel)
            grid_method = halo.convolved_y_fft(thetas, da,
                                               epsabs=epsabs,
                                               epsrel=epsrel)

            diff = integration_method - grid_method
            frac_diff = np.abs(diff / integration_method)

            assert np.all((np.abs(diff) < 2*epsabs) | (frac_diff < 0.01))


@pytest.mark.skip(reason='plots')
def test_2halo():
    cosmo = get_cosmology(0)
    rs = np.geomspace(0.1, 10, 20)
    ks = np.geomspace(0.1, 20, 100)
    z = 0.2
    # Create mass def interpolation table
    m200c_lo = convert_mass(cosmo['hmf_m'].min(), z,
                            mdef_in='200m', mdef_out='200c')
    m200c_hi = convert_mass(cosmo['hmf_m'].max(), z,
                            mdef_in='200m', mdef_out='200c')
    m200c = np.geomspace(m200c_lo * 0.95, m200c_hi * 1.05, cosmo['hmf_m'].size)
    m200m = convert_mass(m200c, z, mdef_in='200c', mdef_out='200m')
    # Do two-halo computation
    cosmo['hmf_f'] = cosmo['hmf_dndm']
    th = pp.TwoHaloProfile(cosmo, m200m, m200c)
    two_halo = th.two_halo(rs, ks, z)
    plt.plot(rs, two_halo)
    plt.loglog()
    plt.show()


@pytest.mark.skip(reason='plots')
def test_convolved_2h():
    cosmo = get_cosmology(0)
    rs_2h = np.geomspace(0.01, 40, 200)
    rs_proj = np.geomspace(0.05, 20, 60)
    ks = np.geomspace(0.1, 20, 100)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True,
                            gridspec_kw={'height_ratios': [2, 1]})
    axs[0].loglog()

    for z in [0.1, 0.2, 0.4]:
        # Create mass def interpolation table
        m200c_lo = convert_mass(cosmo['hmf_m'].min(), z,
                                mdef_in='200m', mdef_out='200c')
        m200c_hi = convert_mass(cosmo['hmf_m'].max(), z,
                                mdef_in='200m', mdef_out='200c')
        m200c = np.geomspace(m200c_lo * 0.95, m200c_hi * 1.05,
                             cosmo['hmf_m'].size)
        m200m = convert_mass(m200c, z, mdef_in='200c', mdef_out='200m')
        # Do two-halo computation
        cosmo['hmf_f'] = cosmo['hmf_dndm']
        th = pp.TwoHaloProfile(cosmo, m200m, m200c)

        da = cosmo['d_a_i'](z)
        # Compare the mock-image-fft method to the analytic Fourier-space
        # convolution
        thetas = rs_proj * (180 / np.pi) * (60 / da)
        two_halo_fft_convolved = th.convolved_y(thetas, da, z, rs_2h, ks)
        two_halo_psl_convolved = th.convolved_y_FT(thetas, da, z, ks)
        axs[0].plot(thetas, two_halo_fft_convolved,
                    label='FFT method, z={}'.format(z))
        axs[0].plot(thetas, two_halo_psl_convolved,
                    label='Proj.-slice metho, z={}'.format(z))
        axs[0].loglog()
        axs[1].plot(thetas,
                    (two_halo_fft_convolved - two_halo_psl_convolved)
                    / two_halo_psl_convolved,
                    label='Fractional residual z={}'.format(z))

    axs[0].legend()
    axs[0].set_ylabel('Compton-$y$')
    axs[1].legend()
    axs[1].set_ylim((-0.02, 0.02))
    axs[1].set_ylabel('Fractional residual')
    axs[1].set_xlabel('Radius from cluster ($Mpc$)')
    plt.show()
