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

    bbps = pp.BBPSProfile(M, z, cosmo['omega_b'], cosmo['omega_m'], cosmo['h0'])

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
        bbps = pp.BBPSProfile(M200, z,
                              cosmo['omega_b'], cosmo['omega_m'], cosmo['h0'])
        for r, P in zip(bin_.r, bin_.P):
            ourP = bbps.pressure(r)
            # Convert to dimensionless `y` (see pp.projected_y_BBPS)
            ourP *= 1.61574202e+15
            # Make unitful
            # Adjust P_{gas} to P_{electron}
            ourP *= (2 * Xh + 2) / (5 * Xh + 3)
            assert abs((ourP - P) / P) < 5e-3


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
        bbps = pp.BBPSProfile(M200, z,
                              cosmo['omega_b'], cosmo['omega_m'], cosmo['h0'])
        for r, y in zip(bin_.r, bin_.y):
            oury = bbps.compton_y(r * cosmo['h0']**(2/3))
            # Convert to dimensionless `y` (see pp.projected_y_BBPS)
            oury *= cosmo['h0']**(8/3)
            assert abs((oury - y) / y) < 1e-2


def test_fourier():
    cosmo = get_cosmology(0)
    Ms = [2e13, 5e14, 5e14, 2e15]
    zs = [0.1, 0.2, 0.3]
    # Units: Mpc
    rmin, nr = 20, 500
    for m, z in product(Ms, zs):
        halo = pp.BBPSProfile(m, z,
                              cosmo['omega_b'], cosmo['omega_m'], cosmo['h0'])
        # Compare both Python and C versions of the Fourier transform
        ks, ps = halo.fourier_pressure(rmin, nr)
        ps_C = halo._C_fourier_pressure(ks)
        msk = ks < 10
        assert np.all(np.abs((ps[msk] - ps_C[msk]) / ps[msk]) < 1e-2)


def test_inverse_fourier():
    # Create our test halo
    M, z = 1e14, 0.2
    omega_b, omega_m = 0.04, 0.28
    h0 = 0.7
    halo = pp.BBPSProfile(M, z, omega_b, omega_m, h0)

    # Create our real and Fourier space evaluation grids
    rs = np.geomspace(0.01, 15, 100)
    ks = np.geomspace(0.01, 1000, 1000)

    true_pressure = halo.pressure(rs)

    # Convert the pressure profile to Fourier space, and back again
    Fs = halo._C_fourier_pressure(ks)
    epsabs = halo.pressure(halo.R_delta * 2)
    ifft_pressure = pp.inverse_spherical_fourier_transform(rs, ks, Fs,
                                                           epsabs=epsabs)

    # Check that the answer is right
    diff = np.abs(true_pressure - ifft_pressure)
    passes = ((diff / true_pressure) < 1e-1) | (diff < epsabs)
    assert np.all(passes)


def test_two_way_fourier():
    rs = np.geomspace(0.1, 5, 75)
    real = np.exp(-rs*rs / 2)
    ks = np.geomspace(1 / (5 * 2 * np.pi), 20, 75)

    epsabs = 1e-3
    ftd = pp.forward_spherical_fourier_transform(ks, rs, real, epsabs=epsabs)
    back = pp.inverse_spherical_fourier_transform(rs, ks, ftd, epsabs=epsabs)

    diff = np.abs(real - back)
    passes = ((diff / real) < 1e-2) | (diff < epsabs)
    assert np.all(passes)


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
        truth = 2 * np.sqrt(a*a - rs[rs<=a]**2)

        assert (np.abs(truth - result[rs<=a]) / truth <= 1e-2).all()
        assert (result[rs>a] == 0).all()


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


@pytest.mark.skip(reason='plots')
def test_convolution_convergence():
    cosmo = get_cosmology(0)
    ns = (800, 400, 200, 100, 50, 25)
    Ms = [1e13, 4e13, 8e13, 2e14, 6e14, 1e15]
    zs = [0.2]
    da_interp = interp1d(cosmo['z_chi'], cosmo['d_a'])
    for (M, z) in zip(Ms, zs):
        halo = pp.BBPSProfile(M, z,
                              cosmo['omega_b'], cosmo['omega_m'], cosmo['h0'])
        convolved = [halo.convolved_y(da=da_interp(z), n=n) for n in ns]
        fig, axs = plt.subplots(nrows=2, figsize=(8, 6), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1]})
        # Plot first
        axs[0].loglog()
        for i, (rs, vals) in enumerate(convolved):
            axs[0].plot(rs, vals, label='n = {}'.format(ns[i]))
        axs[0].legend()

        fid = interp1d(convolved[0][0], convolved[0][1], bounds_error=False)
        for i, (rs, vals) in enumerate(convolved):
            exp = fid(rs)
            axs[1].plot(rs, (vals - exp) / exp, label='n = {}'.format(ns[i]))
        axs[1].set_ylim((-0.02, 0.02))

    plt.show()


@pytest.mark.skip(reason='plots')
def test_2halo():
    cosmo = get_cosmology(0)
    rs = np.geomspace(0.1, 10, 20)
    ks = np.geomspace(0.1, 20, 100)
    z = 0.2
    # Create mass def interpolation table
    m200c_lo = convert_mass(cosmo['hmf_m'].min(), z, mdef_in='200m', mdef_out='200c')
    m200c_hi = convert_mass(cosmo['hmf_m'].max(), z, mdef_in='200m', mdef_out='200c')
    m200c = np.geomspace(m200c_lo * 0.95, m200c_hi * 1.05, cosmo['hmf_m'].size)
    m200m = convert_mass(m200c, z, mdef_in='200c', mdef_out='200m')
    # Do two-halo computation
    th = pp.TwoHaloProfile(cosmo['omega_b'], cosmo['omega_m'], cosmo['h0'],
                           cosmo['hmb_m'], cosmo['hmb_z'], cosmo['hmb_b'],
                           cosmo['hmf_m'], cosmo['hmf_z'], cosmo['hmf_dndm'],
                           cosmo['P_lin_k'], cosmo['P_lin_z'], cosmo['P_lin_p'],
                           m200m, m200c)
    two_halo = th.two_halo(rs, ks, z)
    plt.plot(rs, two_halo)
    plt.loglog()
    plt.show()
