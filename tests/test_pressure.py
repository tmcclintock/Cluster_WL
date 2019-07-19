from cluster_toolkit import pressure as pp
from itertools import product
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pytest
import random
from scipy.interpolate import interp1d


def get_cosmology(n):
    '''
    Loads computed data for a cosmology 1 <= n < 7.
    '''
    dir_ = os.path.join(os.path.dirname(__file__),
                        'data_for_testing', 'cosmology{}'.format(n))
    # Load in table of comoving dist vs. redshift
    z_chis = np.loadtxt(os.path.join(dir_, 'distances/z.txt'))
    chis = np.loadtxt(os.path.join(dir_, 'distances/d_m.txt'))
    d_a = np.loadtxt(os.path.join(dir_, 'distances/d_a.txt'))
    # Load in parameters Omega_b, Omega_m
    with open(os.path.join(dir_, 'cosmological_parameters/values.txt')) as f:
        for line in f.readlines():
            name, val = line.split(' = ')
            if name == 'omega_b':
                omega_b = float(val)
            if name == 'omega_m':
                omega_m = float(val)
            if name == 'h0':
                h0 = float(val)
    return (omega_b, omega_m, h0), z_chis, chis, d_a


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
    (Omega_b, Omega_m, h0), z_chis, chis, d_as = get_cosmology(n)
    r, M, z = sample_rMz()

    bbps = pp.BBPSProfile(M, z, Omega_b, Omega_m)

    # Compute the 'true' value
    expected = bbps._projected_pressure_real(r, chis, z_chis,
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
    (Omega_b, Omega_m, h0), z_chis, chis, d_as = get_cosmology(0)
    profiles = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                        'data_for_testing/y_profiles.csv'))
    for ibin in profiles.ibin.unique():
        bin_ = profiles[profiles.ibin == ibin]
        Xh = 0.76
        # TODO better way to get this?
        row1 = next(bin_.iterrows())[1]
        M200, z = row1.M200, row1.z
        bbps = pp.BBPSProfile(M200, z, Omega_b, Omega_m)
        for r, P in zip(bin_.r, bin_.P):
            ourP = bbps.pressure(r * h0**(2/3))
            # Convert to dimensionless `y` (see pp.projected_y_BBPS)
            ourP *= 1.61574202e+15
            # Make unitful
            ourP *= h0**(8/3)
            # Adjust P_{gas} to P_{electron}
            ourP *= (2 * Xh + 2) / (5 * Xh + 3)
            assert abs((ourP - P) / P) < 5e-3


@pytest.mark.skip(reason='fiducial table used different integration method')
def test_y_projection():
    (Omega_b, Omega_m, h0), z_chis, chis, d_as = get_cosmology(0)
    profiles = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                        'data_for_testing/y_profiles.csv'))
    for ibin in profiles.ibin.unique():
        bin_ = profiles[profiles.ibin == ibin]
        # TODO better way to get this?
        row1 = next(bin_.iterrows())[1]
        M200, z = row1.M200, row1.z
        bbps = pp.BBPSProfile(M200, z, Omega_b, Omega_m)
        for r, y in zip(bin_.r, bin_.y):
            oury = bbps.compton_y(r * h0**(2/3))
            # Convert to dimensionless `y` (see pp.projected_y_BBPS)
            oury *= h0**(8/3)
            assert abs((oury - y) / y) < 1e-2


def test_fourier():
    (Omega_b, Omega_m, h0), z_chis, chis, d_as = get_cosmology(0)
    Ms = [2e13, 5e14, 5e14, 2e15]
    zs = [0.1, 0.2, 0.3]
    # Units: Mpc
    rmin, nr = 20, 500
    for m, z in product(Ms, zs):
        halo = pp.BBPSProfile(m, z, Omega_b, Omega_m)
        # Compare both Python and C versions of the Fourier transform
        ks, ps = halo.fourier_pressure(rmin, nr)
        ps_C = halo._C_fourier_pressure(ks)
        msk = ks < 10
        assert np.all(np.abs((ps[msk] - ps_C[msk]) / ps[msk]) < 1e-2)


def test_inverse_fourier():
    # Create our test halo
    M, z = 1e14, 0.2
    Omega_b, Omega_m = 0.04, 0.28
    halo = pp.BBPSProfile(M, z, Omega_b, Omega_m)

    # Create our real and Fourier space evaluation grids
    rs = np.exp(np.linspace(np.log(0.01), np.log(15), 100))
    ks = np.exp(np.linspace(np.log(0.01), np.log(1000), 1000))

    true_pressure = halo.pressure(rs)

    # Convert the pressure profile to Fourier space, and back again
    Fs = halo._C_fourier_pressure(ks)
    epsabs = halo.pressure(halo.R_delta * 2)
    ifft_pressure = pp.inv_spherical_fourier_transform(rs, ks, Fs,
                                                       epsabs=epsabs)

    # Check that the answer is right
    diff = np.abs(true_pressure - ifft_pressure)
    passes = ((diff / true_pressure) < 1e-1) | (diff < epsabs)
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




@pytest.mark.skip()
def test_convolution_convergence():
    (Omega_b, Omega_m, h0), z_chis, chis, d_as = get_cosmology(0)
    ns = (800, 400, 200, 100, 50, 25)
    Ms = [1e13, 4e13, 8e13, 2e14, 6e14, 1e15]
    zs = [0.2]
    da_interp = interp1d(z_chis, d_as)
    for (M, z) in zip(Ms, zs):
        halo = pp.BBPSProfile(M, z, Omega_b, Omega_m)
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
