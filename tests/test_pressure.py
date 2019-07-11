from cluster_toolkit import pressure as pp
import math
import numpy as np
import os
import pandas as pd
import pytest
import random


def get_cosmology(n):
    '''
    Loads computed data for a cosmology 1 <= n < 7.
    '''
    dir_ = os.path.join(os.path.dirname(__file__),
                        'data_for_testing', 'cosmology{}'.format(n))
    # Load in table of comoving dist vs. redshift
    z_chis = np.loadtxt(os.path.join(dir_, 'distances/z.txt'))
    chis = np.loadtxt(os.path.join(dir_, 'distances/d_m.txt'))
    # Load in parameters Omega_b, Omega_m, Omega_lambda
    with open(os.path.join(dir_, 'cosmological_parameters/values.txt')) as f:
        for line in f.readlines():
            name, val = line.split(' = ')
            if name == 'omega_b':
                omega_b = float(val)
            if name == 'omega_m':
                omega_m = float(val)
            if name == 'omega_lambda':
                omega_lambda = float(val)
            if name == 'h0':
                h0 = float(val)
    return (omega_b, omega_m, omega_lambda, h0), z_chis, chis


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
    (Omega_b, Omega_m, Omega_lambda, h0), z_chis, chis = get_cosmology(n)
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
    (Omega_b, Omega_m, Omega_lambda, h0), z_chis, chis = get_cosmology(0)
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
    (Omega_b, Omega_m, Omega_lambda, h0), z_chis, chis = get_cosmology(0)
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
            # TODO check projected
