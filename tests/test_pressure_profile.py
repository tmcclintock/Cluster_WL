from cluster_toolkit import pressure_profile as pp
import math
import numpy as np
import os
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
    return (omega_b, omega_m, omega_lambda), z_chis, chis


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
    (Omega_b, Omega_m, Omega_lambda), z_chis, chis = get_cosmology(n)
    r, M, z = sample_rMz()

    # Compute the 'true' value
    expected = pp.projected_P_BBPS_real(r, M, z,
                                        Omega_b, Omega_m,
                                        chis, z_chis,
                                        epsrel=epsrel*0.01)

    # Compute the approximate value
    actual = pp.projected_P_BBPS(r, M, z,
                                 Omega_b, Omega_m,
                                 epsrel=epsrel*0.01)

    # Check that the relative difference is acceptable
    assert abs((expected - actual) / expected) < epsrel


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
