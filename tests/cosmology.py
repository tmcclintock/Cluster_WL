import numpy as np
import os
from scipy.interpolate import interp2d


def get_cosmology(n):
    '''
    Loads computed data for a cosmology 1 <= n < 7.
    '''
    dir_ = os.path.join(os.path.dirname(__file__),
                        'data_for_testing', 'cosmology{}'.format(n))
    cosmo = {}

    # Load in table of comoving dist vs. redshift
    cosmo['z_chi'] = np.loadtxt(os.path.join(dir_, 'distances/z.txt'))
    cosmo['chi'] = np.loadtxt(os.path.join(dir_, 'distances/d_m.txt'))
    cosmo['d_a'] = np.loadtxt(os.path.join(dir_, 'distances/d_a.txt'))

    # Get halo mass function
    cosmo['hmf_z'] = np.loadtxt(os.path.join(dir_, 'mass_function/z.txt'))
    cosmo['hmf_m'] = np.loadtxt(os.path.join(dir_, 'mass_function/m_h.txt'))
    cosmo['hmf_dndm'] = np.loadtxt(os.path.join(dir_, 'mass_function/dndlnmh.txt'))

    # (Convert to dn/dm from dn/d(lnm))
    for i in range(cosmo['hmf_dndm'].shape[0]):
        cosmo['hmf_dndm'][i, :] /= cosmo['hmf_m']
    cosmo['hmf'] = interp2d(cosmo['hmf_m'], cosmo['hmf_z'], cosmo['hmf_dndm'])

    # Get the halo mass bias
    cosmo['hmb_z'] = np.loadtxt(os.path.join(dir_, 'tinker_bias_function/z.txt'))
    cosmo['hmb_m'] = np.exp(np.loadtxt(os.path.join(dir_, 'tinker_bias_function/ln_mass.txt')))
    cosmo['hmb_b'] = np.loadtxt(os.path.join(dir_, 'tinker_bias_function/bias.txt'))
    cosmo['hmb'] = interp2d(cosmo['hmb_m'], cosmo['hmb_z'], cosmo['hmb_b'])

    # Get the matter power spectrum
    cosmo['P_lin_k'] = np.loadtxt(os.path.join(dir_, 'matter_power_lin/k_h.txt'))
    cosmo['P_lin_z'] = np.loadtxt(os.path.join(dir_, 'matter_power_lin/z.txt'))
    cosmo['P_lin_p'] = np.loadtxt(os.path.join(dir_, 'matter_power_lin/p_k.txt'))
    cosmo['P_lin'] = interp2d(cosmo['P_lin_k'], cosmo['P_lin_z'], cosmo['P_lin_p'])

    # Load in parameters Omega_b, Omega_m
    with open(os.path.join(dir_, 'cosmological_parameters/values.txt')) as f:
        for line in f.readlines():
            name, val = line.split(' = ')
            if name in ['omega_b', 'omega_m', 'h0']:
                cosmo[name] = float(val)

    return cosmo
