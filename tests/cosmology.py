import numpy as np
import os
from scipy.interpolate import interp1d, interp2d

try:
    from colossus.halo import concentration, mass_defs
    from colossus.cosmology import cosmology
    have_colossus = True
except ImportError:
    have_colossus = False


def convert_mass(m, z, mdef_in='200c', mdef_out='200m',
                 concentration_model='diemer19', profile='nfw'):
    '''
    Converts between mass definitions.
    '''
    if not have_colossus:
        raise Exception('Colossus is necessary for mass definition changes')
    c = concentration.concentration(m, mdef_in, z,
                                    model=concentration_model,
                                    conversion_profile=profile)
    return mass_defs.changeMassDefinition(m, c, z, mdef_in, mdef_out,
                                          profile=profile)[0]


def get_cosmology(n):
    '''
    Loads computed data for a cosmology 1 <= n < 7.
    '''
    dir_ = os.path.join(os.path.dirname(__file__),
                        'data_for_testing', 'cosmology{}'.format(n))
    cosmo = {}

    # Load in parameters Omega_b, Omega_m
    with open(os.path.join(dir_, 'cosmological_parameters/values.txt')) as f:
        for line in f.readlines():
            name, val = line.split(' = ')
            if name in ['omega_b', 'omega_m', 'h0', 'n_s', 'sigma_8']:
                cosmo[name] = float(val)
                if name == 'h0':
                    h0 = float(val)
                if name == 'omega_m':
                    omega_m = float(val)

    # Set the colossus cosmology to this
    # Note: this assumes a flat \Lambda CDM
    if have_colossus:
        params = {'Om0': omega_m,
                  'Ob0': cosmo['omega_b'],
                  'H0': 100 * h0,
                  'sigma8': cosmo['sigma_8'],
                  'ns': cosmo['n_s']}
        cosmology.setCosmology('clusterToolkitText', params)

    def load_path(fname):
        return np.loadtxt(os.path.join(dir_, fname))

    # Load in table of comoving dist vs. redshift
    cosmo['z_chi'] = load_path('distances/z.txt')
    cosmo['chi'] = load_path('distances/d_m.txt')
    cosmo['d_a'] = load_path('distances/d_a.txt')
    cosmo['d_a_i'] = interp1d(cosmo['z_chi'], cosmo['d_a'])

    # Get halo mass function
    # NB: mass definition is _MEAN MASS OVERDENSITY_, not _CRITICAL MASS
    # OVERDENSITY_
    cosmo['hmf_z'] = load_path('mass_function/z.txt')
    cosmo['hmf_m'] = load_path('mass_function/m_h.txt') * omega_m / h0
    cosmo['hmf_dndm'] = load_path('mass_function/dndlnmh.txt') * h0**3

    # (Convert to dn/dm from dn/d(lnm))
    for i in range(cosmo['hmf_dndm'].shape[0]):
        cosmo['hmf_dndm'][i, :] /= cosmo['hmf_m']
    cosmo['hmf'] = interp2d(cosmo['hmf_m'], cosmo['hmf_z'], cosmo['hmf_dndm'])

    # Get the halo mass bias
    # As with HMF, NB: mass definition is _MEAN MASS OVERDENSITY_, not
    # _CRITICAL MASS OVERDENSITY_
    cosmo['hmb_z'] = load_path('tinker_bias_function/z.txt')
    cosmo['hmb_m'] = np.exp(load_path('tinker_bias_function/ln_mass_h.txt'))/h0
    cosmo['hmb_b'] = load_path('tinker_bias_function/bias.txt')
    cosmo['hmb'] = interp2d(cosmo['hmb_m'], cosmo['hmb_z'], cosmo['hmb_b'])

    # Get the matter power spectrum
    cosmo['P_lin_k'] = load_path('matter_power_lin/k_h.txt') * h0
    cosmo['P_lin_z'] = load_path('matter_power_lin/z.txt')
    cosmo['P_lin_p'] = load_path('matter_power_lin/p_k.txt') / (h0**3)
    cosmo['P_lin'] = interp2d(cosmo['P_lin_k'], cosmo['P_lin_z'],
                              cosmo['P_lin_p'])

    return cosmo
