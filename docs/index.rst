******************************
Cluster Toolkit Documentation
******************************

Cluster Toolkit is a Python package specifically built for calculating weak lensing signals from galaxy clusters and cluster cosmology.
It consists of a Python front end wrapped around a well optimized back end in C, merged with `cffi <https://cffi.readthedocs.io/en/latest/>`_.
The core functionality of the package includes:

 * 3D density functions :math:`\rho(r)`
 * 3D correlation functions :math:`\xi(r)`
 * Halo bias models :math:`b(M)`
 * Projected density and differential profiles :math:`\Sigma(R)` and :math:`\Delta\Sigma`
 * Radially averaged profiles :math:`\overline{\Delta\Sigma}`
 * Boost factor models :math:`\mathcal{B} = (1-f_{\rm cl})^{-1}`
 * Miscentering effects on projected profiles :math:`R_{\rm mis}`
 * Halo mass functions :math:`\frac{dn}{dM}(M,z)`
 * Mass-concentration relations :math:`M-c`
 * Sunyaev-Zel'dovich (SZ) cluster signals :math:`Y_{SZ}` (in development)
 * Cluster magnification :math:`\kappa(\theta)` and shear profiles :math:`\gamma(\theta)` (in development)

The source code is publically available at https://github.com/tmcclintock/cluster_toolkit.

.. note::
   Unless stated otherwise, all distances are assumed to be :math:`{\rm Mpc}/h` comoving and masses :math:`{\rm M}_\odot/h`. Furthermore, power spectra :math:`P(k)` must be in units of :math:`({\rm Mpc}/h)^3` with wavenumber :math:`k` in units of :math:`h/{\rm Mpc}`.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/installation
   source/density_profiles
   source/pressure_profiles
   source/correlation_functions
   source/halo_bias
   source/projected_density_profiles
   source/averaged_profiles
   source/boostfactors
   source/miscentering
   source/massfunction
   source/concentration

.. toctree::
   :maxdepth: 1
   :caption: Reference

   Cluster Toolkit Reference/API<api/modules>
   How to use CLASS<source/using_class>
   How to use CAMB<source/using_camb>
   Frequently Asked Questions<source/FAQ>
