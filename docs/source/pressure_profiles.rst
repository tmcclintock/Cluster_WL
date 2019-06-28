******************************
Pressure Profiles
******************************

This module contains routines for calcalating the Battaglia 2012 (`ads <https://ui.adsabs.harvard.edu/abs/2012ApJ...758...75B/abstract>`_, hereafter BBPS) pressure profile.

Battaglia Profile
==================

The Battaglia profile (BBPS equation 10) is a 3D pressure profile given by:

.. math::
   P_{\rm BBPS}(r) = P_{\Delta} P_0 \Big(\frac{r}{r_{\Delta} \cdot x_c}\Big)^\gamma \Big[1 + \Big(\frac{r}{r_{\Delta} \cdot x_c}\Big)^\alpha\Big]^{-\beta}

The free parameters are :math:`P_0, \gamma, x_c, \alpha`, and :math:`\beta`. The thermal pressure :math:`P_{200}` and the radius :math:`r_{\Delta}` are cosmology- and mass-dependent parameters given in BBPS. Due to a strong degeneracy between the parameters, BBPS fixed math:`\alpha = 1` and :math:`\gamma = -0.3`

.. note::
   The pressure profiles can use :math:`\Delta\neq 200`. Note, however, that the BBPS best fit was calibrated for :math:`\Delta = 200`.

To use this, you would do:

.. code::

   from cluster_toolkit import pressure_profile as pp
   # Mass in Msun
   mass, redshift = 1e14, 0.3
   # Plausible cosmology. Used in P_\Delta calculation
   Omega_b, Omega_m, Omega_lambda = 0.044, 0.3, 0.7
   # Radius is Mpc
   radius = 0.5
   P_bbps = pp.projected_P_BBPS(radii, mass, redshift, Omega_b, Omega_m, Omega_lambda)
