/** @file C_density.c
 *  @brief Halo density profiles.
 *
 *  These functions are different models of density profiles for halos.
 *
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_density.h"
#include "C_xi.h"

#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_gamma.h"
#include <math.h>
#include <stdio.h>

#define rhomconst 2.77533742639e+11
//#define rhomconst 2.775808e+11 //old value used. Disregard
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are Msun h^2/Mpc^3

/** @brief The NFW density profile.
 * 
 *  NEED TO UPDATE THIS COMMENT
 *
 *  The NFW density profile of a halo a distance R from the center,
 *  assuming the halo has a given mass and concentration. It works 
 *  with any overdensity parameter and arbitrary matter fraction.
 *  This function calls xi_nfw_at_r().
 * 
 *  @param r Distance from the center of the halo in Mpc/h comoving.
 *  @param M Halo mass in Msun/h.
 *  @param c Concentration.
 *  @delta Halo overdensity.
 *  @Omega_m Matter fraction.
 *  @return NFW halo density.
 */
void calc_rho_nfw(double*r, int Nr, double Mass, double conc, int delta, double Omega_m, double*rho_nfw){
  int i;
  double rhom = Omega_m*rhomconst;//Msun h^2/Mpc^3
  calc_xi_nfw(r, Nr, Mass, conc, delta, Omega_m, rho_nfw); //rho_nfw actually holds xi_nfw here
  for(i = 0; i < Nr; i++){
    rho_nfw[i] = rhom*(1+rho_nfw[i]);
  }
}

void calc_rho_einasto(double*R, int NR, double Mass, double rhos, double conc, double alpha, int delta, double Omega_m, double*rho_einasto){
  double rhom = rhomconst*Omega_m; //SM h^2/Mpc^3
  double Rdelta = pow(Mass/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  double rs = Rdelta / conc; //compute scale radius from concentration
  if (rhos < 0)
    rhos = rhos_einasto_at_M(Mass, conc, alpha, delta, Omega_m);
  int i;
  double x;
  for(i = 0; i < NR; i++){
    x = 2./alpha * pow(R[i]/rs, alpha);
    rho_einasto[i] = rhos * exp(-x);
  }
}
