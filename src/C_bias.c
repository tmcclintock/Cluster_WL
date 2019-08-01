/** @file C_bias.c
 *  @brief Halo bias functions.
 * 
 *  These functions are the halo bias for different
 *  types of inputs including halo mass, halo radius,
 *  and halo peak height.
 *  
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_bias.h"
#include "C_peak_height.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define delta_c 1.686 //Critical collapse density
#define rhocrit 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are Msun h^2/Mpc^3

///////////BIAS FUNCTIONS///////////
/**
 * \brief Compute the bias of a halo with peak height nu for an array
 * of peak heights.
 *
 * This is the Tinker et al. (2010) bias model.
 */
void bias_at_nu_arr(double*nu, int Nnu, int delta, double*bias){
  double y = log10(delta);
  double xp = exp(-1.0*pow(4./y,4.));
  double A = 1.+0.24*y*xp, a = 0.44*y-0.88;
  double B = 0.183, b = 1.5;
  double C = 0.019+0.107*y+0.19*xp, c = 2.4;
  int i;
  for(i = 0; i < Nnu; i++)
    bias[i] = 1 - A*pow(nu[i],a)/(pow(nu[i],a)+pow(delta_c,a)) + B*pow(nu[i],b) + C*pow(nu[i],c);
}

/**
 * \brief Compute the bias of a halo with Lagrangian radius R for an array
 * of radii.
 *
 * This is the Tinker et al. (2010) bias model.
 */
void bias_at_R_arr(double*R, int NR, int delta, double*k, double*P, int Nk,
		  double*bias){
  double*nu = malloc(sizeof(double)*NR);
  nu_at_R_arr(R, NR, k, P, Nk, nu);
  bias_at_nu_arr(nu, NR, delta, bias);
  free(nu);
}

/**
 * \brief Compute the bias of a halo with mass Mfor an array
 * of masses.
 *
 * This is the Tinker et al. (2010) bias model.
 */
void bias_at_M_arr(double*M, int NM, int delta, double*k, double*P, int Nk,
		  double Omega_m, double*bias){
  double*nu = malloc(sizeof(double)*NM);
  nu_at_M_arr(M, NM, k, P, Nk, Omega_m, nu);
  bias_at_nu_arr(nu, NM, delta, bias);
  free(nu);
}

/**
 * \brief Compute the bias of a halo with peak height nu for an array
 * of peak heights, with arbitrary free parameters in a Tinker-like model.
 *
 */
void bias_at_nu_arr_FREEPARAMS(double*nu, int Nnu, int delta,
			      double A, double a, double B, double b,
			      double C, double c, double*bias){
  int i;
  for(i = 0; i < Nnu; i++)
    bias[i] = 1 - A*pow(nu[i],a)/(pow(nu[i],a)+pow(delta_c,a)) + B*pow(nu[i],b) + C*pow(nu[i],c);
}

/******
Derivatives of the bias
 *****/
/**
 * \brief Compute (d/dnu)bias for an array of peak heights.
 *
 * This is the Tinker et al. (2010) bias model.
 */
void dbiasdnu_at_nu_arr(double*nu, int Nnu, int delta, double*deriv){
  double y = log10(delta);
  double xp = exp(-1.0*pow(4./y,4.));
  double A = 1.+0.24*y*xp, a = 0.44*y-0.88;
  double B = 0.183, b = 1.5;
  double C = 0.019+0.107*y+0.19*xp, c = 2.4;
  int i;
  for(i = 0; i < Nnu; i++)
    deriv[i] = a*A*pow(delta_c, a) * pow(nu[i], a-1)*
      pow(pow(nu[i], a) + pow(delta_c, a), -2) +
      B*b*pow(nu[i], b-1) + C*c*pow(nu[i], c-1);
}

/**
 * \brief Compute (d/dnu)bias for an array of peak heights.
 *
 * This is the Tinker et al. (2010) bias model.
 */
void dbiasdM_at_M_arr(double*M, int NM, int delta, double*k, double*P, int Nk,
		  double Omega_m, double*deriv){
  double*nu = malloc(sizeof(double)*NM);
  double*dbiasdnu = malloc(sizeof(double)*NM);
  double*sigma2 = malloc(sizeof(double)*NM);
  double*dsigma2dM = malloc(sizeof(double)*NM);
  nu_at_M_arr(M, NM, k, P, Nk, Omega_m, nu);
  dbiasdnu_at_nu_arr(nu, NM, delta, dbiasdnu);
  sigma2_at_M_arr(M, NM, k, P, Nk, Omega_m, sigma2);
  dsigma2dM_at_M_arr(M, NM, k, P, Nk, Omega_m, dsigma2dM);
  //Note: dnu/dsigma2 = -delta_c /(2*sigma^3)
  int i;
  for(i = 0; i < NM; i ++){
    deriv[i] = -delta_c*0.5 * pow(sigma2[i], -1.5) * dbiasdnu[i] *
      dsigma2dM[i];
  }
  free(nu);
  free(dbiasdnu);
  free(sigma2);
  free(dsigma2dM);
}
