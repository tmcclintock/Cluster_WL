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
 * \brief Compute the bias of a halo of a given peak height, nu.
 *
 * This is the header for computing the bias of a single halo
 * of a known peak height. This function just calls the vectorized
 * version, but otherwise does nothing.
 *
 * This is the Tinker et al. (2010) bias model.
 */
double bias_at_nu(double nu, int delta){
  double bias = 0.0;
  bias_at_nu_arr(&nu, 1, delta, &bias);
  return bias;
}

/**
 * \brief Compute the bias of a halo of a given Lagrangian radius, R.
 *
 * This is the header for computing the bias of a single halo
 * with a Lagrangian radius, R. This function just calls the vectorized
 * version, but otherwise does nothing.
 *
 * This is the Tinker et al. (2010) bias model.
 */
double bias_at_R(double R, int delta, double*k, double*P, int Nk){
  double bias = 0.0;
  bias_at_R_arr(&R, 1, delta, k, P, Nk, &bias);
  return bias;
}

/**
 * \brief Compute the bias of a halo of mass, M.
 *
 * This is the header for computing the bias of a halo
 * of mass M. This calls bias_at_nu(). 
 * This is the Tinker et al. (2010) bias model.
 */
double bias_at_M(double M, int delta, double*k, double*P, int Nk, double om){
  return bias_at_nu(nu_at_M(M, k, P, Nk, om), delta);
}

/**
 * \brief Compute the bias of a halo with peak height nu for an array
 * of peak heights.
 *
 * This is the Tinker et al. (2010) bias model.
 */
int bias_at_nu_arr(double*nu, int Nnu, int delta, double*bias){
  double y = log10(delta);
  double xp = exp(-1.0*pow(4./y,4.));
  double A = 1.+0.24*y*xp, a = 0.44*y-0.88;
  double B = 0.183, b = 1.5;
  double C = 0.019+0.107*y+0.19*xp, c = 2.4;
  int i;
  for(i = 0; i < Nnu; i++)
    bias[i] = 1 - A*pow(nu[i],a)/(pow(nu[i],a)+pow(delta_c,a)) + B*pow(nu[i],b) + C*pow(nu[i],c);
  return 0;
}

/**
 * \brief Compute the bias of a halo with Lagrangian radius R for an array
 * of radii.
 *
 * This is the Tinker et al. (2010) bias model.
 */
int bias_at_R_arr(double*R, int NR, int delta, double*k, double*P, int Nk,
		  double*bias){
  double*nu = malloc(sizeof(double)*NR);
  nu_at_R_arr(R, NR, k, P, Nk, nu);
  bias_at_nu_arr(nu, NR, delta, bias);
  free(nu);
  return 0;
}

/**
 * \brief Compute the bias of a halo with mass Mfor an array
 * of masses.
 *
 * This is the Tinker et al. (2010) bias model.
 */
int bias_at_M_arr(double*M, int NM, int delta, double*k, double*P, int Nk,
		  double Omega_m, double*bias){
  double*nu = malloc(sizeof(double)*NM);
  nu_at_M_arr(M, NM, k, P, Nk, Omega_m, nu);
  bias_at_nu_arr(nu, NM, delta, bias);
  free(nu);
  return 0;
}

/**
 * \brief Compute the bias of a halo with peak height nu for an array
 * of peak heights, with arbitrary free parameters in a Tinker-like model.
 *
 */
int bias_at_nu_arr_FREEPARAMS(double*nu, int Nnu, int delta,
			      double A, double a, double B, double b,
			      double C, double c, double*bias){
  int i;
  for(i = 0; i < Nnu; i++)
    bias[i] = 1 - A*pow(nu[i],a)/(pow(nu[i],a)+pow(delta_c,a)) + B*pow(nu[i],b) + C*pow(nu[i],c);
  return 0;
}
