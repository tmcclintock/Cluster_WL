/** @file C_boostfactors.c
 *  @brief Models of boost factor distributions.
 *   
 *  This file contains models of boost factors.
 *  All functions implemented here are simple
 *  analytic functions of radius that include
 *  various numbers of free parameters.
 * 
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */
#include "C_boostfactors.h"

#include <math.h>
#include <stdio.h>

/**
 *\brief Boost factor assuming a projected NFW profile at an array of radii.
 *
 * Used in McClintock et al. (2018).
 */
void boost_nfw_at_R_arr(double*R, int NR, double B0, double Rs,
		       double*boost){
  int i;
  double x2m1;
  double sqx2m1;
  for(i = 0; i < NR; i++){
    x2m1 = R[i]*R[i]/(Rs*Rs)-1;
    if (x2m1 > 0){ //x > 1
      sqx2m1 = sqrt(x2m1); //sqrt(x*x-1)
      boost[i] = 1. + B0/x2m1 * (1-atan(sqx2m1)/sqx2m1);
    } else if (x2m1 < 0){ // x < 1
      sqx2m1 = sqrt(-x2m1); //sqrt(1-x*x)
      boost[i] =  1. + B0/x2m1 * (1-atanh(sqx2m1)/sqx2m1);
    }  else{ // x = 1
      boost[i] = 1;
    }
  }
}

/**
 *\brief Boost factor assuming a power law at an array of radii.
 *
 * Used in Melchior et al. (2017).
 */
void boost_powerlaw_at_R_arr(double*R, int NR, double B0, double Rs,
			    double alpha, double*boost){
  int i;
  for(i = 0; i < NR; i++){
    boost[i] = 1+B0*pow(R[i]/Rs,alpha);
  }
}
