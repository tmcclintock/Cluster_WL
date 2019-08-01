#include "C_exclusion.h"
#include "C_xi.h"
#include "C_peak_height.h"
#include "C_power.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf.h" //for erfc
#include "gsl/gsl_errno.h"
#include <math.h>
#include <stdio.h>

#define rhocrit 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are SM h^2/Mpc^3
#define sqrt2    1.4142135623730951 //sqrt(2), for computational efficiency
#define invsqrt2 0.7071067811865475 //1/sqrt(2), for computational efficiency
#define sqrtPI   1.77245385091 //sqrt(M_PI), for speed
#define pi6_64 61528.9083888 // pi^6 * 64
#define pi4      12.5663706144 //pi*4
#define Nk 1000 //number of wavenumbers
#define Nrm 1000 //number of radial sampling points
#define rm_min 0.0001 //Mpc/h minimum of the radial splines
#define rm_max 10000. //Mpc/h maximum of the radial splines

void xi_hm_exclusion_at_r_arr(double*r, int Nr,
			    double M, double c, double alpha,
			    double rt, double D,
			    double r_eff, double D_eff,
			    double r_A, double r_B, double D_ex,
			    double bias, double*ximm, int delta,
			    double Omega_m, double*xihm){
  int i;
  double*xi_1h  = malloc(sizeof(double)*Nr);
  double*xi_2h  = malloc(sizeof(double)*Nr);
  double*xi_C  = malloc(sizeof(double)*Nr);
  xi_1h_at_r_arr(r, Nr, M, c, alpha, rt, D, delta, Omega_m, xi_1h);
  xi_2h_at_r_arr(r, Nr, r_eff, D_eff, bias, ximm, xi_2h);
  xi_C_at_r_arr(r, Nr, r_A, r_B, D_ex, xi_2h, xi_C);
  //Resum all terms
  for(i = 0; i < Nr; i++){
    xihm[i] = xi_1h[i] + xi_2h[i] + xi_C[i];
  }
  free(xi_1h);
  free(xi_2h);
  free(xi_C);
}

void xi_1h_at_r_arr(double*r, int Nr, double M, double c, double alpha,
		   double rt, double D, int delta, double Omega_m,
		   double*xi_1h){
  int i;
  double*theta = malloc(sizeof(double)*Nr);
  calc_xi_einasto(r, Nr, M, -1, c, alpha, delta, Omega_m, xi_1h);
  theta_erfc_at_r_arr(r, Nr, rt, D, theta);
  for(i = 0; i < Nr; i++){
    xi_1h[i] = (1+xi_1h[i]) * theta[i];
  }
  free(theta);
}

void xi_2h_at_r_arr(double*r, int Nr, double r_eff, double D_eff,
		   double bias, double*ximm, double*xi2h){
  int i;
  double*theta_eff  = malloc(sizeof(double)*Nr);
  theta_erfc_at_r_arr(r, Nr, r_eff, D_eff, theta_eff);
  for(i = 0; i < Nr; i++){
    xi2h[i] = (1-theta_eff[i]) * bias * ximm[i];
  }
  free(theta_eff);
}

void xi_C_at_r_arr(double*r, int Nr, double r_A, double r_B, double D,
		  double*xi_2h, double*xi_C){
  int i;
  double*theta_A  = malloc(sizeof(double)*Nr);
  double*theta_B  = malloc(sizeof(double)*Nr);
  theta_erfc_at_r_arr(r, Nr, r_A, D, theta_A);
  theta_erfc_at_r_arr(r, Nr, r_B, D, theta_B);
  for(i = 0; i < Nr; i++){
    xi_C[i] = -theta_A[i] * xi_2h[i] - theta_B[i];
  }
  free(theta_A);
  free(theta_B);
}


/////////////////////////////////////////////
//Support functions for the correction term//
/////////////////////////////////////////////

int theta_erfc_at_r_arr(double*r, int Nr, double rt, double D,
			double*theta){
  int i;
  double invD_rt = 1./(D*rt);
  for(i = 0; i < Nr; i++){
    gsl_sf_result erf_result;
    int rc = gsl_sf_erfc_e((r[i]-rt) * invD_rt * invsqrt2,
                           &erf_result);
    if (rc != GSL_SUCCESS)
      return rc;
    theta[i] = 0.5*erf_result.val;
  }
  return GSL_SUCCESS;
}

///////////////////////////////////
//Currently unused exclusion term//
///////////////////////////////////
double r_exclusion(double r1, double r2, int scheme){
  switch(scheme){
  case 0 : //max(r1, r2)
    if (r1 > r2) return r1;
    else         return r2;
  case 1: // (r1^3 + r2^3)^(1/3)
    return pow(r1*r1*r1 + r2*r2*r2, 0.333333333333);
  case 2: //sum(r1, r2)
    return r1+r2;
  }
  return -1; //this is bad, and it means the user passed in a BS scheme
}
