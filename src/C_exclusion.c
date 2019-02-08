#include "C_exclusion.h"
#include "C_xi.h"
#include "C_peak_height.h"
#include "C_power.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_sf.h" //for erfc
#include "gsl/gsl_errno.h"
#include <math.h>
#include <stdio.h>

#define rhocrit 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are SM h^2/Mpc^3
#define sqrt2    1.4142135623730951 //sqrt(2), for computational efficiency
#define invsqrt2 0.7071067811865475 //1/sqrt(2), for computational efficiency
#define pi4      12.5663706144 //pi*4
#define Nk 1000 //number of wavenumbers

int xihm_exclusion_at_r_arr(double*r, int Nr, double M, double c,
			    double rt, double beta,
			    double Ma, double ca, double Mb, double cb,
			    double bias, double*ximm, int delta, double Omega_m,
			    int scheme, double*xihm){
  int i;
  double*xi_1h  = malloc(sizeof(double)*Nr);
  double*xi_2h  = malloc(sizeof(double)*Nr);
  double*xi_c   = malloc(sizeof(double)*Nr);
  xi_1h_at_r_arr(r, Nr, M, c, rt, beta, delta, Omega_m, xi_1h);
  xi_2h_at_r_arr(r, Nr, bias, ximm, xi_2h);
  xi_correction_at_r_arr(r, Nr, M, rt, Ma, ca, Mb, cb, bias, ximm, delta, Omega_m, scheme, xi_c);
  //Resum all terms
  for(i = 0; i < Nr; i++){
    xihm[i] = xi_1h[i] + xi_2h[i] + xi_c[i];
  }
  free(xi_1h);
  free(xi_2h);
  free(xi_c);

  return 0;
}

int ut_conv_thetat_at_r_arr(double*r, int Nr, double M1, double rt,
			    double M2, double c2,
			    int delta, double Omega_m, int scheme,
			    double*out_arr){  
  double rhom = Omega_m * rhocrit; //SM h^2/Mpc^3 comoving
  int i;
  //Compute r_delta for M1 and M2
  double rdelta1 = pow(M1/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  double rdelta2 = pow(M2/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  double rt1 = rt; //can get rid of this variable
  double ratio1 = rt1/rdelta1;
  double rt2 = ratio1*rdelta2;
  double rc = rt2/c2; //scale radius of M2 halo
  double re = r_exclusion(rt1, rt2, scheme);

  //Create the wavenumbers to do the convolution over
  static int init_flag = 0;
  static double*k = NULL;
  if (init_flag == 0){
    init_flag = 1;
    k = malloc(sizeof(double)*Nk);
    double dlogk = 8./(Nk-1); //step size in log10(k)
    for(i = 0; i < Nk; i++){
      k[i] = pow(10, -4. + i*dlogk);
    }
  }
  //Calculate mu = k*rc
  //Compute Fourier transform of u_t
  //Compute Fourier transform of thetat
  double prefactor = 1./(log(1+c2) - c2/(1+c2)); //save computation time
  double*mu         = malloc(sizeof(double)*Nk);
  double*Put        = malloc(sizeof(double)*Nk);
  double*Pthetat    = malloc(sizeof(double)*Nk);
  double*PutPthetat = malloc(sizeof(double)*Nk);
  for(i = 0; i < Nk; i++){
    mu[i] = rc * k[i]; // k[i] * rt2/c2
    Put[i] = prefactor * (cos(mu[i]) * (gsl_sf_Ci(mu[i]*(1+c2)) - gsl_sf_Ci(mu[i])) +
			  sin(mu[i]) * (gsl_sf_Si(mu[i]*(1+c2)) - gsl_sf_Si(mu[i])) -
			  sin(mu[i]*c2) / (mu[i]*(1+c2)));
    Pthetat[i] = pi4*(sin(k[i] * re) - k[i] * re * cos(k[i] * re))/(k[i]*k[i]*k[i]);
    PutPthetat[i] = Put[i] * Pthetat[i];
  }
  //Transform the fourier space profiles back to realspace, for high and low scales
  double*out_low  = malloc(sizeof(double)*Nr);
  double*out_high = malloc(sizeof(double)*Nr);
  calc_xi_mm(r, Nr, k, PutPthetat, Nk, out_low,  8800, 1e-6);
  calc_xi_mm(r, Nr, k, PutPthetat, Nk, out_high, 7000, 1e-5);
  for(i = 0; i < Nr; i++){
    if (r[i] < re) out_arr[i] = out_low[i];
    else           out_arr[i] = out_high[i];
  }
  //Free everything
  free(mu);
  free(Put);
  free(Pthetat);
  free(PutPthetat);
  free(out_low);
  free(out_high);
  return 0;
}

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

int xi_1h_at_r_arr(double*r, int Nr, double M, double c,
		   double rt, double beta, int delta, double Omega_m,
		   double*xi_1h){
  int i;
  double*thetas = malloc(sizeof(double)*Nr);
  calc_xi_nfw(r, Nr, M, c, delta, Omega_m, xi_1h);
  theta_erfc_at_r_arr(r, Nr, rt, beta, thetas);
  for(i = 0; i < Nr; i++){
    xi_1h[i] = (1+xi_1h[i]) * thetas[i];
  }
  free(thetas);
  return 0; //success
}

double xi_1h_at_r(double r, double M, double c,
		  double rt, double beta, int delta, double Omega_m){
  double xi_1h = 0.0;
  xi_1h_at_r_arr(&r, 1, M, c, rt, beta, delta, Omega_m, &xi_1h);
  return xi_1h;
}

int xi_2h_at_r_arr(double*r, int Nr, double bias, double*ximm, double*xi2h){
  int i;
  for(i = 0; i < Nr; i++){
    xi2h[i] = bias*ximm[i];
  }
  return 0;
}

double xi_2h_at_r(double r, double bias, double ximm){
  double xi_2h = 0.0;
  xi_2h_at_r_arr(&r, 1, bias, &ximm, &xi_2h);
  return xi_2h;
}

int xi_correction_at_r_arr(double*r, int Nr, double M, double rt,
			   double Ma, double ca, double Mb, double cb,
			   double bias, double*ximm, int delta, double Omega_m,
			   int scheme, double*xi_c){
  int i;
  double*ut_conv_thetat_a = malloc(sizeof(double)*Nr);
  double*ut_conv_thetat_b = malloc(sizeof(double)*Nr);
  ut_conv_thetat_at_r_arr(r, Nr, M, rt, Ma, ca, delta, Omega_m, scheme, ut_conv_thetat_a);
  ut_conv_thetat_at_r_arr(r, Nr, M, rt, Mb, cb, delta, Omega_m, scheme, ut_conv_thetat_b);
  for(i = 0; i < Nr; i++){
    xi_c[i]  = -bias*ut_conv_thetat_a[i]*ximm[i] - ut_conv_thetat_b[i];
  }
  free(ut_conv_thetat_a);
  free(ut_conv_thetat_b);
  return 0;
}

int theta_erfc_at_r_arr(double*r, int Nr, double rt, double beta, double*theta){
  int i;
  double invbeta = 1./beta;
  for(i = 0; i < Nr; i++){
    theta[i] = 0.5*gsl_sf_erfc((r[i]-rt) * invbeta * invsqrt2);
  }
  return 0;
}

double theta_erfc_at_r(double r, double rt, double beta){
  double thetas = 0.0;
  theta_erfc_at_r_arr(&r, 1, rt, beta, &thetas);
  return thetas;
}
