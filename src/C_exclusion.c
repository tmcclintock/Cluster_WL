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
#define sqrtPI   1.77245385091 //sqrt(M_PI), for speed
#define pi4      12.5663706144 //pi*4
#define Nk 1000 //number of wavenumbers
#define Nrm 1000 //number of radial sampling points
#define rm_min 0.0001 //Mpc/h minimum of the radial splines
#define rm_max 10000. //Mpc/h maximum of the radial splines

int xihm_exclusion_at_r_arr(double*r, int Nr, double M, double c,
			    double rt, double beta,
			    double Ma, double ca, double Mb, double cb,
			    double bias, double*ximm, int delta, double Omega_m,
			    int scheme, double*xihm){
  int i;
  double*xi_1h  = malloc(sizeof(double)*Nr);
  double*xi_2h  = malloc(sizeof(double)*Nr);
  double*xi_2hc  = malloc(sizeof(double)*Nr);
  double*xi_ct1  = malloc(sizeof(double)*Nr);
  double*xi_ct2  = malloc(sizeof(double)*Nr);
  xi_1h_at_r_arr(r, Nr, M, c, rt, beta, delta, Omega_m, xi_1h);//Done
  xi_2h_at_r_arr(r, Nr, bias, ximm, xi_2h);//Done
  xi_2hcorrection_at_r_arr(r, Nr, M, rt, Mb, cb, bias, delta, Omega_m, xi_2hc);
  for(i = 0; i < Nr; i++){
    xi_2h[i] += xi_2hc[i]; //add on the first correction term
  }
  xi_correction_at_r_arr(r, Nr, M, rt, beta, Ma, ca, delta, Omega_m, scheme, xi_ct1);
  xi_correction_at_r_arr(r, Nr, M, rt, beta, Mb, cb, delta, Omega_m, scheme, xi_ct2);

  //Resum all terms
  for(i = 0; i < Nr; i++){
    xihm[i] = xi_1h[i] + xi_2h[i] - xi_ct1[i]*xi_2h[i] - xi_ct2[i];
    //if (i < 0){
    //  printf("%d %.3e   %.3e   %.3e  %.3e \n",i,xi_1h[i],xi_2h[i],xi_ct1[i],xi_ct2[i]);
    //}
  }
  free(xi_1h);
  free(xi_2h);
  free(xi_2hc);
  free(xi_ct1);
  free(xi_ct2);
  return 0;
}

/*Convolution of the NFW profile with the truncation term.
 */
int ut_conv_thetat_at_r_arr(double*r, int Nr, double rt, double conc, double*out_arr){
  //Compute ut(k)
  int i;
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
  
  //Compute (u*t)^2 in fourier space
  double rc = rt/conc;
  double*mu = malloc(sizeof(double)*Nk);
  double*Put = malloc(sizeof(double)*Nk);
  double*PutPut = malloc(sizeof(double)*Nk);
  double prefactor = 1./(log(1+conc) - conc/(1+conc)); //save computation time
  for(i = 0; i < Nk; i++){
    mu[i] = rc * k[i]; // k[i] * rt2/c2
    Put[i] = prefactor * (cos(mu[i]) * (gsl_sf_Ci(mu[i]*(1+conc)) - gsl_sf_Ci(mu[i])) +
			  sin(mu[i]) * (gsl_sf_Si(mu[i]*(1+conc)) - gsl_sf_Si(mu[i])) -
			  sin(mu[i]*conc) / (mu[i]*(1+conc)));
    PutPut[i] = Put[i]*Put[i];
  }
  //Transform to realspace
  calc_xi_mm(r, Nr, k, PutPut, Nk, out_arr, 500, 1e-2);
  //Free everything
  free(mu);
  free(Put);
  free(PutPut);
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

int xi_2h_at_r_arr(double*r, int Nr, double bias, double*ximm, double*xi2h){
  int i;
  for(i = 0; i < Nr; i++){
    xi2h[i] = bias*ximm[i];
  }
  return 0;
}

int xi_2hcorrection_at_r_arr(double*r, int Nr, double M1, double rt,
			     double M2, double conc2, double bias,
			     int delta, double Omega_m, double*xi_2hc){
  double rhom = Omega_m * rhocrit; //SM h^2/Mpc^3 comoving
  int i;
  //Compute rt2
  double rt2 = rt * pow(M2/M1, 0.33333333333);// rt * rdelta2/rdelta1;
  ut_conv_thetat_at_r_arr(r, Nr, rt2, conc2, xi_2hc);
  for(i = 0; i < Nr; i++){
    xi_2hc[i] *= -M2 / rhom; //xi_c1 is now unitless
  }
  return 0;
}

/////////////////////////////////////////////
//Support functions for the correction term//
/////////////////////////////////////////////
double I_term(double r, double R, double re, double beta){
  //Note (sqrt2/beta*sqrtPI) has been pulled out
  double v2 = (R+r-re)/(re*beta*sqrt2);
  double v1 = (R-r-re)/(re*beta*sqrt2);
  double expv2 = exp(-v2*v2);
  double expv1 = exp(-v1*v1);
  double erfv2 = gsl_sf_erf(v2);
  double erfv1 = gsl_sf_erf(v1);
  double erfcv2 = gsl_sf_erfc(v2);
  double erfcv1 = gsl_sf_erfc(v1);
  return (erfv2/4 + erfcv2*(v2*v2/2 + v2/(beta*sqrt2)) - expv2/sqrtPI*(v2/2+1)) -
    (erfv1/4 + erfcv1*(v1*v1/2 + v1/(beta*sqrt2)) - expv1/sqrtPI*(v1/2+1));
}

int xi_correction_at_r_arr(double*r, int Nr, double M1, double rt, double beta,
			   double M2, double c2, int delta, double Omega_m,
			   int scheme, double*xict){
  int i;//, j;
  //static gsl_spline*spline = NULL;
  //static gsl_interp_accel*acc = NULL;

  //Step 0 - make radial points to sample the profiles more easily
  //as well as all other static variables
  static int init_flag = 0;
  static double*rm = NULL;
  static double*lnrm = NULL;
  static double*k = NULL;
  if (init_flag == 0){
    init_flag = 1;
    rm = malloc(sizeof(double)*Nrm);
    lnrm = malloc(sizeof(double)*Nrm);
    double log10rm_min = log10(rm_min);
    double log10rm_max = log10(rm_max);
    double dlogrm = (log10rm_max - log10rm_min)/(Nrm-1); //step size in log10(rm)
    for(i = 0; i < Nrm; i++){
      rm[i] = pow(10, log10rm_min + i*dlogrm);
      lnrm[i] = log(rm[i]);
    }
    //spline = gsl_spline_alloc(gsl_interp_cspline, Nrm);
    //acc = gsl_interp_accel_alloc();
    k = malloc(sizeof(double)*Nk);
    double dlogk = 6./(Nk-1); //step size in log10(k)
    for(i = 0; i < Nk; i++){
      k[i] = pow(10, -2. + i*dlogk);
    }
  }

  //Start with the truncated 1halo term for M2
  //note: u = (1+xi) * rhom / M
  double*utr = malloc(sizeof(double)*Nrm);
  calc_xi_nfw(rm, Nrm, M2, c2, delta, Omega_m, utr);//contains xi_nfw, not rho_nfw for now
  double rhom = Omega_m * rhocrit; //SM h^2/Mpc^3 comoving
  double rdelta1 = pow(M1/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  double rdelta2 = pow(M2/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  double rt2 = rt * rdelta2/rdelta1;
  double rs2 = rdelta2/c2; //scale factor of M2
  //upper limit for integral of M2 density profile
  double x2 = rt2 / rs2;
  //Calculate the mass integrated out to rt2 of M2's density profile
  double Mrt2 = M2*(log(1+x2)-x2/(1+x2))/(log(1+c2)-c2/(1+c2));
  //Calculate the ut(r) profile
  double*utrtt = malloc(sizeof(double)*Nrm);
  theta_erfc_at_r_arr(rm, Nrm, rt, beta, utrtt);
  for( i = 0; i < Nrm; i++){
    utr[i] = (utr[i]+1)*utrtt[i]*rhom/Mrt2; //correctly normalized; units are h^3/Mpc^3
  }
  
  //Get the exclusion radius
  double re = r_exclusion(rt, rt2, scheme);
  //Get the exclusion profile
  double*thetat = malloc(sizeof(double)*Nrm);
  theta_erfc_at_r_arr(rm, Nrm, re, beta, thetat);

  //Take fourier transforms of both utr and thetat
  double*utr_k = malloc(sizeof(double)*Nk);
  double*thetat_k = malloc(sizeof(double)*Nk);
  double*utr_thetat_k = malloc(sizeof(double)*Nk);
  calc_xi_mm(k, Nk, rm, utr, Nrm, utr_k, 500, 1e-5); //missing 8pi^3
  calc_xi_mm(k, Nk, rm, thetat, Nrm, thetat_k, 500, 1e-5); //missing 8pi^3
  for(i = 0; i < Nk; i++){
    utr_thetat_k[i] = 64*pow(M_PI, 6)*utr_k[i]*thetat_k[i];
    //if (i < 10){
    //  printf("%e   %e   \n", utr[i], thetat[i]);
    //}
  }
  //Fourier transform back
  calc_xi_mm(r, Nr, k, utr_thetat_k, Nk, xict, 500, 5e-3);
  
  //Free everything
  free(utr);
  free(utrtt);
  free(thetat);
  free(utr_k);
  free(thetat_k);
  free(utr_thetat_k);
  //free(Ir);
  //free(integrand);
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

////////////////////////////////////////////////
//////////////////OLD CODE BELOW////////////////
////////////////////////////////////////////////

/*
int xi_correction_at_r_arr_OLDVERSION(double*r, int Nr, double M, double rt,
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
*/

/*
int ut_conv_thetat_at_r_arr_OLDVERSION(double*r, int Nr, double M1, double rt,
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

double theta_erfc_at_r(double r, double rt, double beta){
  double thetas = 0.0;
  theta_erfc_at_r_arr(&r, 1, rt, beta, &thetas);
  return thetas;
}
*/
