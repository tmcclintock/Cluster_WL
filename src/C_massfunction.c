/** @file C_massfunction.c
 *  @brief Halo mass functions.
 *  
 *  These functions calculate the halo mass function.
 *  
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_massfunction.h"
#include "C_peak_height.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_sf.h"
#include "gsl/gsl_spline.h"
#include <math.h>

#define rhocrit 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are SM h^2/Mpc^3
#define del 1e-6

///////////////// G multiplicity function///////////////////
void G_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double om, double d, double e, double f, double g, double*G){
  double*sigma = (double*)malloc(sizeof(double)*NM);
  sigma2_at_M_arr(M, NM, k, P, Nk, om, sigma);
  int i;
  for(i = 0; i < NM; i++){
    sigma[i] = sqrt(sigma[i]);
  }
  G_at_sigma_arr(sigma, NM, d, e, f, g, G);
  free(sigma);
}

void G_at_sigma_arr(double*sigma, int Ns, double d, double e, double f, double g, double*G){
  //Compute the prefactor B
  double d2 = 0.5*d;
  double gamma_d2 = gsl_sf_gamma(d2);
  double f2 = 0.5*f;
  double gamma_f2 = gsl_sf_gamma(f2);
  double B = 2./(pow(e, d)*pow(g, -d2)*gamma_d2 + pow(g, -f2)*gamma_f2);
  int i;
  for(i = 0; i < Ns; i++){
    G[i] = B*exp(-g/(sigma[i]*sigma[i]))*(pow(sigma[i]/e, -d)+pow(sigma[i], -f));
  }
}

///////////////// dndM functions below ///////////////////

void dndM_sigma2_precomputed(double*M, double*sigma2, double*dsigma2dM, int NM, double Omega_m, double d, double e, double f, double g, double*dndM){
  //This function exists to make emulator tests faster with sigma^2(M) precomputed.
  double rhom = Omega_m*rhocrit; //normalization coefficient
  double*sigma = (double*)malloc(sizeof(double)*NM);
  double*Gsigma = (double*)malloc(sizeof(double)*NM);
  int i;
  for(i = 0; i < NM; i++){
    sigma[i] = sqrt(sigma2[i]);
  }
  G_at_sigma_arr(sigma, NM, d, e, f, g, Gsigma);
  for(i = 0; i < NM; i++){
    dndM[i] = -rhom*Gsigma[i]*dsigma2dM[i]/(2*M[i]*sigma2[i]);
  }
  free(sigma);
  free(Gsigma);
}

void dndM_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double om, double d, double e, double f, double g, double*dndM){
  double*dsigma2dM = (double*)malloc(sizeof(double)*NM);
  double*sigma2 = (double*)malloc(sizeof(double)*NM);
  sigma2_at_M_arr(M, NM, k, P, Nk, om, sigma2);
  dsigma2dM_at_M_arr(M, NM, k, P, Nk, om, dsigma2dM);
  dndM_sigma2_precomputed(M, sigma2, dsigma2dM, NM, om, d, e, f, g, dndM);
  free(sigma2);
  free(dsigma2dM);
}

///////////////// derivatives of the MF below ///////////////////

int d2ndM2_at_M_arr(double*M, int NM, double*k, double*P, int Nk,
		    double Omega_m, double d, double e, double f,
		    double g, double*d2ndM2){
  return 0;
}

///////////////// N in bin functions below ///////////////////
int n_in_bins(double*edges, int Nedges, double*M, double*dndM, int NM, double*N){
  //Note: N is one element less long than edges
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, NM);
  gsl_spline_init(spline, M, dndM, NM);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();

  if (!spline || !acc)
      return GSL_ENOMEM;

  int i, rc = GSL_SUCCESS;
  for(i = 0; i < Nedges-1; i++){
    rc = gsl_spline_eval_integ_e(spline, edges[i], edges[i+1], acc, &N[i]);
    if (rc != GSL_SUCCESS)
        break;
  }

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  return rc;
}
