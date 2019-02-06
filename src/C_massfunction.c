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
double G_at_M(double M, double*k, double*P, int Nk, double om, double d, double e, double f, double g){
  double G = 0.0;
  G_at_M_arr(&M, 1, k, P, Nk, om, d, e, f, g, &G);
  return G;
}

int G_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double om, double d, double e, double f, double g, double*G){
  double*sigma = (double*)malloc(sizeof(double)*NM);
  sigma2_at_M_arr(M, NM, k, P, Nk, om, sigma);
  int i;
  for(i = 0; i < NM; i++){
    sigma[i] = sqrt(sigma[i]);
  }
  G_at_sigma_arr(sigma, NM, d, e, f, g, G);
  free(sigma);
  return 0;
}

double G_at_sigma(double sigma, double d, double e, double f, double g){
  double G = 0.0;
  G_at_sigma_arr(&sigma, 1, d, e, f, g, &G);
  return G;
}

int G_at_sigma_arr(double*sigma, int Ns, double d, double e, double f, double g, double*G){
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
  return 0;
}

///////////////// dndM functions below ///////////////////

int dndM_sigma2_precomputed(double*M, double*sigma2, double*dsigma2dM, int NM, double Omega_m, double d, double e, double f, double g, double*dndM){
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
    dndM[i] = -rhom*Gsigma[i]*dsigma2dM[i]/(M[i]*sigma2[i]);
  }
  free(sigma);
  free(Gsigma);
  return 0;
}

double dndM_at_M(double M, double*k, double*P, int Nk, double om, double d, double e, double f, double g){
  double dndM = 0.0;
  dndM_at_M_arr(&M, 1, k, P, Nk, om, d, e, f, g, &dndM);
  return dndM;
}

int dndM_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double om, double d, double e, double f, double g, double*dndM){
  double*dsigma2dM = (double*)malloc(sizeof(double)*NM);
  double*sigma2 = (double*)malloc(sizeof(double)*NM);
  sigma2_at_M_arr(M, NM, k, P, Nk, om, sigma2);
  dsigma2dM_at_M_arr(M, NM, k, P, Nk, om, dsigma2dM);
  dndM_sigma2_precomputed(M, sigma2, dsigma2dM, NM, om, d, e, f, g, dndM);
  free(sigma2);
  free(dsigma2dM);
  return 0;
}

///////////////// derivatives of the MF below ///////////////////

int d2ndM2_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double Omega_m, double d, double e, double f, double g, double*d2ndM2){
  return 0;
}


///////////////// N in bin functions below ///////////////////

double n_in_bin(double Mlo, double Mhi, double*M, double*dndM, int NM){
  double edges[2] = { Mlo, Mhi };
  double N = 0.0;
  n_in_bins(edges, 2, M, dndM, NM, &N);
  return N;
}

int n_in_bins(double*edges, int Nedges, double*M, double*dndM, int NM, double*N){
  //Note: N is one element less long than edges
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, NM);
  gsl_spline_init(spline, M, dndM, NM);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  int i;
  for(i = 0; i < Nedges-1; i++){
    N[i] = gsl_spline_eval_integ(spline, edges[i], edges[i+1], acc);
  }
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  return 0;
}
