#include "C_peak_height.h"
#include "C_power.h"

#include "gsl/gsl_errno.h"
#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"

#include <math.h>
#include <stdio.h>

#define ABSERR 0.0
#define RELERR 1e-8
#define delta_c 1.686 //Critical collapse density
#define rhocrit 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are Msun h^2/Mpc^3
#define workspace_size 8000
#define KEY 6 //Used for GSL QAG function

typedef struct integrand_params{
  gsl_spline *spline;
  gsl_interp_accel *acc;
  double r;
  double*kp; //pointer to wavenumbers
  double*Pp; //pointer to P(k) array
  int Nk; //length of k and P arrays
}integrand_params;

//Redefined these, even though they appear in C_bias.c
double M_to_R(double M, double Omega_m){
  //Lagrangian radius Mpc/h
  return pow(M/(1.33333333333*M_PI*rhocrit*Omega_m), 0.3333333333);
}

double R_to_M(double R, double Omega_m){
  //Lagrangian mass Msun/h
  return R*R*R*1.33333333333*M_PI*rhocrit*Omega_m;
}

double dRdM_at_M(double M, double Omega_m){
  //Derivative of Lagrangian radius w.r.t. Mass [Msun/h] in Mpc/Msun
  return pow(36*M_PI*rhocrit*Omega_m, -0.33333333333) *
    pow(M,-0.6666666666666);
}

double sigma2_integrand(double lk, void*params){
  //Integrand for calculating sigma^2
  integrand_params pars = *(integrand_params*)params;
  double k = exp(lk);
  double x = k*pars.r;
  //double P = get_P(x, pars.r, pars.kp, pars.Pp, pars.Nk, pars.spline, pars.acc);
  double P = gsl_spline_eval(pars.spline, k, pars.acc);
  double w = (sin(x)-x*cos(x))*3.0/(x*x*x); //Window function
  return k*k*k*P*w*w;
}

double dsigma2dR_integrand(double lk, void*params){
  //Integrand for calculating dsigma^2/dR, where R is the lagrangian radius
  integrand_params pars = *(integrand_params*)params;
  double k = exp(lk);
  double x = k*pars.r;
  double P = gsl_spline_eval(pars.spline, k, pars.acc);
  double sx = sin(x);
  double cx = cos(x);
  double w = (sx-x*cx)*3.0/(x*x*x); //Window function
  double dwdR = k*3*((x*x-3)*sx + 3*x*cx)/(x*x*x*x); //Derivative of w
  return k*k*k*P*w*dwdR;
}

///////////// linar matter variance functions /////////////

double sigma2_at_R(double R, double*k, double*P, int Nk){
  //sigma^2(R) for a single value of R
  double result;
  sigma2_at_R_arr(&R, 1, k, P, Nk, &result);
  return result;
}

double sigma2_at_M(double M, double*k, double*P, int Nk, double om){
  //sigma^2(M) for a single value of M
  double R=M_to_R(M, om);
  return sigma2_at_R(R, k, P, Nk);
}

int sigma2_at_R_arr(double*R, int NR,  double*k, double*P, int Nk, double*s2){
  //sigma^2(R) for an array of R
  //Initialize GSL things and the integrand structure.
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline,Nk);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  gsl_integration_workspace*workspace = gsl_integration_workspace_alloc(workspace_size);

  // Handle allocation failure
  if (!spline || !acc || !workspace)
      return GSL_ENOMEM;

  gsl_function F;
  integrand_params params;
  double lkmin = log(k[0]);
  double lkmax = log(k[Nk-1]);
  double result,abserr;
  double denom_inv = 1./(2*M_PI*M_PI);
  int i, rc;

  rc = gsl_spline_init(spline,k,P,Nk);
  params.spline = spline;
  params.acc = acc;
  params.kp = k;
  params.Pp = P;
  params.Nk = Nk;
  F.function = &sigma2_integrand;
  F.params = &params;

  for(i = 0; i < NR; i++){
    if (rc != GSL_SUCCESS)
      break;
    params.r = R[i];
    rc = gsl_integration_qag(&F, lkmin, lkmax, ABSERR, RELERR,
			workspace_size, KEY, workspace, &result, &abserr);
    s2[i] = result * denom_inv; //divide by 2pi^2
  }
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);
  return rc;
}

int sigma2_at_M_arr(double*M, int NM,  double*k, double*P, int Nk,
		    double Omega_m, double*s2){
  int i;
  double*R = (double*)malloc(sizeof(double)*NM);
  for(i = 0; i < NM; i++){
    R[i] = M_to_R(M[i], Omega_m);
  }
  int rc = sigma2_at_R_arr(R, NM, k, P, Nk, s2);
  free(R);
  return rc;
}

/* The derivative with respect to R of sigma^2. This is needed for the mass
 * function and replaces having to take a numerical derivative.
 */
int dsigma2dR_at_R_arr(double*R, int NR, double*k, double*P, int Nk,
		       double*ds2dR){
  //Initialize GSL things and the integrand structure.
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline,Nk);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  gsl_integration_workspace*workspace = gsl_integration_workspace_alloc(workspace_size);
  if (!spline || !acc || !workspace)
    return GSL_ENOMEM;

  gsl_function F;
  integrand_params params;
  double lkmin = log(k[0]);
  double lkmax = log(k[Nk-1]);
  double result,abserr;
  //Divide by 2pi^2, but note we pull a factor of 2
  //out of the integrand because we take dw^2/dR
  //We also pull a factor of 3 out of the integrand.
  double denom_inv = 1./(M_PI*M_PI);
  int i, rc;

  rc = gsl_spline_init(spline,k,P,Nk);
  params.spline = spline;
  params.acc = acc;
  params.kp = k;
  params.Pp = P;
  params.Nk = Nk;
  F.function = &dsigma2dR_integrand;
  F.params = &params;
  for(i = 0; i < NR; i++){
    if (rc != GSL_SUCCESS)
      break;
    params.r = R[i];
    rc = gsl_integration_qag(&F, lkmin, lkmax, ABSERR, RELERR,
			workspace_size, KEY, workspace, &result, &abserr);
    ds2dR[i] = result * denom_inv; //divide by 2pi^2
  }
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);

  return rc;
}

double dsigma2dR_at_R(double R, double*k, double*P, int Nk){
  //sigma^2(R) for a single value of R
  double ds2dR;
  dsigma2dR_at_R_arr(&R, 1, k, P, Nk, &ds2dR);
  return ds2dR;
}

/* The derivative with respect to M of sigma^2. This is needed for the mass
 * function and replaces having to take a numerical derivative.
 */
int dsigma2dM_at_M_arr(double*M, int NM, double*k, double*P, int Nk,
		       double Omega_m, double*ds2dM){
  int i;
  double*R = (double*)malloc(sizeof(double)*NM);
  double*dRdM = (double*)malloc(sizeof(double)*NM);
  for(i = 0; i < NM; i++){
    R[i] = M_to_R(M[i], Omega_m);
    dRdM[i] = dRdM_at_M(M[i], Omega_m);
  }
  //Note: ds2dM holds ds2dR for now
  int rc = dsigma2dR_at_R_arr(R, NM, k, P, Nk, ds2dM);
  //Chain rule to convert dsigma2dR to dsigma2dM
  for(i = 0; i < NM; i++){
    ds2dM[i] = ds2dM[i] * dRdM[i];
  }
  free(R);
  free(dRdM);
  return rc;
}

double dsigma2dM_at_M(double M, double*k, double*P, int Nk, double Omega_m){
  double ds2dM;
  dsigma2dR_at_R_arr(&M, 1, k, P, Nk, &ds2dM);
  return ds2dM;
}

///////////PEAK HEIGHT FUNCTIONS///////////

double nu_at_R(double R, double*k, double*P, int Nk){
  //peak height at R
  return delta_c/sqrt(sigma2_at_R(R, k, P, Nk));
}

double nu_at_M(double M, double*k, double*P, int Nk, double om){
  //peak height at M
  double R=M_to_R(M, om);
  return nu_at_R(R, k, P, Nk);
}

int nu_at_R_arr(double*R, int NR, double*k, double*P, int Nk, double*nu){
  //peak height at an array of R
  int i;
  double*s2 = (double*)malloc(sizeof(double)*NR);
  int rc = sigma2_at_R_arr(R, NR, k, P, Nk, s2);
  for(i = 0; i < NR; i++){
    nu[i] = delta_c/sqrt(s2[i]);
  }
  free(s2);
  return rc;
}

int nu_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double om, double*nu){
  //peak height at an array of M
  int i;
  double*R = (double*)malloc(sizeof(double)*NM);
  for(i = 0; i < NM; i++){
    R[i] = M_to_R(M[i], om);
  }
  int rc = nu_at_R_arr(R, NM, k, P, Nk, nu);
  free(R);
  return rc;
}
