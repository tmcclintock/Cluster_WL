/** @file C_profile_derivatives.c
 *  @brief Halo density profile derivatives
 *
 *  These functions are derivatives of different models of density profiles for halos.
 *
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_density.h"
#include "C_power.h"
#include "C_profile_derivatives.h"
#include "C_xi.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_gamma.h"
#include <math.h>
#include <stdio.h>

#define rhocrit 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are Msun h^2/Mpc^3

#define workspace_size 8000
#define workspace_num 100
#define ABSERR 0.0
#define RELERR 1.8e-4

double drho_nfw_dr_at_R(double R, double Mass, double conc, int delta, double Omega_m){
  double drhodr = 0.0;
  drho_nfw_dr_at_R_arr(&R, 1, Mass, conc, delta, Omega_m, &drhodr);
  return drhodr;
}

int drho_nfw_dr_at_R_arr(double*R, int NR, double Mass, double conc,
			 int delta, double Omega_m, double*drhodr){
  int i;
  double rhom = Omega_m*rhocrit;//Msun h^2/Mpc^3
  double Rdelta = pow(Mass/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  double Rscale = Rdelta/conc;
  double fc = log(1.+conc)-conc/(1.+conc);
  double rho0 = Mass/(4.*M_PI*Rscale*Rscale*Rscale*fc); //characteristic density
  double R_Rs;
  for(i = 0; i < NR; i++){
    R_Rs = R[i]/Rscale;
    drhodr[i] = - rho0*(1+3*R_Rs) / (Rscale * R_Rs*R_Rs * (1+R_Rs)*(1+R_Rs));
  }
  return 0;
}

typedef struct integrand_params_profile_derivs{
  gsl_spline*spline;
  gsl_interp_accel*acc;
  gsl_integration_workspace * workspace;
  double r; //3d r; Mpc/h, or inverse units of k
  double*kp; //pointer to wavenumbers
  double*Pp; //pointer to P(k) array
  int Nk; //length of k and P arrays
}integrand_params_profile_derivs;


double integrand_dxi_mm_dr_COSINE(double k, void*params){
  integrand_params_profile_derivs pars
    = *(integrand_params_profile_derivs*)params;
  gsl_spline*spline = pars.spline;
  gsl_interp_accel*acc = pars.acc;
  double*kp = pars.kp;
  double*Pp = pars.Pp;
  int Nk = pars.Nk;
  double R = pars.r;
  double x  = k*R;
  double P = get_P(x, R, kp, Pp, Nk, spline, acc);
  return P*k*k/R; //Note - cos(kR) is taken care of in the qawo table
}

double integrand_dxi_mm_dr_SINE(double k, void*params){
  integrand_params_profile_derivs pars
    = *(integrand_params_profile_derivs*)params;
  gsl_spline*spline = pars.spline;
  gsl_interp_accel*acc = pars.acc;
  double*kp = pars.kp;
  double*Pp = pars.Pp;
  int Nk = pars.Nk;
  double R = pars.r;
  double x  = k*R;
  double P = get_P(x, R, kp, Pp, Nk, spline, acc);
  return P*k/(R*R); //Note - sin(kR) is taken care of in the qawo table
}

double dxi_mm_dr_at_R(double R, double*k, double*P, int Nk){
  double dxidr = 0.0;
  dxi_mm_dr_at_R_arr(&R, 1, k, P, Nk, &dxidr);
  return dxidr;
}

int dxi_mm_dr_at_R_arr(double*R, int NR, double*k, double*P, int Nk, double*dxidr){
  static int init_flag = 0;
  static gsl_interp_accel*acc;
  static gsl_integration_workspace*workspace;
  static integrand_params_profile_derivs params;
  if (!init_flag){
    acc = gsl_interp_accel_alloc();
    workspace = gsl_integration_workspace_alloc(workspace_size);
    init_flag = 1;
  }
  
  gsl_spline*Pspl = gsl_spline_alloc(gsl_interp_cspline, Nk);
  gsl_integration_qawo_table*wf_cosine;
  gsl_integration_qawo_table*wf_sine;
 
  double kmax = 4e3;
  double kmin = 5e-8;
  double result_cosine, result_sine, err;
  int i;
  int status;
  gsl_spline_init(Pspl, k, P, Nk);
  params.acc = acc;
  params.spline = Pspl;
  params.kp = k;
  params.Pp = P;
  params.Nk = Nk;

  gsl_function F_cosine;
  gsl_function F_sine;

  F_cosine.function = &integrand_dxi_mm_dr_COSINE;
  F_sine.function   = &integrand_dxi_mm_dr_SINE;

  F_cosine.params = &params;
  F_sine.params   = &params;

  wf_cosine = gsl_integration_qawo_table_alloc(R[0], kmax-kmin, GSL_INTEG_COSINE,
					       (size_t)workspace_num);
  wf_sine   = gsl_integration_qawo_table_alloc(R[0], kmax-kmin, GSL_INTEG_SINE,
					       (size_t)workspace_num);
  for(i = 0; i < NR; i++){
    status = gsl_integration_qawo_table_set(wf_cosine, R[i], kmax-kmin, GSL_INTEG_COSINE);
    status = gsl_integration_qawo_table_set(wf_sine, R[i], kmax-kmin, GSL_INTEG_SINE);
    if (status){
      printf("Error in dxi_mm_dr in qawo_table_set.\n");
      exit(-1);
    }
    params.r=R[i];
    status = gsl_integration_qawo(&F_cosine, kmin, ABSERR, RELERR, (size_t)workspace_num,
				  workspace, wf_cosine, &result_cosine, &err);
    status = gsl_integration_qawo(&F_sine, kmin, ABSERR, RELERR, (size_t)workspace_num,
				  workspace, wf_sine, &result_sine, &err);
    if (status){
      printf("Error in dxi_mm_dr in the integrals.\n");
      exit(-1);
    }

    dxidr[i] = (result_cosine - result_sine)/(M_PI*M_PI*2);
  }

  gsl_spline_free(Pspl);
  gsl_integration_qawo_table_free(wf_cosine);
  gsl_integration_qawo_table_free(wf_sine);
  //These are now static.
  //gsl_interp_accel_free(acc);
  //gsl_integration_workspace_free(workspace);

  return 0;
}
