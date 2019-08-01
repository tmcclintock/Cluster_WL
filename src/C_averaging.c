/** @file C_averaging.c
 *  @brief Average halo profiles in radial bins.
 *  
 *  These functions average projected halo profiles in
 *  radial bins, assuming that all the specified bins
 *  are bounded by the input profiles.
 *  
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_averaging.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include <math.h>
#include <stdio.h>

#define ABSERR 0
#define RELERR 1e-6
#define workspace_size 8000

////////////// AVERAGING FUNCTIONS BELOW////////////////

typedef struct integrand_params{
  gsl_spline*spline;
  gsl_interp_accel*acc;
}integrand_params;

double ave_integrand(double lR, void*params){
  double R = exp(lR);
  integrand_params pars = *(integrand_params *)params;
  gsl_spline*spline = pars.spline;
  gsl_interp_accel*acc = pars.acc;
  return R*R*gsl_spline_eval(spline, R, acc);
}

int average_profile_in_bins(double*Redges, int Nedges, double*R, int NR,
			    double*profile, double*ave_profile){
  gsl_spline *spline = gsl_spline_alloc(gsl_interp_cspline, NR);
  gsl_interp_accel *acc = gsl_interp_accel_alloc();
  gsl_integration_workspace *ws = gsl_integration_workspace_alloc(workspace_size);
  gsl_function F;
  double result, err;

  gsl_spline_init(spline, R, profile, NR);

  integrand_params params;
  params.acc = acc;
  params.spline = spline;
  F.params = &params;
  F.function = &ave_integrand;

  //Loop over bins and compute the average
  int i;
  for(i = 0; i < Nedges-1; i++){
    gsl_integration_qag(&F, log(Redges[i]), log(Redges[i+1]), ABSERR, RELERR,
			workspace_size, 6, ws, &result, &err);
    ave_profile[i] = 2*result/(Redges[i+1]*Redges[i+1]-Redges[i]*Redges[i]);
  }
  
  //Free everything
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(ws);
  return 0;
}
