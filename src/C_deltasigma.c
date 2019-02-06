/** @file C_deltasigma.c
 *  @brief Projected cluster profiles.
 * 
 *  These functions compute projected cluster profiles,
 *  commonly known as "Sigma" profiles or "DeltaSigma"
 *  profiles for differential profiles.
 *  
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_deltasigma.h"
#include "C_xi.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_errno.h"

#include <math.h>
#include <stdio.h>

#define ABSERR 0.0
#define RELERR 1e-4
#define workspace_size 8000
#define ulim 5.0
#define rhocrit 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are SM h^2/Mpc^3
#define KEY 3 //Used for GSL QAG function

////////////// SIGMA(R) FUNCTIONS BELOW////////////////

/**
 * \brief Projected surface mass density Sigma in units
 * of h*Msun/pc^2 assuming an NFW model at a radius R in Mpc/h.
 *
 * Note: all distances are comoving.
 */
double Sigma_nfw_at_R(double R, double M, double c, int delta, double om){
  double result = 0.0;
  Sigma_nfw_at_R_arr(&R, 1, M, c, delta, om, &result);
  return result;
}

/**
 * \brief Projected surface mass density Sigma in units
 * of h*Msun/pc^2 assuming an NFW model at an array of
 * radii R in Mpc/h.
 *
 * Note: all distances are comoving.
 */
int Sigma_nfw_at_R_arr(double*R, int NR, double M, double c, int delta, double om, double*Sigma){
  double rhom = om*rhocrit;//SM h^2/Mpc^3
  double deltac = delta*0.3333333333*c*c*c/(log(1.+c)-c/(1.+c));
  double Rdelta = pow(M/(1.333333333*M_PI*rhom*delta),0.333333333);//Mpc/h
  double Rscale = Rdelta/c;
  double gx = 0;
  double x;
  int i;
  for(i = 0; i < NR; i++){
    x = R[i]/Rscale;
    if(x<1){
      gx = (1 - 2./sqrt(1-x*x)*atanh(sqrt((1-x)/(1+x))))/(x*x-1);
    }else{// if(x>=1){
      gx = (1 - 2./sqrt(x*x-1)* atan(sqrt((x-1)/(1+x))))/(x*x-1);
    }
    Sigma[i] = 2*Rscale*deltac*rhom*gx*1.e-12; //SM h/pc^2
  }
  return 0;
}

typedef struct integrand_params{
  gsl_spline*spline;
  gsl_interp_accel*acc;
  double Rp;
  double M;
  double conc;
  int delta;
  double om;
  double slope;
  double intercept;
}integrand_params;

/**
 * \brief Integrand (r*\xi_hm(r)) for computing
 * Sigma(R) at small scales.
 *
 * Note: all distances are comoving.
 */
double integrand_small_scales(double lRz, void*params){
  double Rz = exp(lRz);
  integrand_params pars=*(integrand_params*)params;
  double Rp = pars.Rp;
  double om = pars.om;
  double M = pars.M;
  double conc = pars.conc;
  int delta = pars.delta;
  return Rz * xi_nfw_at_R(sqrt(Rz*Rz + Rp*Rp), M, conc, delta, om);
}

/**
 * \brief Integrand (r*\xi_hm(r)) for computing
 * Sigma(R) at intermediate scales.
 *
 * Note: all distances are comoving.
 */
double integrand_medium_scales(double lRz, void*params){
  double Rz = exp(lRz);
  integrand_params pars=*(integrand_params*)params;
  double Rp = pars.Rp;
  return Rz * gsl_spline_eval(pars.spline, log(Rz*Rz + Rp*Rp)*0.5, pars.acc);
}

/**
 * \brief Integrand (r*\xi_hm(r)) for computing
 * Sigma(R) at large scales.
 *
 * Note: all distances are comoving.
 */
double integrand_large_scales(double lRz, void*params){
  //Use a power law approximation. This is fine as long as xi_hm
  //doesn't have wiggles in it (i.e. it doesn't stop at BAO)
  double Rz = exp(lRz);
  integrand_params pars=*(integrand_params*)params;
  double Rp = pars.Rp;
  double slope = pars.slope;
  double intercept = pars.intercept;
  return Rz * intercept*pow(sqrt(Rz*Rz + Rp*Rp), slope);
}

/**
 * \brief Projected surface mass density Sigma in units
 * of h*Msun/pc^2 at a radius R in Mpc/h, given a 
 * 3D halo-matter correlation function.
 *
 * Note: all distances are comoving.
 */
double Sigma_at_R(double R, double*Rxi, double*xi, int Nxi, double M, double conc, int delta, double om){
  double result;
  Sigma_at_R_arr(&R, 1, Rxi, xi, Nxi, M, conc, delta, om, &result);
  return result;
}

/**
 * \brief Projected surface mass density Sigma in units
 * of h*Msun/pc^2 at an array of radii R in Mpc/h, given a 
 * 3D halo-matter correlation function.
 *
 * Note: all distances are comoving.
 */
int Sigma_at_R_arr(double*R, int NR, double*Rxi, double*xi, int Nxi, double M, double conc, int delta, double om, double*Sigma){
  gsl_set_error_handler_off();
  double rhom = om*rhocrit*1e-12; //SM h^2/pc^2/Mpc; integral is over Mpc/h
  double Rxi0 = Rxi[0];
  double Rxi_max = Rxi[Nxi-1];
  double ln_z_max;
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, Nxi);
  //linear interpolators should be used when dealing with simulated data...
  //gsl_spline*spline = gsl_spline_alloc(gsl_interp_linear, Nxi);

  gsl_interp_accel*acc= gsl_interp_accel_alloc();
  gsl_integration_workspace*workspace = gsl_integration_workspace_alloc(workspace_size);
  gsl_function F;
  double result1, err1, result2, err2;
  int i;
  double*lnRxi = (double*)malloc(Nxi*sizeof(double));
  for(i = 0; i < Nxi; i++){
    lnRxi[i] = log(Rxi[i]);
  }
  gsl_spline_init(spline, lnRxi, xi, Nxi);

  integrand_params params;
  params.acc = acc;
  params.spline = spline;
  params.M = M;
  params.conc= conc;
  params.delta = delta;
  params.om = om;

  F.params = &params;

  int status;
  for(i = 0; i < NR; i++){
    ln_z_max = log(sqrt(Rxi_max*Rxi_max - R[i]*R[i])); //Max distance to integrate to
    params.Rp = R[i];
    if(R[i] < Rxi0){
      F.function = &integrand_small_scales;
      status = gsl_integration_qag(&F, log(Rxi0)-10, log(sqrt(Rxi0*Rxi0-R[i]*R[i])), ABSERR, RELERR, workspace_size, KEY, workspace, &result1, &err1);
      if (status)
	fprintf(stderr, "Error in C_deltasigma.c in small scales with\n\t%e\n\t%e\n",M,conc);
      F.function = &integrand_medium_scales;
      status = gsl_integration_qag(&F, log(sqrt(Rxi0*Rxi0-R[i]*R[i])), ln_z_max, ABSERR, RELERR, workspace_size, KEY, workspace, &result2, &err2);
      if (status)
	fprintf(stderr, "Error in C_deltasigma.c in medium scales with\n\t%e\n\t%e\n",M,conc);

    }else{ //R[i] > Rxi0
      result1 = 0;
      F.function = &integrand_medium_scales;
      gsl_integration_qag(&F, -10, ln_z_max, ABSERR, RELERR, workspace_size, KEY, workspace, &result2, &err2);
    }
    Sigma[i] = (result1+result2)*rhom*2;
  }

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);
  free(lnRxi);

  return 0;
}

/**
 * \brief Projected surface mass density Sigma in units
 * of h*Msun/pc^2 at an array of radii R in Mpc/h, given a 
 * 3D halo-matter correlation function, and including
 * the large scales.
 *
 * Note: all distances are comoving.
 */
double Sigma_at_R_full(double R, double*Rxi, double*xi, int Nxi, double M, double conc, int delta, double om){
  double result;
  Sigma_at_R_full_arr(&R, 1, Rxi, xi, Nxi, M, conc, delta, om, &result);
  return result;
}

int Sigma_at_R_full_arr(double*R, int NR, double*Rxi, double*xi, int Nxi, double M, double conc, int delta, double om, double*Sigma){
  //This function just adds on the powerlaw part at the end
  //It can actually be done analytically...
  double rhom = om*rhocrit*1e-12; //Msun h^2/pc^2/Mpc; integral is over Mpc/h
  double Rxi_max = Rxi[Nxi-1];
  double ln_z_max;
  double result3, err3;
  gsl_integration_workspace*workspace = gsl_integration_workspace_alloc(workspace_size);
  integrand_params params;
  gsl_function F;
  int i;
  Sigma_at_R_arr(R, NR, Rxi, xi, Nxi, M, conc, delta, om, Sigma);

  params.slope = log(xi[Nxi-1]/xi[Nxi-2])/log(Rxi[Nxi-1]/Rxi[Nxi-2]);
  params.intercept = xi[Nxi-1]/pow(Rxi[Nxi-1], params.slope);

  F.params = &params;
  F.function = &integrand_large_scales;

  for(i = 0; i < NR; i++){
    ln_z_max = log(sqrt(Rxi_max*Rxi_max - R[i]*R[i]));
    params.Rp = R[i];
    gsl_integration_qag(&F, ln_z_max, ln_z_max+ulim, ABSERR, RELERR, workspace_size, KEY, workspace, &result3, &err3);
    Sigma[i] += (result3*rhom*2);
  }

  gsl_integration_workspace_free(workspace);
  return 0;
}

////////////// DELTASIGMA FUNCTIONS BELOW////////////////

double DS_integrand_small_scales(double lR, void*params){
  double R = exp(lR);
  integrand_params pars = *(integrand_params*)params;
  double M = pars.M;
  double conc = pars.conc;
  int delta = pars.delta;
  double om = pars.om;
  return R * R * Sigma_nfw_at_R(R, M, conc, delta, om);
}

double DS_integrand_medium_scales(double lR, void*params){
  double R = exp(lR);
  integrand_params pars = *(integrand_params*)params;
  return R * R * gsl_spline_eval(pars.spline, log(R), pars.acc);
}

double DeltaSigma_at_R(double R, double*Rs, double*Sigma, int Ns, double M, double conc, int delta, double om){
  double DS;
  DeltaSigma_at_R_arr(&R, 1, Rs, Sigma, Ns, M, conc, delta, om, &DS);
  return DS;
}

int DeltaSigma_at_R_arr(double*R, int NR, double*Rs, double*Sigma, int Ns, double M, double conc, int delta, double om, double*DeltaSigma){
  double lrmin = log(Rs[0]);
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(workspace_size);
  double result1, result2, err1, err2;
  gsl_function F;
  int i;
  double*lnRs = (double*)malloc(Ns*sizeof(double));
  for(i = 0; i < Ns; i++){
    lnRs[i] = log(Rs[i]);
  }
  gsl_spline_init(spline, lnRs, Sigma, Ns);

  integrand_params params;
  params.spline = spline;
  params.acc = acc;
  params.M = M;
  params.conc = conc;
  params.delta = delta;
  params.om = om;

  F.params = &params;
  F.function = &DS_integrand_small_scales;
  gsl_integration_qag(&F, lrmin-10, lrmin, ABSERR, RELERR, workspace_size, KEY, workspace, &result1, &err1);
  F.function = &DS_integrand_medium_scales;

  for(i = 0; i < NR; i++){
    gsl_integration_qag(&F, lrmin, log(R[i]), ABSERR, RELERR, workspace_size, KEY, workspace, &result2, &err2);
    DeltaSigma[i] = (result1+result2)*2/(R[i]*R[i]) - gsl_spline_eval(spline, log(R[i]), acc);
  }

  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);
  free(lnRs);
  return 0;
}
