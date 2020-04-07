/** @file C_miscentering
 *  @brief Miscentering effects on projected cluster profiles.
 * 
 *  This file implements the functions that modify projected
 *  galaxy cluster weak lensing profiles. For stacked clusters,
 *  this includes different distributions of miscentering lengths,
 *  or the distribution of the amount that clusters are miscentered.
 *  
 *  @author Tom McClintock (tmcclintock)
 *  @bug No known bugs.
 */

#include "C_miscentering.h"
#include "C_deltasigma.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_errno.h"
#include <math.h>
#include <stdio.h>

#define ABSERR 0.0
#define RELERR 1e-2 // Used for miscentering
#define workspace_size 8000
#define rhomconst 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are SM h^2/Mpc^3
#define KEY 1 //Used for GSL QAG function

////////////// SIGMA(R) FUNCTIONS BELOW////////////////

typedef struct integrand_params{
  //Spline, accelerator and integration workspaces.
  gsl_spline*spline;
  gsl_interp_accel*acc;
  gsl_integration_workspace*workspace;
  gsl_integration_workspace*workspace2;
  double Rp;             //R_perp (i.e. projected separation)
  double Rp2;            //R_perp^2
  double rmin;           //minimum radius of the spline
  double rmax;           //maximum radius of the spline
  double lrmin;          //log of the minimum radius of the spline
  double lrmax;          //log of the maximum radius of the spline
  double M;              //halo mass; Msun/h
  double conc;           //concentration
  int delta;             //overdensity (200 is suggested)
  double Omega_m;        //Omega_m
  double Rmis;           //Miscentering length
  double Rmis2;          //Rmis^2
  double Rp_cos_theta_2; //Variable to hold 2*Rp*cos(theta)
  double slope;          //slope of the power-law inner profile of Sigma(R); can be removed
  double intercept;      //intercept of the power-law inner profile of Sigma(R); can be removed
  gsl_function F_radial; //function for the "inner integral, or the radial part of the miscentering 
}integrand_params;

/** @brief The integrand the miscentered profile of a 
 *         single cluster.
 *
 *  The miscentered profile of a single cluster with a
 *  known offset is calculated by integrating around an annulus.
 *  See McClintock+ (2018) eq. 38.
 *
 *  @param theta The angle the integral is currently on.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand at theta.
 */
double single_angular_integrand(double theta, void*params){
  integrand_params*pars = (integrand_params*)params;
  double arg = sqrt(pars->Rp2 + pars->Rmis2 - 2*pars->Rp*pars->Rmis*cos(theta));
  if(arg < pars->rmin){
    return Sigma_nfw_at_R(arg, pars->M, pars->conc, pars->delta, pars->Omega_m);
  }else if(arg < pars->rmax){
    return gsl_spline_eval(pars->spline, log(arg), pars->acc);
  }
  return 0;//arg > rmax
}

/** @brief Miscentered Sigma profile at radius R in Mpc/h comoving.
 *
 *  The miscentered Sigma profile of a single cluster, given
 *  a centered  mass surface density profile, Sigma(R).
 *  Units of surface density are all in h*Msun/pc^2 comoving.
 *
 *  @param R Radii in Mpc/h comoving.
 *  @param NR Number of radii.
 *  @param Rs Radii at which we know Sigma(R), in Mpc/h comoving.
 *  @param Sigma Surface mass density profile in h*Msun/pc^2 comoving.
 *  @param Ns Number of elements in Sigma_mis and Rs.
 *  @param M Halo mass in Msun/h.
 *  @param conc Halo concentration.
 *  @param delta Halo overdensity.
 *  @param Omega_m Matter fraction.
 *  @param Rmis Halo projected offset from the true center in Mpc/h comoving.
 *  @param Sigma_mis Output array for Sigma_mis(R) in h*Msun/pc^2 comoving.
 *  @return success Integer indicating no errors.
 */
int Sigma_mis_single_at_R_arr(double*R, int NR, double*Rs, double*Sigma, int Ns,
			      double M, double conc, int delta, double Omega_m,
			      double Rmis, double*Sigma_mis){
  int i;
  double result, err;
  gsl_function F;

  //Precomputing to save time
  double*lnRs = (double*)malloc(Ns*sizeof(double));
  for(i = 0; i < Ns; i++){
    lnRs[i] = log(Rs[i]);
  }

  //Allocate things
  static int init_flag = 0;
  // FIXME - the static variables are not thread safe
  static gsl_spline*spline = NULL;
  static gsl_interp_accel*acc = NULL;
  static gsl_integration_workspace*workspace = NULL;
  static integrand_params params;
  if (init_flag == 0){
    spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
    acc = gsl_interp_accel_alloc();
    workspace = gsl_integration_workspace_alloc(workspace_size);
    if (!spline || !acc || !workspace)
      return GSL_ENOMEM;
    init_flag = 1;
  }

  int rc = gsl_spline_init(spline, lnRs, Sigma, Ns);

  params.acc = acc;
  params.spline = spline;
  params.workspace = workspace;
  params.M = M;
  params.conc = conc;
  params.delta = delta;
  params.Omega_m = Omega_m;
  params.Rmis = Rmis;
  params.Rmis2 = Rmis*Rmis;
  params.rmin = Rs[0];
  params.rmax = Rs[Ns-1];
  params.lrmin = log(Rs[0]);
  params.lrmax = log(Rs[Ns-1]);

  //Angular integral
  F.function = &single_angular_integrand;
  F.params = &params;
  
  for(i = 0; i < NR; i++){
    if (rc != GSL_SUCCESS)
      break;

    params.Rp  = R[i];
    params.Rp2 = R[i] * R[i];
    rc = gsl_integration_qag(&F, 0, M_PI, ABSERR, RELERR, workspace_size,
			KEY, workspace, &result, &err);
    Sigma_mis[i] = result/M_PI;
  }

  //Static objects aren't freed
  free(lnRs);
  return rc;
}

/////////////////// SIGMA(R) INTEGRANDS BELOW //////////////////////

/** @brief The integrand the miscentered profile of a 
 *         cluster stack.
 *
 *  The miscentered profile of a cluster stack has an 
 *  integral over the offset distribution, and this
 *  function is that integrand. At the smallest scales
 *  this function returns the NFW Sigma(R) profile by default.
 *
 *  @param Rc Argument to Sigma(R), with Rc^2=R^2+Rmis^2-2*R*Rmis*cos(theta).
 *  @param Rc2 Rc^2, used to reduce computation time.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand, Sigma(R),  
 */
double get_Sigma(double Rc, double Rc2, void*params){
  integrand_params*pars = (integrand_params*)params;
  double arg = sqrt(pars->Rp2 + Rc2 - Rc*pars->Rp_cos_theta_2);
  if(arg < pars->rmin){
    return Sigma_nfw_at_R(arg, pars->M, pars->conc, pars->delta, pars->Omega_m);
  }else if(arg < pars->rmax){
    return gsl_spline_eval(pars->spline, log(arg), pars->acc);
  }
  return 0;//arg > rmax
}

/** @brief Integrand for a stack of miscentered clusters
 *         with an exponential, rather than Raleigh, distribution.
 *
 *  @param lRc Log of the integrand parameter, Rc, in Mpc/h comoving.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand.
 */
double Gamma_integrand(double lRc, void*params){
  integrand_params*pars = (integrand_params*)params;
  double Rc = exp(lRc);
  double Rc2 = Rc*Rc;
  double Rmis = pars->Rmis;
  return Rc2 * exp(-Rc/Rmis) * get_Sigma(Rc, Rc2, pars);//normalized outside
}

/** @brief Integrand for a stack of miscentered clusters
 *         with a Raleigh distribution.
 *
 *  @param lRc Log of the integrand parameter, Rc, in Mpc/h comoving.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand.
 */
double Rayleigh_radial_integrand(double lRc, void*params){
  integrand_params*pars = (integrand_params*)params;
  double Rc = exp(lRc);
  double Rc2 = Rc*Rc;
  double Rmis2 = pars->Rmis2;
  return Rc2 * exp(-0.5 * Rc2/Rmis2) * get_Sigma(Rc, Rc2, pars);//normalized outside
}

/** @brief Integrand for a stack of miscentered clusters
 *         around an annulus, for an arbitrary radial distribution.
 *
 *  @param theta Angle around the annulus.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand.
 */
double angular_integrand(double theta, void*params){
  double result, err;
  integrand_params*pars = (integrand_params*)params;
  pars->Rp_cos_theta_2 = pars->Rp*cos(theta)*2;
  //\int_0^\inf dRc p(Rc|Rmis) Sigma_mis(R, Rc, Rmis)
  gsl_integration_qag(&pars->F_radial, pars->lrmin-10, pars->lrmax, ABSERR, RELERR, workspace_size,
		      KEY, pars->workspace2, &result, &err);
  return result;
}

/** @brief Miscentered Sigma profile at radius R in Mpc/h comoving.
 *
 *  The miscentered Sigma profile of a cluster stack, given
 *  a centered mass surface density profile, Sigma(R).
 *  Units of surface density are all in h*Msun/pc^2 comoving.
 *
 *  This function computes equations 38 and 39 in McClintock+ (2018), the
 *  DES Y1 lensing analysis of redMaPPer clusters.
 *
 *  @param R Radii in Mpc/h comoving.
 *  @param NR Number of radii.
 *  @param Rs Radii at which we know Sigma(R), in Mpc/h comoving.
 *  @param Sigma Surface mass density profile in h*Msun/pc^2 comoving.
 *  @param Ns Number of elements in Sigma_mis and Rs.
 *  @param M Halo mass in Msun/h.
 *  @param conc Halo concentration.
 *  @param delta Halo overdensity.
 *  @param Omega_m Matter fraction.
 *  @param Rmis Halo projected offset from the true center in Mpc/h comoving.
 *  @param Sigma_mis Output array for Sigma_mis(R) in h*Msun/pc^2 comoving.
 *  @return success Integer indicating no errors.
 */
int Sigma_mis_at_R_arr(double*R, int NR, double*Rs, double*Sigma, int Ns,
		       double M, double conc, int delta, double Omega_m, double Rmis,
		       int integrand_switch, double*Sigma_mis){
  int i;
  double result, err;
  gsl_function F;
  gsl_function F_radial;

  //Precomputing to save time
  double*lnRs = (double*)malloc(Ns*sizeof(double));
  for(i = 0; i < Ns; i++){
    lnRs[i] = log(Rs[i]);
  }

  //Allocate things
  static int init_flag = 0;
  static gsl_spline*spline = NULL;
  static gsl_interp_accel*acc = NULL;
  static gsl_integration_workspace*workspace = NULL;
  static gsl_integration_workspace*workspace2 = NULL;
  static integrand_params params;
  if (init_flag == 0){
    spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
    acc = gsl_interp_accel_alloc();
    workspace = gsl_integration_workspace_alloc(workspace_size);
    workspace2 = gsl_integration_workspace_alloc(workspace_size);
    if (!spline || !acc || !workspace || !workspace2)
      return GSL_ENOMEM;
    init_flag = 1;
  }

  int rc = gsl_spline_init(spline, lnRs, Sigma, Ns);

  params.spline = spline;
  params.acc = acc;
  params.workspace = workspace;
  params.workspace2 = workspace2;
  params.M = M;
  params.conc = conc;
  params.delta = delta;
  params.Omega_m = Omega_m;
  params.Rmis = Rmis;
  params.Rmis2 = Rmis*Rmis;
  params.rmin = Rs[0];
  params.rmax = Rs[Ns-1];
  params.lrmin = log(Rs[0]);
  params.lrmax = log(Rs[Ns-1]);

  //Angular integral
  F.function = &angular_integrand;
  //Radial integral. Choice between rayliegh and gamma distributions
  //See Johnston+ 2007, Simet+ 2018, Melchior+ 2018, McClintock+ 2019
  switch(integrand_switch){
  case 0:
    F_radial.function = &Rayleigh_radial_integrand;
    break;
  case 1:
    F_radial.function = &Gamma_integrand;
    break;
  }

  //Assign the params struct to the GSL functions.
  F_radial.params = &params;
  params.F_radial = F_radial;
  F.params = &params;

  //Angular integral first
  for(i = 0; i < NR; i++){
    if (rc != GSL_SUCCESS)
      break;

    params.Rp  = R[i];
    params.Rp2 = R[i] * R[i]; //Optimization

    rc = gsl_integration_qag(&F, 0, M_PI, ABSERR, RELERR, workspace_size,
			KEY, workspace, &result, &err);
    Sigma_mis[i] = result/(M_PI*Rmis*Rmis); //Normalization
  }
  //Static objects aren't freed
  free(lnRs);
  return rc;
}

///////////////////////////////////////////////////
//////////// DELTASIGMA(R) BELOW //////////////////
///////////////////////////////////////////////////

/** @brief The integrand the miecentered DeltaSigma profile.
 *
 *  The miscentered profile of a either a single cluster
 *  or a stack of clusters is calculated by taking the
 *  difference between Sigma_mis(<R) and Sigma_mis(R).
 *  See McClintock+ (2018) eq. 7.
 *
 *  @param lR The natural log of the radius the integral is currently on.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand at ln(R).
 */
double DS_mis_integrand(double lR, void*params){
  double R = exp(lR);
  integrand_params pars = *(integrand_params*)params;
  return R * R * gsl_spline_eval(pars.spline, R, pars.acc);
}

/** @brief DeltaSigma profile at an array of radii R in Mpc/h comoving.
 *
 *  The miscentered DeltaSigma profile of a cluster, given
 *  its miscentered mass surface density profile, Sigma_mis(R).
 *  Units of surface density are all in h*Msun/pc^2 comoving.
 *  This specific function just interfaces DeltaSigma_mis_at_R_arr().
 *
 *  @param R Radii in Mpc/h comoving.
 *  @param NR Number of radii.
 *  @param Rs Radii at which we know Sigma(R), in Mpc/h comoving.
 *  @param Sigma_mis Surface mass density profile in h*Msun/pc^2 comoving.
 *  @param Ns number of elements in Sigma_mis and Rs.
 *  @return DeltaSigma_mis(R) in h*Msun/pc^2 comoving.
 */
int DeltaSigma_mis_at_R_arr(double*R, int NR, double*Rs, double*Sigma_mis, int Ns, double*DeltaSigma_mis){
  int i;
  double lrmin = log(Rs[0]);
  double result,  err;
  gsl_function F;

  //Compute the integral from 0 to Rs[0] assuming that
  //Sigma_mis(R) is a power law
  double slope = log(Sigma_mis[0]/Sigma_mis[1])/log(Rs[0]/Rs[1]);
  double intercept = Sigma_mis[0]*pow(Rs[0], -slope);
  double low_part = intercept*pow(Rs[0], slope+2)/(slope+2);

  //Allocate things
  static int init_flag = 0;
  static gsl_spline*spline = NULL;
  static gsl_interp_accel*acc = NULL;
  static gsl_integration_workspace*workspace = NULL;
  static integrand_params params;
  if (init_flag == 0){
    spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
    acc = gsl_interp_accel_alloc();
    workspace = gsl_integration_workspace_alloc(workspace_size);
    if (!spline || !acc || !workspace)
      return GSL_ENOMEM;
    init_flag = 1;
  }

  int rc = gsl_spline_init(spline, Rs, Sigma_mis, Ns);
  
  params.spline = spline;
  params.acc = acc;
  F.params = &params;
  F.function = &DS_mis_integrand;

  for(i = 0; i < NR; i++){
    if (rc != GSL_SUCCESS)
      break;
    rc = gsl_integration_qag(&F, lrmin, log(R[i]), ABSERR, RELERR, workspace_size,
			KEY, workspace, &result, &err);
    DeltaSigma_mis[i] = (low_part+result)*2/(R[i]*R[i]) - gsl_spline_eval(spline, R[i], acc);
  }

  //No free() since we use static variables.
  return rc; 
}
