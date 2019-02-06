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
#include <math.h>
#include <stdio.h>

#define ABSERR 0.0
#define RELERR 1e-4 // Used for miscentering
#define workspace_size 8000
#define rhomconst 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are SM h^2/Mpc^3
#define KEY 3 //Used for GSL QAG function

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
  double om;             //Omega_m
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
  gsl_spline*spline = pars->spline;
  gsl_interp_accel*acc = pars->acc;
  double Rp = pars->Rp;
  double Rmis = pars->Rmis;
  double arg = sqrt(Rp*Rp + Rmis*Rmis - 2*Rp*Rmis*cos(theta));
  double rmin = pars->rmin,rmax = pars->rmax;
  if (arg < rmin){
    return Sigma_nfw_at_R(arg, pars->M, pars->conc, pars->delta, pars->om);
  }else if(arg < rmax){
    return gsl_spline_eval(spline, log(arg) ,acc);
  }
  return 0;
}

/** @brief Miscentered Sigma profile at radius R in Mpc/h comoving.
 *
 *  The miscentered Sigma profile of a single cluster, given
 *  a centered  mass surface density profile, Sigma(R).
 *  Units of surface density are all in h*Msun/pc^2 comoving.
 *  This specific function just interfaces Sigma_mis_single_at_R_arr().
 *
 *  @param R Radius in Mpc/h comoving.
 *  @param Rs Radii at which we know Sigma(R), in Mpc/h comoving.
 *  @param Sigma Surface mass density profile in h*Msun/pc^2 comoving.
 *  @param Ns Number of elements in Sigma_mis and Rs.
 *  @param M Halo mass in Msun/h.
 *  @param conc Halo concentration.
 *  @param delta Halo overdensity.
 *  @param Omega_m Matter fraction.
 *  @param Rmis Halo projected offset from the true center in Mpc/h comoving.
 *  @return Sigma_mis(R) in h*Msun/pc^2 comoving.
 */
double Sigma_mis_single_at_R(double R, double*Rs, double*Sigma, int Ns, double M, double conc, int delta, double Omega_m, double Rmis){
  double Smis = 0.0;
  Sigma_mis_single_at_R_arr(&R, 1, Rs, Sigma, Ns, M, conc, delta, Omega_m, Rmis, &Smis);
  return Smis;
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
int Sigma_mis_single_at_R_arr(double*R, int NR, double*Rs, double*Sigma, int Ns, double M, double conc, int delta, double Omega_m, double Rmis, double*Sigma_mis){
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  gsl_integration_workspace*workspace = gsl_integration_workspace_alloc(workspace_size);
  gsl_integration_workspace*workspace2 = gsl_integration_workspace_alloc(workspace_size);
  integrand_params params;
  gsl_function F;
  double result, err;
  int i;
  double*lnRs = (double*)malloc(Ns*sizeof(double));
  for(i = 0; i < Ns; i++){
    lnRs[i] = log(Rs[i]);
  }
  gsl_spline_init(spline, lnRs, Sigma, Ns);
  params.acc = acc;
  params.spline = spline;
  params.workspace = workspace;
  params.workspace2 = workspace2;
  params.M = M;
  params.conc = conc;
  params.delta = delta;
  params.om = Omega_m;
  params.Rmis = Rmis;
  params.Rmis2 = Rmis*Rmis;
  params.rmin = Rs[0];
  params.rmax = Rs[Ns-1];
  params.lrmin = log(Rs[0]);
  params.lrmax = log(Rs[Ns-1]);
  F.function=&single_angular_integrand;
  F.params = &params;
  for(i = 0; i < NR; i++){
    params.Rp  = R[i];
    params.Rp2 = R[i] * R[i];
    gsl_integration_qag(&F, 0, M_PI, ABSERR, RELERR, workspace_size, KEY, workspace, &result, &err);
    Sigma_mis[i] = result/M_PI;
  }
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);
  gsl_integration_workspace_free(workspace2);
  free(lnRs);
  return 0;
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
  gsl_spline*spline = pars->spline;
  gsl_interp_accel*acc = pars->acc;
  double rmin = pars->rmin,rmax = pars->rmax;
  double Rp2 = pars->Rp2;
  double Rp_cos_theta_2 = pars->Rp_cos_theta_2;
  double arg = sqrt(Rp2 + Rc2 - Rc*Rp_cos_theta_2);
  double Sigma = 0;
  if(arg > rmin && arg < rmax){
    Sigma = gsl_spline_eval(spline, log(arg), acc);
  }else if(arg < rmin){
    Sigma = Sigma_nfw_at_R(arg, pars->M, pars->conc, pars->delta, pars->om);
  }
  return Sigma;
}

/** @brief Integrand for a stack of miscentered clusters
 *         with an exponential, rather than Raleigh, distribution.
 *
 *  @param lRc Log of the integrand parameter, Rc, in Mpc/h comoving.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand.
 */
double exp_radial_integrand(double lRc, void*params){
  integrand_params*pars = (integrand_params*)params;
  double Rc = exp(lRc);
  double Rc2 = Rc*Rc;
  double Rmis = pars->Rmis;
  double Sigma = get_Sigma(Rc, Rc2, params);
  return Rc2 * exp(-Rc/Rmis) * Sigma; //normalized outside
}

/** @brief Integrand for a stack of miscentered clusters
 *         with a Raleigh distribution.
 *
 *  @param lRc Log of the integrand parameter, Rc, in Mpc/h comoving.
 *  @param params A structure containing the splines and
 *                other inputs to the integral.
 *  @return The integrand.
 */
double g2d_radial_integrand(double lRc, void*params){
  integrand_params*pars = (integrand_params*)params;
  double Rc = exp(lRc);
  double Rc2 = Rc*Rc;
  double Rmis2 = pars->Rmis2;
  double Sigma = get_Sigma(Rc, Rc2, pars);
  return Rc2 * exp(-0.5 * Rc2/Rmis2) * Sigma; //normalized outside
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
  integrand_params*pars = (integrand_params*)params;
  double cos_theta = cos(theta);
  double lrmin = pars->lrmin, lrmax = pars->lrmax;
  gsl_integration_workspace*workspace = pars->workspace2;
  gsl_function F = pars->F_radial;
  double result, err;
  pars->Rp_cos_theta_2 = pars->Rp*cos_theta*2;
  gsl_integration_qag(&F, lrmin-10, lrmax, ABSERR, RELERR, workspace_size, KEY, workspace, &result, &err);
  return result;
}

double Sigma_mis_at_R(double R, double*Rs, double*Sigma, int Ns, double M, double conc, int delta, double om, double Rmis, int integrand_switch){
  double Sigma_mis = 0.0;
  Sigma_mis_at_R_arr(&R, 1, Rs, Sigma, Ns, M, conc, delta, om, Rmis, integrand_switch, &Sigma_mis);
  return Sigma_mis;
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
int Sigma_mis_at_R_arr(double*R, int NR, double*Rs, double*Sigma, int Ns, double M, double conc, int delta, double om, double Rmis, int integrand_switch, double*Sigma_mis){
  //Create the GSL spline objects and integration workspace.
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  gsl_integration_workspace*workspace = gsl_integration_workspace_alloc(workspace_size);
  gsl_integration_workspace*workspace2 = gsl_integration_workspace_alloc(workspace_size);
  //Create the integration_params structure, which holds extra information when performing integrals.
  //See the top of this file where the struct is defined.
  integrand_params params;
  //GSL integrals need pointers to the integrands. The variables for these functions are defined here.
  gsl_function F;
  gsl_function F_radial;
  //Define variable to hold the result and the numerical error.
  double result, err;
  int i; //iteration variable
  //Define a new array to hold the log of the radii.
  double*lnRs = (double*)malloc(Ns*sizeof(double));
  for(i = 0; i < Ns; i++){
    lnRs[i] = log(Rs[i]);
  }
  //Create the pline and add all information to the params struct.
  gsl_spline_init(spline, lnRs, Sigma, Ns);
  params.acc = acc;
  params.spline = spline;
  params.workspace = workspace;
  params.workspace2 = workspace2;
  params.M = M;
  params.conc = conc;
  params.delta = delta;
  params.om = om;
  params.Rmis = Rmis;
  params.Rmis2 = Rmis*Rmis;
  params.rmin = Rs[0];
  params.rmax = Rs[Ns-1];
  params.lrmin = log(Rs[0]);
  params.lrmax = log(Rs[Ns-1]);
  //The "outer" integral is the angular integral.
  F.function = &angular_integrand;
  //The "inner" integral is radial over R_mis.
  //The if statement below swaps between two distributions: a Rayleigh
  //distribution and a gamma distribution. See the text near eq. 40 in McClintock+ (2018).
  F_radial.function = &g2d_radial_integrand; //integrand_switch == 0
  if(integrand_switch == 1){
    F_radial.function = &exp_radial_integrand;
  }
  //Assign the params struct to the GSL functions.
  F_radial.params = &params;
  params.F_radial = F_radial;
  F.params = &params;
  //Do the integral, with Rp^2 ( precomputed.
  for(i = 0; i < NR; i++){
    params.Rp  = R[i];
    params.Rp2 = R[i] * R[i];
    //The "outer" integral.
    gsl_integration_qag(&F, 0, M_PI, ABSERR, RELERR, workspace_size, KEY, workspace, &result, &err);
    Sigma_mis[i] = result/(M_PI*Rmis*Rmis); //Normalization
  }
  //Free objects and arrays.
  gsl_spline_free(spline);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);
  gsl_integration_workspace_free(workspace2);
  free(lnRs);
  return 0; //return success
}

//////////// DELTASIGMA(R) BELOW //////////////////

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
  gsl_spline*spline = pars.spline;//Sigma(R) spline
  gsl_interp_accel*acc = pars.acc;
  return R * R * gsl_spline_eval(spline, R, acc);
}

/** @brief DeltaSigma profile at radius R in Mpc/h comoving.
 *
 *  The miscentered DeltaSigma profile of a cluster, given
 *  its miscentered mass surface density profile, Sigma_mis(R).
 *  Units of surface density are all in h*Msun/pc^2 comoving.
 *  This specific function just interfaces DeltaSigma_mis_at_R_arr().
 *
 *  @param R Radius in Mpc/h comoving.
 *  @param Rs Radii at which we know Sigma(R), in Mpc/h comoving.
 *  @param Sigma_mis Surface mass density profile in h*Msun/pc^2 comoving.
 *  @param Ns number of elements in Sigma_mis and Rs.
 *  @return DeltaSigma_mis(R) in h*Msun/pc^2 comoving.
 */
double DeltaSigma_mis_at_R(double R, double*Rs, double*Sigma_mis, int Ns){
  double DSm = 0.0;
  DeltaSigma_mis_at_R_arr(&R, 1, Rs, Sigma_mis, Ns, &DSm);
  return DSm;
}

int DeltaSigma_mis_at_R_arr(double*R, int NR, double*Rs, double*Sigma, int Ns, double*DeltaSigma_mis){
  double lrmin = log(Rs[0]);
  gsl_spline*spline = gsl_spline_alloc(gsl_interp_cspline, Ns);
  gsl_interp_accel*acc = gsl_interp_accel_alloc();
  gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(workspace_size);
  integrand_params params;
  double slope = log(Sigma[0]/Sigma[1])/log(Rs[0]/Rs[1]);
  double intercept = Sigma[0]*pow(Rs[0], -slope);
  double low_part = intercept*pow(Rs[0], slope+2)/(slope+2);
  double result,  err;
  gsl_function F;
  int i;
  gsl_spline_init(spline, Rs, Sigma, Ns);
  params.spline = spline;
  params.acc = acc;
  F.params = &params;
  F.function = &DS_mis_integrand;
  for(i = 0; i < NR; i++){
    gsl_integration_qag(&F, lrmin, log(R[i]), ABSERR, RELERR, workspace_size, KEY, workspace, &result, &err);
    DeltaSigma_mis[i] = (low_part+result)*2/(R[i]*R[i]) - gsl_spline_eval(spline, R[i], acc);
  }
  gsl_spline_free(spline),gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);
  return 0;
}
