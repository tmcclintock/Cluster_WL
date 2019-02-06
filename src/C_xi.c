#include "C_xi.h"
#include "C_peak_height.h"
#include "C_power.h"

#include "gsl/gsl_integration.h"
#include "gsl/gsl_spline.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_errno.h"
#include <math.h>
#include <stdio.h>

#define rhomconst 2.77533742639e+11
//1e4*3.*Mpcperkm*Mpcperkm/(8.*PI*G); units are SM h^2/Mpc^3


/** @brief The NFW correlation function profile.
 * 
 *  The NFW correlation function profile of a halo a distance R from the center,
 *  assuming the halo has a given mass and concentration. It works 
 *  with any overdensity parameter and arbitrary matter fraction.
 *  This function calls xi_nfw_at_R_arr().
 * 
 *  @param r Distance from the center of the halo in Mpc/h comoving.
 *  @param M Halo mass in Msun/h.
 *  @param c Concentration.
 *  @delta Halo overdensity.
 *  @Omega_m Matter fraction.
 *  @return NFW halo correlation function.
 */
double xi_nfw_at_R(double R, double Mass, double conc, int delta, double om){
  double xi = 0.0;
  calc_xi_nfw(&R, 1, Mass, conc, delta, om, &xi);
  return xi;
}

int calc_xi_nfw(double*R, int NR, double Mass, double conc, int delta, double om, double*xi_nfw){
  int i;
  double rhom = om*rhomconst;//SM h^2/Mpc^3
  double Rdelta = pow(Mass/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  double Rscale = Rdelta/conc;
  double fc = log(1.+conc)-conc/(1.+conc);
  double R_Rs;
  for(i = 0; i < NR; i++){
    R_Rs = R[i]/Rscale;
    xi_nfw[i] = Mass/(4.*M_PI*Rscale*Rscale*Rscale*fc)/(R_Rs*(1+R_Rs)*(1+R_Rs))/rhom - 1.0;
  }
  return 0;
}

double rhos_einasto_at_M(double Mass, double conc, double alpha, int delta, double om){
  double rhom = om*rhomconst;//Msun h^2/Mpc^3
  // Rdelta in Mpc/h comoving
  double Rdelta = pow(Mass/(1.3333333333333*M_PI*rhom*delta), 0.333333333333);
  double rs = Rdelta/conc; //Scale radius in Mpc/h
  double x = 2./alpha * pow(conc, alpha); //pow(Rdelta/rs, alpha); 
  double a = 3./alpha;
  //printf("alpha = %.3e    x = %.3e    R\n",alpha, x);fflush(stdout);
  double gam = gsl_sf_gamma(a) - gsl_sf_gamma_inc(a, x);
  double num = delta * rhom * Rdelta*Rdelta*Rdelta * alpha * pow(2./alpha, a);
  double den = 3. * rs*rs*rs * gam;
  return num/den;
}

double xi_einasto_at_R(double R, double Mass, double rhos, double conc, double alpha, int delta, double om){
  double xi = 0.0;
  calc_xi_einasto(&R, 1, Mass, rhos, conc, alpha, delta, om, &xi);
  return xi;
}

int calc_xi_einasto(double*R, int NR, double Mass, double rhos, double conc, double alpha, int delta, double om, double*xi_einasto){
  double rhom = om*rhomconst;//SM h^2/Mpc^3
  double Rdelta = pow(Mass/(1.3333333333333*M_PI*rhom*delta), 0.333333333333);
  double rs = Rdelta/conc; //Scale radius in Mpc/h
  double x;
  int i;
  if (rhos < 0)
    rhos = rhos_einasto_at_M(Mass, conc, alpha, delta, om);
  for(i = 0; i < NR; i++){
    x = 2./alpha * pow(R[i]/rs, alpha);
    xi_einasto[i] = rhos/rhom * exp(-x) - 1;
  }
  return 0;
}

int calc_xi_2halo(int NR, double bias, double*xi_mm, double*xi_2halo){
  int i;
  for(i = 0; i < NR; i++){
    xi_2halo[i] = bias * xi_mm[i];
  }
  return 0;
}

int calc_xi_hm(int NR, double*xi_1h, double*xi_2h, double*xi_hm, int flag){
  //Flag specifies how to combine the two terms
  int i;
  if (flag == 0) { //Take the max
    for(i = 0; i < NR; i++){
      if(xi_1h[i] >= xi_2h[i]) xi_hm[i] = xi_1h[i];
      else xi_hm[i] = xi_2h[i];
    }
  } else if (flag == 1) { //Take the sum
    for(i = 0; i < NR; i++){
      xi_hm[i] = 1 + xi_1h[i] + xi_2h[i];
    }
  }
  return 0;
}

int calc_xi_mm(double*R, int NR, double*k, double*P, int Nk, double*xi, int N, double h){
  int i,j;
  double PI_h = M_PI/h;
  double PI_2 = M_PI*0.5;
  gsl_spline*Pspl = gsl_spline_alloc(gsl_interp_cspline, Nk);
  gsl_interp_accel*acc= gsl_interp_accel_alloc();
  gsl_spline_init(Pspl, k, P, Nk);
  double sum;
  
  static int init_flag = 0;
  static double h_old = -1;
  static int N_old = -1;
  static double*x      = NULL;
  static double*xsinx  = NULL;
  static double*dpsi   = NULL;
  static double*xsdpsi = NULL;
  double t, psi, PIsinht;
  if ((init_flag == 0) || (h_old != h) || (N_old < N)){
    h_old = h;
    N_old = N;
    init_flag = 1; //been initiated
    
    if (x!=NULL)      free(x);
    if (xsinx!=NULL)  free(xsinx);
    if (dpsi!=NULL)   free(dpsi);
    if (xsdpsi!=NULL) free(xsdpsi);
    
    x      = malloc(N*sizeof(double));
    xsinx  = malloc(N*sizeof(double));
    dpsi   = malloc(N*sizeof(double));
    xsdpsi = malloc(N*sizeof(double));
    for(i = 0; i < N; i++){
      t = h*(i+1);
      psi = t*tanh(sinh(t)*PI_2);
      x[i] = psi*PI_h;
      xsinx[i] = x[i]*sin(x[i]);
      PIsinht = M_PI*sinh(t);
      dpsi[i] = (M_PI*t*cosh(t) + sinh(PIsinht))/(1+cosh(PIsinht));
      if (dpsi[i]!=dpsi[i]) dpsi[i]=1.0;
      xsdpsi[i] = xsinx[i]*dpsi[i];
    }
  }
  for(j = 0; j < NR; j++){
    sum = 0;
    for(i = 0; i < N; i++){
      sum += xsdpsi[i] * get_P(x[i], R[j], k, P, Nk, Pspl, acc);
    }
    xi[j] = sum/(R[j]*R[j]*R[j]*M_PI*2);
  }
  
  /* //original code below, left in for testing for now
  double zero,psi,x,t,dpsi,f,PIsinht;
  for(j = 0; j < NR; j++){
    sum = 0;
    for(i = 0; i < N; i++){
      zero = i+1;
      psi = h*zero*tanh(sinh(h*zero)*PI_2);
      x = psi*PI_h;
      t = h*zero;
      PIsinht = M_PI*sinh(t);
      dpsi = (M_PI*t*cosh(t)+sinh(PIsinht))/(1+cosh(PIsinht));
      if (dpsi != dpsi) dpsi=1.0;
      f = x*get_P(x,R[j],k,P,Nk,Pspl,acc);
      sum += f*sin(x)*dpsi;
    }
    xi[j] = sum/(R[j]*R[j]*R[j]*M_PI*2);
  }
  */

  gsl_spline_free(Pspl);
  gsl_interp_accel_free(acc);
  return 0; //Note: factor of pi picked up in the quadrature rule
  //See Ogata 2005 for details, especially eq. 5.2
}

///////Functions for calc_xi_mm/////////

double xi_mm_at_R(double R, double*k, double*P, int Nk, int N, double h){
  double xi = 0.0;
  calc_xi_mm(&R, 1, k, P, Nk, &xi, N, h);
  return xi;
}

//////////////////////////////////////////
//////////////Xi(R) exact below //////////
//////////////////////////////////////////

#define workspace_size 8000
#define workspace_num 100
#define ABSERR 0.0
#define RELERR 1.8e-4

typedef struct integrand_params_xi_mm_exact{
  gsl_spline*spline;
  gsl_interp_accel*acc;
  gsl_integration_workspace * workspace;
  double r; //3d r; Mpc/h, or inverse units of k
  double*kp; //pointer to wavenumbers
  double*Pp; //pointer to P(k) array
  int Nk; //length of k and P arrays
}integrand_params_xi_mm_exact;

double integrand_xi_mm_exact(double k, void*params){
  integrand_params_xi_mm_exact pars = *(integrand_params_xi_mm_exact*)params;
  gsl_spline*spline = pars.spline;
  gsl_interp_accel*acc = pars.acc;
  double*kp = pars.kp;
  double*Pp = pars.Pp;
  int Nk = pars.Nk;
  double R = pars.r;
  double x  = k*R;
  double P = get_P(x, R, kp, Pp, Nk, spline, acc);
  return P*k/R; //Note - sin(kR) is taken care of in the qawo table
}

int calc_xi_mm_exact(double*R, int NR, double*k, double*P, int Nk, double*xi){
  gsl_spline*Pspl = gsl_spline_alloc(gsl_interp_cspline, Nk);
  gsl_interp_accel*acc= gsl_interp_accel_alloc();
  gsl_integration_workspace*workspace = gsl_integration_workspace_alloc(workspace_size);
  gsl_integration_qawo_table*wf;
  integrand_params_xi_mm_exact params;
  gsl_function F;
  double kmax = 4e3;
  double kmin = 5e-8;
  double result, err;
  int i;
  int status;
  gsl_spline_init(Pspl, k, P, Nk);
  params.acc = acc;
  params.spline = Pspl;
  params.kp = k;
  params.Pp = P;
  params.Nk = Nk;

  F.function = &integrand_xi_mm_exact;
  F.params = &params;

  wf = gsl_integration_qawo_table_alloc(R[0], kmax-kmin, GSL_INTEG_SINE, (size_t)workspace_num);
  for(i = 0; i < NR; i++){
    status = gsl_integration_qawo_table_set(wf, R[i], kmax-kmin, GSL_INTEG_SINE);
    if (status){
      printf("Error in calc_xi_mm_exact, first integral.\n");
      exit(-1);
    }
    params.r=R[i];
    status = gsl_integration_qawo(&F, kmin, ABSERR, RELERR, (size_t)workspace_num, workspace, wf, &result, &err);
    if (status){
      printf("Error in calc_xi_mm_exact, second integral.\n");
      exit(-1);
    }

    xi[i] = result/(M_PI*M_PI*2);
  }

  gsl_spline_free(Pspl);
  gsl_interp_accel_free(acc);
  gsl_integration_workspace_free(workspace);
  gsl_integration_qawo_table_free(wf);

  return 0;
}

double xi_mm_at_R_exact(double R, double*k, double*P, int Nk){
  double xi = 0.0;
  calc_xi_mm_exact(&R, 1, k, P, Nk, &xi);
  return xi;
}

/*
 * Diemer-Kravtsov 2014 profiles below.
 */

int calc_xi_DK(double*R, int NR, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double*xi){
  double rhom = rhomconst*om; //SM h^2/Mpc^3
  //Compute R200m
  double Rdelta = pow(M/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  //double rs = Rdelta / conc; //compute scale radius from concentration
  xi[0] = 0;
  double*rho_ein = malloc(NR*sizeof(double));
  double*f_trans = malloc(NR*sizeof(double));
  double*rho_outer = malloc(NR*sizeof(double));
  int i;
  double nu = nu_at_M(M, k, P, Nk, om);
  if (alpha < 0){ //means it wasn't passed in
    alpha = 0.155 + 0.0095*nu*nu;
  }
  if (beta < 0){ //means it wasn't passed in
    beta = 4;
  }
  if (gamma < 0){ //means it wasn't passed in
    gamma = 8;
  }
  if (rhos < 0){ //means it wasn't passed in
    rhos = rhos_einasto_at_M(M, conc, alpha, delta, om);
  }
  double g_b = gamma/beta;
  double r_t = (1.9-0.18*nu)*Rdelta; //NEED nu for this
  calc_xi_einasto(R, NR, M, rhos, conc, alpha, delta, om, rho_ein);
  //here convert xi_ein to rho_ein
  for(i = 0; i < NR; i++){
    rho_ein[i] = rhom*(1+rho_ein[i]); //rho_ein had xi_ein in it
    f_trans[i] = pow(1+pow(R[i]/r_t,beta), -g_b);
    rho_outer[i] = rhom*(be*pow(R[i]/(5*Rdelta), -se) + 1);
    xi[i] = (rho_ein[i]*f_trans[i] + rho_outer[i])/rhom - 1;
  }
  free(rho_ein);
  free(f_trans);
  free(rho_outer);
  return 0;
}

double xi_DK(double R, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om){
  double xi = 0.0;
  calc_xi_DK(&R, 1, M, rhos, conc, be, se, alpha, beta, gamma, delta, k, P, Nk, om, &xi);
  return xi;
}

//////////////////////////////
//////Appendix version 1//////
//////////////////////////////
int calc_xi_DK_app1(double*R, int NR, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double bias, double*xi_mm, double*xi){
  double rhom = rhomconst*om; //SM h^2/Mpc^3
  //Compute R200m
  double Rdelta = pow(M/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  //double rs = Rdelta / conc; //compute scale radius from concentration
  xi[0] = 0;
  double*rho_ein = malloc(NR*sizeof(double));
  double*f_trans = malloc(NR*sizeof(double));
  double*rho_outer = malloc(NR*sizeof(double));
  int i;
  double nu = nu_at_M(M, k, P, Nk, om);
  if (alpha < 0){ //means it wasn't passed in
    alpha = 0.155 + 0.0095*nu*nu;
  }
  if (beta < 0){ //means it wasn't passed in
    beta = 4;
  }
  if (gamma < 0){ //means it wasn't passed in
    gamma = 8;
  }
  if (rhos < 0){ //means it wasn't passed in
    rhos = rhos_einasto_at_M(M, conc, alpha, delta, om);
  }
  double g_b = gamma/beta;
  double r_t = (1.9-0.18*nu)*Rdelta; //NEED nu for this
  calc_xi_einasto(R, NR, M, rhos, conc, alpha, delta, om, rho_ein);
  //here convert xi_ein to rho_ein
  for(i = 0; i < NR; i++){
    rho_ein[i] = rhom*(1+rho_ein[i]); //rho_ein had xi_ein in it
    f_trans[i] = pow(1+pow(R[i]/r_t,beta), -g_b);
    rho_outer[i] = rhom*(be*pow(R[i]/(5*Rdelta), -se) * bias * xi_mm[i] + 1);
    xi[i] = (rho_ein[i]*f_trans[i] + rho_outer[i])/rhom - 1;
  }
  free(rho_ein);
  free(f_trans);
  free(rho_outer);
  return 0;
}

double xi_DK_app1(double R, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double bias, double*xi_mm){
  double xi = 0.0;
  calc_xi_DK_app1(&R, 1, M, rhos, conc, be, se, alpha, beta, gamma, delta, k, P, Nk, om, bias, xi_mm, &xi);
  return xi;
}

//////////////////////////////
//////Appendix version 2//////
//////////////////////////////
int calc_xi_DK_app2(double*R, int NR, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double bias, double*xi_mm, double*xi){
  double rhom = rhomconst*om; //SM h^2/Mpc^3
  //Compute R200m
  double Rdelta = pow(M/(1.33333333333*M_PI*rhom*delta), 0.33333333333);
  //double rs = Rdelta / conc; //compute scale radius from concentration
  xi[0] = 0;
  double*rho_ein = malloc(NR*sizeof(double));
  double*f_trans = malloc(NR*sizeof(double));
  double*rho_outer = malloc(NR*sizeof(double));
  int i;
  double nu = nu_at_M(M, k, P, Nk, om);
  if (alpha < 0){ //means it wasn't passed in
    alpha = 0.155 + 0.0095*nu*nu;
  }
  if (beta < 0){ //means it wasn't passed in
    beta = 4;
  }
  if (gamma < 0){ //means it wasn't passed in
    gamma = 8;
  }
  if (rhos < 0){ //means it wasn't passed in
    rhos = rhos_einasto_at_M(M, conc, alpha, delta, om);
  }
  double g_b = gamma/beta;
  double r_t = (1.9-0.18*nu)*Rdelta; //NEED nu for this
  calc_xi_einasto(R, NR, M, rhos, conc, alpha, delta, om, rho_ein);
  //here convert xi_ein to rho_ein
  for(i = 0; i < NR; i++){
    rho_ein[i] = rhom*(1+rho_ein[i]); //rho_ein had xi_ein in it
    f_trans[i] = pow(1+pow(R[i]/r_t,beta), -g_b);
    rho_outer[i] = rhom*((1+be*pow(R[i]/(5*Rdelta), -se))*bias*xi_mm[i] + 1);
    xi[i] = (rho_ein[i]*f_trans[i] + rho_outer[i])/rhom - 1;
  }
  free(rho_ein);
  free(f_trans);
  free(rho_outer);
  return 0;
}

double xi_DK_app2(double R, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double bias, double*xi_mm){
  double xi = 0.0;
  calc_xi_DK_app2(&R, 1, M, rhos, conc, be, se, alpha, beta, gamma, delta, k, P, Nk, om, bias, xi_mm, &xi);
  return xi;
}
