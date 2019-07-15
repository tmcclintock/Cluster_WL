#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <math.h>

// G in Mpc^3 Msun^{-1} s^{-2}
// (source: astropy's constants module and unit conversions) 
#define G (4.51710305e-48)

// Constant used to convert from integrated pressure to the Compton y-param.
// \sigma_T / (m_e c^2) in Msun^{-1} s^2
// (source: astropy's constants module and unit conversions) 
#define P_TO_Y (1.61574202e+15)

// Critical density of the universe.
// The constant is 3 * (100 km / s / Mpc)**2 / (8 * pi * G)
// in units of Msun h^2 Mpc^{-3}
// (source: astropy's constants module and unit conversions) 
#define RHO_CRIT (2.77536627e+11)

// Computes the critical density of the universe, in units of
// RHO_CRIT. Assumes a flat (omega_k == 0) universe.
static double
rho_crit_z(double z, double omega_m)
{
    const double omega_lambda = 1.0 - omega_m,
                 inv_a = 1.0 + z;
    return RHO_CRIT * ((omega_m * inv_a * inv_a * inv_a)
                       + omega_lambda);
}

// Calculated the virial radius of a halo of mass M_delta, of overdensity
// delta, at a given redshift, in a universe of a given omega_m.
//
// Units: Mpc h^{-2/3}
static double
R_delta(double M_delta, double z, double omega_m, double delta)
{
    const double volume = M_delta / (delta * rho_crit_z(z, omega_m));
    return pow(3.0 * volume / (4.0 * M_PI), 1.0/3.0);
}

double
P_delta(double M_delta, double z, double omega_b, double omega_m, double delta)
{
    return G * M_delta * delta * rho_crit_z(z, omega_m)
        * (omega_b / omega_m) / (2 * R_delta(M_delta, z, omega_m, delta));
}

void
P_BBPS(double *P_out,
       const double *r, unsigned Nr,
       double M_delta, double z,
       // Cosmological parameters
       double omega_b, double omega_m,
       // Fit parameters
       double P_0, double x_c, double beta,
       double alpha, double gamma,
       // Halo definition
       double delta)
{
    const double R_del = R_delta(M_delta, z, omega_m, delta),
                 P_amp = P_delta(M_delta, z, omega_b, omega_m, delta);

    for (unsigned i = 0; i < Nr; i++) {
        const double x = r[i] / R_del;
        P_out[i] = P_amp * P_0 * pow(x / x_c, gamma)
                 * pow(1.0 + pow(x / x_c, alpha), -beta);
    }
}

int
projected_P_BBPS(double *P_out, double *P_err_out,
                 // Inputs - array of radii, number of radii, mass and redshift
                 const double *r, unsigned Nr, double M_delta, double z,
                 // Cosmological parameters
                 double omega_b, double omega_m,
                 // Fit parameters
                 double P_0, double x_c, double beta,
                 double alpha, double gamma,
                 // Halo definition
                 double delta,
                 // Integration param
                 unsigned limit,
                 double epsabs, double epsrel)
{
    if ((P_out == NULL) || (r == NULL))
        return GSL_FAILURE;

    gsl_set_error_handler_off();
    gsl_integration_workspace *wkspc = gsl_integration_workspace_alloc(limit);
    if (!wkspc)
        return GSL_FAILURE;

    double
    integrand(double chi, void *params)
    {
        const double this_r = *((const double *) params),
                     central_distance = sqrt(this_r*this_r + chi*chi);
        double P = 0.0;
        P_BBPS(&P,
               &central_distance, 1,
               M_delta, z,
               omega_b, omega_m,
               P_0, x_c, beta,
               alpha, gamma,
               delta);
        return P;
    }

    gsl_function fn;
    fn.params = NULL;
    fn.function = integrand;

    int retcode = GSL_SUCCESS;
    for (unsigned i = 0; i < Nr; i++) {
        double this_r = r[i];
        fn.params = &this_r;
        double result = 0.0, err = 0.0;
        retcode = gsl_integration_qagi(&fn,
                                       epsabs, epsrel, limit,
                                       wkspc, &result, &err);

        // Handle any errors
        if (retcode != GSL_SUCCESS)
            break;

        P_out[i] = result / (1 + z);
        if (P_err_out)
            P_err_out[i] = err;
    }

    gsl_integration_workspace_free(wkspc);
    return retcode;
}


// Computes the 3D fourier transform of the BBPS pressure profile, at a
// series of wavenumbers `k`.
int
fourier_P_BBPS(double *up_out, double *up_err_out,
               const double *ks, unsigned Nk,
               double M_delta,
               double z, double omega_b, double omega_m,
               double P_0, double x_c, double beta,
               double alpha, double gamma, double delta,
               unsigned limit, double epsabs)
{
    if ((up_out == NULL) || (ks == NULL))
        return GSL_FAILURE;

    gsl_set_error_handler_off();
    // Allocate our needed workspaces
    gsl_integration_workspace *wkspc = gsl_integration_workspace_alloc(limit);

    // We also need a "cycle workspace" for the QAWF algorithm
    gsl_integration_workspace *cycle = gsl_integration_workspace_alloc(limit);

    // L is ignored by the function `gsl_integration_qawf`. So make it 1 full cycle for now.
    gsl_integration_qawo_table *tbl =
        gsl_integration_qawo_table_alloc(ks[0], M_2_PI / ks[0], GSL_INTEG_SINE, limit);
    if (!wkspc || !cycle || !tbl)
        return GSL_FAILURE;

    double
    integrand(double r, void *params)
    {
        const double k = *((const double *) params);
        double P = 0.0;
        P_BBPS(&P,
               &r, 1,
               M_delta, z,
               omega_b, omega_m,
               P_0, x_c, beta,
               alpha, gamma,
               delta);
        return P * 4 * M_PI * (r / k);
    }

    gsl_function fn;
    fn.params = NULL;
    fn.function = integrand;

    int retcode = GSL_SUCCESS;
    for (unsigned i = 0; i < Nk; i++) {
        double this_k = ks[i];
        fn.params = &this_k;
        double result = 0.0, err = 0.0;

        // Update table for new iteration speed `k`. Again, L is ignored, so we
        // make it 1 full cycle for simplicity.
        retcode = gsl_integration_qawo_table_set(tbl, this_k, M_2_PI / this_k, GSL_INTEG_SINE);
        if (retcode != GSL_SUCCESS)
            break;

        // The qawf function performs a Fourier transform
        retcode = gsl_integration_qawf(&fn,
                                       // Integrate from 0
                                       0.0,
                                       // Algorithm precision parameters
                                       epsabs, limit,
                                       // Workspace & table for sinusoid integration
                                       wkspc, cycle, tbl,
                                       // Results
                                       &result, &err);

        // Handle any errors
        if (retcode != GSL_SUCCESS)
            break;

        up_out[i] = result;
        if (up_err_out)
            up_err_out[i] = err;
    }

    gsl_integration_qawo_table_free(tbl);
    gsl_integration_workspace_free(wkspc);
    return retcode;
}
