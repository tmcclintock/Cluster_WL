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
    if (P_out == NULL)
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

    for (unsigned i = 0; i < Nr; i++) {
        double this_r = r[i];
        fn.params = &this_r;
        double result = 0.0, err = 0.0;
        int retcode = gsl_integration_qagi(&fn,
                                           epsabs, epsrel, limit,
                                           wkspc, &result, &err);

        // Handle any errors
        if (retcode != GSL_SUCCESS) {
            gsl_integration_workspace_free(wkspc);
            return retcode;
        }

        P_out[i] = result / (1 + z);
        if (P_err_out)
            P_err_out[i] = err;
    }

    gsl_integration_workspace_free(wkspc);
    return GSL_SUCCESS;
}
