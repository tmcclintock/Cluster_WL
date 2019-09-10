#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_sf_bessel.h>
#include <math.h>

#include "C_pressure_profile.h"

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


struct interp_integrand_args {
    gsl_interp *F;
    const double *xs;
    const double *ys;
    int retcode;
};

static double interp_integrand(double x, void *params);
static int interp_integrand_args_alloc(struct interp_integrand_args *args,
                                       const double *xs, const double *ys,
                                       unsigned N);
static int interp_integrand_args_free(struct interp_integrand_args *);

// Computes the (unnormalized) Fourier transform of a spherically symmetric
// function f_r at a set of wavenumbers ks.
static int
spherical_fourier_transform(double *out, double *out_err,
                            const double *ks, unsigned Nk,
                            gsl_function *f_r,
                            unsigned limit, double epsabs);

// Computes the (unnormalized) Fourier transform of a circularly symmetric
// function f_r at a set of wavenumbers ks.
static int
circular_fourier_transform(double *out, double *out_err,
                           const double *ks, unsigned Nk,
                           gsl_function *f_r,
                           unsigned limit, double epsabs, double epsrel);

// Computes the line-of-sight projection of a spherically symmetric function
// f_r at a set of transverse radii rs.
static int
abel_transform(double *out, double *out_err,
               const double *rs, unsigned Nr,
               gsl_function *f_r,
               unsigned limit, double epsabs, double epsrel);

// Computes the critical density of the universe, in units of
// RHO_CRIT. Assumes a flat (omega_k == 0) universe.
static double
rho_crit_z(double z, double omega_m, double h)
{
    const double omega_lambda = 1.0 - omega_m,
                 inv_a = 1.0 + z;
    return RHO_CRIT * h*h *((omega_m * inv_a * inv_a * inv_a)
                             + omega_lambda);
}

// Calculated the virial radius of a halo of mass M_delta, of overdensity
// delta, at a given redshift, in a universe of a given omega_m.
//
// Units: Mpc h^{-2/3}
static double
R_delta(double M_delta, double z, double omega_m, double h, double delta)
{
    const double volume = M_delta / (delta * rho_crit_z(z, omega_m, h));
    return pow(3.0 * volume / (4.0 * M_PI), 1.0/3.0);
}

double
P_delta(double M_delta, double z, double omega_b, double omega_m, double h, double delta)
{
    return G * M_delta * delta * rho_crit_z(z, omega_m, h)
        * (omega_b / omega_m) / (2 * R_delta(M_delta, z, omega_m, h, delta));
}

void
P_BBPS(double *P_out,
       const double *r, unsigned Nr,
       double M_delta, double z,
       // Cosmological parameters
       double omega_b, double omega_m, double h,
       // Fit parameters
       double P_0, double x_c, double beta,
       double alpha, double gamma,
       // Halo definition
       double delta)
{
    const double R_del = R_delta(M_delta, z, omega_m, h, delta),
                 P_amp = P_delta(M_delta, z, omega_b, omega_m, h, delta);

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
                 double omega_b, double omega_m, double h,
                 // Fit parameters
                 double P_0, double x_c, double beta,
                 double alpha, double gamma,
                 // Halo definition
                 double delta,
                 // Integration param
                 unsigned limit,
                 double epsabs, double epsrel)
{
    double
    integrand(double chi, void *params)
    {
        double P = 0.0;
        P_BBPS(&P,
               &chi, 1,
               M_delta, z,
               omega_b, omega_m, h,
               P_0, x_c, beta,
               alpha, gamma,
               delta);
        return P;
    }

    gsl_function fn;
    fn.params = NULL;
    fn.function = integrand;

    int rc = abel_transform(P_out, P_err_out,
                            r, Nr,
                            &fn,
                            limit, epsabs, epsrel);

    if (rc != GSL_SUCCESS)
        return rc;

    for (unsigned i = 0; i < Nr; i++) {
        P_out[i] /= 1 + z;
        if (P_err_out)
            P_err_out[i] /= 1 + z;
    }

    return rc;
}


// Computes the 3D fourier transform of the BBPS pressure profile, at a
// series of wavenumbers `k`.
int
fourier_P_BBPS(double *up_out, double *up_err_out,
               const double *ks, unsigned Nk,
               double M_delta,
               double z, double omega_b, double omega_m, double h,
               double P_0, double x_c, double beta,
               double alpha, double gamma, double delta,
               unsigned limit, double epsabs)
{
    double
    integrand(double r, void *params)
    {
        double P = 0.0;
        P_BBPS(&P, &r, 1,
               M_delta, z,
               omega_b, omega_m, h,
               P_0, x_c, beta,
               alpha, gamma,
               delta);
        return P;
    }

    gsl_function f_r;
    f_r.params = NULL;
    f_r.function = integrand;

    int rc = spherical_fourier_transform(up_out, up_err_out, ks, Nk,
                                         &f_r,
                                         limit, epsabs / (4 * M_PI));

    if (rc != GSL_SUCCESS)
        return rc;

    // Apply normalization for forward Fourier transform
    // (See comment above spherical_fourier_transform)
    for (unsigned i = 0; i < Nk; i++) {
        up_out[i] *= 4 * M_PI;
        if (up_err_out)
            up_err_out[i] *= 4 * M_PI;
    }

    return rc;
}


/// Performs an inverse fourier-transform on the function F(k)
/// specified in the table of ks and Fs
int
inverse_spherical_fourier_transform(double *out, double *out_err,
                                    const double *rs, unsigned Nr,
                                    const double *ks, const double *Fs, unsigned Nk,
                                    unsigned limit, double epsabs)
{
    struct interp_integrand_args args;
    int rc = interp_integrand_args_alloc(&args, ks, Fs, Nk);
    if (rc != GSL_SUCCESS)
        return rc;

    gsl_function f_k;
    f_k.params = &args;
    f_k.function = interp_integrand;

    rc = spherical_fourier_transform(out, out_err,
                                     rs, Nr,
                                     &f_k,
                                     limit, epsabs * 2 * M_PI * M_PI);

    // Handle errors in integrand (as opposed to integrator)
    if (args.retcode != GSL_SUCCESS)
        rc = args.retcode;

    if (rc == GSL_SUCCESS) {
        // Apply normalization for reverse Fourier transform
        // (See comment above spherical_fourier_transform)
        for (unsigned i = 0; i < Nr; i++) {
            out[i] /= 2 * M_PI * M_PI;
            if (out_err)
                out_err[i] /= 2 * M_PI * M_PI;
        }
    }

    interp_integrand_args_free(&args);
    return rc;
}


/// Performs a forward fourier-transform on the function f(r)
/// specified in the table of rs and fs
int
forward_spherical_fourier_transform(double *out, double *out_err,
                                    const double *ks, unsigned Nk,
                                    const double *rs, const double *fs, unsigned Nr,
                                    unsigned limit, double epsabs)
{
    struct interp_integrand_args args;
    int rc = interp_integrand_args_alloc(&args, rs, fs, Nr);
    if (rc != GSL_SUCCESS)
        return rc;

    gsl_function f_r;
    f_r.params = &args;
    f_r.function = interp_integrand;

    rc = spherical_fourier_transform(out, out_err,
                                     ks, Nk,
                                     &f_r,
                                     limit, epsabs / (4 * M_PI));

    // Handle integrand error
    if (args.retcode != GSL_SUCCESS)
        rc = args.retcode;

    if (rc == GSL_SUCCESS) {
        // Apply normalization for reverse Fourier transform
        // (See comment above spherical_fourier_transform)
        for (unsigned i = 0; i < Nk; i++) {
            out[i] *= 4 * M_PI;
            if (out_err)
                out_err[i] *= 4 * M_PI;
        }
    }

    interp_integrand_args_free(&args);
    return rc;
}


/// Performs a (2D) inverse fourier-transform on the function F(k)
/// specified in the table of ks and Fs
int
inverse_circular_fourier_transform(double *out, double *out_err,
                                    const double *rs, unsigned Nr,
                                    const double *ks, const double *Fs, unsigned Nk,
                                    unsigned limit, double epsabs, double epsrel)
{
    struct interp_integrand_args args;
    int rc = interp_integrand_args_alloc(&args, ks, Fs, Nk);
    if (rc != GSL_SUCCESS)
        return rc;

    gsl_function f_k;
    f_k.params = &args;
    f_k.function = interp_integrand;

    rc = circular_fourier_transform(out, out_err,
                                    rs, Nr,
                                    &f_k,
                                    limit, epsabs * 2 * M_PI, epsrel);

    // Handle errors in integrand (as opposed to integrator)
    if (args.retcode != GSL_SUCCESS)
        rc = args.retcode;

    if (rc == GSL_SUCCESS) {
        // Apply normalization for reverse Fourier transform
        // (See comment above spherical_fourier_transform)
        for (unsigned i = 0; i < Nr; i++) {
            out[i] /= 2 * M_PI;
            if (out_err)
                out_err[i] /= 2 * M_PI;
        }
    }

    interp_integrand_args_free(&args);
    return rc;
}


/// Performs a forward (2D) fourier-transform on the function f(r)
/// specified in the table of rs and fs
int
forward_circular_fourier_transform(double *out, double *out_err,
                                   const double *ks, unsigned Nk,
                                   const double *rs, const double *fs, unsigned Nr,
                                   unsigned limit, double epsabs, double epsrel)
{
    struct interp_integrand_args args;
    int rc = interp_integrand_args_alloc(&args, rs, fs, Nr);
    if (rc != GSL_SUCCESS)
        return rc;

    gsl_function f_r;
    f_r.params = &args;
    f_r.function = interp_integrand;

    rc = circular_fourier_transform(out, out_err,
                                    ks, Nk,
                                    &f_r,
                                    limit,
                                    epsabs / (2 * M_PI),
                                    epsrel);

    // Handle errors in integrand (as opposed to integrator)
    if (args.retcode != GSL_SUCCESS)
        rc = args.retcode;

    if (rc == GSL_SUCCESS) {
        // Apply normalization for reverse Fourier transform
        // (See comment above spherical_fourier_transform)
        for (unsigned i = 0; i < Nk; i++) {
            out[i] *= 2 * M_PI;
            if (out_err)
                out_err[i] *= 2 * M_PI;
        }
    }

    interp_integrand_args_free(&args);
    return rc;
}

int
integrate_spline(const double *xs, const double *ys, unsigned Ny,
                 double a, double b,
                 double *result)
{
    gsl_interp *F_interp = gsl_interp_alloc(gsl_interp_cspline, Ny);
    if (!F_interp)
        return GSL_ENOMEM;

    int rc = gsl_interp_init(F_interp, xs, ys, Ny);
    if (rc != GSL_SUCCESS)
        return rc;

    return gsl_interp_eval_integ_e(F_interp, xs, ys, a, b, NULL, result);
}


// Computes the Abel transform (LOS projection) of a function defined on the
// grid (r_grid, f_r), at a series of transverse points (rs, Nr).
int
abel_transform_interp(double *out, double *out_err,
                      const double *r_grid, const double *f_r, unsigned Nr_grid,
                      const double *rs, unsigned Nr,
                      unsigned limit, double epsabs, double epsrel)
{
    gsl_interp *F_interp = gsl_interp_alloc(gsl_interp_cspline, Nr_grid);

    if (!F_interp)
        return GSL_ENOMEM;

    int rc = gsl_interp_init(F_interp, r_grid, f_r, Nr_grid);
    if (rc != GSL_SUCCESS)
        return rc;

    double integrand(double r, void *params) {
        if (r < r_grid[0])
            return f_r[0];

        if (r > r_grid[Nr_grid - 1])
            return 0.0;

        return gsl_interp_eval(F_interp, r_grid, f_r, r, NULL);
    }

    gsl_function fn;
    fn.params = NULL;
    fn.function = integrand;

    return abel_transform(out, out_err,
                          rs, Nr,
                          &fn,
                          limit, epsabs, epsrel);
}


/// Computes the Fourier transform of a spherically symmetric distribution by:
///
/// F(k) = \int_0^\infty dr 4pi r^2 j_0(kr) f(r)
///      = \int_0^\infty dr 4pi r^2 (sin(kr)/kr) f(r)
///      = 4pi / k \int_0^\infty dr r sin(kr) f(r)
///
/// With no normalization factor (so, to compute F(k), multiply by 4pi). The
/// inverse of this is:
///
/// f(r) = \int_0^\infty dk/(2pi^2) k^2 j_0(kr) F(k)
///      = \int_0^\infty dk/(2pi^2) k^2 (sin(kr)/kr) F(k)
///      = 1 / (2 pi^2 r) \int_0^\infty dk k sin(kr) F(k)
///
/// So, to compute f(r) from F(k), use this function then divide by 2 pi^2.
static int
spherical_fourier_transform(double *out, double *out_err,
                            const double *ks, unsigned Nk,
                            gsl_function *f_r,
                            unsigned limit, double epsabs)
{
    if ((out == NULL) || (ks == NULL))
        return GSL_FAILURE;

    gsl_set_error_handler_off();
    // Allocate our needed workspaces
    gsl_integration_workspace *wkspc = gsl_integration_workspace_alloc(limit);

    // We also need a "cycle workspace" for the QAWF algorithm
    gsl_integration_workspace *cycle = gsl_integration_workspace_alloc(limit);

    // L is ignored by the function `gsl_integration_qawf`. So make it 1 full cycle for now.
    gsl_integration_qawo_table *tbl =
        gsl_integration_qawo_table_alloc(ks[0], 2 * M_PI / ks[0], GSL_INTEG_SINE, limit);
    if (!wkspc || !cycle || !tbl)
        return GSL_ENOMEM;

    // Our integrand is (f(r) * r / k)
    double
    integrand(double r, void *params)
    {
        const double k = *((const double *) params);
        const double f = GSL_FN_EVAL(f_r, r);
        return f * r / k;
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
        retcode = gsl_integration_qawo_table_set(tbl, this_k, 2 * M_PI / this_k, GSL_INTEG_SINE);
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

        out[i] = result;
        if (out_err)
            out_err[i] = err;
    }

    // Clean up
    gsl_integration_qawo_table_free(tbl);
    gsl_integration_workspace_free(cycle);
    gsl_integration_workspace_free(wkspc);
    return retcode;
}


/// Computes the Fourier transform of a circularly symmetric distribution by:
///
/// F(k) = \int_0^\infty dr 2pi r J_0(kr) f(r)
///
/// Where J_0 is the Bessel function of the first kind.
/// With no normalization factor (so, to compute F(k), multiply by 2pi). The
/// inverse of this is:
///
/// f(r) = \int_0^\infty dk/(2pi) k J_0(kr) F(k)
///
/// So, to compute f(r) from F(k), use this function then divide by 2 pi^2.
static int
circular_fourier_transform(double *out, double *out_err,
                            const double *ks, unsigned Nk,
                            gsl_function *f_r,
                            unsigned limit, double epsabs, double epsrel)
{
    if ((out == NULL) || (ks == NULL))
        return GSL_FAILURE;

    gsl_set_error_handler_off();
    // Allocate our needed workspaces
    gsl_integration_workspace *wkspc = gsl_integration_workspace_alloc(limit);

    if (!wkspc)
        return GSL_ENOMEM;

    struct circular_ft_integrand_params {
        double k;
        gsl_function *f_r;
        int retcode;
    };

    // Our integrand is (f(r) * r * J_0(kr))
    double
    integrand(double r, void *params)
    {
        struct circular_ft_integrand_params *p = params;

        const double k = p->k;
        gsl_function *f_r = p->f_r;
        const double f = GSL_FN_EVAL(f_r, r);

        gsl_sf_result res;
        int retcode = gsl_sf_bessel_J0_e(k * r, &res);
        if (retcode != GSL_SUCCESS) {
            p->retcode = retcode;
            return 0.0;
        }

        return f * r * res.val;
    }

    gsl_function fn;
    fn.params = NULL;
    fn.function = integrand;

    int retcode = GSL_SUCCESS;
    for (unsigned i = 0; i < Nk; i++) {
        // Set params for this iteration
        struct circular_ft_integrand_params params = {
            .k = ks[i],
            .f_r = f_r,
            .retcode = GSL_SUCCESS
        };
        fn.params = (void *) &params;

        // The qawf function performs a Fourier transform
        double result = 0.0, err = 0.0;
        retcode = gsl_integration_qagiu(&fn,
                                        // Integrate from 0 to +\inf
                                        0.0,
                                        // Algorithm precision parameters
                                        epsabs, epsrel, limit,
                                        // Workspace & table for sinusoid integration
                                        wkspc,
                                        // Results
                                        &result, &err);

        // Propogate any errors inside integrand
        if (params.retcode != GSL_SUCCESS)
            retcode = params.retcode;

        // Handle any errors
        if (retcode != GSL_SUCCESS)
            break;

        out[i] = result;
        if (out_err)
            out_err[i] = err;
    }

    // Clean up
    gsl_integration_workspace_free(wkspc);
    return retcode;
}

// Performs the Abel transform (LOS projection):
//
// F(r) = \int_{-\inf}^{\inf} dk f(\sqrt{r^2 + k^2})
static int
abel_transform(double *out, double *out_err,
               const double *rs, unsigned Nr,
               gsl_function *f_r,
               unsigned limit, double epsabs, double epsrel)
{
    if ((out == NULL) || (rs == NULL))
        return GSL_FAILURE;

    gsl_set_error_handler_off();
    // Allocate our needed workspaces
    gsl_integration_workspace *wkspc = gsl_integration_workspace_alloc(limit);
    if (!wkspc)
        return GSL_ENOMEM;

    // Our integrand is (f(R = sqrt(r*r + k*k))
    double
    integrand(double r, void *params)
    {
        const double chi = *((const double *) params),
                     this_dist = sqrt(chi*chi + r*r);

        return GSL_FN_EVAL(f_r, this_dist);
    }

    gsl_function fn;
    fn.params = NULL;
    fn.function = integrand;

    int retcode = GSL_SUCCESS;
    for (unsigned i = 0; i < Nr; i++) {
        double this_r = rs[i];
        fn.params = &this_r;

        // Integrate on infinite range
        double result = 0.0, err = 0.0;
        retcode = gsl_integration_qagi(&fn,
                                       epsabs, epsrel, limit,
                                       wkspc, &result, &err);

        // Handle any errors
        if (retcode != GSL_SUCCESS)
            break;

        out[i] = result;
        if (out_err)
            out_err[i] = err;
    }

    // Clean up
    gsl_integration_workspace_free(wkspc);
    return retcode;
}

static double
interp_integrand(double x, void *params)
{
    struct interp_integrand_args *args = params;
    if (args->retcode != GSL_SUCCESS)
        return 0.0;

    double F = 0.0;
    int retcode = gsl_interp_eval_e(args->F, args->xs, args->ys,
                                    x, NULL, &F);

    // If we are *below* our x range, we use the leftmost y value,
    // but if we are *above* our x range, we use 0.0.
    // This is because this is used in a lot of infinite integrals (0 -> inf)
    // which would diverge if nonzero at infinity, but have important behavior
    // close to zero.
    if (retcode == GSL_EDOM) {
        if (x < args->xs[0])
            return args->ys[0];
        return 0.0;
    } else if (retcode != GSL_SUCCESS) {
        args->retcode = retcode;
        return 0.0;
    }
    return F;
}

static int
interp_integrand_args_alloc(struct interp_integrand_args *args,
                            const double *xs, const double *ys,
                            unsigned N)
{
    if (!args || !xs || !ys)
        return -1;

    args->F = gsl_interp_alloc(gsl_interp_linear, N);
    if (!args->F)
        return GSL_ENOMEM;

    int rc = gsl_interp_init(args->F, xs, ys, N);
    if (rc != GSL_SUCCESS)
        return rc;

    args->xs = xs;
    args->ys = ys;
    args->retcode = GSL_SUCCESS;

    return GSL_SUCCESS;
}

static int
interp_integrand_args_free(struct interp_integrand_args *args)
{
    if (!args)
        return -1;
    if (!args->F)
        return -1;
    gsl_interp_free(args->F);
    return GSL_SUCCESS;
}
