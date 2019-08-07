void P_BBPS(double *P_out, const double *r, unsigned Nr, double M_delta, double z, double omega_b, double omega_m, double P_0, double x_c, double beta, double alpha, double gamma, double delta);
int projected_P_BBPS(double *P_out, double *P_err_out, const double *r, unsigned Nr, double M_delta, double z, double omega_b, double omega_m, double P_0, double x_c, double beta, double alpha, double gamma, double delta, unsigned limit, double epsabs, double epsrel);
int fourier_P_BBPS(double *up_out, double *up_err_out, const double *ks, unsigned Nk, double M_delta, double z, double omega_b, double omega_m, double P_0, double x_c, double beta, double alpha, double gamma, double delta, unsigned limit, double epsabs);
int inverse_spherical_fourier_transform(double *out, double *out_err, const double *rs, unsigned Nr, const double *ks, const double *Fs, unsigned Nk, unsigned limit, double epsabs);
int forward_spherical_fourier_transform(double *out, double *out_err, const double *ks, unsigned Nk, const double *rs, const double *fs, unsigned Nr, unsigned limit, double epsabs);
int integrate_spline(const double *xs, const double *ys, unsigned Ny, double a, double b, double *result);
