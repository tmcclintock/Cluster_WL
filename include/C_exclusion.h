int xihm_exclusion_at_r_arr(double*r, int Nr, double M, double c, double rt, double beta, double Ma, double ca, double Mb, double cb, double bias, double*ximm, int delta, double Omega_m, int scheme, double*xihm);

int xi_1h_at_r_arr(double*r, int Nr, double M, double c,
		   double rt, double beta, int delta, double Omega_m,
		   double*xi_1h);

int xi_2h_at_r_arr(double*r, int Nr, double bias, double*ximm, double*xi2h);

int xi_2hcorrection_at_r_arr(double*r, int Nr, double M1, double rt,
			     double M2, double conc2, double bias,
			     int delta, double Omega_m, double*xi_2hc);

double I_term(double r, double R, double re, double beta);

int xi_correction_at_r_arr(double*r, int Nr, double M1, double rt, double beta,
			   double M2, double c2, int delta, double Omega_m,
			   int scheme, double*xict);

int theta_erfc_at_r_arr(double*r, int Nr, double rt, double beta, double*theta);

double r_exclusion(double r1, double r2, int scheme);
