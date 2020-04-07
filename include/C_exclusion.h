void xi_hm_exclusion_at_r_arr(double*r, int Nr,
			    double M, double c, double alpha,
			    double rt, double D,
			    double r_eff, double D_eff,
			    double r_A, double r_B, double D_ex,
			    double bias, double*ximm, int delta,
			     double Omega_m, double*xihm);

void xi_1h_at_r_arr(double*r, int Nr, double M, double c, double alpha,
		   double rt, double D, int delta, double Omega_m,
		   double*xi_1h);

void xi_2h_at_r_arr(double*r, int Nr, double r_eff, double D_eff,
		   double bias, double*ximm, double*xi2h);

void xi_C_at_r_arr(double*r, int Nr, double r_A, double r_B, double D,
		  double*xi_2h, double*xi_C);

int theta_erfc_at_r_arr(double*r, int Nr, double rt, double D,
			double*theta);

double r_exclusion(double r1, double r2, int scheme);
