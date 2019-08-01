void G_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double om,
	       double d, double e, double f, double g, double*G);

void G_at_sigma_arr(double*sigma, int Ns, double d, double e,
		   double f, double g, double*G);

void dndM_at_M_arr(double*M, int NM, double*k, double*P, int Nk, double om,
		  double d, double e, double f, double g, double*dndM);

int n_in_bins(double*edges, int Nedges, double*M, double*dndM,
	      int NM, double*N);

//int dndM_sigma2_precomputed(double*M, double*sigma2, double*sigma2_top, double*sigma2_bot, int NM, double Omega_m, double d, double e, double f, double g, double*dndM);
void dndM_sigma2_precomputed(double*M, double*sigma2, double*dsigma2dM,
			    int NM, double Omega_m, double d, double e,
			    double f, double g, double*dndM);
