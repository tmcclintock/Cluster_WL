double xi_nfw_at_r(double, double, double, int, double);
void calc_xi_nfw(double*, int, double, double, int, double, double*);

void calc_xi_einasto(double*, int, double, double, double, double, int, double, double*);
double rhos_einasto_at_M(double Mass, double conc, double alpha, int delta, double om);

double xi_mm_at_r_exact(double r, double*k, double*P, int Nk);
int calc_xi_mm_exact(double*r, int Nr, double*k, double*P, int Nk, double*xi);

void calc_xi_2halo(int, double, double*, double*);
void calc_xi_hm(int, double*, double*, double*, int);
int calc_xi_mm(double*, int, double*, double*, int, double*, int, double);

void calc_xi_DK(double*r, int Nr, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double*xi);

void calc_xi_DK_app1(double*r, int Nr, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double bias, double*xi_mm, double*xi);

void calc_xi_DK_app2(double*r, int Nr, double M, double rhos, double conc, double be, double se, double alpha, double beta, double gamma, int delta, double*k, double*P, int Nk, double om, double bias, double*xi_mm, double*xi);
