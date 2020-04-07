double Sigma_nfw_at_R(double R, double M, double c, int delta, double om);
void Sigma_nfw_at_R_arr(double*R, int NR, double M,double c, int delta, double om, double*Sigma);

int Sigma_at_R_arr(double*R, int NR, double*Rxi, double*xi, int Nxi, double M, double conc, int delta, double om, double*Sigma);

int Sigma_at_R_full_arr(double*R, int NR, double*Rxi, double*xi, int Nxi, double M, double conc, int delta, double om, double*Sigma);

int DeltaSigma_at_R_arr(double*R, int NR, double*Rs, double*Sigma, int Ns, double M, double conc, int delta, double om, double*DeltaSigma);
