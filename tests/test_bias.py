import pytest
from cluster_toolkit import bias
from cluster_toolkit import peak_height as peaks
from os.path import dirname, join
import numpy as np
import numpy.testing as npt

#Halo properties that are inputs
rhomconst = 2.77533742639e+11 #units are SM h^2/Mpc^3
Mass = 1e14 #Msun/h
Omega_m = 0.3 #arbitrary
datapath = "./data_for_testing/"
klin = np.loadtxt(join(dirname(__file__),datapath+"klin.txt"))#h/Mpc; wavenumber
plin = np.loadtxt(join(dirname(__file__),datapath+"plin.txt"))#[Mpc/h]^3 linear power spectrum
Ma = np.array([1e13, 1e14, 1e15]) #Msun/h
Ra = (Ma/(4./3.*np.pi*rhomconst*Omega_m))**(1./3.)

def test_exceptions_bias_at_M():
    with pytest.raises(TypeError):
        bias.bias_at_M()
    with pytest.raises(TypeError):
        bias.bias_at_M(Mass, klin, plin)
    with pytest.raises(TypeError):
        bias.bias_at_M(Mass, klin, plin, Omega_m, Omega_m)
    #with pytest.raises(TypeError):
    #    bias.bias_at_M("a string", klin, plin, Omega_m)

def test_outputs_bias_at_M():
    #List vs. numpy.array
    npt.assert_array_equal(bias.bias_at_M(Ma, klin, plin, Omega_m),
                           bias.bias_at_M(Ma.tolist(), klin, plin, Omega_m))
    #Single value vs numpy.array
    arrout = bias.bias_at_M(Ma, klin, plin, Omega_m)
    for i in range(len(Ma)):
        npt.assert_equal(bias.bias_at_M(Ma[i], klin, plin, Omega_m), arrout[i])

def test_derivatives():
    M = np.logspace(13, 14, 1000)
    dbdM = bias.dbiasdM_at_M(M, klin, plin, Omega_m)
    npt.assert_equal(sorted(dbdM), dbdM[::-1])
    b = bias.bias_at_M(M, klin, plin, Omega_m)
    dM = M[1:] - M[:-1]
    db = b[1:] - b[:-1]
    deriv = db/dM
    pd = dbdM[:-1] / deriv
    npt.assert_array_almost_equal(pd, np.ones_like(pd), 1e-2)

def test_s2_and_nu_functions():
    #Test the mass calls
    s2 = peaks.sigma2_at_M(Mass, klin, plin, Omega_m)
    nu = peaks.nu_at_M(Mass, klin, plin, Omega_m)
    npt.assert_equal(1.686/np.sqrt(s2), nu)
    s2 = peaks.sigma2_at_M(Ma, klin, plin, Omega_m)
    nu = peaks.nu_at_M(Ma, klin, plin, Omega_m)
    npt.assert_array_equal(1.686/np.sqrt(s2), nu)
    out = bias.bias_at_M(Ma, klin, plin, Omega_m)
    out2 = bias.bias_at_nu(nu)
    npt.assert_array_equal(out, out2)
    #Now test the R calls
    R = 1.0 #Mpc/h; arbitrary
    s2 = peaks.sigma2_at_R(R, klin, plin)
    nu = peaks.nu_at_R(R, klin, plin)
    npt.assert_equal(1.686/np.sqrt(s2), nu)
    out = bias.bias_at_R(R, klin, plin)
    out2 = bias.bias_at_nu(nu)
    npt.assert_array_equal(out, out2)

def test_single_vs_array():
    #First sigma2
    a1 = peaks.sigma2_at_M(Ma, klin, plin, Omega_m)
    a2 = np.array([peaks.sigma2_at_M(Mi, klin, plin, Omega_m) for Mi in Ma])
    npt.assert_array_equal(a1, a2)
    a1 = peaks.sigma2_at_R(Ra, klin, plin)
    a2 = np.array([peaks.sigma2_at_R(Ri, klin, plin) for Ri in Ra])
    npt.assert_array_equal(a1, a2)
    #Now the bias
    a1 = bias.bias_at_M(Ma, klin, plin, Omega_m)
    a2 = np.array([bias.bias_at_M(Mi, klin, plin, Omega_m) for Mi in Ma])
    npt.assert_array_equal(a1, a2)
    a1 = bias.bias_at_R(Ra, klin, plin)
    a2 = np.array([bias.bias_at_R(Ri, klin, plin) for Ri in Ra])
    npt.assert_array_equal(a1, a2)


def test_R_vs_M():
    R = 1.0 #Mpc/h
    M = 4.*np.pi/3. * Omega_m * rhomconst * R**3
    npt.assert_almost_equal(bias.bias_at_R(R, klin, plin), bias.bias_at_M(M, klin, plin, Omega_m))
    R = np.array([1.2, 1.4, 1.5])
    M = 4.*np.pi/3. * Omega_m * rhomconst * R**3
    npt.assert_array_almost_equal(bias.bias_at_R(R, klin, plin), bias.bias_at_M(M, klin, plin, Omega_m))

def test_mass_dependence():
    masses = np.logspace(13, 15, num=100)
    arrout = bias.bias_at_M(masses, klin, plin, Omega_m)
    for i in range(len(masses)-1):
        assert arrout[i] < arrout[i+1]
    Rs = (masses/(4./3.*np.pi*Omega_m*rhomconst))**(1./3.)
    arrout = bias.bias_at_R(Rs, klin, plin)
    for i in range(len(masses)-1):
        assert arrout[i] < arrout[i+1]
    nus = peaks.nu_at_M(masses, klin, plin, Omega_m)
    arrout = bias.bias_at_nu(nus)
    for i in range(len(masses)-1):
        assert arrout[i] < arrout[i+1]

def test_Cordering():
    Me,be = np.loadtxt(join(dirname(__file__),datapath+"bias_camb_z0.0.dat")).T
    ke,pe = np.loadtxt(join(dirname(__file__),datapath+"ps_camb_z0.0.dat")).T
    pe = 2*np.pi**2*pe/ke**3
    #Me = np.ascontiguousarray(Me) #test passes if this is not required
    #ke = np.ascontiguousarray(ke) #test passes if this is not required
    #pe = np.ascontiguousarray(pe) #test passes if this is not required
    bt = bias.bias_at_M(Me, ke, pe, Omega_m)
    r = be/bt
    npt.assert_array_almost_equal(r, np.ones_like(r), decimal=3)

if __name__ == "__main__":
    #test_Cordering()
    test_derivatives()
