import warnings
import unittest

import numpy as np
from . import constants as cc
from scipy import special

class NFW(object):
    """ Define an NFW halo with mass, concentration, background 
    density and overdensity factor.

    Parameters
    ----------
    mass: float, scalar
        Halo virial mass.
    c: float, scalar
        Halo concentration.
    rho_mean: float, scalar
        Background density (or a.k.a mean density, :math:`\Omega_m \\times \\rho_{crit} `)
    DELTA_HALO: scalar, optional
        Overdensity factor, default to be 200, which defines halos
        as having an average density of 200*rho_mean.

    """
    def __init__(self, mass, c, rho_mean, DELTA_HALO=200.0):
        self.mass    = mass
        self.c       = c
        self.rho_vir = rho_mean*DELTA_HALO
        self.r_vir   = self._r_vir()
        self.r_s     = self.r_vir/self.c
        self.rho_0   = self.mass/self._mass()
        self.v_cir_vir = np.sqrt(cc.G_const_Mpc_Msun_km_s*self.mass/(self.r_vir**2))

    def __str__(self):
        return(("Halo mass: %g Concentration: %g R_vir: %g R_s: %g Mass enclosed by 2*R_vir: %g")%(
                 self.mass, self.c, self.r_vir, self.r_s, self.total_mass(2.*self.r_vir)))

    def profile(self, r):
        """ Returns the normalized NFW profile.
        """
        return(self._profile(r, rho_0=self.rho_0))

    def v_cir(self, r):
        """ circular velocity at r.
        """
        _ratio  = (np.log((self.r_s + r)/self.r_s) - r/(self.r_s + r))/(np.log(1.0 + self.c) - self.c/(self.c + 1.0))
        return(self.v_cir_vir * np.sqrt(_ratio*self.r_vir/r))

    def total_mass(self, r_max):
        """ Total mass inside of some radius :math:`r_{max}`.

        Parameters
        ----------
        r_max: float, scalar
            Maximum radius to which NFW profile is to be integrated to 
            get the enclosed mass.

        Returns
        -------
        mass: float, scalar
            Enclosed mass.

        """
        return(4.0*cc.pi*self.rho_0*self.r_s**3*(np.log((self.r_s + r_max)/self.r_s) - r_max/(self.r_s + r_max)))

    def potential(self, r) :
        """ Gravitational potential at r in units of  (km/s)^2
        """
        _reff = r/self.r_s
        return(-4.0*cc.pi*cc.G_const_Mpc_Msun_km_s*self.rho_0*self.r_s**2*(np.log(1.+_reff)/_reff))

    def v_esc(self, r) :
        """ Escape velocity at r in units of  km/s
        """
        return(np.sqrt(0.0 - 2.0 * self.potential(r)))

    def log_rho_gradient(self, r) :
        """ Gradient of the logarithmic density profile
        """
        _reff  = r/self.r_s
        _gamma = -(3.0*_reff + 1.0)/(_reff + 1.0)
        return(_gamma)

    def y(self, k, rmax2rvir=1.0) :
        """ y is the fourier transform of the NFW profile 
        normalized to unit mass (but depends on the upper limit of integration).
        """
        r_max = rmax2rvir * self.r_vir
        return(self._fourier_profile(k, rmax2rvir=rmax2rvir, rho_0=self.rho_0)/self.total_mass(r_max))

    def DelSig(self, r) :
        """ Following the analytic formalism in Bartelmann96 or Wright&Braierd00.
        """
        _r = np.atleast_1d(r)
        x = _r/self.r_s
        _g = self._glt_or_ggt(x)
        return(self.r_s * self.rho_0 * _g)

    def _mass(self, rho_0=1.0):
        """ Internal function, returns the virial mass normalized by rho_0.
        """
        return(4.0*cc.pi*rho_0*self.r_s**3*(np.log(1.+self.c)-self.c/(1.+self.c)))

    def _r_vir(self):
        """ Returns the virial radius.
        """
        # given mean density, we can solve for the virial radius from input mass.
        return(np.power( (3.0/(4.0*cc.pi*self.rho_vir)) * self.mass, 1./3.))

    def _profile(self, r, rho_0=1.0):
        """ Internal function, returns the NFW profile normalized by rho_0.
        """
        _reff = r/self.r_s
        return(rho_0/(_reff*(1.0+_reff)**2))

    def _fourier_profile(self, k, rmax2rvir=1.0, rho_0=1.0) :
        """ Internal function, returns the fourier transform of a NFW profile.

        http://integrals.wolfram.com/index.jsp?expr=sin%28k*x%29%2F%28%281%2Bx%2Fs%29%5E2%29&random=false

        """
        r_max = rmax2rvir * self.r_vir # the upper limit of the integration
        kappa = k * self.r_s
        iota  = k * r_max
        si_k,   ci_k  = special.sici(kappa)
        si_ki,  ci_ki = special.sici(kappa+iota)
        _g_k = np.cos(kappa) * (ci_ki - ci_k) + np.sin(kappa)*(si_ki - si_k) - np.sin(iota)/(kappa+iota)
        return(4.*cc.pi*rho_0*self.r_s**3*_g_k)

    def _glt_or_ggt(self, x):
        """ Eqn. 15 and 16 in Wright & Brainerd 2000

        x is the scaled radius r/r_s, and is expected to be an ndarray

        """
        x2  = x*x
        pub = (4./x2) * np.log(x/2.0) 
        _g = np.empty_like(x)
        at = np.empty_like(x)
        ilt   = x <  1.0
        igt   = x >  1.0
        ieq   = x == 1.0
        ineq  = ilt | igt
        if np.any(ilt) :
            at[ilt] = np.arctanh( np.sqrt((1.0-x[ilt])/(1.0+x[ilt])) ) / np.sqrt(1.-x2[ilt])
        if np.any(igt) :
            at[igt] = np.arctan( np.sqrt((x[igt]-1.0)/(x[igt]+1.0)) ) / np.sqrt(x2[igt] - 1.)
        if np.any(ieq) :
            _g[ieq] = pub[ieq]+10./3.
        _g[ineq] = 4.0*at[ineq]*(2./x2[ineq]  + 1./(x2[ineq]-1.0)) + pub[ineq] - 2.0/(x2[ineq]-1.0)
        return(_g)

    
def plotvesc():
    import matplotlib.pyplot as plt
    from .param import CosmoParams
    from .density import Density
    DELTA_HALO = 200.0
    ndbin = 30
    dmin  = 0.1
    dmax  = 4
    darr  = np.linspace(np.log10(dmin), np.log10(dmax),
              num=ndbin)
    darr  = np.power(10.0, darr)

    cosmo = CosmoParams()
    den   = Density(cosmo=cosmo)
    z     = 0.
    rho_m = den.rho_mean_z(z)
    mass  = 1.0e14
    c     = 5
    nfw   = NFW(mass=mass, c=c, rho_mean=rho_m, DELTA_HALO=DELTA_HALO)
    vesc1 = nfw.v_esc(darr)
    vcir1 = nfw.v_cir(darr)
    rs1   = nfw.r_s
    rv1   = nfw.r_vir
    fig1 = plt.figure(1, figsize=(8,6))
    ax1  = fig1.add_subplot(1,1,1)
    ax1.plot(darr, vesc1, 'r-', label="escape")
    ax1.plot(darr, vcir1, 'b-', label="circular")
    ax1.axvline(2.*nfw.r_s)
    ax1.legend(loc="best")
#    ax1.plot(darr/nfw.r_s, nfw.log_rho_gradient(darr), 'r-')
    ax1.set_xscale('log')
#    ax1.set_yscale('log')
    ax1.set_xlabel('r')
    ax1.set_ylabel(r'$v$')
    ax1.set_title("Escape and Circular Velocity")
    ax1.set_xlim(xmax=4)
    plt.show()

def plotprofile():
    """
    """
    import matplotlib.pyplot as plt
    from .param import CosmoParams
    from .density import Density

    DELTA_HALO = 200.0
    ndbin = 60
    dmin  = 0.0001
    dmax  = 2
    darr  = np.linspace(np.log10(dmin), np.log10(dmax),
              num=ndbin)
    darr  = np.power(10.0, darr)
    cosmo = CosmoParams()
    den   = Density(cosmo=cosmo)
    z     = 0.
    rho_m = den.rho_mean_z(z)
    mass  = 1e10
    c     = 30
    nfw   = NFW(mass=mass, c=c, rho_mean=rho_m, DELTA_HALO=DELTA_HALO)
    prof1 = nfw.profile(darr)
    rs1   = nfw.r_s
    rv1   = nfw.r_vir
    mass  = 1e12
    nfw   = NFW(mass=mass, c=c, rho_mean=rho_m, DELTA_HALO=DELTA_HALO)
    prof2 = nfw.profile(darr)
    rs2   = nfw.r_s
    rv2   = nfw.r_vir
    c     = 5
    nfw   = NFW(mass=mass, c=c, rho_mean=rho_m, DELTA_HALO=DELTA_HALO)
    prof3 = nfw.profile(darr)
    rs3   = nfw.r_s
    rv3   = nfw.r_vir

    fig1 = plt.figure(1, figsize=(8,6))
    ax1  = fig1.add_subplot(1,1,1)
    ax1.plot(darr, prof1, 'r-', darr, prof2,'bo', darr, prof3, 'g-')
    ax1.axvline(rs1, color='r', ls='--')
    ax1.axvline(rv1, color='r', ls='-')
    ax1.axvline(rs2, color='b', ls='--', lw=4, alpha=0.5)
    ax1.axvline(rv2, color='b', ls='-',  lw=4, alpha=0.5)
    ax1.axvline(rs3, color='g', ls='--')
    ax1.axvline(rv3, color='g', ls='-')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('r')
    ax1.set_ylabel(r'$\rho$')
    ax1.set_title("NFW")
    ax1.set_xlim(xmax=1.0)
    plt.show()

def plotmass():
    """  I wanna see how the enclosed mass varies as function of scale.
    """
    import matplotlib.pyplot as plt
    from .param import CosmoParams
    from .density import Density

    DELTA_HALO = 200.0
    ndbin = 60
    dmin  = 0.0001
    dmax  = 1e10
    darr  = np.linspace(np.log10(dmin), np.log10(dmax),
              num=ndbin)
    darr  = np.power(10.0, darr)
    cosmo = CosmoParams()
    den   = Density(cosmo=cosmo)
    z     = 0.
    rho_m = den.rho_mean_z(z)
    mass  = 1e10
    c     = 30
    nfw   = NFW(mass=mass, c=c, rho_mean=rho_m, DELTA_HALO=DELTA_HALO)
    prof1 = nfw.profile(darr)
    mass1 = nfw.total_mass(darr)
    rs1   = nfw.r_s
    rv1   = nfw.r_vir

    fig1 = plt.figure(1, figsize=(8,6))
    ax1  = fig1.add_subplot(1,1,1)
    # ax1.plot(darr, prof1, 'r-')
    # ax1.plot(darr, mass1, 'r-')
    ax1.plot(darr, mass1/mass, 'r-')
    ax1.axvline(rs1, color='r', ls='--')
    ax1.axvline(rv1, color='r', ls='-')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('r')
    ax1.set_ylabel(r'$\rho$')
    ax1.set_title("NFW")
    # ax1.set_xlim(xmax=1.0)
    plt.show()

def plotydm():
    """  
    """
    import matplotlib.pyplot as plt
    from .param import CosmoParams
    from .density import Density

    DELTA_HALO = 200.0
    nkbin = 60
    kmin  = 0.001
    kmax  = 1e3
    karr  = np.linspace(np.log10(kmin), np.log10(kmax),
              num=nkbin)
    karr  = np.power(10.0, karr)
    cosmo = CosmoParams()
    den   = Density(cosmo=cosmo)
    z     = 0.
    rho_m = den.rho_mean_z(z)
    mass  = 1e10
    c     = 30
    nfw   = NFW(mass=mass, c=c, rho_mean=rho_m, DELTA_HALO=DELTA_HALO)
    y1    = nfw.y(karr)
    y2    = nfw.y(karr, rmax2rvir=2.0)
    y5    = nfw.y(karr, rmax2rvir=5.0)
    y10   = nfw.y(karr, rmax2rvir=10.)
    rs1   = nfw.r_s
    rv1   = nfw.r_vir

    fig1 = plt.figure(1, figsize=(8,6))
    ax1  = fig1.add_subplot(1,1,1)
    ax1.plot(karr,  y1, 'r-', label="1")
    ax1.plot(karr,  y2, 'k-', label="2")
    ax1.plot(karr,  y5, 'b-', label="5")
    ax1.plot(karr, y10, 'g-', label="10")
    # ax1.plot(darr, mass1/mass, 'r-')
    ax1.axvline(rs1, color='r', ls='--')
    ax1.axvline(rv1, color='r', ls='-')
    ax1.legend(loc="best")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('k')
    ax1.set_ylabel(r'$y$')
    ax1.set_title("NFW")
    ax1.set_ylim(ymax=1.2)
    plt.show()

def plotdelsig() :
    """  I wanna see how the enclosed mass varies as function of scale.
    """
    import matplotlib.pyplot as plt
    from .param import CosmoParams
    from .density import Density
    from DeltaSigmaR import DeltaSigmaR 

    DELTA_HALO = 200.0
    ndbin = 60
    dmin  = 0.01
    dmax  = 100.
    darr  = np.linspace(np.log10(dmin), np.log10(dmax),
              num=ndbin)
    darr  = np.power(10.0, darr)
    cosmo = CosmoParams()
    den   = Density(cosmo=cosmo)
    z     = 0.
    rho_m = den.rho_mean_z(z)
    mass  = 1e14
    c     = 6
    nfw   = NFW(mass=mass, c=c, rho_mean=rho_m, DELTA_HALO=DELTA_HALO)
    prof1 = nfw.profile(darr)
    ds1   = nfw.DelSig(darr)
    rp, ds0   = DeltaSigmaR(darr, prof1, rp_max=50.0)[:2]
    rs1   = nfw.r_s
    rv1   = nfw.r_vir
    fig1 = plt.figure(1, figsize=(8,6))
    ax1  = fig1.add_subplot(1,1,1)
    ax1.plot(darr, ds1, 'r-')
    ax1.plot(rp, ds0, 'k-')
    ax1.axvline(rs1, color='r', ls='--')
    ax1.axvline(rv1, color='r', ls='-')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('R')
    ax1.set_ylabel(r'$\Delta\Sigma$')
    ax1.set_title("NFW")
    ax1.set_xlim(dmin, dmax)
    plt.show()

class NFWTest(unittest.TestCase):
    def test_simple_analytical_properties(self) :
        mass       = 1.0
        c          = 5.0
        rho_mean   = 1.0
        DELTA_HALO = 1.0
        nfw = NFW(mass, c, rho_mean, DELTA_HALO)
        print nfw
        mass       = 1.0
        c          = 50.0
        rho_mean   = 1.0
        DELTA_HALO = 1.0
        nfw = NFW(mass, c, rho_mean, DELTA_HALO)
        print nfw


if __name__ == "__main__":
    """
    """
    # unittest.main()
    # plotprofile()
    # plotvesc()
    # plotmass()
    # plotydm()
    plotdelsig()
