#!/usr/bin/env python
#Last-modified: 12 Dec 2013 22:32:34

import warnings
import unittest

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
from math import log, log10, pi, exp

from .param import CosmoParams
from .density import Density
from .lineartheory import LinearTheory
from . import cex

#----------------------------------------------------------------------


class HaloStat(object):
    """ A population of dark matter halos.

    Parameters
    ----------
    cosmo: CosmoParams object, optional
        Cosmology for calculating the current background density and mass variance, not used if both `rho_mean_0` and `sigma_r` are provided.
    z: scalar, optional
        Redshift (default: 0). This is necessary in the fitting formula for the redshift evolution of halo statistics, but optional in cosmology-related predictions.
    DELTA_HALO: scalar, optional
        Overdensity factor (default: 200).
    rho_mean_0 : scalar, optional
        Background density at today. It will overwrite the value
        calculated from cosmo (default: None).
    sigma_r: callable object, optional
        Function to calculate the r.m.s. of linear density field 
        at z. It will overwrite the function derived from cosmo
        (default: None).
    mass: nparray, optional
        Mass array, need to be equal-spaced in log (default: None).
    dlogm: float, optional
        Logarithmic spacing of *mass*, required when *mass* is
        set (default: None).
    mmin: float, optional
        Minimum halo mass, not to be used when mass and dlogm is set
        (default: 1.e8 Msun).
    mmax: float, optional
        Maximum halo mass, not to be used when mass and dlogm is set 
        (default: 1.e16 Msun).
    nmbin: int, optional
        Number of mass bins, not to be used when mass and dlogm is set 
        (default: 100)

    """
    def __init__(self, cosmo=None, z=0.0, DELTA_HALO=200.0, 
                       rho_mean_0=None, sigma_r=None, 
                       mass=None, dlogm=None,
                       mmin=1.e8, mmax=1.e16, nmbin=100):
        if cosmo is None:
            cp = CosmoParams(set_flat=True)
        else:
            cp = cosmo
            if not cp.isflat:
                warnings.warn("Non-flat Universe encountered!")
#       redshift
        self.z = z
#       halo definition
        self.DELTA_HALO = DELTA_HALO
#       background density
        if (rho_mean_0 is None):
            den = Density(cosmo=cp)
            self.rho_mean_0 = den.rho_mean_z(z=0.0)
        else:
            self.rho_mean_0 = rho_mean_0

#       converstion from mass to radius
        self.r3conv = 3.0/(4.0*pi*self.rho_mean_0)

        if sigma_r is None:
#           linear theory at z
            lin = LinearTheory(cosmo=cp, z=z)
            self.sigma_r = lin.sigma_r
        else:
            if not hasattr(sigma_r, "__call__"):
                raise cex.NonCallableObject(sigma_r)
            self.sigma_r = sigma_r

#       initialize the masses and the sigma_r accordingly
        if (mass is None):
            self.reset_halo_mass(mmin, mmax, nmbin)
        else:
            if (dlogm is None):
                cex.ParametersNotPairSet("Need to set dlogm with mass")
            else:
                self.mass  = mass
                self.dlogm = dlogm
                self.reset_sig()


#----------------------------------------------------------------------


#----------------------------------------------------------------------

    def reset_halo_mass(self, mmin, mmax, nmbin):
        """ Initialize or reset the mass array.

        .. note ::

            Register *HaloStat.mass* with an array of halo masses. Subsequent
            calculations of mass function and bias function will both be 
            matched to *HaloStat.mass*.

        Parameters
        ----------
        mmin: float
            Minimum halo mass.
        mmax: float
            Maximum halo mass.
        nmbin: int
            Number of mass bins.

        """
        log10mass, dlog10mass = np.linspace(log10(mmin), log10(mmax), 
                                            num=nmbin, retstep=True)
#       log10(e) = 0.434294482
        self.dlogm = dlog10mass/0.434294482
        self.mass = np.power(10.0, log10mass)
#       scales that correspond to halo masses in the linear density
#       field extrapolated to z=0
        self.reset_sig()

#----------------------------------------------------------------------

    def reset_sig(self):
        self.rm   = np.power(self.r3conv*self.mass, 1.0/3.0)
        self.sig = self.sigma_r(self.rm)

#----------------------------------------------------------------------

    @property
    def mass_function(self):
        """ 
        Halo mass function from Tinker et al. (2008) fitting formula.

        """
        self._initialize_mass_function()
        mlo    = 0.99*self.mass
        mhi    = 1.01*self.mass
        rlo    = np.power(self.r3conv*mlo, 1.0/3.0)
        rhi    = np.power(self.r3conv*mhi, 1.0/3.0)
        slo    = self.sigma_r(rlo)
        shi    = self.sigma_r(rhi)
#       ds/dM
        dsdM   = (shi-slo)/(mhi-mlo)

        self.fsig = (self.a1*(np.power(self.sig/self.a3,-self.a2)+1.0)*
                  np.exp(-self.a4/self.sig/self.sig))
        dndM   = -self.fsig*self.rho_mean_0/self.mass/self.sig*dsdM
        return(dndM)


    def _initialize_mass_function(self):
        """ Find the four fitting parameters for Tinker05 mass function.
        """
        _logDELTA_HALO = log(self.DELTA_HALO)
        x       = np.arange(1.0,10.0,1.0)
        odd     = np.arange(0,9,2)
        even    = np.arange(1,9,2)
        x[odd]  = np.log(200.0*np.power(2.0,(x[odd]-1)/2.0))
        x[even] = np.log(300.0*np.power(2.0,(x[even]-2)/2.0))
        y = np.array([1.858659e-01,
                      1.995973e-01,
                      2.115659e-01,
                      2.184113e-01,
                      2.480968e-01,
                      2.546053e-01,
                      2.600000e-01,
                      2.600000e-01,
                      2.600000e-01,
                      ])
        f1 = spline1d(x, y) 
        a1 = f1(_logDELTA_HALO)
        if(self.DELTA_HALO>=1600):
            a1 = 0.26
        y = np.array([1.466904e+00,
                      1.521782e+00,
                      1.559186e+00,
                      1.614585e+00,
                      1.869936e+00,
                      2.128056e+00,
                      2.301275e+00,
                      2.529241e+00,
                      2.661983e+00,
                      ])
        f2 = spline1d(x, y) 
        a2 = f2(_logDELTA_HALO)
        y = np.array([2.571104e+00, 
                      2.254217e+00, 
                      2.048674e+00, 
                      1.869559e+00, 
                      1.588649e+00, 
                      1.507134e+00, 
                      1.464374e+00, 
                      1.436827e+00, 
                      1.405210e+00, 
                      ])
        f3 = spline1d(x, y) 
        a3 = f3(_logDELTA_HALO)
        y = np.array([1.193958e+00,
                      1.270316e+00,
                      1.335191e+00,
                      1.446266e+00,
                      1.581345e+00,
                      1.795050e+00,
                      1.965613e+00,
                      2.237466e+00,
                      2.439729e+00,
                      ])
        f4 = spline1d(x, y) 
        a4 = f4(_logDELTA_HALO)
        ztemp = self.z > 3.0 and 3.0 or self.z
        a1 = a1*pow(1.0+ztemp, -0.14)
        a2 = a2*pow(1.0+ztemp, -0.06)
        at  = -pow(0.75/log10(self.DELTA_HALO/75.0),1.2)
        at  = pow(10.0,at)
        # Eduardo's buggy version, sigh...
#        at  = -pow(0.75/log(self.DELTA_HALO/75.0),1.2)
#        at  = exp(at)
        a3 = a3*pow(1.0+ztemp, -at)
        self.a1, self.a2, self.a3, self.a4 = (a1,a2,a3,a4)

#----------------------------------------------------------------------

    @property
    def bias_function(self):
        """ 
        Halo bias function from Tinker et al. (2010) fitting formula.

        """
#        bias_a, bias_A, bias_b, bias_B, bias_c, bias_C = self._initialize_bias_function()
        self._initialize_bias_function()
        a = np.power(self.sig, -self.bias_a)
        b = (1 - self.bias_A*a/(a+1) + self.bias_B*np.power(self.sig, -self.bias_b) + 
             self.bias_C*np.power(self.sig,-self.bias_c))
        return(b)

    def db_ddeltasc(self, kind="ST99"):
        """
        Derivative of mean bias to :math:`\delta_\mathrm{sc}`.

        .. math ::

            \\frac{\partial\\bar{b}(M)}{\partial\delta_\mathrm{sc}}


        From peak-background split, the mean bias of halos with mass :math:`M`
        can be obtained by, to first order, 

        .. math ::

            \\bar{b}(M) = 1 - \\frac{\partial\ln\\bar{n}(M)}{\partial\delta_\mathrm{sc}} 

        
        from which we can derive the fluctuation of halo bias induced by 
        large wavelenght perturbation, i.e., sample variance,
    
        .. math ::

            \\bar{b}(M) + \delta b = 
            1 - \\frac{\partial}{\partial \delta_\mathrm{sc}}(\ln[\\bar{n}(M)(1+\\bar{b}(M)\delta)])= 
            \\bar{b}(M) - \delta \\frac{\partial\\bar{b}(M)}{\partial\delta_\mathrm{sc}}

        where :math:`\delta` is the underlying matter overdensity.

        Parameters
        ----------
        kind: string, optional
            Formula to use for calculating the derivative. Three options are:
            "ST99": canonical Eulerian bias formula used in ST99; 
            "Tinker10": new fitting formula proposed in Tinker10; 
            "Tinker10EQ15": canonical Eulerian bias formula but using the new halo mass
            function from Tinker08.
            "ST99" and "Tinker10EQ15" are very similar but "Tinker10" deviates from them way off. Not
            sure which is actually right, but I assume derivative from peak-background split is at leaset
            physically grounded, so either "ST99" and "Tinker10EQ15" is fine. (default: "ST99")

        Returns
        -------
        derivative: nparray
            :math:`\partial\\bar{b}(M)/\partial\delta_\mathrm{sc}`

        """
#        deltasc = 1.6864702
        deltasc = 1.686
        if kind is "ST99":
            # use the Sheth & Tormen (1999) formula for b, more physically intuitive.
            p = 0.3; q = 0.707; 
#            p = 0.3; q = 0.75; 
            sig2   = np.power(self.sig, 2)
            dsc2   = np.power(deltasc, 2)
            x      = np.power(q*dsc2/sig2, p)
            retval =  q/sig2 + (1.0-(1.0+x*(2.0*p+1.0))*2.0*p/np.power(1.0+x, 2))/dsc2
        elif kind is "Tinker10":
            # use the Tinker et al. (2010) formula EQ 6 for b, the one I used to calculate b.
            self._initialize_bias_function()
            sigb = np.power(self.sig, self.bias_b)
            sigc = np.power(self.sig, self.bias_c)
            retval = (self.bias_B*self.bias_b/sigb + self.bias_C*self.bias_c/sigc)/deltasc
        elif kind is "Tinker10EQ15":
            # use the Tinker et al. (2010) formula EQ 15 for b, peak-background split 
            # theory from Tinker Halo mass function. 
            if (self.DELTA_HALO != 200.0):
                raise cex.ParameterOutsideDefaultRange("Currently only implemented for DELTA_HALO=200,\
                                                        you can look for the full parameterization\
                                                        at Table 4 of Tinker et (2010).")
            evo   =  1.+self.z
            beta  =  0.589*evo**0.20
            psi   = -0.729*evo**(-0.08)
            eta   = -0.243*evo**0.27
            gamma =  0.864*evo**(-0.01)
            sig2   = np.power(self.sig, 2)
            dsc2   = np.power(deltasc, 2)
            x      = np.power(beta*beta*dsc2/sig2, psi)
            retval = gamma/sig2 + (1.0+2.0*eta-(1.0+x*(2.0*psi+1.0))*2.0*psi/np.power(1.0+x, 2))/dsc2
        return(retval)


    def _initialize_bias_function(self):
        """ Find the fitting parameters for Tinker10 bias function.
        """
        # the detection here simply because if could be used by either bias_function or db_ddeltasc
        if not hasattr(self, "bias_is_initialized"):
            y       = log10(self.DELTA_HALO)
            y44     = exp(-pow(4.0/y,4.0))
#           the capital coefficients deviate from Table 2 of Tinker et al.
#           (2010), simply because we absorbed the \delta_c term into them.
            bias_A  = 1.00 + 0.24*y*y44
            bias_a  = (y-2.0)*0.44
            bias_B  = 0.4
            bias_b  = 1.5
            bias_C  = ((y-2.6)*0.4 + 1.11 + 0.7*y*y44)*0.94
            bias_c  = 2.4
            self.bias_a, self.bias_A, self.bias_b, self.bias_B, self.bias_c, self.bias_C = (bias_a, 
                    bias_A, bias_b, bias_B, bias_c, bias_C)
            self.bias_is_initialized = True
        else:
            pass
    
#----------------------------------------------------------------------



#----------------------------------------------------------------------

def testUniversalMF():
    from .param import CosmoParams
    from .lineartheory import LinearTheory
    import matplotlib.pyplot as plt

    DELTA_HALO=200.
    z=0.0

    halostat  = HaloStat(z=z, DELTA_HALO=DELTA_HALO)
    invsig1   = 1./halostat.sig
    mf1       = halostat.mass_function
    fsig1     = halostat.fsig

    z=1.25
    halostat  = HaloStat(z=z, DELTA_HALO=DELTA_HALO)
    invsig2   = 1./halostat.sig
    mf3       = halostat.mass_function
    fsig2     = halostat.fsig

    invsig1 = np.log10(invsig1)
    invsig2 = np.log10(invsig2)
    fsig1 = np.log10(fsig1)
    fsig2 = np.log10(fsig2)

    fig1 = plt.figure(1, figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(invsig1, fsig1, 'g-', lw=4, alpha=0.5, label="z=0" )
    ax1.plot(invsig2, fsig2, 'ro', label="z=1.25" )
    ax1.set_ylabel('$\log\, f(\sigma)$')
    ax1.set_xlabel('$\log (1/\sigma)$')
    ax1.set_xlim(-0.65, 0.5)
    ax1.set_ylim(-3.2, 0)
    ax1.set_title("Halo Mass Function using scaled mass")
    ax1.legend(loc = "lower left")
    plt.show()

def plotwarren(z=0.0, DELTA_HALO=280.0):
    import matplotlib.pyplot as plt
    cp = CosmoParams(omega_M_0=0.3, omega_b_0=0.04, n=1, h=0.7, sigma_8=0.9,set_flat=True)
    halostat  = HaloStat(cosmo=cp, z=z, DELTA_HALO=DELTA_HALO)
    halomass  = halostat.mass
    massfunc  = halostat.mass_function
    dndlnm    = massfunc*halomass
    dndlnm_h  = dndlnm/cp.h**3
    mass_h    = halomass*cp.h
    fig1 = plt.figure(1, figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(mass_h, dndlnm_h, 'go')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel('dn/dlnM [h$^3\odot Mpc^{-3}$]')
    ax1.set_xlabel('M [$M_\odot$/h]')
    ax1.set_xlim(1.e10, 1.e16)
    ax1.set_ylim(1.e-10, 1.)
    ax1.set_title("Warren Halo Mass Function")
    plt.show()

def plotrozo(z=0.23, DELTA_HALO=200.0):
    import matplotlib.pyplot as plt
    rozodat    = "test/mf0.23.dat"
#    rozodat    = "test/mf.dat"
    mass, dndm = np.genfromtxt(rozodat, unpack=True) 
    dlogm = np.log(mass[1]/mass[0])
#    print(dlogm)
#    print(np.log(mass[600]/mass[599]))
#    quit()

    cp = CosmoParams(omega_M_0=0.272, n=0.963, h=0.7, sigma_8=0.8,set_flat=True)
#    halostat  = HaloStat(cosmo=cp, z=z, DELTA_HALO=DELTA_HALO, mass=mass[599:700], dlogm=dlogm)
    halostat  = HaloStat(cosmo=cp, z=z, DELTA_HALO=DELTA_HALO, rho_mean_0=2.775e11*0.7*0.7*0.272)

    mass2  = halostat.mass
    dndm2  = halostat.mass_function

#    print(dndm2/dndm[599:700])

#    halostat  = HaloStat(cosmo=cp, z=0, DELTA_HALO=DELTA_HALO)
#    mass3  = halostat.mass
#    dndm3  = halostat.mass_function

    jeremydat    = "test/mf0.23.jeremy.dat"
    mass4, dndm4 = np.genfromtxt(jeremydat, unpack=True) 


    fig1 = plt.figure(1, figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(mass,   dndm, 'g-', label="rozo mf")
    ax1.plot(mass2, dndm2, 'r-', label="zu mf")
#    ax1.plot(mass2*cp.h, dndm2/cp.h**4, 'ro')
#    ax1.plot(mass3, dndm3, 'b-')
#    ax1.plot(mass4, dndm4, 'k-', label="tinker mf")
    ax1.plot(mass4/cp.h, dndm4*cp.h**4, 'k-', label="tinker mf")
    ax1.legend(loc=2)
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel('dn/dM [Mpc$^{-3}$]')
    ax1.set_xlabel('M [$M_\odot$]')
    ax1.set_xlim(1.e10, 1.e16)
#    ax1.set_ylim(1.e-10, 1.)
    ax1.set_title("Rozo Halo Mass Function")
    plt.show()


def plotmf(z=0.0, DELTA_HALO=200.):
    import matplotlib.pyplot as plt
    halostat  = HaloStat(z=z, DELTA_HALO=DELTA_HALO)
    halomass1 = halostat.mass
    massfunc1 = halostat.mass_function
    fig1 = plt.figure(1, figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot(halomass1, massfunc1, 'go')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_ylabel('dn/dM [$M^{-1}_\odot Mpc^{-3}$]')
    ax1.set_xlabel('M [$M_\odot$]')
    ax1.set_title("Halo Mass Function")
    mass0 = 1.e15
    indx0 = np.searchsorted(halomass1, mass0)
    slope,intercept=np.polyfit(np.log(halomass1[indx0-1: indx0+2]),
                               np.log(massfunc1[indx0-1: indx0+2]),1)
    print("slope at lg(M) = 15 is %6.3f" % slope)
    plt.show()


def plotbf(z=0.0, DELTA_HALO=200.):
    import matplotlib.pyplot as plt
    halostat  = HaloStat(z=z, DELTA_HALO=DELTA_HALO)
    halomass1 = halostat.mass
    biasfunc1 = halostat.bias_function
    fig2 = plt.figure(2, figsize=(8,6))
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot(halomass1, biasfunc1, 'go')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylabel('b')
    ax2.set_xlabel('M [$M_\odot$]')
    ax2.set_title("Halo Bias Function")
    mass0 = 1.e15
    indx0 = np.searchsorted(halomass1, mass0)
    slope,intercept=np.polyfit(np.log(halomass1[indx0-1: indx0+2]),
                               np.log(biasfunc1[indx0-1: indx0+2]),1)
    print("slope at lg(M) = 15 is %6.3f" % slope)
    plt.show()

def mf_redshift():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    cp = CosmoParams(set_flat=True)
    color_list   = ["blue", 
                    "cyan", 
                    "green", 
                    "black", 
                    "yellow", 
                    "magenta", 
                    "red"
                   ]
    redshift_list =[0.100,                                   
                    0.133,                                   
                    0.167,                                   
                    0.200,
                    0.233,                                   
                    0.267,                                   
                    0.300,
                   ]
    label_list   = ["$z=0.100$",                                   
                    "$z=0.133$",                                   
                    "$z=0.167$",                                   
                    "$z=0.200$",                                   
                    "$z=0.233$",                                   
                    "$z=0.267$",                                   
                    "$z=0.300$",
                   ]
    lin = LinearTheory(cosmo=cp)
    den = Density(cosmo=cp)
    sigma_r_0 = lin.sigma_r_0_interp()
    for redshift, color, label in zip(redshift_list, color_list, 
                                                     label_list):
        fgrowth = den.growth_factor(redshift)
        sigma_r = lambda x:sigma_r_0(x)*fgrowth
        hs = HaloStat(cosmo=cp, z=redshift, sigma_r=sigma_r)
        mass = hs.mass
        dndm = hs.mass_function
        ax.plot(mass, dndm, color=color, ls="-", lw=2, label=label)
    ax.legend(loc="lower left") 
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e12, 1e16)
    ax.set_ylim(1e-30, 1e-14)
    ax.set_ylabel('dn/dM [$M^{-1}_\odot Mpc^{-3}$]')
    ax.set_xlabel('M [$M_\odot$]')
    ax.set_title("Dependence of Halo Mass Function on redshift") 
    plt.show()

def mf_sigma8():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    cp = CosmoParams(set_flat=True)
    color_list   = ["blue", 
                    "cyan", 
                    "green", 
                    "black", 
                    "yellow", 
                    "magenta", 
                    "red"
                   ]
    sigma_8_list = [0.694,                                   
                    0.734,                                   
                    0.775,                                   
                    0.816,
                    0.857,                                   
                    0.898,                                   
                    0.938,
                   ]
    label_list   = ["$\sigma_8=0.694$",                                   
                    "$\sigma_8=0.734$",                                   
                    "$\sigma_8=0.775$",                                   
                    "$\sigma_8=0.816$",                                   
                    "$\sigma_8=0.857$",                                   
                    "$\sigma_8=0.898$",                                   
                    "$\sigma_8=0.938$",
                   ]
    for sigma_8, color, label in zip(sigma_8_list, color_list, 
                                                   label_list):
        cp.sigma_8 = sigma_8
        cp.set_flat()
        hs = HaloStat(cosmo=cp)
        mass = hs.mass
        dndm = hs.mass_function
        ax.plot(mass, dndm, color=color, ls="-", lw=2, label=label)
    ax.legend(loc="lower left")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e12, 1e16)
    ax.set_ylim(1e-30, 1e-14)
    ax.set_ylabel('dn/dM [$M^{-1}_\odot Mpc^{-3}$]')
    ax.set_xlabel('M [$M_\odot$]')
    ax.set_title("Dependence of Halo Mass Function on $\sigma_8$")
    plt.show()

def bf_sigma8():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    cp = CosmoParams(set_flat=True)
    color_list   = ["blue", 
                    "cyan", 
                    "green", 
                    "black", 
                    "yellow", 
                    "magenta", 
                    "red"
                   ]
    sigma_8_list = [0.694,                                   
                    0.734,                                   
                    0.775,                                   
                    0.816,
                    0.857,                                   
                    0.898,                                   
                    0.938,
                   ]
    label_list   = ["$\sigma_8=0.694$",                                   
                    "$\sigma_8=0.734$",                                   
                    "$\sigma_8=0.775$",                                   
                    "$\sigma_8=0.816$",                                   
                    "$\sigma_8=0.857$",                                   
                    "$\sigma_8=0.898$",                                   
                    "$\sigma_8=0.938$",
                   ]
    for sigma_8, color, label in zip(sigma_8_list, color_list, 
                                                   label_list):
        cp.sigma_8 = sigma_8
        cp.set_flat()
        hs = HaloStat(cosmo=cp)
        mass = hs.mass
        bias = hs.bias_function
        ax.plot(mass, bias, color=color, ls="-", lw=2, label=label)

    ax.legend(loc="upper left")
    ax.set_xscale('log')
    ax.set_xlim(1e12, 1e16)
    ax.set_ylim(0, 30)
    ax.set_ylabel('b')
    ax.set_xlabel('M [$M_\odot$]')
    ax.set_title("Dependence of Halo Bias Function on $\sigma_8$")
    plt.show()

def mf_omegam_PS():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    cp = CosmoParams(set_flat=True)
    lin = LinearTheory(cosmo=cp)
    sigma_r = lin.sigma_r
    color_list   = ["blue", 
                    "cyan", 
                    "green", 
                    "black", 
                    "yellow", 
                    "magenta", 
                    "red"
                   ]
    omega_m_list = [                                   
                    0.315,                                   
                    0.301,                                   
                    0.288,                                   
                    0.274,                                   
                    0.260,                                    
                    0.247,                                   
                    0.233,
                   ]
    label_list   = [                                   
                    "$\Omega_m=0.315$",                                   
                    "$\Omega_m=0.301$",                                   
                    "$\Omega_m=0.288$",                                   
                    "$\Omega_m=0.274$",                                   
                    "$\Omega_m=0.260$",                                    
                    "$\Omega_m=0.247$",                                   
                    "$\Omega_m=0.233$",
                   ]
    for omega_m, color, label in zip(omega_m_list, color_list, 
                                                   label_list):
        cp.omega_M_0 = omega_m
        cp.set_flat()
        hs = HaloStat(cosmo=cp, sigma_r=sigma_r)
        mass = hs.mass
        dndm = hs.mass_function
        ax.plot(mass, dndm, color=color, ls="-", lw=2, label=label)

    ax.legend(loc="lower left")
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e12, 1e16)
    ax.set_ylim(1e-30, 1e-14)
    ax.set_ylabel('dn/dM [$M^{-1}_\odot Mpc^{-3}$]')
    ax.set_xlabel('M [$M_\odot$]')
    ax.set_title("Dependence of Halo Mass Function on $\Omega_m$ [P(k) prior]")
    plt.show()


def bf_omegam_PS():
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    cp = CosmoParams(set_flat=True)
    lin = LinearTheory(cosmo=cp)
    sigma_r = lin.sigma_r
    color_list   = ["blue", 
                    "cyan", 
                    "green", 
                    "black", 
                    "yellow", 
                    "magenta", 
                    "red"
                   ]
    omega_m_list = [                                   
                    0.315,                                   
                    0.301,                                   
                    0.288,                                   
                    0.274,                                   
                    0.260,                                    
                    0.247,                                   
                    0.233,
                   ]
    label_list   = [                                   
                    "$\Omega_m=0.315$",                                   
                    "$\Omega_m=0.301$",                                   
                    "$\Omega_m=0.288$",                                   
                    "$\Omega_m=0.274$",                                   
                    "$\Omega_m=0.260$",                                    
                    "$\Omega_m=0.247$",                                   
                    "$\Omega_m=0.233$",
                   ]
    for omega_m, color, label in zip(omega_m_list, color_list, 
                                                   label_list):
        cp.omega_M_0 = omega_m
        cp.set_flat()
        hs = HaloStat(cosmo=cp, sigma_r=sigma_r)
        mass = hs.mass
        bias = hs.bias_function
        ax.plot(mass, bias, color=color, ls="-", lw=2, label=label)

    ax.legend(loc="upper left")
    ax.set_xscale('log')
    ax.set_xlim(1e12, 1e16)
    ax.set_ylim(0, 30)
    ax.set_ylabel('b')
    ax.set_xlabel('M [$M_\odot$]')
    ax.set_title("Dependence of Halo Bias Function on $\Omega_m$ [P(k) prior]")
    plt.show()
#-------------------------------------------

def plotslopes():
    import matplotlib.pyplot as plt
    z=0.0
    DELTA_HALO=200
    halostat  = HaloStat(z=z, DELTA_HALO=DELTA_HALO)
    halostat.reset_halo_mass(1.e12, 1.e16, 100)
    halomass = halostat.mass
    massfunc = halostat.mass_function
    biasfunc = halostat.bias_function
    # to make it dn/dlnM from dn/dM
    massfunc = massfunc*halomass

    nmass = len(halomass)
    new_mass = halomass[1:-2]
    mf_slope = np.zeros(nmass-3)
    bf_slope = np.zeros(nmass-3)
    for i in range(nmass-3):
        mf_slope[i],intercept=np.polyfit(np.log(halomass[i: i+3]),
                                         np.log(massfunc[i: i+3]),1)
        bf_slope[i],intercept=np.polyfit(np.log(halomass[i: i+3]),
                                         np.log(biasfunc[i: i+3]),1)

    fig1 = plt.figure(1, figsize=(8,8))
    ax1 = fig1.add_subplot(211)
    mf_slope = - mf_slope
    # 2*alpha*alpha
    alpha = 0.757
    alpha2 = 2.*0.757*0.757
    ax1.plot(new_mass, mf_slope, 'm-', label="$n(M)$ slope: $\gamma$")
    ax1.plot(new_mass, bf_slope, 'b-', label="$b(M)$ slope: $\\beta$")
    ax1.plot(new_mass, (mf_slope-bf_slope)*bf_slope/alpha2, 'k-',
          label="$\\frac{(\gamma-\\beta)*\\beta}{2\\alpha^2}$", lw=2)
    ax1.plot(new_mass, (mf_slope-bf_slope)/alpha, 'g-',
          label="$\\frac{\gamma-\\beta}{\\alpha}$", lw=2)
    ax1.set_ylabel('$[1/\delta\sigma^2_{\ln N}]\\times [\delta \ln b]$')
    ax1.set_ylabel('Slope')
    ax1.set_xlabel('M [$M_\odot$]')
    ax1.set_xlim(1e12, 5e15)
    ax1.set_ylim(0, 7)
    ax1.set_xscale('log')
    ax1.legend(loc="upper left")
    ax1.grid(True)

    ax2 = fig1.add_subplot(212)
    scatter = 0.357
    eps = (2.*scatter)**2 - scatter**2
    ax2.plot(new_mass, -eps*(mf_slope-bf_slope)*bf_slope/alpha2,'r-',
            label="$\sigma_{\ln N}$: $0.357 \\times 2$")
    eps = (1.5*scatter)**2 - scatter**2
    ax2.plot(new_mass, -eps*(mf_slope-bf_slope)*bf_slope/alpha2,'m-',
            label="$\sigma_{\ln N}$: $0.357 \\times 1.5$")
    eps = (1.2*scatter)**2 - scatter**2
    ax2.plot(new_mass, -eps*(mf_slope-bf_slope)*bf_slope/alpha2,'y-',
            label="$\sigma_{\ln N}$: $0.357 \\times 1.2$")
    eps = (0.8*scatter)**2 - scatter**2
    ax2.plot(new_mass, -eps*(mf_slope-bf_slope)*bf_slope/alpha2,'g-',
            label="$\sigma_{\ln N}$: $0.357 \\times 0.8$")
    eps = (0.5*scatter)**2 - scatter**2
    ax2.plot(new_mass, -eps*(mf_slope-bf_slope)*bf_slope/alpha2,'b-',
            label="$\sigma_{\ln N}$: $0.357 \\times 0.5$")
    eps = (0.0*scatter)**2 - scatter**2
    ax2.plot(new_mass, -eps*(mf_slope-bf_slope)*bf_slope/alpha2,'k-',
            label="$\sigma_{\ln N}$: $0.357 \\times 0.0$")
    ax2.set_ylabel("$\delta b/b \sim \\frac{-(\gamma-\\beta)\\beta}{2\\alpha^2}  \delta\,\sigma^2_{\ln N}$")
    ax2.set_xlim(1e12, 5e15)
    ax2.set_ylim(-1.5, 1)
    ax2.set_xscale('log')
    ax2.legend(loc="lower left", ncol=2)
    ax2.set_xlabel('M [$M_\odot$]')
    ax2.grid(True)
#    ax2.set_yscale('log')
    plt.show()

def plotderivbias(z=0.0, DELTA_HALO=200.):
    import matplotlib.pyplot as plt
    halostat  = HaloStat(z=z, DELTA_HALO=DELTA_HALO)
    halomass  = halostat.mass
    derivb1   = halostat.db_ddeltasc(kind="ST99")
    derivb2   = halostat.db_ddeltasc(kind="Tinker10")
    derivb3   = halostat.db_ddeltasc(kind="Tinker10EQ15")
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(halomass, derivb1, 'r-', label="ST99")
    ax.plot(halomass, derivb2, 'g-', label="Tinker10")
    ax.plot(halomass, derivb3, 'b-', label="Tinker10EQ15")
    ax.legend(loc=4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('M [$M_\odot$]')
    ax.set_ylabel('derivative')
    ax.set_title("$\partial\\bar{b}(M)/\partial\delta_\mathrm{sc}$")
    plt.show()

def mf_oms8(alpha=0.4):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig = plt.figure(figsize=(8,6))
    ax  = fig.add_subplot(111)
    cp = CosmoParams(set_flat=True)
    const = 0.85*0.30**alpha
    s8arr = np.arange(0.2, 1.2, 0.1)
    omarr = np.power(const/s8arr, 1./alpha)
    j = 0
    for sigma_8, omega_M_0 in zip(s8arr, omarr):
        cp.sigma_8 = sigma_8
        cp.set_flat()
        hs = HaloStat(cosmo=cp)
        mass = hs.mass
        dndm = hs.mass_function
        color = cm.jet(1.*j/len(s8arr))
        label = "$\sigma_8="+format(sigma_8, "3.2f")+"$"
        ax.plot(mass, dndm*mass, color=color, alpha=0.5, ls="-", lw=1, label=label)
        j+=1
    ax.legend(loc="lower left")
    ax.text(0.1, 0.1, "$\sigma_8\omega_m^{"+format(alpha, "3.2f")+"}$", transform = ax.transAxes)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(1e12, 1e16)
#    ax.set_ylim(1e-30, 1e-14)
    ax.set_ylim(1e-10, 5e-2)
#    ax.set_ylabel('dn/dM [$M^{-1}_\odot Mpc^{-3}$]')
    ax.set_xlabel('M [$M_\odot$]')
#    ax.set_title("Dependence of Halo Mass Function on $\sigma_8$")
    plt.show()


if __name__ == "__main__":
#    plotmf()
#    plotbf()
#    testUniversalMF()
#    mf_sigma8()
#    bf_sigma8()
#    mf_omegam_PS()
#    bf_omegam_PS()
#    plotslopes()
#    plotderivbias()
#    plotwarren()
#    plotrozo()
#    mf_redshift()
    mf_oms8(alpha=0.4)
