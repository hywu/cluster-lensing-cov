#!/usr/bin/env python
#Last-modified: 15 Dec 2011 06:25:40 PM

import warnings
import unittest

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
from math import log, log10, exp, sqrt
import scipy.integrate as si

from .param import CosmoParams
from .density import Density
from .lineartheory import LinearTheory
from .frw import FRW
from . import cex
from . import constants as cc

#----------------------------------------------------------------------

class HaloGrowth(object):
    """ Growth history of dark matter halos, as described by
    Zhao et al. (2009).

    .. warning ::
    
        *lg* for :math:`\log_{10}`, *ln* for :math:`\ln` (i.e.,
        :math:`\log_e`)

    Parameters
    ----------
    cosmo: CosmoParams object, optional
        Cosmology for calculating the current background density (default:
        None).
    rho_mean_0: scalar, optional
        Current background density.  It will overwrite the value
        calculated from cosmo (default: None).
    sigma_r_0: callable object, optional
        Function to calculate the r.m.s. of current linear
        density field. It will overwrite the function derived
        from cosmo (default: None).
    delta_crit: callable object, optional
        Function to calculate the collapse barrier in peak
        formalism. It will overwrite the function derived from
        cosmo (default: None).
    age: callable object, optional
        Function to calculate the age of the universe. It will
        overwrite the function derived from cosmo (default: None).
    mmin: float, optional                                           
        Minimum halo mass (default: 1.e5 Msun).                     
    mmax: float, optional                                           
        Maximum halo mass (default: 1.e17 Msun).                    
    nmbin: int, optional                                            
        Number of mass bins (default: 50)      
    zmin: float, optional                                           
        Minimum redshift (default: 0).                     
    zmax: float, optional                                           
        Maximum redshift (default: 9).                    
    nzbin: int, optional                                            
        Number of z bins (default: 100)      

    """
    def __init__(self, cosmo=None, rho_mean_0=None, sigma_r_0=None, 
                       delta_crit=None, age=None,
                       mmin=1.e5, mmax=1.e17, nmbin=50,
                       zmin=0.0, zmax=9.0, nzbin=100):
        if cosmo is None:
            cp = CosmoParams(set_flat=True)
        else:
            cp = cosmo
            if not cp.isflat:
                warnings.warn("Non-flat Universe encountered!")

        self.h = cp.h
        self.m0 = cp.omega_M_0
        self.q0 = cp.omega_lambda_0
        self.k0 = cp.omega_k_0
        self.frac_magic = 0.04

        if (rho_mean_0 is None) or (delta_crit is None):
            # get all the related densities
            den = Density(cosmo=cp)
            if rho_mean_0 is None:
                rho_mean_0 = den.rho_mean_z(z=0.0)

            if delta_crit is None:
                self.delta_crit = den.delta_crit
            else:
                if not hasattr(delta_crit, "__call__"):
                    raise cex.NonCallableObject(delta_crit)
                self.delta_crit = delta_crit

        if sigma_r_0 is None:
            # the initial perturbation (extrapolated to z=0)
            lin = LinearTheory(cosmo=cp, z=0.0)
            self.sigma_r_0 = lin.sigma_r
        else:
            if not hasattr(sigma_r_0, "__call__"):
                raise cex.NonCallableObject(sigma_r_0)
            self.sigma_r_0 = sigma_r_0

        if age is None:
            # calculate t-z relation in FRW metric
            self.age = FRW(cosmo=cp).age
        else:
            if not hasattr(age, "__call__"):
                raise cex.NonCallableObject(age)
            self.age = age

        # converstion from mass to radius 
        self.r3conv = 3.0/(4.0*cc.pi*rho_mean_0)
        # initialize mass and scale range
        self._init_mass(mmin, mmax, nmbin)
        # initialize redshift and expansion factor range
        self._init_time(zmin, zmax, nzbin)
        # sigma(M) of the initial perturbation (extrapolated to z=0)
        self._get_sigma_M()
        # delta_crit
        self._get_deltacrit()
        # s(M)
        self._get_s_M()

    def _init_mass(self, mmin, mmax, nmbin):
        """ Initialize the mass array.                         
        """
        self.lgmass, self.dlgmass = np.linspace(log10(mmin), log10(mmax),  
                                             num=nmbin, retstep=True)
        self.mass  = np.power(10.0, self.lgmass) 
        self.scale = np.power(self.r3conv*self.mass, 1.0/3.0) 
        self.lgmmax = self.lgmass[-1]
        self.lgmmin = self.lgmass[0]

    def _init_time(self, zmin, zmax, nzbin):
        """ Initialize the redshift array.

        .. warning ::
            
            lgz is actually lg(1+z) in the source code.
        """
        # lgz is actually lg(1+z)
        # this actually very ill-defined as we really want to pack the early time with more 
        # sampling points...
#        self.lgz = np.linspace(log10(zmin+1), log10(zmax+1), num=nzbin)
#        self.lgzmax = self.lgz[-1]
#        self.lgzmin = self.lgz[0]
#        self.z   = np.power(10., self.lgz) - 1.

        _lgzrev = np.linspace(log10(zmin+1), log10(zmax+1), num=nzbin)
        _zrev   = zmin + zmax - (np.power(10.0, _lgzrev) - 1)
        self.z  = _zrev[::-1] 
        self.lgz    = np.log10(self.z+1)
        self.lgzmax = self.lgz[-1]
        self.lgzmin = self.lgz[0]


    def _get_sigma_M(self):
        """
        """
        sig    = self.sigma_r_0(self.scale)
        self.lgsig  = np.log10(sig)
        self.lgsig_reverse = self.lgsig[::-1]

        self.dlgsig = self.lgsig[2:] - self.lgsig[:-2]
        self.f_lgsig         = spline1d(self.lgmass, self.lgsig)
        self.f_lgsig_reverse = spline1d(self.lgsig_reverse, 
                                        self.lgmass[::-1])

    def _get_deltacrit(self):
        deltacrit = self.delta_crit(self.z)
        lgdc = np.log10(deltacrit)
        self.f_lgdc_reverse = spline1d(lgdc, self.lgz)
        self.f_lgdc = spline1d(self.lgz, lgdc)
        min_interval_lgdc = np.min(lgdc[1:]-lgdc[:-1])
        self.stp_lgdc  = 2.*min_interval_lgdc

    def _get_s_M(self):
        """
        """
        lgs = self.lgsig[1:-1] + self.dlgsig/(self.dlgmass*2.0)
        self.f_lgs = spline1d(self.lgsig_reverse[1:-1], lgs[::-1])


    def _w_lg(self, lgsig, lgdc):
        """
        """
        _deltacrit = np.power(10.0, lgdc)
        _s = np.power(10.0, self.f_lgs(lgsig))
        _w_lg = _deltacrit/_s
        return(_w_lg)

    def MAH(self, mobs, zobs, mseed=None, set_history=False):
        """ Mass accretion history of a halo with mass mobs at zobs.

        Parameters
        ----------
        mobs: scalar
            Halo virial mass at observing epoch.
        zobs: scalar
            Redshift of Observing epoch.
        mseed: scalar, optional
            The mininum mass of the protohalos required, default is
            1% of mobs.
        set_history: Boolean, optional
            True to return the mass accretion history in a 2d array.
            Default is False.

        Returns
        -------
        cvir: scalar
            Concentration at observing epoch.
        mah_history: array_like, optional
            2d array with the 1st axis as masses and the 2nd axis as
            redshifts in the accretion history. Returned if
            set_history is True.

        """
        lgmobs = log10(mobs)
        lgzobs = log10(zobs+1.)
        if ((lgmobs > self.lgmmax) or 
            (lgmobs < self.lgmmin)):
            print("lgmobs beyond range lgmobs %10.3f lgmmin %10.3f lgmmax %10.3f "%(lgmobs, self.lgmmin, self.lgmmax))
            raise cex.ParameterOutsideDefaultRange(mobs)
        if ((lgzobs > self.lgzmax) or 
            (lgzobs < self.lgzmin)):
            raise cex.ParameterOutsideDefaultRange(zobs)
        # starting mass
        if mseed is None:
            lgmseed = lgmobs - 2.0
        else:
            lgmseed = log10(mseed)
        if (lgmseed < self.lgmmin):
            print("lgmmin too large lgmseed %10.3f lgmmin %10.3f"%(lgmseed, self.lgmmin))
            raise cex.ParameterOutsideDefaultRange(mseed)
        # for concentration
        m_magic = mobs*self.frac_magic
        lgmmagic = log10(m_magic)
        if (lgmmagic < lgmseed):
            raise cex.ParameterOutsideDefaultRange(m_magic)

        lgz_magic, lgm_history, lgz_history = self._MAH_lg(
                                lgmobs, lgzobs, lgmseed, lgmmagic)
        t_magic = self.age(10.**lgz_magic-1.)
        t_obs   = self.age(zobs)
#        t_magic = self._age(10.**lgz_magic-1.)
#        t_obs   = self._age(zobs)
        cvir = self._cvir_fit(t_magic, t_obs)
        if set_history:
            m_history = np.power(10., lgm_history)
            z_history = np.power(10., lgz_history)-1.
            mah_history = np.vstack([m_history, z_history])
            return(cvir, mah_history)
        else:
            return(cvir)

#    def _age(self, z):
#        ul = 0.
#        dl = 1./(1.+z)**1.5
#        uniage1d, err = si.quad(self._ctfunc, ul, dl, limit=1000)
#        uniage1d = uniage1d*9.78e11/self.h*2./3.
#        return(uniage1d)

#    def _ctfunc(self, y):
#        _ct=1./sqrt(self.m0+self.q0*y**2.+
#                    (1.-self.m0-self.q0)*y**(2./3.))
#        return(_ct)


    def _MAH_lg(self, lgmobs, lgzobs, lgmseed, lgmmagic):
        lgsig_obs  = self.f_lgsig(lgmobs)
        lgdc_obs   = self.f_lgdc(lgzobs)
        lgsigseed  = self.f_lgsig(lgmseed)
        lgsigmagic = self.f_lgsig(lgmmagic)

        lgsig_acc = lgsig_obs
        lgdc_acc  = lgdc_obs
        lgdc_step = self.stp_lgdc
        lgsig_history = [ lgsig_obs ]
        lgdc_history  = [ lgdc_obs ]

        i = 0
        magic_id = -1
        while(lgsig_acc < lgsigseed):
            _dlgsig_dlgdc = self.dlgsig_dlgdeltacrit(lgsig_obs, lgdc_obs, 
                                                     lgsig_acc, lgdc_acc)

            lgsig_acc  = lgsig_acc + lgdc_step*_dlgsig_dlgdc
            lgdc_acc   = lgdc_acc + lgdc_step
            lgsig_history.append(lgsig_acc)
            lgdc_history.append(lgdc_acc)
            if (lgsig_acc > lgsigmagic) and (magic_id == -1):
                magic_id = i
            i = i + 1

        if magic_id == -1:
            raise cex.ConditionNotReached("cannot track to the magic time of\
                    halo growth")

        lgsig_history = np.asarray(lgsig_history).ravel()
        lgdc_history  = np.asarray(lgdc_history).ravel()

        lgm_history = self.f_lgsig_reverse(lgsig_history)
        lgz_history = self.f_lgdc_reverse(lgdc_history)

        # linear interpolation
        lgz_magic = (((lgmmagic-lgm_history[magic_id])*
                       (lgz_history[magic_id+1]-lgz_history[magic_id])/
                       (lgm_history[magic_id+1]-lgm_history[magic_id]))+
                       lgz_history[magic_id])
        return(lgz_magic, lgm_history, lgz_history)

        
    def _cvir_fit(self, t_magic, t_obs):
        _cvir = 4.*pow(1. + pow(t_obs/(3.74447097*t_magic), 8.4), 0.125)
        return(_cvir)


    def dlgsig_dlgdeltacrit(self, lgsigobs, lgdcobs, lgsigacc, lgdcacc):
        """ Function to calculate the growth in terms of scaled variables.
        """
        _d = ((self._w_lg(lgsigacc, lgdcacc) - 
               self._p_z(lgdcacc, lgsigobs, lgdcobs))/
              5.85)
        return(_d)

    def _p_z(self, lgdc, lgsigobs, lgdcobs):
        _w_0 = self._w_lg(lgsigobs, lgdcobs)
        _p_0 = self._p_obs(_w_0)

        _x = 1.-((lgdc - lgdcobs)/
                 (0.272/_w_0))
        _p = _p_0*max(0.0, _x)
        return(_p)

    def _p_obs(self, wobs):
        _p = wobs/(2.*(1.+pow(wobs/4., 6)))
        return(_p)

#-------------------------------------------

def plotcm():
    import matplotlib.pyplot as plt
    hg = HaloGrowth()
    mmin=1.e10; mmax=1.e16; nmbin=20
    log10mass = np.linspace(log10(mmin), log10(mmax),   
                             num=nmbin)    
    mass  = np.power(10.0, log10mass)
    zobs  = 0.0
    cvir  = np.zeros(nmbin)
    for i in range(nmbin):
        cvir[i] = hg.MAH(mass[i]/hg.h, zobs, set_history=False)
        print(np.log10(mass[i])),
        print(cvir[i])

    fig1 = plt.figure(1, figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)                                       
    ax1.plot(mass, cvir, 'ro')                                
    ax1.set_yscale('log')                                               
    ax1.set_xscale('log')                                               
    ax1.set_xlim(1e10, 1e16)
    ax1.set_ylim(1, 20)
    ax1.set_ylabel('c')                               
    ax1.set_xlabel('M [$h^{-1} M_\odot$]')                                     
    ax1.set_title("C-M")                                 
    plt.show()                      

def plotcmevo():
    import matplotlib.pyplot as plt
    fig1 = plt.figure(1, figsize=(8,6))
    ax1 = fig1.add_subplot(1,1,1)                                       
    hg = HaloGrowth()
    mmin=1.e10; mmax=1.e16; nmbin=20
    log10mass = np.linspace(log10(mmin), log10(mmax),   
                             num=nmbin)    
    mass  = np.power(10.0, log10mass)
    cvir  = np.zeros(nmbin)
    zobs  = 0.1
    for i in range(nmbin):
        cvir[i] = hg.MAH(mass[i]/hg.h, zobs, set_history=False)
    ax1.plot(mass/hg.h, cvir, 'ro', label="z=0.1")                                
    zobs  = 0.3
    for i in range(nmbin):
        cvir[i] = hg.MAH(mass[i]/hg.h, zobs, set_history=False)
    ax1.plot(mass/hg.h, cvir, 'go', label="z=0.3")                                
    ax1.legend(loc=1)
    ax1.set_yscale('log')                                               
    ax1.set_xscale('log')                                               
    ax1.set_xlim(1e10, 1e16)
    ax1.set_ylim(1, 20)
    ax1.set_ylabel('c')                               
    ax1.set_xlabel('M [$h^{-1} M_\odot$]')                                     
    ax1.set_title("C-M")                                 
    plt.show()                      

def profiling():
    cp = CosmoParams()
    lin = LinearTheory(cosmo=cp)
    sigma_r_0 = lin.sigma_r_0 
    hg1 = HaloGrowth()

#@profile
def testMAH():
#    import matplotlib.pyplot as plt
    cp = CosmoParams()
    lin = LinearTheory(cosmo=cp)                                        
    sigma_r_0 = lin.sigma_r_0_interp()
    hg = HaloGrowth(sigma_r_0=sigma_r_0)
    mass = 1.e15
#    cvir,mah = hg.MAH(mass, 0.0, set_history=True)
    cvir = hg.MAH(mass, 0.0, set_history=False)
#    fig = plt.figure(figsize=(8,6))
#    ax = fig.add_subplot(1,1,1)                                       
#    ax.plot(mah[1]+1, mah[0], 'k-', lw=4, alpha=0.5)
#    ax.set_yscale('log')                                               
#    ax.set_xscale('log')                                               
#    ax.set_xlim(0.9, 5)
#    ax.set_ylim(1e12, 2e15)
#    ax.set_xlabel('z+1')                               
#    ax.set_ylabel('M [$M_\odot$]')                                     
#    plt.show()                      

def plotmah():
    import matplotlib.pyplot as plt
    cp = CosmoParams()
    lin = LinearTheory(cosmo=cp)                                        
    sigma_r_0 = lin.sigma_r_0_interp()

    hg1 = HaloGrowth()
    mpower = 15
    mass = 10**mpower*hg1.h
    zobs = 0.0
    cvir,mah1 = hg1.MAH(mass, zobs, mseed=2.e7, set_history=True)

    cp = CosmoParams(sigma_8 = 1.1)                                
    cp.set_flat()                                                    
    lin = LinearTheory(cosmo=cp, z=0)                                
    sigma_r_0 = lin.sigma_r_0_interp()
    hg2 = HaloGrowth(sigma_r_0=sigma_r_0)
    mpower = 15
    mass = 10**mpower*hg2.h
    zobs = 0.0
    cvir,mah2 = hg2.MAH(mass, zobs, mseed=2.e7, set_history=True)

    cp = CosmoParams(omega_M_0 = 0.4)
    cp.set_flat()                                                    
    hg3 = HaloGrowth(cosmo=cp, sigma_r_0=sigma_r_0)
    mpower = 15
    mass = 10**mpower*hg3.h
    zobs = 0.0
    cvir,mah3 = hg3.MAH(mass, zobs, mseed=2.e7, set_history=True)

    fig2 = plt.figure(2, figsize=(8,6))
    ax2 = fig2.add_subplot(1,1,1)                                       
    ax2.plot(mah1[1]+1, mah1[0], 'k-', label="fiducial", lw=4, alpha=0.5)
    ax2.plot(mah2[1]+1, mah2[0], 'g-', label="high $\sigma_8$") 
    ax2.plot(mah3[1]+1, mah3[0], 'r-', label="high $\Omega_m$, P(k) prior")
    ax2.legend(loc="lower left")
    ax2.set_yscale('log')                                               
    ax2.set_xscale('log')                                               
    ax2.set_xlim(0.5, 20)
    ax2.set_ylim(1e8, 1e16)
    ax2.set_xlabel('z+1')                               
    ax2.set_ylabel('M [$M_\odot$]')                                     
    plt.show()                      


def plotcvir_resolution():
    import matplotlib.pyplot as plt
    zobs  = 0.0
    mmin=1.e10; mmax=1.e16; nmbin=20
    log10mass = np.linspace(log10(mmin), log10(mmax),
                             num=nmbin)
    mass  = np.power(10.0, log10mass)
    cp = CosmoParams()
    lin = LinearTheory(cosmo=cp)                                        
    sigma_r_0 = lin.sigma_r_0_interp()

    hg = HaloGrowth(mmin=1.e8, mmax=1.e16, nmbin=40, zmin=0, zmax=9, nzbin=200)
    cvir1  = np.zeros(nmbin)
    for i in range(nmbin):                                              
        cvir1[i] = hg.MAH(mass[i], zobs, set_history=False) 

#    hg = HaloGrowth(mmin=1.e8, mmax=1.e16, nmbin=20)
#    cvir2  = np.zeros(nmbin)
#    for i in range(nmbin):                                              
#        cvir2[i] = hg.MAH(mass[i], zobs, set_history=False) 

#    hg = HaloGrowth(zmin=0, zmax=9, nzbin=100)
#    cvir3  = np.zeros(nmbin)
#    for i in range(nmbin):                                              
#        cvir3[i] = hg.MAH(mass[i], zobs, set_history=False) 

    hg = HaloGrowth(mmin=1.e8, mmax=1.e16, nmbin=20, zmin=0, zmax=9, nzbin=20)
    cvir4  = np.zeros(nmbin)
    for i in range(nmbin):                                              
        cvir4[i] = hg.MAH(mass[i], zobs, set_history=False) 

    fig2 = plt.figure(2, figsize=(8,6))
    ax2 = fig2.add_subplot(1,1,1)                                       
    ax2.plot(mass, cvir1/cvir1, 'k-', label="fiducial", lw=4, alpha=0.5)
#    ax2.plot(mass, cvir2, 'g-', label="coarse mass") 
#    ax2.plot(mass, cvir3, 'r-', label="coarse time")
    ax2.plot(mass, cvir4/cvir1, 'b-', label="coarse mass and time")
    ax2.legend(loc="lower left")
    ax2.set_xscale('log')                                               
    ax2.set_xlim(1e10, 1e16)
#    ax2.set_yscale('log')                                               
#    ax2.set_ylim(1, 30)
    ax2.set_ylim(0.9, 1.1)
    ax2.set_ylabel('c')                               
    ax2.set_xlabel('M [$M_\odot$]')                                     
    ax2.set_title("C-M with different parameter resolution")                                 
    plt.show()                      


def plotcvir():
    import matplotlib.pyplot as plt
    zobs  = 0.0
    mmin=1.e10; mmax=1.e16; nmbin=20
    log10mass = np.linspace(log10(mmin), log10(mmax),
                             num=nmbin)
    mass  = np.power(10.0, log10mass)

    cp = CosmoParams()
    lin = LinearTheory(cosmo=cp)                                        
    sigma_r_0 = lin.sigma_r_0_interp()

    hg1 = HaloGrowth()
    cvir1  = np.zeros(nmbin)
    for i in range(nmbin):                                              
        cvir1[i] = hg1.MAH(mass[i], zobs, set_history=False) 

    cp = CosmoParams(sigma_8 = 1.1)                                
    cp.set_flat()                                                    
    lin = LinearTheory(cosmo=cp, z=0)                                
    sigma_r_0 = lin.sigma_r_0_interp()
    hg2 = HaloGrowth(sigma_r_0=sigma_r_0)
    cvir2  = np.zeros(nmbin)
    for i in range(nmbin):                                              
        cvir2[i] = hg2.MAH(mass[i], zobs, set_history=False) 

    cp = CosmoParams(omega_M_0 = 0.4)
    cp.set_flat()                                                    
    hg3 = HaloGrowth(cosmo=cp, sigma_r_0=sigma_r_0)
    cvir3  = np.zeros(nmbin)
    for i in range(nmbin):                                              
        cvir3[i] = hg3.MAH(mass[i], zobs, set_history=False) 

    fig2 = plt.figure(2, figsize=(8,6))
    ax2 = fig2.add_subplot(1,1,1)                                       
    ax2.plot(mass, cvir1, 'k-', label="fiducial", lw=4, alpha=0.5)
    ax2.plot(mass, cvir2, 'g-', label="high $\sigma_8$") 
    ax2.plot(mass, cvir3, 'r-', label="high $\Omega_m$, P(k) prior")
    ax2.legend(loc="lower left")
    ax2.set_yscale('log')                                               
    ax2.set_xscale('log')                                               
    ax2.set_xlim(1e10, 1e16)
    ax2.set_ylim(1, 30)
    ax2.set_ylabel('c')                               
    ax2.set_xlabel('M [$M_\odot$]')                                     
    ax2.set_title("C-M with different cosmology")                                 
    plt.show()                      

if __name__ == "__main__":
    """
    """
#    profiling()
#    plotmah()
    plotcm()
#    plotcvir()
#    testMAH()
#    plotcmevo()
#    plotcvir_resolution()
