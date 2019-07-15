#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from clens.ying.param import CosmoParams
from clens.ying.lineartheory import LinearTheory
from clens.ying.density import Density
from clens.ying.nfw import NFW
from clens.ying.halostat import HaloStat

import clens.util.constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.scaling_relation import RichnessSelection, FiducialScalingRelation



class CorrelationFunctions3D(object):
    """
    calculating xi_mm
    xi_hm (without richness-mass relation)
    xi_cm (*with* richness-mass relation)
    """
    def __init__(self, z, co, nu, lambda_min=1e14, lambda_max=2e14):
        self.z = z
        self.co = co
        self.nu = nu

        self.cosmo_ying = CosmoParams(omega_M_0=self.co.OmegaM, omega_b_0=self.co.OmegaB, omega_lambda_0=self.co.OmegaDE, h=self.co.h, sigma_8=self.co.sigma8, n=self.co.ns, tau=self.co.tau) # Ying's
        self.lin = LinearTheory(cosmo=self.cosmo_ying, z=self.z)
        self.den = Density(cosmo=self.cosmo_ying)

        self.set_up_radius()
        self.set_up_halos(self.z)
        rho_crit = cn.rho_crit_with_h * self.co.h**2  # h-free
        self.rho_mean_z = rho_crit * self.co.OmegaM # comoving density!
        self.fsr = FiducialScalingRelation(self.nu)
        #print('lambda_min, lambda_max %e %e'%(lambda_min, lambda_max))
        self.rs = RichnessSelection(scaling_relation=self.fsr, lambda_min=lambda_min, lambda_max=lambda_max)

    def set_up_radius(self):
        # set up 3D r
        rmin = 0.0001 #3D r for xi(r), need to be small!
        rmax = 1.e+4#1000.0
        self.nrbin = 400#200 
        x = np.linspace(np.log10(rmin), np.log10(rmax), num=self.nrbin)
        self.radius = np.power(10.0, x)

    def set_up_halos(self, zl):
        #mmin = 1e13
        #mmax = 5e16
        #nmbin = 91 # TODO! less sampling for 
        #M_arr = np.logspace(np.log10(mmin), np.log10(mmax), nmbin)
        #lgM_arr = np.log10(M_arr)
        #lnM_arr = np.log(M_arr)
        #dlogm = np.log(M_arr[1]) - np.log(M_arr[0])
        nM = 100
        self.lnM_arr = np.linspace(np.log(5e13), np.log(2e16), nM)
        M_arr = np.exp(self.lnM_arr)
        dlogm = self.lnM_arr[1] - self.lnM_arr[0]
        hs = HaloStat(cosmo=self.cosmo_ying, z=zl, mass=M_arr, dlogm=dlogm)
        dndM_arr = hs.mass_function
        bias_arr = hs.bias_function
        self.bias_lnM_interp = interp1d(self.lnM_arr, bias_arr)
        self.dndlnM_arr = dndM_arr * M_arr
        #self.lnM_arr = lgM_arr*np.log(10.)
        self.dndlnM_lnM_interp = interp1d(self.lnM_arr, self.dndlnM_arr)

    def xi_mm(self):
        pk_lin = self.lin.power_spectrum
        h = self.co.h
        k = 10**np.linspace(-4,2) # h-free
        pk = pk_lin(k) # h-free
        xi_arr = np.zeros(self.nrbin)
        for ir, r in enumerate(self.radius):
            kmin = 1e-3
            kmax = 200./r
            dlnk = 0.01
            lnk = np.arange(np.log(kmin),np.log(kmax),dlnk)
            k = np.exp(lnk)
            integrand = pk_lin(k) * k**3/(2*np.pi**2) * np.sin(k*r)/(k*r)
            xi_arr[ir] = np.trapz(integrand, x=lnk)
        return xi_arr

    def c200m_M200m(self, m200m):
        # Bhattacharya Table 2, last column
        growthD = self.den.growth_factor(self.z)
        # TODO!! this den.growth_factor is not accurate
        nu = (1./growthD) * (1.12*(m200m*self.co.h/5e+13)**0.3 + 0.53)
        return (growthD**1.15) * 9.0 * nu**(-0.29) 

    def xi_hm_1h(self, lnM, c=None):
        if c==None:
            c = self.c200m_M200m(m200m=np.exp(lnM))
        nfw = NFW(mass=np.exp(lnM), c=c, rho_mean=self.rho_mean_z)
        profile = nfw.profile(self.radius)
        xihm1h = profile/self.rho_mean_z - 1.0
        return(xihm1h)

    def xi_hm_2h(self, ximm, lnM):
        bias = self.bias_lnM_interp(lnM)
        xihm2h = bias * ximm
        return xihm2h

    def xi_hm(self, lnM=np.log(1.e14), c=None):
        xi_mm = self.xi_mm()
        xi_hm_2h = self.xi_hm_2h(xi_mm, lnM)
        xi_hm_1h = self.xi_hm_1h(lnM, c)
        xi_hm = np.maximum(xi_hm_1h, xi_hm_2h)
        return xi_hm


    def xi_cm(self):
        nM = len(self.lnM_arr)
        lnM_selection_arr = self.rs.lnM_selection(self.lnM_arr)
        dndlnM_arr = self.dndlnM_lnM_interp(self.lnM_arr)
        xi_mass_radius = np.zeros([nM,self.nrbin])
        for iM, lnM in enumerate(self.lnM_arr):
            xi_mass_radius[iM,:] = self.xi_hm(lnM)
        sum_xi = np.zeros(self.nrbin)
        for ir in range(self.nrbin):
            sum_xi[ir] = np.trapz(dndlnM_arr*lnM_selection_arr*xi_mass_radius[:,ir], x=self.lnM_arr)
        n = np.trapz(dndlnM_arr*lnM_selection_arr, x=self.lnM_arr)
        xi_mean = sum_xi/n
        return xi_mean


    def mean_density_and_bias(self):
        nM = 20
        #lnM_arr = np.linspace(np.log(1e13), np.log(1e15), nM)
        lnM_selection_arr = self.rs.lnM_selection(self.lnM_arr)
        dndlnM_arr = self.dndlnM_lnM_interp(self.lnM_arr)
        bias_arr = self.bias_lnM_interp(self.lnM_arr)
        # number density
        n = np.trapz(dndlnM_arr*lnM_selection_arr, x=self.lnM_arr)
        number_density = n
        print('number_density', number_density)
        # mean bias
        bn = np.trapz(bias_arr*dndlnM_arr*lnM_selection_arr, x=self.lnM_arr) 
        mean_bias = bn/n
        #print('mean_bias', mean_bias)
        return number_density, mean_bias


    def plot_sanity(self):
        #plt.figure(figsize=(14,14))
        #plt.subplot(221)
        xi_mm = self.xi_mm()
        plt.plot(self.radius, xi_mm, label='mm')


        # xi_hm
        xi_hm = self.xi_hm()
        plt.plot(self.radius, xi_hm, label='hm')

        # xi_cm
        xi_cm = self.xi_cm()
        plt.plot(self.radius, xi_cm, label='cm')

        plt.xscale('log')
        plt.yscale('log')
        plt.legend()

        # Ying's 
        # from zypy.zycosmo.halo_matter_correlation import xi_hm
        # xi3d = self.lin.correlation_func_interp()
        # r_arr = self.radius
        # ximm = xi3d(r_arr)
        # plt.plot(r_arr, ximm, label='Ying')
        # plt.xscale('log')
        # plt.yscale('log')
        
        

if __name__ == "__main__":
    #nu = NuisanceParameters()
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)# Mobs=Mtrue
    co = CosmoParameters()
    cf3d = CorrelationFunctions3D(z=0.5, co=co, nu=nu, lambda_min=1e14, lambda_max=2e14)
    cf3d.plot_sanity()

    plt.show()