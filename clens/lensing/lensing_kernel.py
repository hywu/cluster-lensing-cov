#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

from clens.ying.param import CosmoParams
from clens.ying.lineartheory import LinearTheory

from clens.util import constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.survey import Survey


"""
calaculating lensing kernel, making various sanity checks 
"""

class LensingKernel(object):
    def __init__(self, co, su):
        self.co = co
        self.su = su

        # astropy
        astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        self.chi = astropy_dist.comoving_distance

        self.cosmo_ying = CosmoParams(omega_M_0=self.co.OmegaM, omega_b_0=self.co.OmegaB, omega_lambda_0=self.co.OmegaDE, h=self.co.h, sigma_8=self.co.sigma8, n=self.co.ns, tau=self.co.tau) # Ying's

        self.calc_kernel()

    def calc_kernel(self):
        zl_min = 0.1
        # if self.su.zs_max is None:
        #     zs_max = 5.
        # else:
        if self.su.top_hat == True:
            print('using tophat source distribution')

        zs_max = self.su.zs_max
        zl_list = np.linspace(zl_min, zs_max-0.01, 100)
        kernel_list = np.zeros(len(zl_list))
        for iz, zl in enumerate(zl_list):
            #if self.su.zs_min is None:
            zs_list = np.linspace(max(zl+0.01,self.su.zs_min), zs_max, 100)
            #else:
            #    zs_list = np.linspace(self.su.zs_min, self.su.zs_max, 100)
                
            ns_list = self.su.pz_src(zs_list)
            chi_l = self.chi(z=zl).value
            chi_s = self.chi(z=zs_list).value
            #print('zl', zl, zs_list)
            Sigma_crit = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_l /(chi_s - chi_l)/(1.+zl) #Msun/Mpc^2
            integrand = ns_list/Sigma_crit
            kernel_list[iz] = np.trapz(integrand, x=zs_list)
        self.kernel_z_interp = interp1d(zl_list, kernel_list)


    def calc_kernel_Sigma(self, zh): # used for C_ell_Sigma # depend on redshift of cluster # ugly!!!
        chi_h = self.chi(z=zh).value
        #print('kernel for Sigma')

        zl_min = 0.1
        if self.su.top_hat == True:
            print('lensing kernel uses top-hat source distribution')

        #zs_max = self.su.zs_max
        zl_list = np.linspace(zl_min, self.su.zs_max-0.01, 100)
        kernel_list = np.zeros(len(zl_list))
        for iz, zl in enumerate(zl_list):
            zs_list = np.linspace(max(zl+0.01,self.su.zs_min), self.su.zs_max, 100)
            ns_list = self.su.pz_src(zs_list)
            chi_l = self.chi(z=zl).value
            chi_s = self.chi(z=zs_list).value
            #print('zl', zl, zs_list)
            Sigma_crit = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_l /(chi_s - chi_l)/(1.+zl) #Msun/Mpc^2
            Sigma_crit_halo = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_h /(chi_s - chi_h)/(1.+zh) #Msun/Mpc^2
            integrand = ns_list / Sigma_crit * Sigma_crit_halo
            kernel_list[iz] = np.trapz(integrand, x=zs_list)
        self.kernel_Sigma_z_interp = interp1d(zl_list, kernel_list)


    def mean_Sigma_crit(self, zh):
        chi_h = self.chi(z=zh).value
        zs_list = np.linspace(max(zh+0.01,self.su.zs_min), self.su.zs_max, 100)
        ns_list = self.su.pz_src(zs_list)
        chi_s = self.chi(z=zs_list).value
        Sigma_crit_halo = (cn.c)**2/(4.*np.pi*cn.G)*chi_s / chi_h /(chi_s - chi_h)/(1.+zh) #Msun/Mpc^2
        integrand = ns_list * Sigma_crit_halo
        return np.trapz(integrand, x=zs_list)


    def distance_sanity(self):
        plt.figure(figsize=(7, 7))
        # "z vs. chi"
        z_list = np.linspace(0.1,3)
        plt.minorticks_on()
        chi_list = self.chi(z_list)
        plt.plot(z_list, chi_list)
        #plt.plot(z_list, z_list*3000./self.co.h, ls=':', c='gray')
        plt.xlabel('z')
        plt.ylabel('comoving distance [Mpc, no h]')
        #plt.yscale('log')
        plt.grid(True)
        plt.savefig('../../plots/lensing/chi_vs_z.pdf')


        plt.figure(figsize=(14, 14))
        "theta vs. r"
        plt.subplot(221)
        lnrp = np.linspace(np.log(0.1), np.log(100))
        rp = np.exp(lnrp)
        for zl in np.arange(0.2, 1.2, 0.2):
            chi_l = self.chi(zl)
            theta_l = rp/chi_l
            plt.plot(rp, theta_l, label=r'$\rm z_l=%g$'%(zl))
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\rm r_p$'+'  [Mpc, no h]')
        plt.ylabel(r'$\rm \theta\ [radian]$')
        plt.minorticks_on()

        "theta[arcmin] vs. r"
        plt.subplot(223)
        lnrp = np.linspace(np.log(0.1), np.log(100))
        rp = np.exp(lnrp)
        for zl in np.arange(0.2, 1.2, 0.2):
            chi_l = self.chi(zl)
            theta_l = rp/chi_l
            plt.plot(rp, theta_l*cn.radian_to_arcmin, label=r'$\rm z_l=%g$'%(zl))
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\rm r_p$'+'  [Mpc, no h]')
        plt.ylabel(r'$\rm \theta\ [arcmin]$')
        plt.minorticks_on()

        "ell vs. k"
        plt.subplot(222)
        lnk = np.linspace(np.log(1e-4), np.log(30))
        k = np.exp(lnk)
        for zl in np.arange(0.2, 1.2, 0.2):
            chi_l = self.chi(zl)
            ell = k*chi_l
            plt.plot(k, ell, label=r'$\rm z_l=%g$'%(zl))
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\rm k$'+'  [1/Mpc, no h]')
        plt.ylabel(r'$\rm \ell$')
        plt.minorticks_on()

        
        "P_mm vs. k"
        plt.subplot(224)
        for zl in np.arange(0.2, 1.2, 0.2):
            lin = LinearTheory(cosmo=self.cosmo_ying, z=zl)
            pk_lin = lin.power_spectrum
            plt.plot(k, pk_lin(k), label=r'$\rm z_l=%g$'%(zl))
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\rm k$'+'  [1/Mpc, no h]')
        plt.ylabel(r'$\rm P(k)$')
        plt.minorticks_on()
        
        plt.savefig('../../plots/lensing/distance_sanity.pdf')


    def kernel_sanity(self):
        zs_max = 2
        plt.figure(figsize=(14, 14))
        plt.subplot(221)
        plt.title('source distribution vs. z')
        zs_list = np.linspace(0.1,zs_max)
        ns_list = self.su.pz_src(zs_list)
        plt.plot(zs_list, ns_list)
        plt.xlabel(r'$\rm z_{s}$')
        plt.ylabel(r'$\rm n_s = dn/dz_s \ (normalized)$')
        #print('test normalization', np.trapz(ns_list, x=zs_list))
        plt.minorticks_on()
        plt.xlim(0,None)

        plt.subplot(222)
        plt.title('source distribution vs. '+r'$\rm\chi$')
        dchi_dz = 3000./self.co.h/np.sqrt(self.co.OmegaM*(1+zs_list)**3 + self.co.OmegaDE)
        chi_s = self.chi(z=zs_list).value
        dn_dchi = ns_list / dchi_dz
        plt.plot(chi_s, dn_dchi)
        plt.xlabel(r'$\rm \chi_s\ [Mpc]$')
        plt.ylabel(r'$\rm dn/d\chi_{s} \ (normalized)$')
        #print('test normalization', np.trapz(dn_dchi, x=chi_s))
        plt.minorticks_on()
        plt.xlim(0,None)

        plt.subplot(223)
        plt.title('lensing kernel vs. z')
        zl_list = np.linspace(0.1, zs_max-0.01, 100)
        kernel_list = self.kernel_z_interp(zl_list)
        plt.plot(zl_list, kernel_list)
        plt.xlabel(r'$\rm z_{lens}$')
        plt.ylabel(r'$\rm Kernel(z_l) = \displaystyle\int dz_s n(z_s)/\Sigma_{crit}(z_s, z_l)$')
        plt.minorticks_on()
        plt.xlim(0,None)

        plt.subplot(224)
        plt.title('lensing kernel vs. '+r'$\rm\chi$')
        chi_l_list = self.chi(z=zl_list).value
        plt.plot(chi_l_list, kernel_list)
        plt.minorticks_on()
        plt.xlabel(r'$\rm \chi_{lens}\ [Mpc]$')
        plt.ylabel(r'$\rm Kernel(\chi_l)$')
        plt.minorticks_on()
        plt.xlim(0,None)
        
        plt.savefig('../../plots/lensing/kernel_sanity.pdf')




if __name__ == "__main__":
    co = CosmoParameters()
    su = Survey()
    lk = LensingKernel(co=co, su=su)
    lk.distance_sanity()
    lk.kernel_sanity()

    plt.show()
