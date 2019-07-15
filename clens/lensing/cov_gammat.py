#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

import clens.util.constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.survey import Survey

from clens.lensing.angular_power_spectra import AngularPowerSpectra
from clens.lensing.bessel_for_cov_theta import BesselForCovTheta


class Covgammat(object):
    """
    Cluster lensing covariance matrix based on angular power spectrum
    Equation 18 in Jeong, Komatsu and Jain (2009)
    default output is for all sky
    NOTE: all units are h-free
    """
    def __init__(self, co, nu, su, no_halo=False, halo_shot_noise_only=False, calc_kappa=False, slicing=False, dz_slicing=0):
        """
        Args:
            co: cosmology object
            nu: nuisance parameter object
            su: survey object
            no_halo (bool): cosmic shear only
            halo_shot_noise_only (bool): no clustering
            calc_kappa (bool): if True, calculate the covariance for kappa
        """
        self.co = co
        self.nu = nu
        self.su = su
        self.no_halo = no_halo
        self.halo_shot_noise_only = halo_shot_noise_only
        self.calc_kappa = calc_kappa
        self.slicing = slicing
        self.dz_slicing = dz_slicing

        self.aps = AngularPowerSpectra(co=self.co, nu=self.nu, su=self.su)
        self.bf = BesselForCovTheta()
        self.fsky = 1.

    def _calc_C_ell_integration(self, thmin1, thmax1, thmin2, thmax2, zh_min, zh_max, Mmin, Mmax):
        """
        calling "angular_power_spectra.py" to calculate various C_ell's for *one* angular bin 
        Args:
            thmin1 (float): theta_min for the first bin in radian
            thmax1 (float): theta_max, ditto
            thmin2 (float): theta_min for the second bin in radian
            thmax2 (float): theta_max, ditto
            zh_min (float): min redshift of halos
            zh_max (float): max redshift of halos
            Mmin (float): min mass of halos, in Msun (no h)
            Mmin (float): max mass of halos, in Msun (no h)
        """
        
        ## making the multiple (ell) range wide enough
        thmax = max(thmax1, thmax2)
        thmin = min(thmin1, thmin2)
        lnell_list = np.arange(np.log(self.bf.scaling_for_ell_min/thmax), np.log(self.bf.scaling_for_ell_max/thmin), self.bf.dlnell)
        ell_list = np.exp(lnell_list)

        ## calculating cosmic shear contribution, from 0.1 to zs_max (zs_max can be up to 2), and interpolate
        if self.slicing == False: ## default: LSS from 0 to zs_max
            self.aps.calc_C_ell_kappa(zl_min=0.1, zl_max=min(2,self.su.zs_max-0.1))
        if self.slicing == True: ## slicing: from zh-dz to zh+dz
            zh_mid = 0.5*(zh_min+zh_max)
            self.aps.calc_C_ell_kappa(zl_min=zh_mid-self.dz_slicing, zl_max=zh_mid+self.dz_slicing)

        C_ell_kappa_interp = interp1d(self.aps.ell_kappa, self.aps.C_ell_kappa)

        if self.no_halo == True: ## only calculating cosmic shear + shape noise
            halo = 1
            print('no halo at all, cosmic shear')
        if self.no_halo == False: ## calculating the contribution from halo counts: C_ell_h and shot noise
            self.aps.calc_C_ell_h(zh_min=zh_min, zh_max=zh_max, Mmin=Mmin, Mmax=Mmax)
            C_ell_h_interp = interp1d(self.aps.ell_h, self.aps.C_ell_h)
            # print('shot noise', self.aps.shot_noise)
            # print('max(ell_list)=%e'%max(ell_list))
            # print('max(self.aps.ell_h)=%e'%max(self.aps.ell_h))
            halo = C_ell_h_interp(ell_list) + self.aps.shot_noise

            ## calculating the contribution from halo profile: C_ell_hk
            self.aps.calc_C_ell_h_kappa(zh_min=zh_min, zh_max=zh_max, Mmin=Mmin, Mmax=Mmax)
            C_ell_h_kappa_interp = interp1d(self.aps.ell_h_kappa, self.aps.C_ell_h_kappa)

        if self.halo_shot_noise_only == True:  ## calculating shot noise only, C_ell_h = 0
            halo = self.aps.shot_noise
            print('shot noise only')

        if self.calc_kappa == False:  ## calculating gammat cov, using J2
            geometry = self.bf.j2_bin(ell_list, thmin1, thmax1) * self.bf.j2_bin(ell_list, thmin2, thmax2) * ell_list**2 / (2.*np.pi)
        if self.calc_kappa == True:  ## calculating kappa cov, using J0
            geometry = self.bf.j0_bin(ell_list, thmin1, thmax1) * self.bf.j0_bin(ell_list, thmin2, thmax2) * ell_list**2 / (2.*np.pi)

        ## contribution from cosmic shear, (C_ell^h + halo shot) * C_ell^kappa
        integrand_cosmic_shear = halo * C_ell_kappa_interp(ell_list) * geometry

        ## contribution from shape noise, (C_ell^h + halo shot) * shape noise
        integrand_shape_noise = halo * self.aps.shape_noise * geometry

        # print('compare integrand')
        # print('cosmic shear', halo * C_ell_kappa_interp(ell_list))
        # print('intrinsic', C_ell_h_kappa_interp(ell_list)**2)
        # plt.loglog(ell_list, halo * C_ell_kappa_interp(ell_list))
        # plt.loglog(ell_list, C_ell_h_kappa_interp(ell_list)**2)
        # plt.show()
        # exit()

        ## contribution from halo intrinsic profile, C_ell^hk ** 2 
        if self.no_halo == False:
            integrand_halo_intrinsic = C_ell_h_kappa_interp(ell_list)**2 * geometry

        ## integration over ell!
        self.cosmic_shear_integration = np.trapz(integrand_cosmic_shear, x=lnell_list)
        self.shape_noise_integration = np.trapz(integrand_shape_noise, x=lnell_list)
        if self.no_halo == False:
            self.halo_intrinsic_integration = np.trapz(integrand_halo_intrinsic, x=lnell_list)

        print('compare integrtion')
        print('cosmic shear', self.cosmic_shear_integration)
        print('intrinsic', self.halo_intrinsic_integration)
        


    def calc_cov_gammat_integration(self, thmin, thmax, nth, zh_min, zh_max, Mmin, Mmax, diag_only=False):
        """ calling self._calc_C_ell_integration, calculating the cov for *multiple bins*
        Args:
            thmin (float): theta_min in radian
            thmax (float): theta_max in radian
            nth (int): number of log-spaced bins between thmin and thmax
            zh_min (float): min redshift of halos
            zh_max (float): max redshift of halos
            Mmin (float): min mass of halos, in Msun (no h)
            Mmin (float): max mass of halos, in Msun (no h)
            diag_only (bool): if True, calculating only the diagnal elements


        Returns:
            if diag_only==True: returns variance array
            var_cosmic_shear, var_shape_noise, var_halo_intrinsic

            if diag_only==False: returns covariance matrixes
            cov_cosmic_shear, cov_shape_noise, cov_halo_intrinsic

            one can also directly access theose as attributes.
        """

        ## setting up the angular bins
        lnth_list = np.linspace(np.log(thmin), np.log(thmax), nth+1)
        th_list = np.exp(lnth_list)
        self.thmin_list = th_list[:-1]
        self.thmax_list = th_list[1:]
        self.thmid_list = np.sqrt(self.thmin_list*self.thmax_list)

        factor = 1./(4.*np.pi*self.fsky)
        #print('fsky', self.fsky)

        self.cov_cosmic_shear = np.zeros([nth, nth])
        self.cov_shape_noise = np.zeros([nth, nth])
        self.cov_halo_intrinsic = np.zeros([nth, nth])

        if diag_only==True: ## only calculating the diagonal elements
            for ith1 in range(nth):
                ith2 = ith1
                self._calc_C_ell_integration(thmin1=self.thmin_list[ith1], thmax1=self.thmax_list[ith1], thmin2=self.thmin_list[ith2], thmax2=self.thmax_list[ith2], zh_min=zh_min, zh_max=zh_max, Mmin=Mmin, Mmax=Mmax)
                self.cov_shape_noise[ith1,ith1] = factor*self.shape_noise_integration
                self.cov_cosmic_shear[ith1,ith1] = factor*self.cosmic_shear_integration
                self.cov_halo_intrinsic[ith1,ith1] = factor*self.halo_intrinsic_integration
            self.var_shape_noise = self.cov_shape_noise.diagonal()
            self.var_cosmic_shear = self.cov_cosmic_shear.diagonal()
            self.var_halo_intrinsic = self.cov_halo_intrinsic.diagonal()

        if diag_only==False: ## calculating the off-diagonal elements

            for ith1 in range(nth):
                for ith2 in range(ith1, nth): ## only the upper right corner
                    self._calc_C_ell_integration(thmin1=self.thmin_list[ith1], thmax1=self.thmax_list[ith1], thmin2=self.thmin_list[ith2], thmax2=self.thmax_list[ith2], zh_min=zh_min, zh_max=zh_max, Mmin=Mmin, Mmax=Mmax)
                    self.cov_shape_noise[ith1, ith2] = factor*self.shape_noise_integration
                    self.cov_cosmic_shear[ith1, ith2] = factor*self.cosmic_shear_integration
                    self.cov_halo_intrinsic[ith1, ith2] = factor*self.halo_intrinsic_integration

            ## copy the upper right corner to the lower left corner
            for ith1 in range(nth):
                for ith2 in range(ith1):
                    self.cov_cosmic_shear[ith1, ith2] = self.cov_cosmic_shear[ith2, ith1]
                    self.cov_shape_noise[ith1, ith2] = self.cov_shape_noise[ith2, ith1]
                    self.cov_halo_intrinsic[ith1, ith2] = self.cov_halo_intrinsic[ith2, ith1]


        #self.th_list = th_list
        self.cov_sum = self.cov_cosmic_shear + self.cov_shape_noise + self.cov_halo_intrinsic
        self.var_sum = self.cov_sum.diagonal()
        return self.cov_cosmic_shear, self.cov_shape_noise, self.cov_halo_intrinsic


def demo_var():
    co = CosmoParameters(OmegaM=0.286, sigma8=0.82, h=0.7, OmegaDE=0.714)
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey()
    cj = Covgammat(co=co, nu=nu, su=su)
    thmin = 5e-4
    thmax = 5e-2
    nth = 2#10#4
    cj.calc_cov_gammat_integration(thmin=thmin, thmax=thmax, nth=nth, zh_min=0.1, zh_max=0.3, Mmin=1e14, Mmax=1e16, diag_only=True)
    plt.figure()
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.var_cosmic_shear, label='cosmic shear')
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.var_shape_noise, label='shape noise')
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.var_halo_intrinsic, label='halo intrinsic')
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.var_sum, label='sum')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\rm\theta\ [arcmin]$')
    plt.ylabel(r'$\rm Var[\gamma_t]$')


def demo_var_slicing():
    co = CosmoParameters(OmegaM=0.286, sigma8=0.82, h=0.7, OmegaDE=0.714)
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey()
    cj = Covgammat(co=co, nu=nu, su=su, slicing=True, dz_slicing=0.03)
    thmin = 5e-4
    thmax = 5e-2
    nth = 2#10#4
    cj.calc_cov_gammat_integration(thmin=thmin, thmax=thmax, nth=nth, zh_min=0.1, zh_max=0.3, Mmin=1e14, Mmax=1e16, diag_only=True)
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.var_cosmic_shear, label='cosmic shear, slicing',ls='--')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\rm\theta\ [arcmin]$')
    plt.ylabel(r'$\rm Var[\gamma_t]$')



def demo_cov(save_files=False):
    co = CosmoParameters(OmegaM=0.286, sigma8=0.82, h=0.7, OmegaDE=0.714)
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey()
    cj = Covgammat(co=co, nu=nu, su=su)
    thmin = 5e-4
    thmax = 5e-2
    nth = 2#10#4
    zh_min = 0.1
    zh_max = 0.3
    Mmin = 1e14
    Mmax = 1e16
    cj.calc_cov_gammat_integration(thmin=thmin, thmax=thmax, nth=nth, zh_min=zh_min, zh_max=zh_max, Mmin=Mmin, Mmax=Mmax)
    plt.figure(figsize=(14,7))
    plt.subplot(121)
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.cov_cosmic_shear.diagonal(), label='cosmic shear')
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.cov_shape_noise.diagonal(), label='shape noise')
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.cov_halo_intrinsic.diagonal(), label='halo intrinsic')
    plt.plot(cj.thmid_list*cn.radian_to_arcmin, cj.cov_sum.diagonal(), label='sum')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\rm\theta\ [arcmin]$')
    plt.ylabel(r'$\rm Var[\gamma_t]$')

    plt.subplot(122)
    plt.imshow(np.log10(cj.cov_sum))
    plt.colorbar()

    if save_files==True:
        plot_loc = '../../plots/lensing/'
        if os.path.isdir(plot_loc)==False: os.makedirs(plot_loc)
        output_loc = '../../output/lensing/'
        if os.path.isdir(output_loc)==False: os.makedirs(output_loc)


        plt.savefig(plot_loc+'cov_cosmic_shear_nbins_%i.pdf'%(nth))
        ## save var_cosmic_shear
        outfile = open(output_loc+'var_cosmic_shear_fullsky_z_%g_%g_M_%.0e_%.0e.dat'%(zh_min,zh_max,Mmin,Mmax),'w')
        outfile.write('# theta[radian], var gammat \n')
        for it in range(len(cj.thmid_list)):
            outfile.write('%g %g\n'%(cj.thmid_list[it], cj.cov_cosmic_shear.diagonal()[it]))
        outfile.close()

        ## save var_shape_noise
        outfile = open(output_loc+'var_shape_noise_fullsky_z_%g_%g_M_%.0e_%.0e.dat'%(zh_min,zh_max,Mmin,Mmax),'w')
        outfile.write('# theta[radian], var gammat \n')
        for it in range(len(cj.thmid_list)):
            outfile.write('%g %g\n'%(cj.thmid_list[it], cj.cov_shape_noise.diagonal()[it]))
        outfile.close()

        ## save the entire cov
        outfile = open(output_loc+'cov_sum_fullsky_z_%g_%g_M_%.0e_%.0e.dat'%(zh_min,zh_max,Mmin,Mmax),'w')
        np.savetxt(outfile, cj.cov_sum)
        outfile.close()

if __name__ == "__main__":
    #demo_var()
    #demo_cov()
    demo_var_slicing()
    plt.show()