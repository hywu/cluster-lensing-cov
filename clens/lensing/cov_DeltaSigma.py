#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

import clens.util.constants as cn
from clens.util.parameters import CosmoParameters#, NuisanceParameters
from clens.util.scaling_relation import FiducialScalingRelation, Costanzi21ScalingRelation
from clens.util.survey import Survey

from clens.lensing.angular_power_spectra import AngularPowerSpectra
from clens.lensing.bessel_for_cov_theta import BesselForCovTheta
from clens.lensing.lensing_profiles import LensingProfiles


class CovDeltaSigma(object):
    """
    cluster lensing (DeltaSigma) covariance matrix based on angular power spectrum
    works for a wide range of source redshift (specified in Survey class)
    works for a *thin* slice of halo redshift
    NOTE: all units are h-free
    NOTE: there's no intrinsic variance yet
    """
    def __init__(self, co, su, sr, fsky, slicing=False, dz_slicing=0, halo_shot_noise_only=False, cosmic_shear_no_shot=False):
        """
        Args:
            co: CosmoParameters object
            su: Survey object
            sr: ScalingRelation object
        """
        self.co = co
        self.su = su
        self.sr = sr
        self.fsky = fsky
        self.slicing = slicing
        self.dz_slicing = dz_slicing
        self.halo_shot_noise_only = halo_shot_noise_only
        if self.halo_shot_noise_only == True:
            self.cosmic_shear_no_shot == False ##only one can be True
        else:
            self.cosmic_shear_no_shot = cosmic_shear_no_shot

        self.aps = AngularPowerSpectra(co=self.co, su=self.su, sr=self.sr)
        self.bf = BesselForCovTheta()
        astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        self.chi = astropy_dist.comoving_distance

    def calc_mean(self, lambda_min, lambda_max, zh_min, zh_max, rp_min=0.1, rp_max=100):
        lp = LensingProfiles(co=self.co, su=self.su, zh_min=zh_min, zh_max=zh_max, lambda_min=lambda_min, lambda_max=lambda_max)
        rp, DeltaSigma = lp.calc_DeltaSigma()
        select = (rp >= rp_min)&(rp <= rp_max)
        return rp[select], DeltaSigma[select]

    def _calc_C_ell_integration(self, thmin1, thmax1, thmin2, thmax2, zh_min, zh_max, lambda_min, lambda_max):
        ## making the multiple (ell) range wide enough
        thmax = max(thmax1, thmax2)
        thmin = min(thmin1, thmin2)
        lnell_list = np.arange(np.log(self.bf.scaling_for_ell_min/thmax), np.log(self.bf.scaling_for_ell_max/thmin), self.bf.dlnell)
        ell_list = np.exp(lnell_list)

        ## calculating cosmic shear contribution, from 0.1 to zs_max (zs_max can be up to 2), and interpolate
        zh_mid = 0.5*(zh_min+zh_max)
        if self.slicing == False: ## default: LSS from 0 to zs_max
            self.aps.calc_C_ell_Sigma(zl_min=0.1, zl_max=min(2,self.su.zs_max-0.1), zh=zh_mid)
        if self.slicing == True: ## slicing: from zh-dz to zh+dz
            self.aps.calc_C_ell_Sigma(zl_min=zh_mid-self.dz_slicing, zl_max=zh_mid+self.dz_slicing, zh=zh_mid)
            print('zh_mid', zh_mid)
            print('slicing z', zh_mid-self.dz_slicing, zh_mid+self.dz_slicing)

        C_ell_Sigma_interp = interp1d(self.aps.ell_Sigma, self.aps.C_ell_Sigma)
        
        ## calculating the contribution from halo counts: C_ell_h and shot noise
        self.aps.calc_C_ell_h(zh_min=zh_min, zh_max=zh_max, lambda_min=lambda_min, lambda_max=lambda_max)
        C_ell_h_interp = interp1d(self.aps.ell_h, self.aps.C_ell_h)
        halo = C_ell_h_interp(ell_list) + self.aps.shot_noise

        if self.halo_shot_noise_only == True:  ## calculating shot noise only, C_ell_h = 0
            halo = self.aps.shot_noise
            print('shot noise only')
        if self.cosmic_shear_no_shot == True:  ## calculating cosmic shear only, no shot noise
            halo = C_ell_h_interp(ell_list)
            print('cosmic shear only')

        '''
        ## TODO!
        ## calculating the contribution from halo profile: C_ell_hk
        self.aps.calc_C_ell_h_Sigma(zh_min=zh_min, zh_max=zh_max, Mmin=Mmin, Mmax=Mmax)
        C_ell_h_Sigma_interp = interp1d(self.aps.ell_h_Sigma, self.aps.C_ell_h_Sigma)
        '''
        ## the J2 part of the integration
        geometry = self.bf.j2_bin(ell_list, thmin1, thmax1) * self.bf.j2_bin(ell_list, thmin2, thmax2) * ell_list**2 / (2.*np.pi)

        ## contribution from cosmic shear, (C_ell^h + halo shot) * C_ell^kappa
        integrand_cosmic_shear = halo * C_ell_Sigma_interp(ell_list) * geometry

        ## contribution from shape noise, (C_ell^h + halo shot) * shape noise
        integrand_shape_noise = halo * self.aps.shape_noise_for_Sigma * geometry

        ## contribution from halo intrinsic profile, C_ell^hk ** 2  # TODO
        #integrand_halo_intrinsic = C_ell_h_Sigma_interp(ell_list)**2 * geometry

        ## integration over ell!
        self.cosmic_shear_integration = np.trapz(integrand_cosmic_shear, x=lnell_list)
        self.shape_noise_integration = np.trapz(integrand_shape_noise, x=lnell_list)
        #self.halo_intrinsic_integration = np.trapz(integrand_halo_intrinsic, x=lnell_list) # TODO

    def calc_cov(self, rp_min, rp_max, n_rp, zh_min, zh_max, lambda_min, lambda_max, diag_only=False):
        """ calling self._calc_C_ell_integration, calculating the cov for *multiple bins*
        Args:
            rp_min (float): min projected separation in comoving Mpc (no h)
            rp_max (float): max 
            nth (int): number of log-spaced bins between rp_min and rp_max
            zh_min (float): min redshift of halos
            zh_max (float): max redshift of halos
            lambda_min (float): min lambda of halos
            lambda_min (float): max lambda_ of halos
            diag_only (bool): if True, calculating only the diagnal elements, off-diag will be zero

        Returns:
            cov_cosmic_shear, cov_shape_noise, cov_halo_intrinsic
            units all in Msun^2/pc^4
            one can also directly access these as attributes
        """
        zh_mid = 0.5*(zh_min+zh_max)
        chi_h = self.chi(z=zh_mid).value

        thmin = rp_min / chi_h
        thmax = rp_max / chi_h
        nth = n_rp

        ## setting up the angular bins
        lnth_list = np.linspace(np.log(thmin), np.log(thmax), nth+1)
        th_list = np.exp(lnth_list)
        thmin_list = th_list[:-1]
        thmax_list = th_list[1:]
        thmid_list = np.sqrt(thmin_list*thmax_list)

        factor = 1./(4.*np.pi*self.fsky)
        print('fsky', self.fsky)

        self.cov_cosmic_shear = np.zeros([nth, nth])
        self.cov_shape_noise = np.zeros([nth, nth])
        #self.cov_halo_intrinsic = np.zeros([nth, nth])

        if diag_only==True: ## only calculating the diagonal elements
            for ith1 in range(nth):
                ith2 = ith1
                self._calc_C_ell_integration(thmin1=thmin_list[ith1], thmax1=thmax_list[ith1], thmin2=thmin_list[ith2], thmax2=thmax_list[ith2], zh_min=zh_min, zh_max=zh_max, lambda_min=lambda_min, lambda_max=lambda_max)
                self.cov_shape_noise[ith1,ith1] = factor*self.shape_noise_integration
                self.cov_cosmic_shear[ith1,ith1] = factor*self.cosmic_shear_integration
                #self.cov_halo_intrinsic[ith1,ith1] = factor*self.halo_intrinsic_integration

        if diag_only==False: ## calculating the off-diagonal elements
            for ith1 in range(nth):
                for ith2 in range(ith1, nth): ## only the upper right corner
                    self._calc_C_ell_integration(thmin1=thmin_list[ith1], thmax1=thmax_list[ith1], thmin2=thmin_list[ith2], thmax2=thmax_list[ith2], zh_min=zh_min, zh_max=zh_max, lambda_min=lambda_min, lambda_max=lambda_max)
                    self.cov_shape_noise[ith1, ith2] = factor*self.shape_noise_integration
                    self.cov_cosmic_shear[ith1, ith2] = factor*self.cosmic_shear_integration
                    #self.cov_halo_intrinsic[ith1, ith2] = factor*self.halo_intrinsic_integration

            ## copy the upper right corner to the lower left corner
            for ith1 in range(nth):
                for ith2 in range(ith1):
                    self.cov_cosmic_shear[ith1, ith2] = self.cov_cosmic_shear[ith2, ith1]
                    self.cov_shape_noise[ith1, ith2] = self.cov_shape_noise[ith2, ith1]
                    #self.cov_halo_intrinsic[ith1, ith2] = self.cov_halo_intrinsic[ith2, ith1]

        self.cov_cosmic_shear *= (1e-24) # make it (Msun/pc^2)^2
        self.cov_shape_noise *= (1e-24)
        #self.cov_halo_intrinsic *= (1e-24)
        self.cov_sum = self.cov_cosmic_shear + self.cov_shape_noise #+ self.cov_halo_intrinsic

        self.rp_min_list = thmin_list * chi_h
        self.rp_max_list = thmax_list * chi_h
        self.rp_mid_list = thmid_list * chi_h
        return self.rp_mid_list, self.cov_cosmic_shear, self.cov_shape_noise #, self.cov_halo_intrinsic



# def demo_var_slicing():
#     co = CosmoParameters()#OmegaM=0.286, sigma8=0.82, h=0.7, OmegaDE=0.714)
#     nu = NuisanceParameters()#sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
#     su = Survey(zs_min=0.56, zs_max=0.65, top_hat=True, n_src_arcmin=10, sigma_gamma=0.3)
#     rp_min = 1
#     rp_max = 10
#     n_rp = 2 #5
#     cds = CovDeltaSigma(co=co, nu=nu, su=su, slicing=True, dz_slicing=0.033)
#     cds.calc_cov(rp_min=rp_min, rp_max=rp_max, n_rp=n_rp, zh_min=0.2, zh_max=0.35, lambda_min=20, lambda_max=30, diag_only=True)
#     print(cds.cov_cosmic_shear)

#     #cds = CovDeltaSigma(co=co, nu=nu, su=su, slicing=False)
#     #cds.calc_cov(rp_min=rp_min, rp_max=rp_max, n_rp=n_rp, zh_min=0.1, zh_max=0.3, Mmin=1e14, Mmax=1e16, diag_only=False)



def demo_cov(plotting=False):
    co = CosmoParameters()#OmegaM=0.286, sigma8=0.82, h=0.7, OmegaDE=0.714)
    #nu = NuisanceParameters()#sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    sr = Costanzi21ScalingRelation()
    #sr = FiducialScalingRelation(nu)
    su = Survey(zs_min=0.56, zs_max=0.65, top_hat=True, n_src_arcmin=10, sigma_gamma=0.3)
    fsky = 5000. / 41253.
    cds = CovDeltaSigma(co=co, su=su, sr=sr, fsky=fsky)
    rp_min = 1
    rp_max = 10
    n_rp = 2 #5
    cds.calc_cov(rp_min=rp_min, rp_max=rp_max, n_rp=n_rp, zh_min=0.2, zh_max=0.35, lambda_min=20, lambda_max=30, diag_only=True)
    print(cds.cov_cosmic_shear)
    print(cds.cov_shape_noise)
    '''
    ### compare with the old code
    from clens.lensing.cov_DeltaSigma_old import CovDeltaSigmaOld
    cdso = CovDeltaSigmaOld(co=co, nu=nu, su=su)
    ## cov
    output = cdso.calc_cov_thin_slice(rp_min=rp_min, rp_max=rp_max, n_rp=n_rp, zh_min=0.1, zh_max=0.3, Mmin=1e14, Mmax=1e16)
    #rp_mid, cov_cosmic_shear, cov_shape_noise, x = output
    print(cdso.cov_cosmic_shear)
    print(cdso.cov_shape_noise)
    
    if plotting==True:
        plt.plot(cds.rp_mid_list, cds.cov_cosmic_shear.diagonal(), label='cosmic shear')
        plt.plot(cds.rp_mid_list, cds.cov_shape_noise.diagonal(), label='shape noise')
        #plt.plot(cds.rp_mid_list, cds.cov_halo_intrinsic.diagonal(), label='halo intrinsic')
        #plt.plot(cds.thmid_list*cn.radian_to_arcmin, cds.cov_sum.diagonal(), label='sum')
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\rm r_p\ [Mpc]$')
        plt.ylabel(r'$\rm Var[\Delta\Sigma]$')
        plt.show()
    '''
if __name__ == "__main__":
    demo_cov()#plotting=True)
    #demo_var_slicing()