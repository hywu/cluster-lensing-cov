#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

from clens.ying.param import CosmoParams
from clens.ying.lineartheory import LinearTheory
#from zypy.zycosmo.halofit import HALOFIT # doesn't work yet

from clens.util import constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.survey import Survey
from clens.util.cluster_counts import ClusterCounts
from clens.lensing.lensing_kernel import LensingKernel
from clens.lensing.pk_hm_halomodel import PowerSpectrumHaloMatter


class AngularPowerSpectra(object):
    """
    calaculating various C_ell's that will be used in covariance matrices
    """
    def __init__(self, co, nu, su, use_halofit=False):
        self.co = co
        self.nu = nu
        self.su = su
        self.use_halofit = use_halofit

        rho_crit = cn.rho_crit_with_h * co.h**2
        self.rho_mean = rho_crit * self.co.OmegaM # comoving!

        # astropy
        astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        self.chi = astropy_dist.comoving_distance

        self.cosmo_ying = CosmoParams(omega_M_0=self.co.OmegaM, omega_b_0=self.co.OmegaB, omega_lambda_0=self.co.OmegaDE, h=self.co.h, sigma_8=self.co.sigma8, n=self.co.ns, tau=self.co.tau) # Ying's
        #print('sigma8', self.cosmo_ying.sigma_8)
        self.lk = LensingKernel(co=self.co, su=self.su)
        #print('aps n_src_arcmin', self.su.n_src_arcmin)
        #print('aps sigma_gamma', self.su.sigma_gamma)

        self.ln_ell = np.linspace(np.log(1e-1), np.log(1e7), 8000) 
        ## 1000 bins per decade. 
        ## ell_max=1e7 sufficient for rp=0.03 at zh=1.4
        self.ell = np.exp(self.ln_ell)
        #print('sigma_lambda',nu.sigma_lambda) 

    ###### matter-matter ######
    def calc_C_ell_kappa(self, zl_min, zl_max):
        print('calculating C_ell_kappa')
        # edge of the z bins, 20 bins per dz=1
        nzl = max(int(20*(zl_max-zl_min)), 1)# at least one redshift slice
        #print('z range', zl_min, zl_max, 'nz', nzl)
        zl_bins = np.linspace(zl_min, zl_max, nzl+1)
        zl_min_list = zl_bins[:-1]
        zl_max_list = zl_bins[1:]

        C_ell_sum = np.zeros(len(self.ell)) # using direct summation of chi instead of integration
        for izl in range(nzl):
            zl_min = zl_min_list[izl]
            zl_max = zl_max_list[izl]
            zl_mid = 0.5*(zl_min+zl_max)
            chi_l = self.chi(z=zl_mid).value
            kernel = self.lk.kernel_z_interp(zl_mid)
            dchi_l = self.chi(z=zl_max).value-self.chi(z=zl_min).value
            # power spectrum
            lin = LinearTheory(cosmo=self.cosmo_ying, z=zl_mid)
            #if self.use_halofit==False:
            pk_lin = lin.power_spectrum
            #else:
            # pshm = PowerSpectrumHaloModel()
            # pshm.calc_uk()
            # pshm.calc_Pk()
            # self.pk_lin = pshm.Pk_k_interp
            '''
            nonlin = HALOFIT(lin.power_spectrum, omega_M_z=self.co.OmegaM, omega_lambda_z=1-self.co.OmegaM)
            lnk_list = np.linspace(np.log(1e-6), np.log(2e+4), 1000)
            k = np.exp(lnk_list)
            pknl   = nonlin.pk_NL(k)
            pk_lin = interp1d(k, pknl)
            '''

            #pk_lin = lin.power_spectrum
            k = self.ell/chi_l
            #print('k range', min(k), max(k))
            Pmm_ell = pk_lin(k)
            summand =  dchi_l * kernel**2 / chi_l**2 * Pmm_ell
            C_ell_sum += summand
        C_ell_sum *= self.rho_mean**2 #* (1e-12) # (1e-12): Mpc^-2 to pc^-2 
        self.ell_kappa = self.ell
        self.C_ell_kappa = C_ell_sum

        n_src_sr = self.su.n_src_arcmin/cn.arcmin_to_radian**2
        self.shape_noise = self.su.sigma_gamma**2/n_src_sr
        #print('n_src_sr', n_src_sr)


    def calc_C_ell_Sigma(self, zl_min, zl_max, zh):# basically kappa * Sigma_crit
        print('calculating C_ell_Sigma')
        self.lk.calc_kernel_Sigma(zh)
        # edge of the z bins, 20 bins per dz=1
        nzl = max(int((zl_max-zl_min)/0.1), 1)# at least one redshift slice
        #print('z range', zl_min, zl_max, 'nz', nzl)
        zl_bins = np.linspace(zl_min, zl_max, nzl+1)


        zl_min_list = zl_bins[:-1]
        zl_max_list = zl_bins[1:]

        C_ell_sum = np.zeros(len(self.ell)) # using direct summation of chi instead of integration
        for izl in range(nzl):
            zl_min = zl_min_list[izl]
            zl_max = zl_max_list[izl]
            zl_mid = 0.5*(zl_min+zl_max)
            chi_l = self.chi(z=zl_mid).value
            kernel = self.lk.kernel_Sigma_z_interp(zl_mid)
            dchi_l = self.chi(z=zl_max).value-self.chi(z=zl_min).value
            # power spectrum
            lin = LinearTheory(cosmo=self.cosmo_ying, z=zl_mid)
            if self.use_halofit==False:
                pk_lin = lin.power_spectrum
            else:
                # pshm = PowerSpectrumHaloModel()
                # pshm.calc_uk()
                # pshm.calc_Pk()
                # self.pk_lin = pshm.Pk_k_interp
                nonlin = HALOFIT(lin.power_spectrum, omega_M_z=self.co.OmegaM, omega_lambda_z=1-self.co.OmegaM)
                lnk_list = np.linspace(np.log(1e-6), np.log(2e+4), 1000)
                k = np.exp(lnk_list)
                pknl   = nonlin.pk_NL(k)
                pk_lin = interp1d(k, pknl)

            #pk_lin = lin.power_spectrum
            k = self.ell/chi_l
            #print('k range', min(k), max(k))
            Pmm_ell = pk_lin(k)
            summand =  dchi_l * kernel**2 / chi_l**2 * Pmm_ell
            C_ell_sum += summand
        C_ell_sum *= self.rho_mean**2 
        self.ell_Sigma = self.ell
        self.C_ell_Sigma = C_ell_sum

        n_src_sr = self.su.n_src_arcmin/cn.arcmin_to_radian**2
        mean_Sigma_crit = self.lk.mean_Sigma_crit(zh=zh)
        #print('mean_Sigma_crit', mean_Sigma_crit)
        self.shape_noise_for_Sigma = self.su.sigma_gamma**2/n_src_sr * mean_Sigma_crit**2 

    ###### halo-halo ######
    def calc_C_ell_h(self, zh_min, zh_max, Mmin, Mmax):
        print('calculating C_ell_h')
        # edge of the z bins, 20 bins per dz=1
        nzh = max(int((zh_max-zh_min)/0.1),1) # at least one redshift slice...
        print('nzh', nzh)
        zh_bins = np.linspace(zh_min, zh_max, nzh+1)
        zh_min_list = zh_bins[:-1]
        zh_max_list = zh_bins[1:]

        # get the halo number density and bias
        survey_area_sq_deg = 41253./48.# exact value doesn't matter
        cc = ClusterCounts(cosmo_parameters=self.co, nuisance_parameters=self.nu)
        cc.calc_counts(lambda_min=Mmin, zmin=zh_min, zmax=zh_max, survey_area_sq_deg=survey_area_sq_deg, lambda_max=Mmax)
        #n_h_Mpc3 = cc.cluster_number_density
        area_sr = 4.*np.pi*survey_area_sq_deg/41253.
        print('cc.counts', cc.counts)
        n_h_sr = cc.counts/area_sr
        b = cc.cluster_mean_bias
        print('bias', b)
        #exit()
        
        C_ell_sum = np.zeros(len(self.ell)) # using direct summation of chi instead of integration
        # get the volume
        vol_sum = 0
        for izh in range(nzh):
            zh_min = zh_min_list[izh]
            zh_max = zh_max_list[izh]
            zh_mid = 0.5*(zh_min+zh_max)
            chi_h = self.chi(z=zh_mid).value
            dchi_h = self.chi(z=zh_max).value-self.chi(z=zh_min).value
            vol_sum += dchi_h * chi_h**2
            # power spectrum
            lin = LinearTheory(cosmo=self.cosmo_ying, z=zh_mid)
            pk_lin = lin.power_spectrum
            k = self.ell/chi_h
            Pmm_ell = pk_lin(k)
            summand = dchi_h * chi_h**2 * b**2 * Pmm_ell
            C_ell_sum += summand
        C_ell_sum /= vol_sum**2

        self.ell_h = self.ell
        self.C_ell_h = C_ell_sum
        self.shot_noise = 1./n_h_sr

    ###### halo-matter ######
    def calc_C_ell_h_kappa(self, zh_min, zh_max, Mmin, Mmax):
        print('calculating C_ell_h_kappa')
        ## the zbin should be the narrower of (zs_min, zs_max) and (zh_min, zh_max)
        ## if not doing this, the slicing results would be inconsistent
        zint_min = max(self.su.zs_min, zh_min) # min for integration
        zint_max = min(self.su.zs_max, zh_max) # max for integration
        nzh = max(int((zint_max-zint_min)/0.1),1)
        zh_bins = np.linspace(zint_min, zint_max, nzh+1)
        #nzh = max(int((zh_max-zh_min)/0.1),1)
        #zh_bins = np.linspace(zh_min, zh_max, nzh+1)
        zh_min_list = zh_bins[:-1]
        zh_max_list = zh_bins[1:]

        # get the halo number density and bias
        survey_area_sq_deg = 1. # not important
        cc = ClusterCounts(cosmo_parameters=self.co, nuisance_parameters=self.nu)
        cc.calc_counts(lambda_min=Mmin, zmin=zh_min, zmax=zh_max, survey_area_sq_deg=survey_area_sq_deg, lambda_max=Mmax)
        # n_h_Mpc3 = cc.cluster_number_density
        # area_sr2 = 4.*np.pi*survey_area_sq_deg/41253.
        # n_h_sr2 = cc.counts/area_sr2
        b = cc.cluster_mean_bias
        #print('for Chk, bias', b)
        #print('n_h_sr2, b', n_h_sr2, b)

        # get the volume
        C_ell_sum = np.zeros(len(self.ell)) # using direct summation of chi instead of integration
        vol_sum = 0
        for izh in range(nzh):
            zh_mid = 0.5*(zh_min_list[izh]+zh_max_list[izh])
            chi_h = self.chi(z=zh_mid).value
            dchi_h = self.chi(z=zh_max_list[izh]).value-self.chi(z=zh_min_list[izh]).value
            vol_sum += dchi_h * chi_h**2

            k = self.ell/chi_h
            #print('zh', zh_mid)
            # power spectrum.  Use pk_hm_halomodel!
            pk_hm = PowerSpectrumHaloMatter(co=self.co, nu=self.nu, su=self.su, zh=zh_mid)
            pk_hm.calc_Pk_hm_full(Mmin=Mmin, Mmax=Mmax)
            
            #print('min(k), max(k)', min(k), max(k))
            Phm_ell = np.exp(pk_hm.lnPk_lnk_interp(np.log(k)))

            kernel = self.lk.kernel_z_interp(zh_mid)
            summand = dchi_h * kernel * Phm_ell
            C_ell_sum += summand
        C_ell_sum = C_ell_sum * self.rho_mean / vol_sum 

        self.ell_h_kappa = self.ell
        self.C_ell_h_kappa = C_ell_sum


    ###### halo-matter ######
    def calc_C_ell_h_Sigma(self, zh_min, zh_max, Mmin, Mmax):
        print('calculating C_ell_h_Sigma')
        self.lk.calc_kernel_Sigma(0.5*(zh_min+zh_max))
        ## the zbin should be the narrower of (zs_min, zs_max) and (zh_min, zh_max)
        ## if not doing this, the slicing results would be inconsistent
        zint_min = max(self.su.zs_min, zh_min) # min for integration
        zint_max = min(self.su.zs_max, zh_max) # max for integration
        nzh = max(int((zint_max-zint_min)/0.1),1)
        zh_bins = np.linspace(zint_min, zint_max, nzh+1)
        #zh_bins = np.linspace(zh_min, zh_max, nzh+1)
        zh_min_list = zh_bins[:-1]
        zh_max_list = zh_bins[1:]
        
        C_ell_sum = np.zeros(len(self.ell)) # using direct summation of chi instead of integration
        vol_sum = 0
        for izh in range(nzh):
            zh_mid = 0.5*(zh_min_list[izh]+zh_max_list[izh])
            chi_h = self.chi(z=zh_mid).value
            dchi_h = self.chi(z=zh_max_list[izh]).value-self.chi(z=zh_min_list[izh]).value
            vol_sum += dchi_h * chi_h**2

            k = self.ell/chi_h
            #print('zh', zh_mid)
            # power spectrum.  Use pk_hm_halomodel!
            pk_hm = PowerSpectrumHaloMatter(co=self.co, nu=self.nu, su=self.su, zh=zh_mid)
            pk_hm.calc_Pk_hm_full(Mmin=Mmin, Mmax=Mmax)
            
            #print('min(k), max(k)', min(k), max(k))
            Phm_ell = np.exp(pk_hm.lnPk_lnk_interp(np.log(k)))
            kernel = self.lk.kernel_Sigma_z_interp(zh_mid)
            #kernel = self.lk.kernel_z_interp(zh_mid)
            summand = dchi_h * kernel * Phm_ell
            C_ell_sum += summand
        C_ell_sum = C_ell_sum * self.rho_mean / vol_sum 

        self.ell_h_Sigma = self.ell
        self.C_ell_h_Sigma = C_ell_sum

if __name__ == "__main__":
    co = CosmoParameters()
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey()
    aps = AngularPowerSpectra(co=co, nu=nu, su=su)
    #aps.calc_C_ell_h(zh_min=0.1, zh_max=0.3, Mmin=1e14, Mmax=1e16)
    # aps.calc_C_ell_h_kappa(zh_min=0.1, zh_max=0.3, Mmin=1e14, Mmax=1e16)
    # plt.loglog(aps.ell_h_kappa, aps.C_ell_h_kappa)
    aps.calc_C_ell_h_Sigma(0.1, 0.3, 1e+14, 1e+16)
    plt.loglog(aps.ell_h_Sigma, aps.C_ell_h_Sigma)

    plt.show()
