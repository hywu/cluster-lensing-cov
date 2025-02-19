#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import RectBivariateSpline # 2D interpolation

from clens.ying.param import CosmoParams
from clens.ying.lineartheory import LinearTheory
from clens.ying.density import Density
from clens.ying.halostat import HaloStat

from clens.util import constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.survey import Survey
from clens.util.cluster_counts import ClusterCounts
from clens.lensing.lensing_kernel import LensingKernel
from clens.lensing.pk_hm_halomodel import PowerSpectrumHaloMatter
from clens.lensing.correlation_functions_3d import CorrelationFunctions3D
from clens.lensing.bessel_for_cov_theta import BesselForCovTheta
import timeit

## TODO: don't use yet

class CovgammatNonGaussian(object):
    def __init__(self, co, nu, su):
        self.co = co
        self.nu = nu
        self.su = su

        rho_crit = cn.rho_crit_with_h * co.h**2
        self.rho_mean = rho_crit * self.co.OmegaM # comoving!

        astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        self.chi = astropy_dist.comoving_distance
        self.lk = LensingKernel(co=self.co, su=self.su)
        self.ln_ell = np.linspace(np.log(1), np.log(1e+5), 500) 
        #self.ln_ell = np.linspace(np.log(1e-4), np.log(1e+7), 50) 
        # 100 bins per decade. 
        # ell_max=1e7 sufficient for rp=0.03 at zh=1.4
        self.ell = np.exp(self.ln_ell)
        self.n_ell = len(self.ell)

        self.bf = BesselForCovTheta()

    def calc_3d_trispectrum_1h(self, zh_mid, Mmin, Mmax):
        Mmax = min(5e+15, Mmax)
        cf3d = CorrelationFunctions3D(zh_mid, self.co, self.nu, Mmin, Mmax) # for concentration

        ## works for single redshift
        self.nk = 300
        self.lnk_list = np.linspace(np.log(0.1), np.log(10000), self.nk)# cannot go beyond 300 ## AFFECTING SPEED!
        self.k_list = np.exp(self.lnk_list)

        # get halo mass functions
        self.cosmo_ying = CosmoParams(omega_M_0=self.co.OmegaM, omega_b_0=self.co.OmegaB, omega_lambda_0=self.co.OmegaDE, h=self.co.h, sigma_8=self.co.sigma8, n=self.co.ns, tau=self.co.tau)
        DELTA_HALO = 200.0
        dlnM = 0.01#01 ## AFFECTING accuracy
        lnM_arr_bins = np.arange(np.log(Mmin), np.log(Mmax), dlnM)
        lnM_arr = lnM_arr_bins + 0.5 * dlnM
        M_arr = np.exp(lnM_arr)
        self.den = Density(cosmo=self.cosmo_ying)
        rho_mean_0 = self.den.rho_mean_z(0.0)
        hs = HaloStat(cosmo=self.cosmo_ying, z=zh_mid, DELTA_HALO=DELTA_HALO, rho_mean_0=rho_mean_0, mass=M_arr, dlogm=dlnM)
        dndM_arr = hs.mass_function
        bias_arr = hs.bias_function
        
        dndlnM_arr = dndM_arr * M_arr
        nM = len(lnM_arr)
        #print('nM', nM)
        nh = np.trapz(dndlnM_arr, lnM_arr)
        #print(nh)
        #exit()
        Tk1k2 = np.zeros([self.nk, self.nk])
        # integrate over mass (using summation)
        start = timeit.default_timer()
        pk_hm = PowerSpectrumHaloMatter(co=self.co, nu=self.nu, su=self.su, zh=zh_mid)
        
        '''
        for iM in range(nM):
            c = cf3d.c200m_M200m(M_arr[iM])
            pk_hm.calc_uk(M200m=M_arr[iM], c=c)
            uk_list = np.exp(pk_hm.lnuk_lnk_interp(self.lnk_list))
            for ik1 in range(self.nk):
                for ik2 in range(self.nk):
                    Tk1k2[ik1, ik2] += (M_arr[iM]/self.rho_mean)**2 * uk_list[ik1]*uk_list[ik2] * dndlnM_arr[iM] * dlnM / nh**2
        '''
        ## need lnuk_lnk_lnM_interp
        pk_hm.calc_uk_M_interp(Mmin=Mmin, Mmax=Mmax)

        for ik1 in range(self.nk):
            for ik2 in range(self.nk):
                lnk1 = self.lnk_list[ik1]
                lnk2 = self.lnk_list[ik2]
                uk1 = np.exp(pk_hm.ln_uk_lnM_lnk_interp(lnM_arr, lnk1)).flatten()
                uk2 = np.exp(pk_hm.ln_uk_lnM_lnk_interp(lnM_arr, lnk2)).flatten()
                Tk1k2_int = (M_arr/self.rho_mean)**2 * uk1*uk2 * dndlnM_arr
                Tk1k2[ik1, ik2] = np.trapz(Tk1k2_int, x=lnM_arr)

        Tk1k2 = Tk1k2 / nh**2
        stop = timeit.default_timer()
        #print('took', stop - start, 'seconds')
        self.ln_Tk1k2_interp = RectBivariateSpline(self.lnk_list, self.lnk_list, np.log(Tk1k2))
        # plt.loglog(self.k_list, Tk1k2.diagonal())
        # plt.xlabel('k')
        # plt.ylabel('T')
        # plt.show()

    def calc_angular_trispectrum(self, zh_min, zh_max, Mmin, Mmax): ##Limber approximation
        nzh = max(int((zh_max-zh_min)/0.2),1) # at least one redshift slice...
        print('nzh', nzh)
        zh_bins = np.linspace(zh_min, zh_max, nzh+1)
        zh_min_list = zh_bins[:-1]
        zh_max_list = zh_bins[1:]

        T_L1_L2_sum = np.zeros([self.n_ell, self.n_ell])
        vol_sum = 0
        #summmand = 0

        for izh in range(nzh):
            zh_min = zh_min_list[izh]
            zh_max = zh_max_list[izh]
            zh_mid = 0.5*(zh_min+zh_max)
            self.calc_3d_trispectrum_1h(zh_mid, Mmin, Mmax)

            chi_h = self.chi(z=zh_mid).value
            dchi_h = self.chi(z=zh_max).value-self.chi(z=zh_min).value
            vol_sum += dchi_h * chi_h**2
        
            T_L1_L2 = np.zeros([self.n_ell, self.n_ell])
            for iL1 in range(self.n_ell):
                for iL2 in range(self.n_ell):
                    lnk1 = np.log(self.ell[iL1]/chi_h)
                    lnk2 = np.log(self.ell[iL2]/chi_h)
                    #print(lnk1, lnk2)
                    #if iL1==iL2: print(np.exp(lnk1), np.exp(lnk2), self.ln_Tk1k2_interp(lnk1, lnk2))
                    T_L1_L2[iL1, iL2] = np.exp(self.ln_Tk1k2_interp(lnk1, lnk2))
            # plt.loglog(self.ell, self.ell**4 * T_L1_L2.diagonal())
            # plt.xlabel('ell')
            # plt.ylabel(r'$\ell^4 T $')
            # plt.loglog(self.ell, self.bf.j2_bin(self.ell, 5e-4, 1e-4), label='J2')
            # plt.show()


            kernel = self.lk.kernel_z_interp(zh_mid)
            w_kappa = kernel * self.rho_mean
                
            #DH = 2997./self.co.h
            #w_cluster = dchi_h**-1 # gives same results (without vol_sum**-2)
            w_cluster = chi_h**2
            print('w_cluster', w_cluster)
            print('w_kappa', w_kappa)
            summand = dchi_h * (chi_h**-6)* (w_cluster**2) * (w_kappa**2) * T_L1_L2

            T_L1_L2_sum += summand

        T_L1_L2_sum *= (vol_sum**-2)
        #T_L1_L2_sum *= (vol_sum**-1)

        # plt.subplot(121)
        # plt.imshow(T_L1_L2)
        # plt.subplot(122)
        # plt.loglog(self.ell, T_L1_L2.diagonal())
        # print(T_L1_L2.diagonal())
        # plt.show()
        fsky = 1.
        T_L1_L2_sum = T_L1_L2_sum/(4.*np.pi*fsky) # actually variance
        self.ln_TL1L2_interp = RectBivariateSpline(self.ln_ell, self.ln_ell, np.log(T_L1_L2_sum))
        

    def _calc_var(self, thmin1, thmax1, thmin2, thmax2, zh_min, zh_max, Mmin, Mmax): ## Hankel transform
        thmax = max(thmax1, thmax2)
        thmin = min(thmin1, thmin2)
        lnell_list = np.arange(np.log(self.bf.scaling_for_ell_min/thmax), np.log(self.bf.scaling_for_ell_max/thmin), 20*self.bf.dlnell) # using 10*self.bf.dlnell makes no difference
        lnL1_list = lnell_list#np.arange(np.log(10),np.log(5e+3), 0.1)
        lnL2_list = lnell_list#np.arange(np.log(10),np.log(5e+3), 0.1)

        int_1 = np.zeros(len(lnL1_list))
        for i1, lnL1 in enumerate(lnL1_list):
            L1 = np.exp(lnL1)
            int_2 = np.zeros(len(lnL2_list))
            for i2, lnL2 in enumerate(lnL2_list):
                L2 = np.exp(lnL2)
                int_2[i2] = np.exp(self.ln_TL1L2_interp(lnL1, lnL2)) * self.bf.j2_bin(L2, thmin2, thmax2) * L2**2
            int_1[i1] = np.trapz(int_2, x=lnL2_list) * self.bf.j2_bin(L1, thmin1, thmax1) * L1**2 
        return np.trapz(int_1, x=lnL1_list)/(4.*np.pi**2)

    '''
    def _calc_var_(self, thmin1, thmax1, thmin2, thmax2, zh_min, zh_max, Mmin, Mmax): ## Hankel transform
        self.calc_angular_trispectrum(zh_min, zh_max, Mmin, Mmax)

        thmax = max(thmax1, thmax2)
        thmin = min(thmin1, thmin2)
        lnell_list = np.arange(np.log(self.bf.scaling_for_ell_min/thmax), np.log(self.bf.scaling_for_ell_max/thmin), 20*self.bf.dlnell)
        ell_list = np.exp(lnell_list)
        n_ell = len(ell_list)
        #print('n_ell', n_ell)
        #print('ell range', min(ell_list), max(ell_list))
        #exit()
        dlnL = lnell_list[1]-lnell_list[0]
        summand = np.zeros([n_ell, n_ell])
        for iL1 in range(n_ell):
            for iL2 in range(n_ell):
                lnL1 = lnell_list[iL1]
                lnL2 = lnell_list[iL2]
                L1 = ell_list[iL1]
                L2 = ell_list[iL2]
                #print(lnk1, lnk2)
                geometry = self.bf.j2_bin(L1, thmin1, thmax1) * self.bf.j2_bin(L2, thmin2, thmax2) * L1**2 * L2**2 * dlnL**2 / (2.*np.pi)**2 
                summand[iL1, iL2] = geometry * np.exp(self.ln_TL1L2_interp(lnL1, lnL2))

        # plt.imshow(np.log10(summand))
        # plt.show()
        fsky = 1.# full sky
        #plt.loglog(ell_list, summand.diagonal())
        #plt.show()
        
        #fac = 1./(4.*np.pi*fsky) 
        fac = 1./(4.*np.pi*fsky) 
        #print(fac*np.sum(summand))
        return fac*np.sum(summand)
        '''


    def calc_var(self, thmin, thmax, nth, zh_min, zh_max, Mmin, Mmax, diag_only=True):
        self.calc_angular_trispectrum(zh_min, zh_max, Mmin, Mmax)
        lnth_list = np.linspace(np.log(thmin), np.log(thmax), nth+1)
        th_list = np.exp(lnth_list)
        self.thmin_list = th_list[:-1]
        self.thmax_list = th_list[1:]
        self.thmid_list = np.sqrt(self.thmin_list*self.thmax_list)

        
        self.cov_trispectrum = np.zeros([nth, nth])
        if diag_only==True: ## only calculating the diagonal elements
            for ith1 in range(nth):
                ith2 = ith1
                start = timeit.default_timer()
                self.cov_trispectrum[ith1,ith1] = self._calc_var(thmin1=self.thmin_list[ith1], thmax1=self.thmax_list[ith1], thmin2=self.thmin_list[ith2], thmax2=self.thmax_list[ith2], zh_min=zh_min, zh_max=zh_max, Mmin=Mmin, Mmax=Mmax)
                stop = timeit.default_timer()
                print('took', stop - start, 'seconds')
                print('cov',self.cov_trispectrum[ith1,ith1])


    def plot_comparison(self, zh_min=0.155, zh_max=0.323):
        theta, var_ng = np.loadtxt('../../output/takahashi_cov/non_gauss/cov_gammat_trispectrum_%s_test.dat'%(which_z), unpack=True)
        fsky = 1./48.
        plt.loglog(theta, var_ng/fsky, label='trispectrum')

        data_loc = '../../data/takahashi_cov/zl_%g_%g_zs_%g_%g/1e+14_1e+16_th5e-04_5e-01_nth15/'%(zh_min, zh_max, self.su.zs_min, self.su.zs_max)
        theta = np.loadtxt(data_loc+'mean_gammat.dat')[:,0]
        var = np.loadtxt(data_loc+'cov_gammat.dat').diagonal()
        plt.loglog(theta, var, '-x', label='Takahashi sim')

        theory_loc = '../../output/takahashi_cov/theory/zl_%g_%g_zs_%g_%g/1e+14_1e+16_th5e-04_5e-01_nth15/'%(zh_min, zh_max, self.su.zs_min, self.su.zs_max)
        data = np.loadtxt(theory_loc+'var_gammat_theory_full.dat')
        theta = data[:,0]
        var  = data[:,1] + data[:,3]
        plt.loglog(theta, var/fsky, label='Gaussian')
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$\rm Var[\gamma_t]$')
        plt.legend()


which_z = 'highz'

if __name__ == "__main__":
    if which_z=='lowz':
        (zh_min, zh_max, zs_min, zs_max)=(0.155, 0.323, 0.508, 0.574)
    if which_z=='midz':
        (zh_min, zh_max, zs_min, zs_max)=(0.323, 0.444, 0.714, 0.789)
    if which_z=='highz':
        (zh_min, zh_max, zs_min, zs_max)=(0.508, 0.574, 1.218, 1.318)
    if which_z=='highz2':
        (zh_min, zh_max, zs_min, zs_max)=(0.714, 0.789, 1.653, 1.779)

    co = CosmoParameters(OmegaM=0.279, sigma8=0.82, h=0.7, OmegaDE=0.721)
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey(zs_min=zs_min, zs_max=zs_max, top_hat=True)
    at = CovgammatNonGaussian(co=co, nu=nu, su=su)
    #at.calc_3d_trispectrum_1h(0.5, 1e+14, 1e+16) 
    #at.calc_angular_trispectrum(0.1, 0.2, 1e+14, 1e+16)

    at.calc_var(thmin=5e-4, thmax=5e-2, nth=10, zh_min=zh_min, zh_max=zh_max, Mmin=1e+14/0.7, Mmax=1e+16/0.7)
    outfile = open('../../output/takahashi_cov/non_gauss/cov_gammat_trispectrum_%s_test.dat'%(which_z),'w')
    for ith1 in range(len(at.thmid_list)):
        outfile.write('%g %g \n'%(at.thmid_list[ith1], at.cov_trispectrum[ith1,ith1]))
    outfile.close()

    at.plot_comparison(zh_min=zh_min, zh_max=zh_max)
    plt.savefig('../../plots/lensing/cov_gammat_trispectrum_%s.pdf'%(which_z))
    plt.show()
