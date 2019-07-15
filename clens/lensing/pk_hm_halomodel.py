#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.interpolate import RectBivariateSpline # 2D interpolation

import clens.util.constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.lensing.correlation_functions_3d import CorrelationFunctions3D
from clens.util.survey import Survey

#from zypy.zycosmo import CosmoParams, Density, LinearTheory
from clens.ying.param import CosmoParams
from clens.ying.lineartheory import LinearTheory
from clens.ying.density import Density
from clens.lensing.correlation_functions_3d import CorrelationFunctions3D


class PowerSpectrumHaloMatter(object):
    def __init__(self, co, nu, su, zh):
        self.co = co
        self.nu = nu
        self.su = su
        self.zh = zh

        rho_crit = cn.rho_crit_with_h * co.h**2
        self.rho_mean = rho_crit * self.co.OmegaM # comoving!

        self.nk = 1000
        self.lnk_list = np.linspace(np.log(1e-5), np.log(2e+4), self.nk)# required by "angular_power_spectra.py"
        self.k_list = np.exp(self.lnk_list)


        self.cf3 = CorrelationFunctions3D(z=0.2, co=co, nu=nu)
        self.cf3.set_up_halos(zl=zh)

        self.cosmo_ying = CosmoParams(omega_M_0=self.co.OmegaM, omega_b_0=self.co.OmegaB, omega_lambda_0=self.co.OmegaDE, h=self.co.h, sigma_8=self.co.sigma8, n=self.co.ns, tau=self.co.tau) # Ying's
        den = Density(cosmo=self.cosmo_ying)
        self.lin = LinearTheory(self.cosmo_ying, z=zh)


    ## calcualte u(k) based on Cooray+Sheth Eq 81
    def _Ci(self, x, uplim=6., dlnt=0.001):
        lnt = np.arange(np.log(x), uplim, dlnt)
        t = np.exp(lnt)
        integrand = np.cos(t)
        return -np.trapz(integrand, x=lnt)

    def _Si(self, x, lowlim=-5, dlnt=0.001):
        if x > 1e+3: x=1e+3 # should be flat above 1e+3
        lnt = np.arange(lowlim, np.log(x), dlnt)
        t = np.exp(lnt)
        integrand = np.sin(t)
        return np.trapz(integrand, x=lnt)

    def _check_convergence_Ci(self):
        plt.figure()
        for uplim in [2, 4, 6]: # 10 is disasterous
            lnx_list = np.linspace(np.log(1e-2), np.log(1e+5), 1000)
            x_list = np.exp(lnx_list)
            Ci_list = np.zeros(len(x_list))
            for ix, x in enumerate(x_list):
                Ci_list[ix] = self._Ci(x, uplim)
            plt.plot(x_list, Ci_list, label=uplim)
            plt.legend()
            plt.xscale('log')
        plt.ylabel('Ci')

    def _check_convergence_Si(self):
        plt.figure()
        for lowlim in [-100,-5,-2]:
            lnx_list = np.linspace(np.log(1e-2), np.log(1e+5), 1000)
            x_list = np.exp(lnx_list)
            #print('max(x_list)', max(x_list))
            Si_list = np.zeros(len(x_list))
            for ix, x in enumerate(x_list):
                Si_list[ix] = self._Si(x, lowlim)
            plt.plot(x_list, Si_list, label=lowlim)
            plt.legend()
            plt.xscale('log')
        plt.ylabel('Si')


    def calc_uk(self, M200m, c, plotting=False):
        def _u_k(k):
            Delta = 200.
            rhom = self.rho_mean
            R = (3.*M200m/(4.*np.pi*rhom*Delta))**(1./3.)
            rs = R/c
            rhos = M200m/(4.*np.pi*rs**3 * (np.log(1.+c)-c/(1.+c)))# Eq 76
            #print('max((1+c)*k*rs)', np.max((1+c)*k*rs))

            term1 = 4.*np.pi * rhos* rs**3 / M200m
            term2 = np.sin(k*rs)*(self._Si((1+c)*k*rs) - self._Si(k*rs)) 
            term2 += -1.*np.sin(c*k*rs)/(1+c)/k/rs 
            term2 += np.cos(k*rs)*(self._Ci((1+c)*k*rs) - self._Ci(k*rs))
            return term1 * term2

        ## need a even longer range of k for interp
        nk_interp = 1000
        lnk_list_interp = np.linspace(min(self.lnk_list), max(self.lnk_list)+1, nk_interp)
        k_list_interp = np.exp(self.lnk_list)

        lnuk_list = np.zeros(nk_interp)
        uk_list = np.zeros(nk_interp)
        for i, lnk in enumerate(lnk_list_interp):
            k = np.exp(lnk)
            uk_list[i] = (_u_k(k))
        
        uk_list_smooth = savgol_filter(uk_list, 21, 3)
        select = [uk_list_smooth > 0]
        self.lnuk_lnk_interp = interp1d(lnk_list_interp[select], np.log(uk_list_smooth[select]))

    def calc_uk_M_interp(self, Mmin, Mmax): ## used by tripsectrum
        lnM_list = np.arange(np.log(Mmin), np.log(Mmax)+0.1, 0.1)
        nM_interp = len(lnM_list)
        c_list = self.cf3.c200m_M200m(np.exp(lnM_list))

        nk_interp = 1000
        lnk_list_interp = np.linspace(min(self.lnk_list), max(self.lnk_list), nk_interp)

        lnuk_M_list = np.zeros([nM_interp, nk_interp])
        for i, lnM in enumerate(lnM_list):
            self.calc_uk(M200m=np.exp(lnM), c=c_list[i])
            lnuk_M_list[i,:] = self.lnuk_lnk_interp(lnk_list_interp) # calc_uk does it on positive u values, not regular k grid, so I need to re-interpolate to a regular grid
        #print(lnuk_M_list)
        self.ln_uk_lnM_lnk_interp = RectBivariateSpline(lnM_list, lnk_list_interp, lnuk_M_list)


    def calc_Pk_1h(self, Mmin, Mmax):
        sum_M_uk_dndlnM = np.zeros(self.nk)
        sum_dndlnM = np.zeros(self.nk)
        nM = int((np.log(Mmax)-np.log(Mmin))/0.5)
        #print('nM', nM)

        lnM_bin = np.linspace(np.log(Mmin), np.log(Mmax), nM+1)
        lnM_min_list = lnM_bin[:-1:]
        lnM_max_list = lnM_bin[1::]
        for iM in range(nM):
            lnMmid = 0.5*(lnM_min_list[iM]+lnM_max_list[iM])
            self.calc_uk(M200m=np.exp(lnMmid), c=4) # TODO: add C-M relation!
            uk = np.exp(self.lnuk_lnk_interp(self.lnk_list))
            dndlnM = self.cf3.dndlnM_lnM_interp(lnMmid)
            sum_M_uk_dndlnM += np.exp(lnMmid) * uk * dndlnM
            sum_dndlnM += dndlnM
        self.P1h_list = sum_M_uk_dndlnM / sum_dndlnM / self.rho_mean

    def calc_Pk_2h(self, Mmin, Mmax):
        lnM_list = np.linspace(np.log(Mmin), np.log(Mmax), 100)
        M_list = np.exp(lnM_list)
        dndlnM_list = self.cf3.dndlnM_lnM_interp(lnM_list)
        b_list = self.cf3.bias_lnM_interp(lnM_list)
        bias = np.trapz(b_list * dndlnM_list, x=lnM_list)/np.trapz(dndlnM_list, x=lnM_list)
        #print('bias', bias)
        self.P2h_list = self.lin.power_spectrum(self.k_list) * bias # not bias**2

    def calc_Pk_hm_full(self, Mmin=1e14, Mmax=2e14):
        self.calc_Pk_1h(Mmin=Mmin, Mmax=Mmax)
        self.calc_Pk_2h(Mmin=Mmin, Mmax=Mmax)
        self.lnPk_lnk_interp = interp1d(self.lnk_list, np.log(self.P1h_list+self.P2h_list))

def demo_uk():
    co = CosmoParameters()
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey()
    pshm = PowerSpectrumHaloMatter(co=co, nu=nu, su=su, zh=0.5)
    pshm._check_convergence_Ci()
    pshm._check_convergence_Si()

    plt.figure()
    pshm.calc_uk(M200m=1e14, c=4, plotting=True)
    pshm.calc_uk(M200m=1e16, c=4, plotting=True)

def demo_uk_interp():
    co = CosmoParameters()
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey()
    pshm = PowerSpectrumHaloMatter(co=co, nu=nu, su=su, zh=0.5)
    pshm.calc_uk_M_interp(1e+14, 2e+14)

def demo_pk():
    co = CosmoParameters()
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey()

    pshm = PowerSpectrumHaloMatter(co=co, nu=nu, su=su, zh=0.5)
    pshm.calc_Pk_hm_full(Mmin=1e14, Mmax=1e16)

    plt.figure()
    plt.loglog(pshm.k_list, pshm.P1h_list)
    plt.loglog(pshm.k_list, pshm.P2h_list)
    plt.loglog(pshm.k_list, np.exp(pshm.lnPk_lnk_interp(pshm.lnk_list)))


if __name__ == "__main__":
    #demo_uk()
    #demo_pk()
    demo_uk_interp()
    plt.show()
