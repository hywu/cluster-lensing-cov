#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import sys
from astropy.cosmology import FlatLambdaCDM


"output in Msun/Mpc^2 to avoid confusion!!!!!"

import clens.util.constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.survey import Survey
from clens.lensing.correlation_functions_3d import CorrelationFunctions3D

from clens.ying.DeltaSigmaR import DeltaSigmaR, xi_rp

class LensingProfiles(object):
    def __init__(self, co, nu, su, zh_min, zh_max, lambda_min, lambda_max):
        self.co = co
        self.nu = nu
        self.su = su

        self.zh_min = zh_min
        self.zh_max = zh_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        # astropy
        astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        self.chi = astropy_dist.comoving_distance

    def calc_DeltaSigma(self):
        zh = 0.5*(self.zh_min + self.zh_max)
        rho_crit = cn.rho_crit_with_h * self.co.h**2  # h-free
        self.rho_mean = rho_crit * self.co.OmegaM # / (1+zh)**3

        self.cf3d = CorrelationFunctions3D(z=zh, co=self.co, nu=self.nu, lambda_min=self.lambda_min, lambda_max=self.lambda_max)
        r = self.cf3d.radius
        xi_cm = self.cf3d.xi_cm()
        rp, DeltaSigma, xi, ximean = DeltaSigmaR(r, xi_cm, rp_max=150, rho=self.rho_mean) # Ying's code

        return rp, DeltaSigma#*(1e-12)


    def calc_Sigma(self):
        zh = 0.5*(self.zh_min + self.zh_max)
        rho_crit = cn.rho_crit_with_h * self.co.h**2  # h-free
        self.rho_mean = rho_crit * self.co.OmegaM # / (1+zh)**3

        self.cf3d = CorrelationFunctions3D(z=zh, co=self.co, nu=self.nu, lambda_min=self.lambda_min, lambda_max=self.lambda_max)
        r = self.cf3d.radius
        xi_cm = self.cf3d.xi_cm()
        rp, DeltaSigma, xi, ximean = DeltaSigmaR(r, xi_cm, rp_max=150, rho=self.rho_mean) # Ying's code
        Sigma = xi * self.rho_mean
        
        return rp, Sigma


    def mean_inv_Sigma_crit(self, zh):
        """
        calculating the averaged 1/Sigma_crit, for *one* lens redshift and *a range of* source redshifts
        """
        ## array of source redshifts
        dz = 0.01
        zs_list = np.arange(zh+0.01, 2.+dz, dz)
        chi_l = self.chi(z=zh).value
        chi_s_list = self.chi(z=zs_list).value
        Sigma_crit = (cn.c)**2/(4*np.pi*cn.G)  * chi_s_list / chi_l /(chi_s_list - chi_l) / (1+zh) #* (1e-12) # Msun/pc^2

        ## source redshift distribution
        f_src = self.su.pz_src(zs_list)
        #print(f_src, f_src)
        mean_inv_Sigma_crit = np.trapz(f_src/Sigma_crit, x=zs_list) / np.trapz(f_src, x=zs_list) 
        return mean_inv_Sigma_crit


    def calc_gammmat(self):
        rp, DeltaSigma = self.calc_DeltaSigma()

        zh = 0.5*(self.zh_min + self.zh_max)
        mean_inv_Sigma_crit = self.mean_inv_Sigma_crit(zh)
        gammat_avg = DeltaSigma * mean_inv_Sigma_crit #* (1e-12)
        chi_l = self.chi(zh).value
        theta = rp/chi_l

        return theta, gammat_avg


if __name__ == "__main__":
    co = CosmoParameters()#OmegaM=0.286, sigma8=0.82, h=0.7, OmegaDE=0.714)
    nu = NuisanceParameters()#sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)#1-1,no scatter
    su = Survey(zs_min=0.508, zs_max=0.574)
    lp = LensingProfiles(co=co, nu=nu, su=su, zh_min=0.2, zh_max=0.35, lambda_min=20, lambda_max=30)
    rp, DeltaSigma = lp.calc_DeltaSigma()
    plt.plot(rp, DeltaSigma)
    #theta, gammat = lp.calc_gammmat()
    #plt.plot(theta, gammat)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()