#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import erfc
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

from clens.ying.param import CosmoParams
from clens.ying.lineartheory import LinearTheory
from clens.ying.density import Density
from clens.ying.halostat import HaloStat

from clens.util import constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters, NuisanceParametersPiecewise
from clens.util.scaling_relation import FiducialScalingRelation, PiecewiseScalingRelation


#renamed from mean_mass/cluster_counts_lambda_min.py


class ClusterCounts(object):
    def __init__(self, cosmo_parameters, nuisance_parameters, piecewise=False):
        self.cp = cosmo_parameters
        self.nu = nuisance_parameters
        self.piecewise = piecewise

    def calc_counts(self, zmin, zmax, lambda_min, lambda_max=100000, survey_area_sq_deg=2000.):
        fsky = survey_area_sq_deg/41253.
        cosmo = FlatLambdaCDM(H0=self.cp.h*100, Om0=self.cp.OmegaM)
        vol = (cosmo.comoving_volume(zmax).value - cosmo.comoving_volume(zmin).value)*fsky
        #print('vol', vol)
        
        z = 0.5*(zmin+zmax)
        # using ying's mass function for now
        self.cosmo = CosmoParams(omega_M_0=self.cp.OmegaM, omega_b_0=self.cp.OmegaB, omega_lambda_0=self.cp.OmegaDE, h=self.cp.h, sigma_8=self.cp.sigma8, n=self.cp.ns, tau=self.cp.tau)
        DELTA_HALO = 200.0

        # mass function
        Mmin = 1e13
        Mmax = 5e15
        dlnM = 0.01
        lnM_arr = np.arange(np.log(Mmin), np.log(Mmax), dlnM)
        M_arr = np.exp(lnM_arr)
        self.den = Density(cosmo=self.cosmo)
        rho_mean_0 = self.den.rho_mean_z(0.)
        hs = HaloStat(cosmo=self.cosmo, z=z, DELTA_HALO=DELTA_HALO, rho_mean_0=rho_mean_0, mass=M_arr, dlogm=dlnM)
        dndM_arr = hs.mass_function
        bias_arr = hs.bias_function
        dndlnM_arr = dndM_arr * M_arr

        if self.piecewise == True:
            psr = PiecewiseScalingRelation(norm_mid=self.nu.norm_mid, 
                alpha_M0=self.nu.alpha_M0, alpha_M1=self.nu.alpha_M1, alpha_M2=self.nu.alpha_M2,
                scatter0=self.nu.scatter0, scatter1=self.nu.scatter1, scatter2=self.nu.scatter2)

        def lnM_selection(lnM):
            if self.piecewise == False:
                sr = FiducialScalingRelation(self.nu)
                lnlambda_mean = sr.lnlambda_lnM(lnM)
                x_lo = (np.log(lambda_min) - lnlambda_mean)/np.sqrt(2.)/self.nu.sigma_lambda
                x_hi = (np.log(lambda_max) - lnlambda_mean)/np.sqrt(2.)/self.nu.sigma_lambda

            if self.piecewise == True:
                lnlambda_mean = np.array([psr.lnlambda_lnM_piecewise(lnM0) for lnM0 in lnM])
                sigma_lambda_piecewise = np.array([psr.scatter_piecewise(lnM0) for lnM0 in lnM])
                x_lo = (np.log(lambda_min) - lnlambda_mean)/np.sqrt(2.)/sigma_lambda_piecewise
                x_hi = (np.log(lambda_max) - lnlambda_mean)/np.sqrt(2.)/sigma_lambda_piecewise

            return 0.5*erfc(x_lo) - 0.5*erfc(x_hi)



        lnM_selection_arr = lnM_selection(lnM_arr)
        self.cluster_number_density = np.trapz(dndlnM_arr*lnM_selection_arr, x=lnM_arr)
        counts = self.cluster_number_density * vol
        self.counts = counts
        # sample variance
        bn = np.trapz(bias_arr*dndlnM_arr*lnM_selection_arr, x=lnM_arr) * vol
        _scale = (3./(4.*np.pi) * vol)**(1./3.)
        f_fgrowth  = self.den.growth_factor
        lin_0 = LinearTheory(cosmo=self.cosmo, z=0, den=self.den, set_warnsig8err=True)
        # spherical mass variance as a function of scale.
        sigma_r_0 = lin_0.sigma_r_0_interp()
        _sigma2_v = (sigma_r_0(_scale) * f_fgrowth(z))**2 
        sv = bn**2 * _sigma2_v


        mean_bias = bn/counts
        self.cluster_mean_bias = mean_bias
        

        return counts, sv, mean_bias # #lnM_mean

if __name__ == "__main__":
    # regular
    cosmo_parameters = CosmoParameters()
    nuisance_parameters = NuisanceParameters()
    cmm = ClusterCounts(cosmo_parameters=cosmo_parameters, nuisance_parameters=nuisance_parameters, piecewise=False)
    cc = cmm.calc_counts(lambda_min=28, zmin=0.1, zmax=0.3)
    print(cc)
    '''
    # piecewise
    nuisance_parameters = NuisanceParametersPiecewise()
    cmm = ClusterCounts(cosmo_parameters=cosmo_parameters, nuisance_parameters=nuisance_parameters, piecewise=True)
    cc = cmm.calc_counts(lambda_min=28, zmin=0.1, zmax=0.3)
    print(cc)
    '''

