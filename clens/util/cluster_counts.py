#!/usr/bin/env python
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
from clens.util.parameters import CosmoParameters
from clens.util.scaling_relation import RichnessSelection, FiducialScalingRelation, Costanzi21ScalingRelation, To21ScalingRelation

class ClusterCounts(object):
    def __init__(self, cosmo_parameters, scaling_relation):
        self.cp = cosmo_parameters
        self.sr = scaling_relation
        #print(nuisance_parameters)

    def calc_counts(self, zmin, zmax, lambda_min, lambda_max, survey_area_sq_deg):
        fsky = survey_area_sq_deg/41253.
        cosmo = FlatLambdaCDM(H0=self.cp.h*100, Om0=self.cp.OmegaM)
        vol = fsky * 4. * np.pi/3. * (cosmo.comoving_distance(zmax).value**3 - cosmo.comoving_distance(zmin).value**3)
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

        rs = RichnessSelection(scaling_relation=self.sr, lambda_min=lambda_min, lambda_max=lambda_max)

        lnM_selection_arr = rs.lnM_selection(lnM_arr, z)
        self.cluster_number_density = np.trapz(dndlnM_arr*lnM_selection_arr, x=lnM_arr)
        #print('self.cluster_number_density', self.cluster_number_density)
        
        self.counts = self.cluster_number_density * vol
        print('counts', self.counts)
        
        # sample variance
        bn = np.trapz(bias_arr*dndlnM_arr*lnM_selection_arr, x=lnM_arr) * vol
        _scale = (3./(4.*np.pi) * vol)**(1./3.)
        f_fgrowth  = self.den.growth_factor
        lin_0 = LinearTheory(cosmo=self.cosmo, z=0, den=self.den, set_warnsig8err=True)
        # spherical mass variance as a function of scale.
        sigma_r_0 = lin_0.sigma_r_0_interp()
        _sigma2_v = (sigma_r_0(_scale) * f_fgrowth(z))**2 
        self.sv = bn**2 * _sigma2_v

        self.cluster_mean_bias = bn/self.counts

        # mean mass
        self.lnM_mean = np.trapz(lnM_arr*dndlnM_arr*lnM_selection_arr, x=lnM_arr) / self.cluster_number_density

        return self.counts, self.sv, self.cluster_mean_bias, self.lnM_mean, self.cluster_number_density 

if __name__ == "__main__":
    #cosmo_parameters = CosmoParameters()
    cosmo_parameters = CosmoParameters(h=0.7, OmegaDE=0.724, OmegaM=0.276, sigma8=0.802)
    #scaling_relation = FiducialScalingRelation()
    #scaling_relation = Costanzi21ScalingRelation()
    scaling_relation = To21ScalingRelation()
    cmm = ClusterCounts(cosmo_parameters=cosmo_parameters, scaling_relation=scaling_relation)
    cc = cmm.calc_counts(zmin=0.2, zmax=0.35, lambda_min=20, lambda_max=30, survey_area_sq_deg=1437)
    print(cc)

