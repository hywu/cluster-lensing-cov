#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.cosmology import FlatLambdaCDM
from clens.util.parameters import CosmoParameters
from clens.util.scaling_relation import PrecalculatedCountsBias
from clens.util.survey import Survey
from clens.lensing.cov_DeltaSigma import CovDeltaSigma
from clens.lensing.cov_gammat import Covgammat
class DemoSPT(object):
    """
    calculating covariance matrix at the given lens redshift, source redshift, radial range,
    taking in lens number counts and bias 
    """
    def __init__(self, co, su, sr, zh_min, zh_max, iz, ilam, ixi, theta_edges, fsky, output_loc): #rp_min_hiMpc, rp_max_hiMpc, n_rp
        self.co = co
        self.su = su
        self.sr = sr
        self.zs_mid = 0.5 * (self.su.zs_min+self.su.zs_max)
        self.iz = iz 
        self.ilam = ilam
        self.ixi = ixi

        self.zh_min = zh_min # kinda redundant....
        self.zh_max = zh_max
        self.zh_mid = 0.5 * (zh_min + zh_min)
        self.fsky = fsky
        # self.rp_min = rp_min_hiMpc/co.h
        # self.rp_max = rp_max_hiMpc/co.h
        # self.n_rp = n_rp

        #self.chi = astropy_dist.comoving_distance

        self.output_loc = output_loc
        if os.path.isdir(self.output_loc)==False: os.makedirs(self.output_loc)
        #print('output', self.output_loc)

        bin_name = f'z{iz}_lam{ilam}_xi{ixi}'

        #### radial bins
        # lnrp_list = np.linspace(np.log(rp_min_hiMpc), np.log(rp_max_hiMpc), n_rp+1)
        # rp_list = np.exp(lnrp_list)
        # rpmin_list = rp_list[:-1]
        # rpmax_list = rp_list[1:]
        # rpmid_list = np.sqrt(rpmin_list*rpmax_list)

        # self.rp_fname = self.output_loc + f'rp_hiMpc_{bin_name}.dat'
        # outfile = open(self.rp_fname, 'w')
        # outfile.write('#rp_min[Mpc/h], rp_max[Mpc/h], rp_mid[Mpc/h] \n')
        # for i in range(n_rp):
        #     outfile.write('%12g %12g %12g \n'%(rpmin_list[i], rpmax_list[i], rpmid_list[i]))
        # outfile.close()
        
        ## output of this code is saved here
        self.theta_edges = theta_edges
        self.cov_cosmic_shear_fname = self.output_loc+f'gammat_cov_cosmic_shear_{bin_name}.dat'
        self.cov_shape_noise_fname = self.output_loc+f'gammat_cov_shape_noise_{bin_name}.dat'
        self.cov_combined_fname = self.output_loc+f'gammat_cov_combined_{bin_name}.dat' ## key results!
        self.counts_fname = self.output_loc+f'counts_{bin_name}.dat'

    def calc_cov_full(self, diag_only):

        arcmin_to_radian = np.pi / 180. / 60.
        self.thmax = max(self.theta_edges) * arcmin_to_radian
        self.thmin = min(self.theta_edges) * arcmin_to_radian
        self.nth = len(self.theta_edges) - 1
        #print('self.nth', self.nth)

        cj = Covgammat(co=self.co, su=self.su, sr=self.sr, fsky=self.fsky)
        output = cj.calc_cov_gammat_integration(thmin=self.thmin, thmax=self.thmax, nth=self.nth, zh_min=self.zh_min, zh_max=self.zh_max, lambda_min=20, lambda_max=30, diag_only=diag_only)
        cov_cosmic_shear, cov_shape_noise = output

        '''
        cds = CovDeltaSigma(co=self.co, su=self.su, sr=self.sr, fsky=fsky)
        output = cds.calc_cov(zh_min=self.zh_min, zh_max=self.zh_max, rp_min=self.rp_min, rp_max=self.rp_max, n_rp=self.n_rp, diag_only=diag_only)
        rp_mid, cov_cosmic_shear, cov_shape_noise = output
        '''
        np.savetxt(self.cov_cosmic_shear_fname, cov_cosmic_shear)
        np.savetxt(self.cov_shape_noise_fname, cov_shape_noise)
        cov_combined = cov_cosmic_shear + cov_shape_noise
        np.savetxt(self.cov_combined_fname, cov_combined) ## key results!
        
'''
# see the notebook
if __name__ == "__main__":
    co = CosmoParameters(h=0.7, OmegaDE=0.7, OmegaM=0.3, sigma8=0.8)
    astropy_dist = FlatLambdaCDM(H0=co.h*100, Om0=co.OmegaM)
    
    file = np.load("fid_bias_ab.npz")
    mass_list = file["mass"]
    bias_list = file["bias"]
    ab_list = file["abundance"]
    area = file["area"]
    zmin_list = file["z_low"]
    zmax_list = file["z_high"]
    zmid_list = 0.5*(file["z_low"] + file["z_high"])
    print(area, "sq deg")

    
    nbins = len(mass_list)
    for ibin in [0]: #range(nbins):
        zh_min = zmin_list[ibin]
        zh_max = zmax_list[ibin]
        counts = ab_list[ibin]
        bias = bias_list[ibin]
        print(zh_min, zh_max, counts, bias)

        sr = PrecalculatedCountsBias(counts, bias)
        output_loc='spt_analytic_counts_bias/'

        zh_mid = 0.5 * (zh_min + zh_max)
        a = 1/(1 + zh_mid)
        rp_min_hiMpc = 0.2 / a * co.h
        rp_max_hiMpc = 15 / a * co.h
        n_rp = 15 
        n_src_arcmin = 5.59 #TODO: cehck 6.28  # DES Y3 survey condition
        sigma_gamma = 0.3 #TODO: check # DES survey condition

        
        
        astropy_dist = FlatLambdaCDM(H0=co.h*100, Om0=co.OmegaM)
        vol_fullsky = astropy_dist.comoving_volume(zh_max) - astropy_dist.comoving_volume(zh_min) # no h
        survey_area = area
        fsky = survey_area / 41253
        print('fsky=', fsky, 'area=', survey_area)

        su = Survey(n_src_arcmin=n_src_arcmin, sigma_gamma=sigma_gamma)#, zs_min=zs_min, zs_max=zs_max)
        cp = DemoSPT(co=co, su=su, sr=sr, zh_min=zh_min, zh_max=zh_max, rp_min_hiMpc=rp_min_hiMpc, rp_max_hiMpc=rp_max_hiMpc, n_rp=n_rp, output_loc=output_loc)

        cp.calc_cov_full(diag_only=True) # takes some time
'''
    