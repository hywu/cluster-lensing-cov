#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.cosmology import FlatLambdaCDM
from clens.util.parameters import CosmoParameters
from clens.util.scaling_relation import PrecalculatedCountsBias
from clens.util.survey import Survey
#from clens.lensing.cov_DeltaSigma import CovDeltaSigma
from clens.lensing.cov_gammat import Covgammat

class DemoAnacal(object):
    """
    calculating covariance matrix at the given lens redshift, source redshift, radial range,
    taking in lens number counts and bias 
    """
    def __init__(self, co, su, sr, zh_min, zh_max, zs_mid, lam_min, lam_max, theta_edges, fsky, output_loc): #rp_min_hiMpc, rp_max_hiMpc, n_rp
        self.co = co
        self.su = su
        self.sr = sr
        self.zs_mid = zs_mid #0.5 * (self.su.zs_min+self.su.zs_max)
        self.lam_min = lam_min
        self.lam_max = lam_max
        #self.iz = iz 
        #self.ilam = ilam
        #self.ixi = ixi

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
        bin_name = f'lam_{lam_min}_{lam_max}'

        #print('output', self.output_loc)

        #bin_name = f'z{iz}_lam{ilam}_xi{ixi}'

        #### radial bins
        # lnrp_list = np.linspace(np.log(rp_min_hiMpc), np.log(rp_max_hiMpc), n_rp+1)
        # rp_list = np.exp(lnrp_list)
        # rpmin_list = rp_list[:-1]
        # rpmax_list = rp_list[1:]
        # rpmid_list = np.sqrt(rpmin_list*rpmax_list)


        self.theta_edges = theta_edges 

        n_theta = len(theta_edges) -1
        self.theta_fname = self.output_loc + f'theta_{bin_name}.dat'
        outfile = open(self.theta_fname, 'w')
        outfile.write('#theta_min, theat_max, theta_mid [radian]\n')
        for i in range(n_theta):
            outfile.write('%12g %12g %12g \n'%(theta_edges[i], theta_edges[i+1], (theta_edges[i]+theta_edges[i+1])/2.))
        outfile.close()
        
        ## output of this code is saved here
        self.cov_cosmic_shear_fname = self.output_loc+f'gammat_cov_cosmic_shear_{bin_name}.dat'
        self.cov_shape_noise_fname = self.output_loc+f'gammat_cov_shape_noise_{bin_name}.dat'
        self.cov_combined_fname = self.output_loc+f'gammat_cov_combined_{bin_name}.dat' ## key results!
        self.counts_fname = self.output_loc+f'counts_{bin_name}.dat'

    def calc_cov_full(self, diag_only):

        self.thmax = max(self.theta_edges) 
        self.thmin = min(self.theta_edges) 
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


if __name__ == "__main__":

    # use colossus 'planck18'
    co = CosmoParameters(h=0.6766, OmegaDE=0.6889, OmegaM=0.3111, sigma8=0.8102)
    astropy_dist = FlatLambdaCDM(H0=co.h*100, Om0=co.OmegaM)
    
    file = np.load("file_counts_bias.npz")
    mass_list = file["mean_mass"]
    bias_list = file["mean_bias"]
    counts_list = file["N_lam"]
    lambda_edges = file["lambda_edges"]
    area = 18000.
    zh_min = 0.2
    zh_max = 0.4
    zs_mid = 1
    n_src_arcmin = 27
    sigma_gamma = 0.3 # TODO: check

    pixel_bin_edges = np.linspace(15.0 / 0.2, 2468, 15 + 1)
    angular_bin_edges = pixel_bin_edges * 0.2
    arcsec_to_radian = np.pi / 180. / 3600.
    theta_edges = angular_bin_edges * arcsec_to_radian

    nbins = len(mass_list)
    for ibin in range(nbins):
        counts = counts_list[ibin]
        bias = bias_list[ibin]
        lam_min = lambda_edges[ibin]
        lam_max = lambda_edges[ibin+1]

        sr = PrecalculatedCountsBias(counts, bias)
        output_loc = 'anacal_analytic_counts_bias/'

        astropy_dist = FlatLambdaCDM(H0=co.h*100, Om0=co.OmegaM)
        vol_fullsky = astropy_dist.comoving_volume(zh_max) - astropy_dist.comoving_volume(zh_min) # no h
        survey_area = area
        fsky = survey_area / 41253
        print('fsky=', fsky, 'area=', survey_area)

        su = Survey(n_src_arcmin=n_src_arcmin, sigma_gamma=sigma_gamma, top_hat=True, zs_min=zs_mid-0.01, zs_max=zs_mid+0.01) # thin slice of source
        cp = DemoAnacal(co, su, sr, zh_min, zh_max, zs_mid, lam_min, lam_max, theta_edges, fsky, output_loc)

        cp.calc_cov_full(diag_only=False) # takes some time
