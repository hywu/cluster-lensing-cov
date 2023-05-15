#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys

from astropy.cosmology import FlatLambdaCDM
from clens.util.parameters import CosmoParameters#, NuisanceParameters
from clens.util.scaling_relation import FiducialScalingRelation, Costanzi21ScalingRelation
from clens.util.survey import Survey
from clens.lensing.cov_DeltaSigma import CovDeltaSigma

class DemoCosmoDC2(object):
    """
    calculating covariance matrix at the given lens redshift, source redshift, lambda range, radial range
    """
    def __init__(self, co, su, sr, zh_min, zh_max, lambda_min, lambda_max, rp_min_hiMpc, rp_max_hiMpc, n_rp, output_loc):
        self.co = co
        self.su = su
        self.sr = sr
        self.zs_mid = 0.5 * (self.su.zs_min+self.su.zs_max)

        self.zh_min = zh_min
        self.zh_max = zh_max
        self.zh_mid = 0.5 * (zh_min + zh_min)

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

        self.rp_min = rp_min_hiMpc/co.h
        self.rp_max = rp_max_hiMpc/co.h
        self.n_rp = n_rp

        astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        self.chi = astropy_dist.comoving_distance

        self.output_loc = output_loc
        if os.path.isdir(self.output_loc)==False: os.makedirs(self.output_loc)
        print('output', self.output_loc)

        #### radial bins
        lnrp_list = np.linspace(np.log(rp_min_hiMpc), np.log(rp_max_hiMpc), n_rp+1)
        rp_list = np.exp(lnrp_list)
        rpmin_list = rp_list[:-1]
        rpmax_list = rp_list[1:]
        rpmid_list = np.sqrt(rpmin_list*rpmax_list)

        self.rp_fname = self.output_loc+'rp_hiMpc.dat'
        outfile = open(self.rp_fname, 'w')
        outfile.write('#rp_min[Mpc/h], rp_max[Mpc/h], rp_mid[Mpc/h] \n')
        for i in range(n_rp):
            outfile.write('%12g %12g %12g \n'%(rpmin_list[i], rpmax_list[i], rpmid_list[i]))
        outfile.close()
        
        ## output of this code is saved here
        self.cov_cosmic_shear_fname = self.output_loc+'DeltaSigma_cov_cosmic_shear.dat'
        self.cov_shape_noise_fname = self.output_loc+'DeltaSigma_cov_shape_noise.dat'
        self.cov_combined_fname = self.output_loc+'DeltaSigma_cov_combined.dat' ## key results!

    def calc_cov_full(self, diag_only):
        cds = CovDeltaSigma(co=self.co, su=self.su, sr=self.sr, fsky=fsky)
        output = cds.calc_cov(lambda_min=self.lambda_min, lambda_max=self.lambda_max, zh_min=self.zh_min, zh_max=self.zh_max, rp_min=self.rp_min, rp_max=self.rp_max, n_rp=self.n_rp, diag_only=diag_only)
        rp_mid, cov_cosmic_shear, cov_shape_noise = output
        cov_combined = cov_cosmic_shear/self.co.h**2 + cov_shape_noise/self.co.h**2

        np.savetxt(self.cov_cosmic_shear_fname, cov_cosmic_shear/self.co.h**2) # h^2Msun^2/Mpc^4
        np.savetxt(self.cov_shape_noise_fname, cov_shape_noise/self.co.h**2)
        np.savetxt(self.cov_combined_fname, cov_combined) ## key results!

if __name__ == "__main__":
    ## ./demo_cosmodc2.py 0.2 0.35 20 30 15
    zh_min = float(sys.argv[1])
    zh_max =  float(sys.argv[2])
    lambda_min = float(sys.argv[3])
    lambda_max = float(sys.argv[4]) 
    n_rp = int(sys.argv[5])
    output_loc = 'temp/'

    zs_mid = 2 * zh_max 

    survey_area = 5000.
    fsky = survey_area / 41253.

    co = CosmoParameters(h=0.701, OmegaDE=0.7352, OmegaM=0.2648, sigma8=0.8) # CosmoDC2
    sr = Costanzi21ScalingRelation()
    su = Survey(top_hat=True, zs_min=zs_mid-0.05, zs_max=zs_mid+0.05, n_src_arcmin=10)

    cp = DemoCosmoDC2(co=co, su=su, sr=sr, zh_min=zh_min, zh_max=zh_max, lambda_min=lambda_min, lambda_max=lambda_max, rp_min_hiMpc=0.1, rp_max_hiMpc=100., n_rp=n_rp, output_loc=output_loc)
    cp.calc_cov_full(diag_only=False) # takes some time
