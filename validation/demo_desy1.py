#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.cosmology import FlatLambdaCDM
from clens.util.parameters import CosmoParameters
from clens.util.scaling_relation import Costanzi21ScalingRelation, To21ScalingRelation, PrecalculatedCountsBias
from clens.util.survey import Survey
from clens.lensing.cov_DeltaSigma import CovDeltaSigma
from clens.util.cluster_counts import ClusterCounts

class DemoDESY1(object):
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

        #astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        #self.chi = astropy_dist.comoving_distance

        self.output_loc = output_loc
        if os.path.isdir(self.output_loc)==False: os.makedirs(self.output_loc)
        print('output', self.output_loc)

        bin_name = f'{zh_min}_{zh_max}_{lambda_min}_{lambda_max}'

        #### radial bins
        lnrp_list = np.linspace(np.log(rp_min_hiMpc), np.log(rp_max_hiMpc), n_rp+1)
        rp_list = np.exp(lnrp_list)
        rpmin_list = rp_list[:-1]
        rpmax_list = rp_list[1:]
        rpmid_list = np.sqrt(rpmin_list*rpmax_list)

        self.rp_fname = self.output_loc + f'rp_hiMpc_{bin_name}.dat'
        outfile = open(self.rp_fname, 'w')
        outfile.write('#rp_min[Mpc/h], rp_max[Mpc/h], rp_mid[Mpc/h] \n')
        for i in range(n_rp):
            outfile.write('%12g %12g %12g \n'%(rpmin_list[i], rpmax_list[i], rpmid_list[i]))
        outfile.close()
        
        ## output of this code is saved here
        self.cov_cosmic_shear_fname = self.output_loc+f'DeltaSigma_cov_cosmic_shear_{bin_name}.dat'
        self.cov_shape_noise_fname = self.output_loc+f'DeltaSigma_cov_shape_noise_{bin_name}.dat'
        self.cov_combined_fname = self.output_loc+f'DeltaSigma_cov_combined_{bin_name}.dat' ## key results!
        self.counts_fname = self.output_loc+f'counts_{bin_name}.dat'

    def calc_cov_full(self, diag_only):
        cds = CovDeltaSigma(co=self.co, su=self.su, sr=self.sr, fsky=fsky)
        output = cds.calc_cov(lambda_min=self.lambda_min, lambda_max=self.lambda_max, zh_min=self.zh_min, zh_max=self.zh_max, rp_min=self.rp_min, rp_max=self.rp_max, n_rp=self.n_rp, diag_only=diag_only)
        rp_mid, cov_cosmic_shear, cov_shape_noise = output

        np.savetxt(self.cov_cosmic_shear_fname, cov_cosmic_shear/self.co.h**2) # h^2Msun^2/pc^4
        np.savetxt(self.cov_shape_noise_fname, cov_shape_noise/self.co.h**2)
        cov_combined = cov_cosmic_shear/self.co.h**2 + cov_shape_noise/self.co.h**2
        np.savetxt(self.cov_combined_fname, cov_combined) ## key results!

    def calc_counts(self):
        try:
            counts = self.sr.lens_counts
            bias = self.sr.lens_bias
            print('use precalculated counts and bias')

        except:
            cmm = ClusterCounts(cosmo_parameters=co, scaling_relation=self.sr)
            cc = cmm.calc_counts(zmin=zh_min, zmax=zh_max, lambda_min=lambda_min, lambda_max=lambda_max, survey_area_sq_deg=survey_area)
            counts = cc.counts 
            bias = cc.cluster_mean_bias

        np.savetxt(self.counts_fname, [counts, bias], header='counts, sv, bias, mean_mass')


if __name__ == "__main__":
    ## ./demo_desy1.py 0.2 0.35 0.75 20 30
    zh_min_list = [0.2, 0.35, 0.5]
    zh_max_list = [0.35, 0.5, 0.65]
    #lambda_min_list = [ 5, 10, 14, 20, 30, 45, 60]
    #lambda_max_list = [10, 14, 20, 30, 45, 60, 1000]

    lambda_min_list = [ 20, 30, 45, 60]
    lambda_max_list = [ 30, 45, 60, 1000]


    counts_list = np.array([[762, 376, 123, 91],
                            [1549, 672, 187, 148],
                            [1612, 687, 205, 92]]) # from Y1 paper
    
    bias_list = np.array([[2.80019023, 3.67337471, 4.20838351, 5.94689383]
                          [2.72935697, 3.56092105, 4.36959595, 5.84793938]
                          [2.54976559, 3.5966532 , 4.39331755, 5.71029621]]) # from Andres

    nz = len(zh_min_list)
    nlam = len(lambda_min_list)
    for iz in range(nz):
        for ilam in range(nlam):

            zh_min = zh_min_list[iz]
            zh_max = zh_max_list[iz]

            lambda_min = lambda_min_list[ilam]
            lambda_max = lambda_max_list[ilam]

            counts = counts_list[iz, ilam]
            bias = bias_list[iz, ilam]

            # Costanzi21 Table 4 column 2 (BKG)
            #co = CosmoParameters(h=0.715, OmegaDE=0.678, OmegaM=0.322, sigma8=0.790) 
            #sr = Costanzi21ScalingRelation()
            #### TODO: bug: too high counts!

            # To & Krause 21 6×2pt+N, h is unconstrained
            #co = CosmoParameters(h=0.7, OmegaDE=0.724, OmegaM=0.276, sigma8=0.802)
            #sr = To21ScalingRelation()
            output_loc='desy1_analytic_To21/'

            co = CosmoParameters(h=0.7, OmegaDE=0.7, OmegaM=0.3, sigma8=0.8)
            sr = PrecalculatedCountsBias(counts, bias)
            output_loc='desy1_analytic_counts_bias/'

            h = co.h
            zh_mid = 0.5 * (zh_min + zh_max)
            a = 1/(1 + zh_mid)
            rp_min_hiMpc = 0.03 / a * h
            rp_max_hiMpc = 30. / a * h
            n_rp = 15 
            n_src_arcmin = 6.28
            #n_src_arcmin = 5.59
            sigma_gamma = 0.3 / sqrt(2) #0.261
            survey_area = 1437.  # 1321+116
            fsky = survey_area / 41253.

            su = Survey(n_src_arcmin=n_src_arcmin, sigma_gamma=sigma_gamma)
            cp = DemoDESY1(co=co, su=su, sr=sr, zh_min=zh_min, zh_max=zh_max, lambda_min=lambda_min, lambda_max=lambda_max, rp_min_hiMpc=rp_min_hiMpc, rp_max_hiMpc=rp_max_hiMpc, n_rp=n_rp, output_loc=output_loc)
            if False: #os.path.exists(cp.cov_combined_fname) == True:
                print('done')
            else:
                print('doing', cp.cov_combined_fname)
                cp.calc_cov_full(diag_only=False) # takes some time
                cp.calc_counts()
            