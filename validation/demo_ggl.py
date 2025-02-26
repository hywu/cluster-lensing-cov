#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.cosmology import FlatLambdaCDM
from clens.util.parameters import CosmoParameters
from clens.util.scaling_relation import PrecalculatedCountsBias
from clens.util.survey import Survey
from clens.lensing.cov_DeltaSigma import CovDeltaSigma

class DemoGGL(object):
    """
    calculating covariance matrix at the given lens redshift, source redshift, radial range,
    taking in lens number counts and bias 
    """
    def __init__(self, co, su, sr, zh_min, zh_max, rp_min_hiMpc, rp_max_hiMpc, n_rp, output_loc):
        self.co = co
        self.su = su
        self.sr = sr
        self.zs_mid = 0.5 * (self.su.zs_min+self.su.zs_max)

        self.zh_min = zh_min
        self.zh_max = zh_max
        self.zh_mid = 0.5 * (zh_min + zh_min)

        self.rp_min = rp_min_hiMpc/co.h
        self.rp_max = rp_max_hiMpc/co.h
        self.n_rp = n_rp

        #self.chi = astropy_dist.comoving_distance

        self.output_loc = output_loc
        if os.path.isdir(self.output_loc)==False: os.makedirs(self.output_loc)
        print('output', self.output_loc)

        bin_name = f'{zh_min}_{zh_max}'

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
        output = cds.calc_cov(zh_min=self.zh_min, zh_max=self.zh_max, rp_min=self.rp_min, rp_max=self.rp_max, n_rp=self.n_rp, diag_only=diag_only)
        rp_mid, cov_cosmic_shear, cov_shape_noise = output

        np.savetxt(self.cov_cosmic_shear_fname, cov_cosmic_shear/self.co.h**2) # h^2Msun^2/pc^4
        np.savetxt(self.cov_shape_noise_fname, cov_shape_noise/self.co.h**2)
        cov_combined = cov_cosmic_shear/self.co.h**2 + cov_shape_noise/self.co.h**2
        np.savetxt(self.cov_combined_fname, cov_combined) ## key results!


if __name__ == "__main__":
    co = CosmoParameters(h=0.673, OmegaDE=0.684, OmegaM=0.316, sigma8=0.812)
    astropy_dist = FlatLambdaCDM(H0=co.h*100, Om0=co.OmegaM)
    '''
    zh_min = 0.2
    zh_max = 0.35
    density_hiMpc3 = 0.0018261871
    vol_survey_Mpc3 = 1000**3 # no h
    counts = density_hiMpc3 * vol_survey_Mpc3 * co.h**3
    bias = 1.45827

    sr = PrecalculatedCountsBias(counts, bias)
    output_loc='ggl_analytic_counts_bias/'

    zh_mid = 0.5 * (zh_min + zh_max)
    a = 1/(1 + zh_mid)
    rp_min_hiMpc = 0.03 / a * co.h
    rp_max_hiMpc = 30. / a * co.h
    n_rp = 15 
    n_src_arcmin = 6.28  # DES survey condition
    sigma_gamma = 0.3 # DES survey condition

    astropy_dist = FlatLambdaCDM(H0=co.h*100, Om0=co.OmegaM)
    vol_fullsky = astropy_dist.comoving_volume(zh_max) - astropy_dist.comoving_volume(zh_min) # no h
    fsky = vol_survey_Mpc3 / vol_fullsky.value 
    survey_area = 41253 * fsky
    print('fsky=', fsky, 'area=', survey_area)
    '''

    #### compare with Singh 2017 (Sec 3.2) SDSS LOWZ
    zh_min = 0.16
    zh_max = 0.36
    #density_hiMpc3 = 3e-4
    
    bias = 1.77 # TODO  Singh 2015 (IA paper)

    zh_mid = 0.5 * (zh_min + zh_max)
    a = 1/(1 + zh_mid)
    rp_min_hiMpc = 1.
    rp_max_hiMpc = 100.
    n_rp = 16 ## Singh 17 Fig 1
    n_src_arcmin = 1.18 # Reyes 12 Table 1   ## TODO: is this the same as 8 h2Mpc-2 ?
    sigma_gamma = 0.36 # Reyes 12 Table 1# divided by 2R ?? no way.
    fsky = 7131./ 41253. # Reyes 12  Table 1
    vol_fullsky = astropy_dist.comoving_volume(zh_max) - astropy_dist.comoving_volume(zh_min) # no h
    counts = 225181 # from Singh

    zs_min = 0.4 # Reyes 12 Fig 3.
    zs_max = 0.42

    sr = PrecalculatedCountsBias(counts, bias)
    output_loc='ggl_analytic_Singh17/'


    su = Survey(n_src_arcmin=n_src_arcmin, sigma_gamma=sigma_gamma, zs_min=zs_min, zs_max=zs_max)
    cp = DemoGGL(co=co, su=su, sr=sr, zh_min=zh_min, zh_max=zh_max, rp_min_hiMpc=rp_min_hiMpc, rp_max_hiMpc=rp_max_hiMpc, n_rp=n_rp, output_loc=output_loc)

    cp.calc_cov_full(diag_only=True) # takes some time
            