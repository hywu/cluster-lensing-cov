#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

import clens.util.constants as cn
from clens.util.parameters import CosmoParameters, NuisanceParameters
from clens.util.survey import Survey
from clens.lensing.cov_DeltaSigma import CovDeltaSigma
from clens.lensing.angular_power_spectra import AngularPowerSpectra

z = np.arange(0,1.6,0.01)
cosmo = FlatLambdaCDM(H0=67.3, Om0=0.314)
chi = cosmo.comoving_distance(z).value
z_chi_interp = interp1d(chi, z)
chi_z_interp = interp1d(z, chi)

from abacus_theory_counts_lensing import AbacusTheoryCountsLensing

'''
* prerequisite: 
    run demo_analytic.py
    run abacus 
    copy abacus to this repo

* grafting them together
* plotting the matrix
'''

class AbacusAnalyticGraftingInterp(object):
    def __init__(self, co, nu, su, zh_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp, boxsize, scatter):
        self.co = co
        self.nu = nu
        self.su = su
        self.zh_mid = zh_mid
        self.zs_mid = 0.5 * (self.su.zs_min+self.su.zs_max)

        self.Mmin_hiMsun = Mmin_hiMsun
        self.Mmax_hiMsun = Mmax_hiMsun
        self.rp_min_hiMpc = rp_min_hiMpc
        self.rp_max_hiMpc = rp_max_hiMpc
        self.n_rp = n_rp
        self.boxsize = boxsize
        self.scatter = scatter

        
        # astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        # self.chi = astropy_dist.comoving_distance

        ## output of analytic
        m1, x = np.modf(Mmin_hiMsun/(1e+14))
        m2, x = np.modf(Mmax_hiMsun/(1e+14))
        if m1 > 1e-4 or m2 > 1e-4:
            self.output_loc = '../output/analytic_abacus_scatter%g/zh%g_zs%g/%.2e_%.2e_R%g_%g_nrp%i/'%(scatter, zh_mid, self.zs_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
        else:
            self.output_loc = '../output/analytic_abacus_scatter%g/zh%g_zs%g/%.e_%.e_R%g_%g_nrp%i/'%(scatter, zh_mid, self.zs_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
        print(self.output_loc)
        #if os.path.isdir(self.output_loc)==False: os.makedirs(self.output_loc)
        
        
        ## radial bins
        lnrp_list = np.linspace(np.log(rp_min_hiMpc), np.log(rp_max_hiMpc), n_rp+1)
        rp_list = np.exp(lnrp_list)
        rpmin_list = rp_list[:-1]
        rpmax_list = rp_list[1:]
        rpmid_list = np.sqrt(rpmin_list*rpmax_list)

        self.rp_fname = self.output_loc+'rp_hiMpc.dat'
        if os.path.exists(self.rp_fname)==False:
            outfile = open(self.rp_fname, 'w')
            outfile.write('#rp_min[Mpc/h], rp_max[Mpc/h], rp_mid[Mpc/h] \n')
            for i in range(n_rp):
                outfile.write('%12g %12g %12g \n'%(rpmin_list[i], rpmax_list[i], rpmid_list[i]))
            outfile.close()
        
        self.cov_cosmic_shear_fname = self.output_loc+'DeltaSigma_cov_cosmic_shear.dat'
        self.cov_shape_noise_fname = self.output_loc+'DeltaSigma_cov_shape_noise.dat'

        self.cov_cosmic_shear_slice_fname = self.output_loc+'DeltaSigma_cov_cosmic_shear_slice.dat'
        self.cov_halo_intrinsic_slice_fname = self.output_loc+'DeltaSigma_cov_halo_intrinsic_slice.dat'

        self.cov_combined_fname = self.output_loc+'DeltaSigma_cov_combined.dat' ## key results!

        self.cov_cosmic_shear_no_shot_fname = self.output_loc+'DeltaSigma_cov_cosmic_shear_no_shot.dat'##needed for cross-mass bin


        #### get abacus 

        ## where I store abacus output
        ## find the closest redshift
        zh_abacus_list = np.array([0.3, 0.5, 0.7, 1.0])
        if Mmin_hiMsun > 3.99e+14:
            zh_abacus_list = np.array([0.3, 0.5, 0.7])


        diff = abs(zh_abacus_list - self.zh_mid)
        #### this is totally horrible ####
        if min(diff) < 0.01:
            m1, x = np.modf(Mmin_hiMsun/(1e+14))
            m2, x = np.modf(Mmax_hiMsun/(1e+14))
            if m1 > 1e-4 or m2 > 1e-4:
                self.abacus_loc = '../data/abacus_scatter%g/%i_planck_10percent/z%g/%.2e_%.2e_R%g_%g_nrp%i/'%(scatter, boxsize, self.zh_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
            else:
                self.abacus_loc = '../data/abacus_scatter%g/%i_planck_10percent/z%g/%.e_%.e_R%g_%g_nrp%i/'%(scatter, boxsize, self.zh_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
            self.cov_abacus = np.loadtxt(self.abacus_loc+'cov_DeltaSigma.dat') 
        elif self.zh_mid < 0.3:
            self.abacus_loc = '../data/abacus_scatter%g/%i_planck_10percent/z%g/%.e_%.e_R%g_%g_nrp%i/'%(scatter, boxsize, self.zh_mid+0.1, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
            self.cov_abacus = np.loadtxt(self.abacus_loc+'cov_DeltaSigma.dat')
        elif self.zh_mid < 0.7:
            self.abacus_loc = '../data/abacus_scatter%g/%i_planck_10percent/z%g/%.e_%.e_R%g_%g_nrp%i/'%(scatter, boxsize, self.zh_mid-0.1, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
            self.abacus_loc2 = '../data/abacus_scatter%g/%i_planck_10percent/z%g/%.e_%.e_R%g_%g_nrp%i/'%(scatter, boxsize, self.zh_mid+0.1, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
            cov_abacus1 = np.loadtxt(self.abacus_loc+'cov_DeltaSigma.dat')
            cov_abacus2 = np.loadtxt(self.abacus_loc2+'cov_DeltaSigma.dat')
            self.cov_abacus = 0.5 * (cov_abacus1 + cov_abacus2)
        else:
            zh_abacus = zh_abacus_list[diff==min(diff)]
            self.abacus_loc = '../data/abacus_scatter%g/%i_planck_10percent/z%g/%.e_%.e_R%g_%g_nrp%i/'%(scatter, boxsize, zh_abacus, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
            ### get the counts and lensing from theory
            ai_zmid = AbacusTheoryCountsLensing(co, nu, su, zh_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp, boxsize, scatter)
            ai_zmid.calc_N_DS()
            self.counts_zmid = ai_zmid.counts
            self.DeltaSigma_zmid = ai_zmid.DeltaSigma
            
            ai_abacus = AbacusTheoryCountsLensing(co, nu, su, zh_abacus, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp, boxsize, scatter)
            ai_abacus.calc_N_DS()
            self.counts_abacus = ai_abacus.counts
            self.DeltaSigma_abacus = ai_abacus.DeltaSigma

            cov_abacus_orig = np.loadtxt(self.abacus_loc+'cov_DeltaSigma.dat') 
            ### abacus #####

        
            # for z < 0.7 the small scale looks almost constant.
            ## need to scale: N cov / (DeltaSigma)^2 is constant
            ## do it radius by radius
            self.cov_abacus = np.zeros([self.n_rp, self.n_rp])
            for irp in range(self.n_rp):
                for jrp in range(self.n_rp):
                    redshift_scaling = (self.counts_abacus/self.counts_zmid) * (self.DeltaSigma_zmid[irp]/self.DeltaSigma_abacus[irp]) * (self.DeltaSigma_zmid[jrp]/self.DeltaSigma_abacus[jrp])
                    #print('redshift_scaling', redshift_scaling)
                    self.cov_abacus[irp, jrp] = cov_abacus_orig[irp, jrp] * redshift_scaling





    def grafting(self):
        cov_theory = np.loadtxt(self.cov_cosmic_shear_fname)
        cov_slice = np.loadtxt(self.cov_cosmic_shear_slice_fname)
        #+np.loadtxt(self.cov_halo_intrinsic_slice_fname))
        self.cov_theory_excised = cov_theory - cov_slice

        rpmid_list_check, DS_mean = np.loadtxt(self.abacus_loc+'mean_DeltaSigma.dat', unpack=True)
        rpmin_list, rpmax_list, rpmid_list = np.loadtxt(self.rp_fname, unpack=True)
        if max(abs(rpmid_list-rpmid_list_check)/rpmid_list) > 0.01:
            print('rp mismatch!!')
        





        cov_combined = self.cov_abacus + self.cov_theory_excised
        np.savetxt(self.cov_combined_fname, cov_combined) ## key results!
        print('saved', self.cov_combined_fname)

        # plt.figure()
        # plt.loglog(rpmid_list, cov_abacus.diagonal(), label='abacus')
        # plt.loglog(rpmid_list, self.cov_theory_excised.diagonal(), label='theory')
        # plt.loglog(rpmid_list, cov_combined.diagonal(), label='combined', c='k')
        # plt.loglog(rpmid_list, cov_theory.diagonal(), c='gray', ls='--', label='theory-full')
        # plt.loglog(rpmid_list, cov_slice.diagonal(), c='gray', ls=':', label='theory-slice')
        # plt.title(r'$\rm\rm z_l=%g, z_s=%g, %.e < M < %.e$'%(self.zh_mid, self.zs_mid, self.Mmin_hiMsun, self.Mmax_hiMsun))
        # plt.savefig('../temp_plots/grafting_zh%g_zs%g_%.e_%.e_R%g_%g_nrp%i.png'%(self.zh_mid, self.zs_mid, self.Mmin_hiMsun, self.Mmax_hiMsun, self.rp_min_hiMpc, self.rp_max_hiMpc, self.n_rp))  ## for eyeballing bugs

    def plot_abacus_plus_analytic(self):

        plt.figure(figsize=(14,7))
        rpmin_list, rpmax_list, rpmid_list = np.loadtxt(self.rp_fname, unpack=True)
        cov_combined = np.loadtxt(self.cov_combined_fname)

        nbins = len(rpmid_list)
        ax = plt.subplot(121)
        plt.imshow(np.log10(cov_combined))
        plt.colorbar()
        tick_locs = np.arange(0,nbins,5)
        tick_lbls = rpmid_list[0:-1:5]
        
        ax.set_xticks(tick_locs)
        ax.set_yticks(tick_locs)
        ax.set_xticklabels(["$%.2g$" % x for x in tick_lbls]) # use LaTeX #, fontsize=18
        ax.set_yticklabels(["$%.2g$" % x for x in tick_lbls]) # use LaTeX #, fontsize=18
        plt.title(r'$\rm log_{10} Cov[\Delta\Sigma(r_p), \Delta\Sigma(r_p^{\prime})]$')
        plt.xlabel(r'$\rm r_p^{\prime}\ [Mpc/h]$')
        plt.ylabel(r'$\rm r_p\ [Mpc/h]$')

        ax = plt.subplot(122)
        nbins = len(rpmid_list)
        print('nbins', nbins)
        diag = np.zeros(nbins)
        for i in range(nbins): 
            diag[i] = cov_combined[i,i]

        for dist_from_diag in range(6):
            slanted = np.zeros(nbins-dist_from_diag)
            for i in range(nbins-dist_from_diag):
                slanted[i] = cov_combined[i, i+dist_from_diag]
            if dist_from_diag==0:
                plt.loglog(rpmid_list[:nbins-dist_from_diag], slanted, label='Diagonal')
            else:
                plt.loglog(rpmid_list[:nbins-dist_from_diag], slanted, label='offset=%i'%(dist_from_diag))

        plt.legend()
        plt.xlabel(r'$\rm r_p\ [Mpc/h]$')
        plt.ylabel(r'$\rm Cov[\Delta\Sigma(r_p), \Delta\Sigma(r_p^\prime)]\ [M_\odot^2/pc^4]$')
        plt.title(r'$\rm\rm z_l=%g, z_s=%g, %.e < M < %.e$'%(self.zh_mid, self.zs_mid, self.Mmin_hiMsun, self.Mmax_hiMsun))

# ./abacus_analytic_grafting.py 0.5 1.25 1e+14 1e16 15 720

# zh_mid = float(sys.argv[1])
# zs_mid = float(sys.argv[2]) 
# Mmin_hiMsun = float(sys.argv[3])
# Mmax_hiMsun = float(sys.argv[4]) 
# n_rp = int(sys.argv[5])


if __name__ == "__main__":
    co = CosmoParameters(h=0.673, OmegaDE=0.686, OmegaM=0.314, sigma8=0.83) # Abacus cosmology
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1) # 1-1,no scatter

    boxsize = 720
    #Mlim_list = [(1e+14, 1e+16)]#, (2e+14, 1e+16)]#, (4e+14, 1e+16)]

    #Mlim_list = [(1e+14, 1e+16), (2e+14, 1e+16), (4e+14, 1e+16), (1e+14, 2e+14), (2e+14, 4e+14), (1e+14, 4e+14), (1e+14, 8e+14), (8e+14, 1e+16)]

    Mlim_list = []
    #for Msplit in [2**0.25, 2**0.5, 2**0.75, 2**1.5]: #
    for Msplit in 2**np.arange(0.5,3.1,0.5):

        Mlim_list.append((1e+14, Msplit*1e+14))
        Mlim_list.append((Msplit*1e+14, 1e+16))
        print("Msplit={:.2e}".format(Msplit))

    for scatter in [0]:#, 0.2, 0.4]: # 0 
        for zh_mid in [0.3]:#[0.5]:#np.arange(0.8, 1.11, 0.1):#[0.2]:#[0.3, 0.5, 0.7]:#[0.3, 0.5, 0.7]: #
            for (Mmin_hiMsun, Mmax_hiMsun) in Mlim_list:
                for zs_mid in [zh_mid*2.5]:#zh_mid*1.5, zh_mid*2, zh_mid*2.5, zh_mid*3]:
                    for n_rp in [30]:#[15]:#, 30]: #
                        su = Survey(top_hat=True, zs_min=zs_mid-0.05, zs_max=zs_mid+0.05, n_src_arcmin=10)

                        aag = AbacusAnalyticGraftingInterp(co=co, nu=nu, su=su, zh_mid=zh_mid, Mmin_hiMsun=Mmin_hiMsun, Mmax_hiMsun=Mmax_hiMsun, rp_min_hiMpc=0.1, rp_max_hiMpc=100., n_rp=n_rp, boxsize=boxsize, scatter=scatter)
                        #abacus_fname = aag.abacus_loc+'mean_DeltaSigma.dat'
                        theory_fname = aag.cov_cosmic_shear_slice_fname
                        if os.path.exists(theory_fname):  #os.path.exists(abacus_fname) and 
                            #pass
                            
                            aag.grafting()

                            #aag.plot_abacus_plus_analytic()
                            
                            # copy over the mean
                            # cmd = 'cp '+aag.abacus_loc+'mean_DeltaSigma.dat '+aag.output_loc+'mean_DeltaSigma.dat'
                            # os.system(cmd)
                            
                        else:
                            print('cannot do'+abacus_fname)


    #plt.show()

        
