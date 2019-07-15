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

z = np.arange(0,1.5,0.01)
cosmo = FlatLambdaCDM(H0=67.3, Om0=0.314)
chi = cosmo.comoving_distance(z).value
z_chi_interp = interp1d(chi, z)
chi_z_interp = interp1d(z, chi)

'''
calculating covariance matrix at the given lens redshift, source redshift, mass range, radial range
using a thin redshift slice 
normalized the volume to 1 (Gpc/h)^3
'''

class DemoAnalytic(object):
    def __init__(self, co, nu, su, zh_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp):
        self.co = co
        self.nu = nu
        self.su = su
        self.zh_mid = zh_mid
        self.zs_mid = 0.5 * (self.su.zs_min+self.su.zs_max)

        self.Mmin_hiMsun = Mmin_hiMsun
        self.Mmax_hiMsun = Mmax_hiMsun
        self.Mmin = Mmin_hiMsun/co.h
        self.Mmax = Mmax_hiMsun/co.h
        self.rp_min = rp_min_hiMpc/co.h
        self.rp_max = rp_max_hiMpc/co.h
        self.n_rp = n_rp
        self.scatter = self.nu.sigma_lambda

        astropy_dist = FlatLambdaCDM(H0=self.co.h*100, Om0=self.co.OmegaM)
        self.chi = astropy_dist.comoving_distance

        if self.scatter < 0.01:
            self.output_loc = '../output/analytic_abacus_scatter%s/zh%g_zs%g/%.e_%.e_R%g_%g_nrp%i/'%(0, zh_mid, self.zs_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)
        else:
            self.output_loc = '../output/analytic_abacus_scatter%s/zh%g_zs%g/%.e_%.e_R%g_%g_nrp%i/'%(self.scatter, zh_mid, self.zs_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)

        if os.path.isdir(self.output_loc)==False: os.makedirs(self.output_loc)
        print('output', self.output_loc)

        #### radial bins
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
        
        ## output of this code is saved here
        self.cov_cosmic_shear_fname = self.output_loc+'DeltaSigma_cov_cosmic_shear.dat'
        self.cov_shape_noise_fname = self.output_loc+'DeltaSigma_cov_shape_noise.dat'
        self.cov_halo_intrinsic_fname = self.output_loc+'DeltaSigma_cov_halo_intrinsic.dat'# not useful

        self.cov_cosmic_shear_slice_fname = self.output_loc+'DeltaSigma_cov_cosmic_shear_slice.dat'
        self.cov_halo_intrinsic_slice_fname = self.output_loc+'DeltaSigma_cov_halo_intrinsic_slice.dat'

        self.cov_combined_fname = self.output_loc+'DeltaSigma_cov_combined.dat' ## key results!

        self.cov_cosmic_shear_no_shot_fname = self.output_loc+'DeltaSigma_cov_cosmic_shear_no_shot.dat'##needed for cross-mass bin

        ## redshift slice for calculating C_ell's
        self.fsky = 1
        self.zh_min = self.zh_mid - 0.05
        self.zh_max = self.zh_mid + 0.05
        self.vol_hiGpc3 = (astropy_dist.comoving_volume(self.zh_max).value - astropy_dist.comoving_volume(self.zh_min).value) * self.co.h**3 * 1e-9
        print('self.vol_hiGpc3 ', self.vol_hiGpc3 )

        '''
        ## where I store abacus output
        self.abacus_loc = '../data/abacus_cov/z%g/%.e_%.e_R%g_%g_nrp%i/'%(zh_mid, Mmin_hiMsun, Mmax_hiMsun, rp_min_hiMpc, rp_max_hiMpc, n_rp)

        ## we used a prism of 1100*1100*1100/25 (Mpc/h)^3
        ## if we assume it's fsky = 1/48. calculate how thick it is.
        ## calculating the correponding zh_min and zh_max
        chi_l = self.chi(z=zh_mid).value # Mpc
        self.fsky = 1./48.
        Vsim = (1100.**3)/25./self.co.h**3 ## Mpc^3
        dchi = Vsim/(4.*np.pi*self.fsky*chi_l**2) ## Mpc
        chi_max = chi_l + 0.5*dchi
        chi_min = chi_l - 0.5*dchi
        self.zh_min = z_chi_interp(chi_min)
        self.zh_max = z_chi_interp(chi_max)
        #print('zh integration range for abacus', self.zh_min, self.zh_max)
        '''
        ### what is the slice we need?
        chi_l = self.chi(z=zh_mid).value # Mpc
        pimax = 100./self.co.h # Mpc, the thickness used when measusring DeltaSigma in abacus  
        zmin = z_chi_interp(chi_l-pimax)
        zmax = z_chi_interp(chi_l+pimax)
        self.dz_slicing = (zmax-zmin)/2.
        print('self.zh_min = ', self.zh_min)
        print('self.zh_max = ', self.zh_max)
        print('self.dz_slicing = ', self.dz_slicing)
        
    def calc_cov_full(self):
        cds = CovDeltaSigma(co=self.co, nu=self.nu, su=self.su, fsky=self.fsky)
        output = cds.calc_cov(Mmin=self.Mmin, Mmax=self.Mmax, zh_min=self.zh_min, zh_max=self.zh_max, rp_min=self.rp_min, rp_max=self.rp_max, n_rp=self.n_rp, diag_only=False)
        rp_mid, cov_cosmic_shear, cov_shape_noise, cov_halo_intrinsic = output

        ## make it 1 (Gpc/h)^3
        cov_cosmic_shear *= self.vol_hiGpc3
        cov_shape_noise *= self.vol_hiGpc3
        cov_halo_intrinsic *= self.vol_hiGpc3

        np.savetxt(self.cov_cosmic_shear_fname, cov_cosmic_shear/self.co.h**2) # h^2Msun^2/Mpc^4
        np.savetxt(self.cov_shape_noise_fname, cov_shape_noise/self.co.h**2)
        np.savetxt(self.cov_halo_intrinsic_fname, cov_halo_intrinsic/self.co.h**2)

    def calc_cov_full_no_shot_noise(self): #### needed for cross-variance
        cds = CovDeltaSigma(co=self.co, nu=self.nu, su=self.su, fsky=self.fsky, cosmic_shear_no_shot=True)
        output = cds.calc_cov(Mmin=self.Mmin, Mmax=self.Mmax, zh_min=self.zh_min, zh_max=self.zh_max, rp_min=self.rp_min, rp_max=self.rp_max, n_rp=self.n_rp, diag_only=False)
        rp_mid, cov_cosmic_shear, cov_shape_noise, cov_halo_intrinsic = output

        ## make it 1 (Gpc/h)^3
        cov_cosmic_shear *= self.vol_hiGpc3
        cov_shape_noise *= self.vol_hiGpc3
        cov_halo_intrinsic *= self.vol_hiGpc3

        np.savetxt(self.cov_cosmic_shear_no_shot_fname, cov_cosmic_shear/self.co.h**2) # h^2Msun^2/Mpc^4

    def calc_cov_slice(self):
        cds = CovDeltaSigma(co=self.co, nu=self.nu, su=self.su, fsky=self.fsky, slicing=True, dz_slicing=self.dz_slicing)
        output = cds.calc_cov(Mmin=self.Mmin, Mmax=self.Mmax, zh_min=self.zh_min, zh_max=self.zh_max, rp_min=self.rp_min, rp_max=self.rp_max, n_rp=self.n_rp, diag_only=False)
        rp_mid, cov_cosmic_shear, cov_shape_noise, cov_halo_intrinsic = output

        ## make it 1 (Gpc/h)^3
        cov_cosmic_shear *= self.vol_hiGpc3
        cov_shape_noise *= self.vol_hiGpc3
        cov_halo_intrinsic *= self.vol_hiGpc3

        np.savetxt(self.cov_cosmic_shear_slice_fname, cov_cosmic_shear/self.co.h**2)
        np.savetxt(self.cov_halo_intrinsic_slice_fname, cov_halo_intrinsic/self.co.h**2)

    def plot_analytic(self):

        plt.figure(figsize=(14,7))
        rpmin_list, rpmax_list, rpmid_list = np.loadtxt(self.rp_fname, unpack=True)
        cov_cosmic_shear = np.loadtxt(self.cov_cosmic_shear_fname)

        plt.subplot(121)
        plt.loglog(rpmid_list, cov_cosmic_shear.diagonal())
        plt.show()

    '''
    def grafting(self):
        cov_theory = np.loadtxt(self.cov_cosmic_shear_fname)
        cov_slice = (np.loadtxt(self.cov_cosmic_shear_slice_fname)+np.loadtxt(self.cov_halo_intrinsic_slice_fname))
        self.cov_theory_excised = cov_theory - cov_slice

        rpmid_list_check, DS_mean = np.loadtxt(self.abacus_loc+'mean_DeltaSigma.dat', unpack=True)
        rpmin_list, rpmax_list, rpmid_list = np.loadtxt(self.rp_fname, unpack=True)
        if max(abs(rpmid_list-rpmid_list_check)/rpmid_list) > 0.01:
            print('rp mismatch!!')
        
        cov_abacus = np.loadtxt(self.abacus_loc+'cov_DeltaSigma.dat') 

        cov_combined = cov_abacus + self.cov_theory_excised
        np.savetxt(self.cov_combined_fname, cov_combined) ## key results!

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
'''




    # cp.calc_cov_slice() # takes some time
    # 
    #cp.grafting() # only when the analytic is done
    

    
    #cp.plot_abacus_plus_analytic()
    
    #
    #cp.calc_cov_full_no_shot_noise() ## for cross covariance

if __name__ == "__main__":
    ## ./demo_analytic.py 0.5 1.25 1e+14 1e16 2 0 #0.4
    zh_mid = float(sys.argv[1])
    zs_mid = float(sys.argv[2]) 
    Mmin_hiMsun = float(sys.argv[3])
    Mmax_hiMsun = float(sys.argv[4]) 
    n_rp = int(sys.argv[5])
    scatter = float(sys.argv[6])


    if scatter==0:
        Mmin_obs = Mmin_hiMsun
    else:
        ## NEW Mobs threshold with scatter
        boxsize = 720
        den, M_lim_noscat, Mobs_lim_scat = np.loadtxt('../data/abacus_scatter%g/%i_planck_10percent/z%g/Mobs_threshold.dat'%(scatter,boxsize,zh_mid), unpack=True)
        select = (M_lim_noscat == Mmin_hiMsun)
        Mmin_obs = Mobs_lim_scat[select]

        


    co = CosmoParameters(h=0.673, OmegaDE=0.686, OmegaM=0.314, sigma8=0.83) # Abacus cosmology
    if scatter==0:
        nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1) # 1-1,no scatter
    else:
        nu = NuisanceParameters(sigma_lambda=scatter, lgM0=0, alpha_M=1, lambda0=1) 

    su = Survey(top_hat=True, zs_min=zs_mid-0.05, zs_max=zs_mid+0.05, n_src_arcmin=10)

    cp = DemoAnalytic(co=co, nu=nu, su=su, zh_mid=zh_mid, Mmin_hiMsun=Mmin_obs, Mmax_hiMsun=Mmax_hiMsun, rp_min_hiMpc=0.1, rp_max_hiMpc=100., n_rp=n_rp)
    cp.calc_cov_full() # takes some time
    #cp.calc_cov_slice()# takes some time
    #cp.calc_cov_full_no_shot_noise()# takes some time
    #cp.plot_analytic()
