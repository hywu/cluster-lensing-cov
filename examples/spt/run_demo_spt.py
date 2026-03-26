#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('MNRAS')
import os, sys

## read Sebastian's data
import numpy as np
import matplotlib.pyplot as plt
import h5py
fname = 'data/cosmo_products_SG(4).h5'
f = h5py.File(fname,'r')
#print(f.keys())
#cali = f['calibrations']
#print(cali.keys())
data_lensing = f['data']
#print(data_lensing.keys())
## source distributions
zs_centers = data_lensing['zs_centers'][:]
P_zs = data_lensing['P_zs']

#print(data_lensing['lbins'][:])
#print(data_lensing['zbins'][:])


## get the mean mass and bias for a given bin
file_zhou = np.load("data/fid_bias_ab.npz")
#mass = file_zhou["mass"]
bias = file_zhou["bias"]
#abun = file_zhou["ab"] # incorrect!!
#areaspt = file_zhou["areaspt"]
#areanospt = file_zhou["areanospt"]
#zmid = 0.5*(file_zhou["z_low"] + file_zhou["z_high"])
#print(areaspt, "sq deg")
#print(areanospt, "sq deg")

#abun_spt = abun[:32].reshape(4, 4, 2) # inside SPT
bias_spt = bias[:32].reshape(4, 4, 2) 

#abun_nospt = abun[-16:].reshape(4, 4) # outside SPT
bias_nospt = bias[-16:].reshape(4, 4)
#print(abun_spt)


#### for covariance matrix ####
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.cosmology import FlatLambdaCDM
from clens.util.parameters import CosmoParameters
from clens.util.scaling_relation import PrecalculatedCountsBias
from clens.util.survey import Survey
from clens.lensing.cov_DeltaSigma import CovDeltaSigma
co = CosmoParameters(h=0.7, OmegaDE=0.7, OmegaM=0.3, sigma8=0.8)


#### All DES, one z bin, one lambda bin #### 
#iz = 0
#ilam = 0
#ixi = 0 # 0: SPT-non-detected
#ixi = 1 # 1: SPT-detected
#ixi = -1 # -1: not in SPT
#ixi = -2 # -2: entire DES # Not used by Zhou

area_total = 3567 + 576  ## CHECK!



z_bins = [0.2,  0.35, 0.5,  0.65, 0.85]
lam_bins = [ 20.,  35.,  45.,  60., 300.]
nz = len(z_bins) - 1
nlam = len(lam_bins) - 1


import fitsio
fname = 'data/redmapperY3-SPTconf.fits'
data, header = fitsio.read(fname, header=True)
#print(header)
lam = data['LAMBDA_CHISQ']
ra = data['RA']
dec = data['DEC']
z = data['Z']
gf = data['SPT-FIELD-DEPTH']
xi = data['SPT-SIGNIF']

def calc_cov_one_bin(iz, ilam, ixi):
    # zmin = data_lensing['zbins'][iz]
    # zmax = data_lensing['zbins'][iz+1]
    # lammin = data_lensing['lbins'][ilam]
    # lammax = data_lensing['lbins'][ilam+1]

    z_min = z_bins[iz]
    z_max = z_bins[iz+1]
    lam_min = lam_bins[ilam]
    lam_max = lam_bins[ilam+1]
    sel = (z >= z_min)&(z < z_max)&(lam >= lam_min)&(lam < lam_max)

    if ixi == -2: #all
        sel2 = (gf > -1)
    if ixi == -1: # outside SPT
        sel2 = (gf < 0.5)
    if ixi == 0: # inside SPT, no detection
        sel2 = (gf > 0.5)&(xi < 5)
    if ixi == 1: # inside SPT, has detection
        sel2 = (gf > 0.5)&(xi > 5)

    counts = len(lam[sel&sel2])

    zmean = 0.5 * (z_min + z_max)

    #### get the counts and bias first ##### 
    if ixi == -2: # entire DES footprint
        theta = data_lensing['theta_arcmin'][iz, ilam]
        gt = data_lensing['gt'][iz, ilam]
        err_gt = data_lensing['err_gt'][iz, ilam]
        area = area_total * 1.
        '''
        area = 3567 + 576  ## CHECK!
        
        #### read redmapper cluster and find counts ####
        import fitsio
        fname = 'redmapperY3-SPTconf.fits'
        data_rm, header = fitsio.read(fname, header=True)
        #print(header)
        lam_all = data_rm['LAMBDA_CHISQ']
        z_all = data_rm['Z']    
        sel = (z_all >= zmin)&(z_all < zmax)&(lam_all >= lammin)&(lam_all < lammax)
        lam = lam_all[sel]
        z = z_all[sel]
        #zmean = 0.264 ## This seems to work np.mean(z)
        zmean = np.mean(z)
        counts = len(lam)
        '''
        bias = 5 # TODO
        
    else:
        if ixi == -1:  # outside SPT
            name = 'nospt'
            #counts = abun_nospt[iz, ilam] # incorrect 
            bias = bias_nospt[iz, ilam]
            #area = file_zhou["areanospt"] # incorrect
            area = area_total * 0.3

        if ixi == 0: # inside SPT, no detection 
            name = 'sptnodet'
            #counts = abun_spt[iz, ilam, ixi] # incorrect 
            bias = bias_spt[iz, ilam, ixi]
            #area = file_zhou["areaspt"] # incorrect
            area = area_total * 0.7

        if ixi == 1: # inside SPT, with detection
            name = 'sptdet'
            #counts = abun_spt[iz, ilam, ixi] # incorrect 
            bias = bias_spt[iz, ilam, ixi]
            #area = file_zhou["areaspt"] # incorrect
            area = area_total * 0.7


        theta = data_lensing[name]['theta_arcmin'][iz, ilam]
        gt = data_lensing[name]['gt'][iz, ilam]
        err_gt = data_lensing[name]['err_gt'][iz, ilam]
    
    
    #plt.loglog(theta, err_gt, 'o-')
    
    #### fit the source distribution
    zs = zs_centers
    Pz = P_zs[iz, ilam]
    normalization = np.trapz(Pz, x=zs)
    Pz = Pz / normalization
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.plot(zs, Pz)
    
    def chisq(para):
        z_star_src, m_src, beta_src = para 
        y = zs**m_src * np.exp(-(zs/z_star_src)**beta_src)
        norm = np.trapz(y, zs)
        y = y / norm
        return np.sum((y-Pz)**2)
    
    from scipy.optimize import minimize
    res = minimize(chisq, x0=(0.6, 2, 1.2), bounds=((0.3, 1),(0.5,3),(0.8,3)))
    #print(res.x)
    z_star_src, m_src, beta_src = res.x
    y = zs**m_src * np.exp(-(zs/z_star_src)**beta_src)
    norm = np.trapz(y, zs)
    y = y / norm
    
    plt.plot(zs, y, label=r'$z_*=%.3g, m=%.3g, \beta=%.3g$'%(z_star_src, m_src, beta_src))
    plt.legend(frameon=False, fontsize=14)
    plt.xlabel(r'$z$')
    plt.ylabel(r'$n(z)$ [normalized]')
    plt.title(r'$z^m exp(-(z/z_*)^\beta)$')
    #plt.ylim(0,1)

    #### get the radius ####
    #rbins = np.insert(np.logspace(np.log10(0.2), np.log10(15), num=15), 0, 0) # physical Mpc/h
    #print(rbins) # Sebastin's first bin is (0,0.2).
    lnrbin = np.linspace(np.log(0.2), np.log(15), num=15) # I use all log-spaced bins. the innermost bin is mismatched with his.
    dlnr = lnrbin[1] - lnrbin[0]
    lnrbin = np.append(lnrbin[0]-dlnr, lnrbin)
    rbins = np.exp(lnrbin)

    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100, Om0=0.3)
    DA = cosmo.angular_diameter_distance(zmean).value
    theta_bins_cov = rbins / DA / np.pi * 180. * 60. # arcmin
    # compare with Sebastian's
    #theta_mid_cov = np.sqrt(theta_bins_cov[:-1]*theta_bins_cov[1:])
    theta_mid_cov = 0.5*(theta_bins_cov[:-1]+theta_bins_cov[1:]) # this is closer to Sebastian's
    diff = (theta - theta_mid_cov)/theta
    #print('frac radius diff', diff)
    
    #### get the covariance matrix
    bias_fake = 0 # ignoring cluster clustering for now, to match Grandis Eq 23
    print('counts', counts)
    print('zmean', zmean)
    
    sr = PrecalculatedCountsBias(counts, bias_fake)
    n_src_arcmin = 3.8 #Gradis Fig 4 ##5.59 #TODO: check 6.28  # DES Y3 survey condition
    sigma_gamma = 0.2635 + 0.0300 * zmean**2 - 0.0008 * zmean #Grandis Eq 18
    su = Survey(z_star_src=z_star_src, m_src=m_src, beta_src=beta_src, n_src_arcmin=n_src_arcmin, sigma_gamma=sigma_gamma)#, zs_min=zs_min, zs_max=zs_max)
    from demo_spt_gammat import DemoSPT
    
    fsky = area/41253.
    theta_edges = theta_bins_cov#[1:] 
    demo1 = DemoSPT(co, su, sr, z_min, z_max, iz, ilam, ixi, theta_edges, fsky, output_loc='temp1/')
    demo1.calc_cov_full(diag_only=True)
    plt.subplot(222)
    plt.plot(theta, err_gt, 'o-', label='Sebastian')
    cov_shape = np.loadtxt(f'temp1/gammat_cov_shape_noise_z{iz}_lam{ilam}_xi{ixi}.dat')
    plt.plot(theta_mid_cov, np.sqrt(np.diag(cov_shape)), 'o-', label='shape noise')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\sigma(\gamma_{\rm t})$')
    plt.loglog()
    plt.legend()

    #### correction factor for the shape noise ####
    #### matching Sebastian's shape noise amplitude (he has no clustering)
    factor = np.mean((err_gt/ np.sqrt(np.diag(cov_shape)))[5::])
    print('shape noise scale factor', factor)
    np.savetxt(f'temp1/factor_z{iz}_lam{ilam}_xi{ixi}.dat', [factor])
    plt.title(f'fac={factor:.2g}, counts={counts}')
    
    #### calculate the full covariance matrix ####
    sr = PrecalculatedCountsBias(counts, bias)
    demo2 = DemoSPT(co, su, sr, z_min, z_max, iz, ilam, ixi, theta_edges, fsky, output_loc='temp2/')
    demo2.calc_cov_full(diag_only=False)#diag_only=True)
    plt.subplot(223)
    plt.plot(theta, err_gt, 'o-', label='Sebastian')
    cov_combined = np.loadtxt(f'temp2/gammat_cov_combined_z{iz}_lam{ilam}_xi{ixi}.dat')
    factor = np.loadtxt(f'temp1/factor_z{iz}_lam{ilam}_xi{ixi}.dat') 
    
    cov_corrected = cov_combined * factor**2
    cov_corrected_fname = f'output/gammat_cov_corrected_z{iz}_lam{ilam}_xi{ixi}.dat'
    np.savetxt(cov_corrected_fname, cov_corrected, fmt='%-12g')
    plt.plot(theta_mid_cov, np.sqrt(np.diag(cov_corrected)), 'o-', label='Heidi')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\sigma(\gamma_{\rm t})$')
    plt.title(f'iz{iz} ilam{ilam} ixi{ixi}')
    plt.loglog()
    plt.legend()

    info_fname = f'output/info_z{iz}_lam{ilam}_xi{ixi}.dat'
    info = [z_min, z_max, ilam, ixi, counts, bias]
    np.savetxt(info_fname, info, fmt='%-12g', header='zmin, zmax, ilam, ixi, counts, bias')

    theta_fname = f'output/theta_z{iz}_lam{ilam}_xi{ixi}.dat'
    np.savetxt(theta_fname, theta_edges, fmt='%-12g', header='theta edges in arcmin')

    plt.subplot(224)
    plt.imshow(np.log(cov_corrected))
    
    plt.savefig(f'plots/cov_z{iz}_lam{ilam}_xi{ixi}.pdf')
    
for iz in range(4): # 4 
    for ilam in range(4): # 4
        for ixi in [-2, -1, 0, 1]: #range(1): # -1, 0, 1
            calc_cov_one_bin(iz=iz, ilam=ilam, ixi=ixi)