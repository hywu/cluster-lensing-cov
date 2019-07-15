#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc



class RichnessSelection(object):
    def __init__(self, scaling_relation, lambda_min, lambda_max):
        self.scaling_relation = scaling_relation
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        #print('scaling relation %g'%(scaling_relation.lgM0))

    def lnM_selection(self, lnM):
        lnlambda_mean = self.scaling_relation.lnlambda_lnM(lnM)
        sigma_lambda = self.scaling_relation.scatter(lnM)
        # print('sigma_lambda', sigma_lambda)
        # print('lambda_mean', np.exp(lnlambda_mean))

        x_lo = (np.log(self.lambda_min) - lnlambda_mean)/np.sqrt(2.)/sigma_lambda
        x_hi = (np.log(self.lambda_max) - lnlambda_mean)/np.sqrt(2.)/sigma_lambda
        return 0.5*erfc(x_lo) - 0.5*erfc(x_hi)


class FiducialScalingRelation(object): # Simet values
    def __init__(self, nuisance):# lgM0 = 14.34, alpha_M = 0.752, lambda0=40, scatter=0.5):
        self.lgM0 = nuisance.lgM0
        self.alpha_M = nuisance.alpha_M
        self.lambda0 = nuisance.lambda0
        self.sigma_lambda = nuisance.sigma_lambda

    def lnlambda_lnM(self, lnM):
        M = np.exp(lnM)
        return np.log(self.lambda0 * (M/10**self.lgM0)**self.alpha_M)

    def scatter(self, lnM):
        return self.sigma_lambda 
        

class PiecewiseScalingRelation(object):
    def __init__(self, nuisance):
    # norm_mid = -21.1415, alpha_M0 = 0.752, alpha_M1 = 0.752, alpha_M2 = 0.752, scatter0=0.5, scatter1=0.5, scatter2=0.5):

        self.norm_mid = nuisance.norm_mid
        self.alpha_M0 = nuisance.alpha_M0
        self.alpha_M1 = nuisance.alpha_M1
        self.alpha_M2 = nuisance.alpha_M2
        self.scatter0 = nuisance.scatter0
        self.scatter1 = nuisance.scatter2
        self.scatter2 = nuisance.scatter2

        self.lnM0 = np.log(1e14)
        self.lnM1 = np.log(2e14)
        self.lnM2 = np.log(4e14)

    def lnlambda_lnM(self, lnM):
        lnM0 = self.lnM0
        lnM1 = self.lnM1
        lnM2 = self.lnM2
        select0 = (lnM > -1)&(lnM < 0.5*(lnM0+lnM1))
        select1 = (lnM > 0.5*(lnM0+lnM1))&(lnM < 0.5*(lnM1+lnM2))
        select2 = (lnM > 0.5*(lnM1+lnM2))
        nM = len(lnM)
        alpha_M = np.zeros(nM)
        norm = np.zeros(nM)
        alpha_M[select0] = self.alpha_M0
        alpha_M[select1] = self.alpha_M1
        alpha_M[select2] = self.alpha_M2
        norm[select0] = (self.alpha_M1-self.alpha_M0)*0.5*(lnM0+lnM1) + self.norm_mid
        norm[select1] = self.norm_mid
        norm[select2] = (self.alpha_M1-self.alpha_M2)*0.5*(lnM1+lnM2) + self.norm_mid
        '''
        if lnM > -1              and lnM < 0.5*(lnM0+lnM1): 
            alpha_M = self.alpha_M0
            norm = (self.alpha_M1-self.alpha_M0)*0.5*(lnM0+lnM1) + self.norm_mid
        if lnM > 0.5*(lnM0+lnM1) and lnM < 0.5*(lnM1+lnM2): 
            alpha_M = self.alpha_M1
            norm = self.norm_mid
        if lnM > 0.5*(lnM1+lnM2): 
            alpha_M = self.alpha_M2
            norm = (self.alpha_M1-self.alpha_M2)*0.5*(lnM1+lnM2) + self.norm_mid
        '''
        return alpha_M * lnM + norm


    def scatter(self, lnM):
        lnM0 = self.lnM0
        lnM1 = self.lnM1
        lnM2 = self.lnM2
        select0 = (lnM > -1)&(lnM < 0.5*(lnM0+lnM1))
        select1 = (lnM > 0.5*(lnM0+lnM1))&(lnM < 0.5*(lnM1+lnM2))
        select2 = (lnM > 0.5*(lnM1+lnM2))

        scatter = np.zeros(len(lnM))
        scatter[select0] = self.scatter0
        scatter[select1] = self.scatter1
        scatter[select2] = self.scatter2
        '''
        # there must be a better way to do this
        if lnM > -1              and lnM < 0.5*(lnM0+lnM1): 
            return self.scatter0
        if lnM > 0.5*(lnM0+lnM1) and lnM < 0.5*(lnM1+lnM2): 
            return self.scatter1
        if lnM > 0.5*(lnM1+lnM2): 
            return self.scatter2
        '''
        return scatter

def plot_lambda_M():
    lgM_arr = np.arange(13.5,15.1,0.01)
    plt.subplot(111)
    # fiducial
    from parameters import NuisanceParameters
    nuisance = NuisanceParameters()
    fsr = FiducialScalingRelation(nuisance)
    plt.plot(10**lgM_arr, np.exp(fsr.lnlambda_lnM(lgM_arr*np.log(10.))), label='original')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\rm M_{200m}$')
    plt.ylabel(r'$\rm \lambda$')
    plt.savefig('../../plots/mean_mass/mass_lambda_simet17.pdf')

    # plotting the piecewise
    from parameters import NuisanceParametersPiecewise
    nuisance = NuisanceParametersPiecewise(alpha_M0 = 0.65, alpha_M1 = 0.752, alpha_M2 = 0.85)
    psr = PiecewiseScalingRelation(nuisance=nuisance)
    plt.plot(10**lgM_arr, np.exp(psr.lnlambda_lnM(lgM_arr*np.log(10.))), label='piecewise')

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\rm M_{200m}$')
    plt.ylabel(r'$\rm \lambda$')
    plt.title('mean relation from Simet 2017')
    plt.savefig('../../plots/mean_mass/mass_lambda_piecewise.pdf')

def plot_selection():
    lnM_arr = np.linspace(np.log(1e13), np.log(1e15))

    # fiducial
    from parameters import NuisanceParameters
    nuisance = NuisanceParameters()    
    fsr = FiducialScalingRelation(nuisance)
    rs1 = RichnessSelection(scaling_relation=fsr, lambda_min=10, lambda_max=100)
    plt.plot(lnM_arr/np.log(10.), rs1.lnM_selection(lnM_arr))

    # piecewise
    from parameters import NuisanceParametersPiecewise
    nuisance = NuisanceParametersPiecewise(alpha_M0 = 0.65, alpha_M1 = 0.752, alpha_M2 = 0.85)
    psr = PiecewiseScalingRelation(nuisance=nuisance)
    rs2 = RichnessSelection(scaling_relation=psr, lambda_min=10, lambda_max=100)
    plt.plot(lnM_arr/np.log(10.), rs2.lnM_selection(lnM_arr))

if __name__ == "__main__":
    #plot_lambda_M()
    plot_selection()
    plt.show()