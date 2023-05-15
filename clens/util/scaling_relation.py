#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc

#from clens.util.parameters import NuisanceParameters

class RichnessSelection(object):
    def __init__(self, scaling_relation, lambda_min, lambda_max):
        self.scaling_relation = scaling_relation
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def lnM_selection(self, lnM, z):
        lnlambda_mean = self.scaling_relation.lnlambda_lnM(lnM, z)
        sigma_lambda = self.scaling_relation.scatter(lnM, z)

        x_lo = (np.log(self.lambda_min) - lnlambda_mean)/np.sqrt(2.)/sigma_lambda
        x_hi = (np.log(self.lambda_max) - lnlambda_mean)/np.sqrt(2.)/sigma_lambda
        return 0.5*erfc(x_lo) - 0.5*erfc(x_hi)


class FiducialScalingRelation(object):
    def __init__(self, lgM0=14.6258, alpha_M=1, lambda0=70, sigma_lambda=0.18):
        self.lgM0 = lgM0
        self.alpha_M = alpha_M
        self.lambda0 = lambda0
        self.sigma_lambda = sigma_lambda

    def lnlambda_lnM(self, lnM, z):
        M = np.exp(lnM)
        return np.log(self.lambda0 * (M/10**self.lgM0)**self.alpha_M)

    def scatter(self, lnM, z):
        return self.sigma_lambda 

class Costanzi21ScalingRelation(object):
    def __init__(self, Alam=72.4, Blam=0.935, Clam=0.51, Dlam=0.207):
        self.Alam = Alam
        self.Blam = Blam
        self.Clam = Clam
        self.Dlam = Dlam

    def lnlambda_lnM(self, lnM, z):
        lnMpivot_Msun = np.log(3e14/0.7)
        return np.log(self.Alam) + self.Blam * (lnM - lnMpivot_Msun) + self.Clam * np.log((1+z)/1.45)

    def scatter(self, lnM, z):
        lam = np.exp(self.lnlambda_lnM(lnM, z))
        sigma_sqr = self.Dlam**2 + (lam - 1)/lam**2
        return np.sqrt(sigma_sqr)

#class Murata19ScalingRelation(object): # TODO


def plot_lambda_M(scaling_relation):
    lgM_arr = np.arange(13.5,15.1,0.01)
    z = 1
    plt.plot(10**lgM_arr, np.exp(scaling_relation.lnlambda_lnM(lgM_arr*np.log(10.), z)), label='original')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\rm M_{200m}$')
    plt.ylabel(r'$\rm \lambda$')


def plot_selection(scaling_relation):
    lnM_arr = np.linspace(np.log(1e13), np.log(1e15))
    rs1 = RichnessSelection(scaling_relation, lambda_min=10, lambda_max=100)
    plt.plot(lnM_arr/np.log(10.), rs1.lnM_selection(lnM_arr, z))


if __name__ == "__main__":

    # fiducial
    nuisance = {"lgM0": 14.6258, "alpha_M": 1, "lambda0": 70, "sigma_lambda": 0.18}
    scaling_relation = FiducialScalingRelation(**nuisance)
    plot_lambda_M(scaling_relation)

    # Costanzi21
    scaling_relation = Costanzi21ScalingRelation(Alam=72.4, Blam=0.935, Clam=0.51, Dlam=0.207)
    plot_lambda_M(scaling_relation)

    plt.show()