#!/usr/bin/env python
import numpy as np
from scipy.special import jv
import matplotlib.pyplot as plt
from clens.util import constants as cn


class BesselForCovTheta(object):
    def __init__(self):
        self.scaling_for_ell_min = 1 #1e-3  # TODO: if changing ell range, need to change P(k) range accordingly.
        self.scaling_for_ell_max = 100 #1e+5 
        self.dlnell = 1e-3 #1e-4 

    def j2_bin(self, ell, thmin, thmax):
        area = np.pi*(thmax**2 - thmin**2)
        out = 2*jv(0, ell*thmin) - 2*jv(0, ell*thmax) 
        out += (ell*thmin*jv(1, ell*thmin) - ell*thmax*jv(1, ell*thmax))
        out *= (2.*np.pi/ell**2)/area
        return out

    def j0_bin(self, ell, thmin, thmax):
        area = np.pi*(thmax**2 - thmin**2)
        out = thmax*jv(1, ell*thmax) - thmin*jv(1, ell*thmin) 
        out *= (2.*np.pi/ell)/area
        return out

    def plot(self):
        ell = np.arange(1000)
        plt.plot(ell, self.j2_bin(ell, 0.1, 0.2))
        plt.plot(ell, jv(2, ell*0.1))


if __name__ == "__main__":
    bf = BesselForCovTheta()
    bf.plot()
    plt.show()