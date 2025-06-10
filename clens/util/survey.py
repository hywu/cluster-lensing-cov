#!/usr/bin/env python
import numpy as np
from scipy.interpolate import interp1d

class Survey(dict):
    def __init__(self, z_star_src=0.5, m_src=2, beta_src=1.4, sigma_gamma = 0.3, n_src_arcmin=10, top_hat=False, zs_min=None, zs_max=None, interp=False, z_interp=False, dndz_interp=False):
        """
        Two choices of source redshift distribution:
        1. whale-shape (default): z_star_src, m_src, beta_src,  parameterization based on Rozo 2011 Eq 14
        2. top-hat (set top_hat=True), zs_min, zs_max
        3. interpolated (set interp=True), z_interp, dndz_interp (should be normalized)
        """

        self.sigma_gamma = sigma_gamma # shape noise
        self.n_src_arcmin = n_src_arcmin
        
        self.top_hat = top_hat
        self.interp = interp
        if zs_min == None:
            zs_min = 0
        if zs_max == None:
            zs_max = 3

        self.zs_min = zs_min
        self.zs_max = zs_max

        if self.interp == False:
            if self.top_hat == True:
                print('using top-hat redshift distribution')

            if self.top_hat == False:
                print('using a whale-shaped p(z)')
                self.z_star_src = z_star_src
                self.m_src = m_src
                self.beta_src = beta_src
                zs_list = np.arange(self.zs_min, self.zs_max, 0.01)
                self.norm = np.trapz(self._pz_src(zs_list), x=zs_list)

        if self.interp == True:
            self.dndz_interp = interp1d(z_interp, dndz_interp)

    def _pz_src(self, zs):
        return zs**self.m_src * np.exp(-(zs/self.z_star_src)**self.beta_src)

    def pz_src(self, zs):
        if self.interp == False:
            if self.top_hat == False:
                out = self._pz_src(zs)/self.norm 
                out[zs>self.zs_max] = 0
                out[zs<self.zs_min] = 0
                return out

            if self.top_hat == True:
                zs = np.atleast_1d(zs)
                out = np.zeros(len(zs))
                out[(zs>=self.zs_min)&(zs<=self.zs_max)] = 1./(self.zs_max-self.zs_min)
                return out

        if self.interp == True:
            return self.dndz_interp(zs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # if you want a delta function at zs: set m_src=10, beta_src=10, z_star_src=zs
    zs_list = np.arange(0.1,2,0.01)
    su = Survey(z_star_src=0.5, m_src=10, beta_src=10)
    plt.plot(zs_list, su.pz_src(zs_list), label='zs=%g,m=%g,beta=%g'%(su.z_star_src,su.m_src,su.beta_src))
    plt.legend()
    plt.show()