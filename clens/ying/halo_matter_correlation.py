#Last-modified: 11 Dec 2011 02:29:37 PM

import numpy as np

""" Hiyashi and White scheme.
"""

def xi_hm(profile, ximm, bias, rho_mean_z, set_retall=False):
    """please make sure profile and ximm are computed at the same radii
    """
    # rho_mean_z has to be the same as the input value in NFW class.
    xihm1h = xi_hm_1h(profile, rho_mean_z)
    xihm2h = xi_hm_2h(ximm, bias)
    # Hayashi & White method
    xihm   = np.maximum(xihm1h, xihm2h)
    # test
    # rcutoff = radius[np.nonzero(xihm2h >= xihm1h)[0][0]]
    if set_retall:
        return(xihm, xihm1h, xihm2h)
    else:
        return(xihm)


def xi_hm_1h(profile, rho_mean_z):
    xihm1h = profile/rho_mean_z - 1.0
    return(xihm1h)


def xi_hm_2h(ximm, bias):
    xihm2h = bias*ximm
    return(xihm2h)
