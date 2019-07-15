#!/usr/bin/env python
#Last-modified: 12 Dec 2013 20:53:47


"""  python wrapper for halofit.
"""

#---------------------------------------------------------------------
import numpy as np
#---------------------------------------------------------------------
from . import constants as cc
from . import cex
#---------------------------------------------------------------------
# from zyhalofit import zyhalofit
import zyhalofit 
#---------------------------------------------------------------------


#---------------------------------------------------------------------
class HALOFIT(object):
    def __init__(self, pklin, omega_M_z, omega_lambda_z, set_reversewn=False):
        """ Python wrapper for Smith et al. (2003)'s HALOFIT routine. 
        
        Parameters
        ----------
        pklin: function
            Linear power spectrum P(k) at desired redshift.
        omega_M_z: scalar
            Matter density at desired redshift.
        omega_lambda_z: scalar
            Vaccum density at desired redshift.
        set_reversewn: bool, optional, DEPRECATED
            The wavenumber array from Fortran zyhalofit.wn is reversed before being sent
            to pklin. This is crucial when pklin is a python interpolation-based function
            which requires input with increasing order, whereas the zyhalofit.wn from Fortran
            is decreasing because of different storage order of Python and Fortran.
        
        """
        self.pklin      = pklin
        # set_reversewn is deprecated, but kept for downward-compatibility
        # XXX I am not hundred percent sure why the heck I had this argument in the first
        # place, but guess I'll find out.
        # if set_reversewn:
            # zyhalofit.ps    = pklin(zyhalofit.wn[::-1])[::-1]
        # else:
            # zyhalofit.ps    = pklin(zyhalofit.wn)
        _zyhalofit = zyhalofit.zyhalofit
        _zyhalofit.ps    = pklin(_zyhalofit.wn)
        _zyhalofit.om_m  = omega_M_z
        _zyhalofit.om_v  = omega_lambda_z
        ifound = _zyhalofit.setup_halofit()
        self.zyhalofit = _zyhalofit
        if ifound == 0:
            raise cex.ConditionNotReached("HALOFIT failed to find rknl")

    def pk_NL(self, k):
        k = np.atleast_1d(k)
        pnl = self.zyhalofit.pnl_vec(k, self.pklin(k))[0]
        return(pnl)

    def pk_NL_all(self, k):
        k = np.atleast_1d(k)
        pnl, p1h, p2h = self.zyhalofit.pnl_vec(k, self.pklin(k))
        return(pnl, p1h, p2h)



                                                    
def testpknl():
    import matplotlib.pyplot as plt
    from .param import CosmoParams
    from .lineartheory import LinearTheory
    from .density import Density
    try :
        from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
    except ImportError:
        from zypy.contrib.cubicSpline import NaturalCubicSpline as spline1d
    cp = CosmoParams(set_flat=True)
    den = Density(cp)
    log10k = np.arange(-3, 1, 0.1)
    k = np.power(10.0, log10k)
    fig = plt.figure(1, figsize=(8, 6))
    ax  = fig.add_subplot(111)
    zs = [0., 0.5, 1., 1.5, 2.]
    cs = ["k", "r", "m", "g",  "c"]
    for z, c in zip(zs, cs):
        lin    = LinearTheory(cp, z=z)
        pklinfunc  = lin.power_spectrum
        pklin  = pklinfunc(k)
        # get an interpolated function as well
        # pklinfunc_interp = spline1d(k, pklin)
        omega_M_z = den.omega_M_z(z)
        omega_lambda_z = den.omega_lambda_z(z)
        nonlin = HALOFIT(pklinfunc, omega_M_z, omega_lambda_z)
        # nonlin = HALOFIT(pklinfunc_interp, omega_M_z, omega_lambda_z, set_reversewn=True)
        pknl,pkq,pkh   = nonlin.pk_NL_all(k)
        ax.plot(k, pknl ,  lw=2, ls="-", color=c, label="z="+str(z))
        ax.plot(k, pklin,  lw=1, ls="--", color=c) 
        # ax.plot(k, pkq,  lw=1, ls="--", color=c) 
        # ax.plot(k, pkh,  lw=1, ls="--", color=c) 
    ax.legend(loc=1, ncol=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.001, 10)
    ax.set_ylim(1e0, 5e5)
    ax.set_xlabel("k [1/Mpc]")
    ax.set_ylabel("$P(k)$")
    plt.show()

                                                    




if __name__ == '__main__':
    testpknl()

