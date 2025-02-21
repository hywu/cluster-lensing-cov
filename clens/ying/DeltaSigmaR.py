#Last-modified: 03 Dec 2013 12:45:14 AM

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager

try:
    from scipy.integrate import trapz
except ImportError:
    from scipy.integrate import trapezoid as trapz

import sys

from .param import CosmoParams
from .density import Density
from .lineartheory import LinearTheory
from .halostat import HaloStat
from .clusterrichness import ClusterRichness
from .nfw import NFW
from .halogrowth import HaloGrowth
from .halo_matter_correlation import xi_hm


""" to project a 3D spherical profile to 2D, in terms of :math:`\\xi_{hm}` (it
could also be 'gm' or 'cm' depending on the application), to the observable
:math:`\Delta\Sigma(R)`.  """


def DeltaSigmaR(r, xi3d, rp_max=10.0, rho=1.0):
    """ This  function calculates the 2D weaking lensing profile :math:`\Delta\Sigma(R)` by projecting the 3D version :math:`\\xi_{hm}(r)` along line-of-sight:

    .. math::

        \Delta\Sigma(R) = \\bar{\Sigma}(<R) - \\bar{\Sigma(R)},

    where

    .. math::

        \\bar{\Sigma}(<R) = \\rho_m \\frac{2}{R^2}\int_{0}^{R}\int_{-\infty}^{+\infty} r\\xi(\sqrt{r^2+z^2})dzdr

    and 

    .. math::

        \\bar{\Sigma(R)} = \\rho_m \int_{-\infty}^{+\infty} \\xi(\sqrt{r^2+z^2})dz


    Parameters
    ----------
    r: ndarray
        Radius of the 3D profile.
    xi3d: ndarray
        :math:`\\xi_{hm}`.
    rp_max: float, optional
        Maximumm projected distance of the output 2D profile (default: 10.0)
    rho: float, optional
        Mean background density (default: 1.0).

    Returns
    -------
    rp: ndarray
        Radius of the 2D profile.
    DeltaSigma(rp): ndarray
        Projected profile at *rp*.
    xirp: ndarray
        Mean projected correlation functoin at *rp*.
    xirp_mean: ndarray
        Mean projected correlation functoin interior to *rp*.

    """
    rp, xirp  = xi_rp(r, xi3d, rp_max=rp_max) 
    xirp_mean = xi_rp_mean(rp, xirp)
    return(rp, rho*(xirp_mean-xirp), xirp, xirp_mean)


def xi_rp(r3d, xi3d, rp_max=10.0):
    """ Project the input profile using trapezoidal rule.
    """
    # a deep copy of the input radius because we only want a portion of the
    # original list.
    r       = r3d.copy()
    if rp_max > r[-1]:
        raise RuntimeError("rp_max outside of r3d range")
        
    # the index of the largest element in r3d that is below rp_max (actually minus 1)
    i_max   = np.searchsorted(r, rp_max)
    # define rp by reference to the original array
    rp      = r[0:i_max]
    xirp  = []
    for i, R in enumerate(rp):
        # to avoid any interpolation, we have to set up z array so 
        # that z*z + rp*rp = r*r
        # I give up integrating in log space because the inclusion of z = 0 point is
        # very important for trapz, but much less so for spline-based method.
        z       = np.sqrt(r[i:]*r[i:] - R*R)
        xirp.append(2.0*np.trapz(xi3d[i:], x=z))
    return(rp, np.asarray(xirp))

def xi_rp_mean(rp, xirp):
    """ This function calculates the surface-averaged pojected correlation function
    from the output of *xi_rp*.

    .. warning::

        The first few points at small scales are not trustworthy due to limited
        resolution of the inner surface.
        
    """
    xirp_mean = []
    for i, R in enumerate(rp):
        rp_int   = rp[:i+1]
        xirp_int = xirp[:i+1]
        t = np.trapz(xirp_int*rp_int, x=rp_int)
        t = t*2.0/(R*R)
        xirp_mean.append(t)
    return(np.asarray(xirp_mean))


def gen_xihm(masswant):
    from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
    if np.isscalar(masswant):
        masswant = np.asarray([masswant,])
    else:
        masswant = np.asarray(masswant)

    cp = CosmoParams(set_flat=True)
    h          = cp.h
    z          = 0.0
    DELTA_HALO = 200.0

    den        = Density(cosmo=cp)
    rho_mean   = den.rho_mean_z(z=z)
    lin        = LinearTheory(cosmo=cp, z=z)

    hs         = HaloStat(cosmo=cp, z=z, DELTA_HALO=DELTA_HALO)
    mass       = hs.mass

    bias_func  = hs.bias_function

    hg         = HaloGrowth(cosmo=cp)
    nvir       = 100
    mvir       = setup_logbin(xmin=mass[0], xmax=mass[-1]*5., nbin=nvir)
    cvir       = np.zeros(nvir)
    m200       = np.zeros(nvir)
    c200       = np.zeros(nvir)
    for i in range(nvir):
        cvir[i] = hg.MAH(mvir[i], z)
        m200[i], c200[i] = den.vir2delta(z, mvir[i], cvir[i], DELTA_HALO)
    lgm200 = np.log10(m200)
    f_cmrelation = spline1d(lgm200, c200)
    c200_func  = f_cmrelation(np.log10(mass))

    indxwant   = np.searchsorted(mass, masswant)
    c200want   = c200_func[indxwant]
    biaswant   = bias_func[indxwant]

    radius_hinv = setup_logbin(0.0001, 200.0, 200)
    radius = radius_hinv/h

    xi3d = lin.correlation_func_interp()
    ximm = xi3d(radius)

    xihm_lin = []
    xihm_all = []
    for m, b, c in zip(masswant, biaswant, c200want):
        this_nfw = NFW(m, c, rho_mean, DELTA_HALO=DELTA_HALO)
        halo_profile = this_nfw.profile(radius)
        this_xihm = xi_hm(radius, halo_profile, ximm, b, rho_mean)
        xihm_all.append(this_xihm)
        this_blin = b*ximm
        xihm_lin.append(this_blin)
    xihm_lin = np.asarray(xihm_lin)
    xihm_all = np.asarray(xihm_all)

    return(radius_hinv, radius, xihm_all, xihm_lin)

def gen_xihm_fast(m, b, c, rcutoff):
    z = 0.0
    DELTA_HALO = 200.0
    cp = CosmoParams(set_flat=True)
    h = cp.h
    den        = Density(cosmo=cp)
    rho_mean   = den.rho_mean_z(z=z)
    lin        = LinearTheory(cosmo=cp, z=z)
    radius_hinv = setup_logbin(0.0001, 200.0, 200)
    radius = radius_hinv/h
    xi3d = lin.correlation_func_interp()
    ximm = xi3d(radius)
    nfw = NFW(m, c, rho_mean, DELTA_HALO=DELTA_HALO)
    mcutoff = nfw.total_mass(rcutoff)
    rvir = nfw.r_vir
    print(rvir)
    halo_profile = nfw.profile(radius)
    xihm = xi_hm(halo_profile, ximm, b, rho_mean)
    blin = b*ximm
    return(radius_hinv, radius, xihm, blin, mcutoff)


def decomposition_demo(m=1.e14, b=2.0, c=6.0, rcutoff=3):
    radius_hinv, radius, xihm, blin, mcutoff = gen_xihm_fast(m, b, c, rcutoff)
    r = radius
    profile = xihm
    pi = 3.14159265
    h = 0.702
    rho_mean = 37473017305.5
    rp0, delsig0, xirp0, xirpmean0 = DeltaSigmaR(r, profile, rho=rho_mean, rp_max = 80)
    profile = blin
    icutoff = np.searchsorted(r, rcutoff)
    profile[0:icutoff] = 0.0
    rp1, delsig1, xirp1, xirpmean1 = DeltaSigmaR(r, profile, rho=rho_mean, rp_max = 80)
#    delsig11 = delsig1 + m/(pi*rp1*rp1)
#    delsig12 = delsig1 + mcutoff/(pi*rp1*rp1)- (4.0/3.0)*rho_mean*rcutoff**3.0/rp1**2
    delsig12 = delsig1 + mcutoff/(pi*rp1*rp1)
    fig = plt.figure(figsize=(8, 6))
    fac = 1.e-12/h
    ax = fig.add_subplot(111)
    ax.plot(rp0*h, delsig0*fac, "k-",  label="$\Delta\Sigma(R)$ from $\\xi_{hm}$")
#    ax.plot(rp1, delsig11*fac, "r--", label="$\Delta\Sigma(R)$ from $\\xi_{mm}$ and $M_h$" )
    ax.plot(rp1*h, delsig12*fac, "b-", label="$\Delta\Sigma(R)$ from $\\xi_{mm}$ and $M_h$, corrected" )
    ax.plot(rp1*h, delsig1*fac, "g--", label="2h term" )
    ax.plot(rp1*h, fac*mcutoff/(pi*rp1*rp1), "r--", label="1h term" )
    ax.axvline(rcutoff, color="y")
    ax.text(0.02, 0.5, str(m)+" $M_\odot$")
#    ax.legend(loc=1, ncol=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 99)
    ax.set_ylim(1e-1, 2e3)
    ax.set_xlabel("$R$ [h$^{-1}$Mpc]")
#    ax.set_xlabel("$R$ [Mpc]")
    ax.set_ylabel("profile")
    plt.show()


def setup_logbin(xmin, xmax, nbin):
    ''' Returns bins with logarithmic spacing determined by             
    range of variable and number of bins.                               
    '''
    x = np.linspace(np.log10(xmin), np.log10(xmax), num=nbin)
    x = np.power(10.0, x)
    return(x)


def profile_demo(mh, color="g"):
    radius_hinv, radius, xihm_all, xihm_lin = gen_xihm([mh,])
    r = radius_hinv
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    rho_mean = 1.e-12*37473017305.5/0.702**2
    profile = xihm_all[0]
    # plot the original 3D profile
    rp, delsig, xirp, xirpmean = DeltaSigmaR(r, profile, rho=rho_mean, rp_max = 30)
    ax.plot(r,  profile,  ":",  color=color, label="$\\xi_{hm}$")
    ax.plot(rp, delsig,   "-",  color=color, label="$\Delta\Sigma(R)$" )
    ax.plot(rp, xirp,     "-.", color=color, label="$w(r_p)$" )
    ax.plot(rp, xirpmean, "--", color=color, label="$\\bar{w(r_p)}$" )
    ax.text(0.02, 0.5, str(mh)+" $M_\odot$")
    ax.legend(loc=1, ncol=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-2, 99)
    ax.set_ylim(1e-1, 2e5)
    ax.set_xlabel("$r$ [h$^{-1}$Mpc]")
    ax.set_ylabel("profile")
    plt.show()


def derivative(x, y):
    xnew = x[1:-1]
    n    = len(xnew)
    der  = np.zeros(n)
    for i in range(n):
        der[i] = (y[i+2] - y[i])/(x[i+2] - x[i])
    return(xnew, der)

def get_break(x, y):
    y[y<0]=0.00001
    xx, yprime = derivative(np.log(x), np.log(y))
    n = len(xx)
    for i in range(n-2):
        if ((yprime[i+1] - yprime[i])*(yprime[i+2] - yprime[i+1])<0.0):
            breakpt = np.exp(xx[i+2])
            print("find break point at %6.3f"%breakpt)
            break
    return(breakpt)


def transition_demo(mass=1.e14):
    from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
    radius_hinv, radius, xihm_all, xihm_lin = gen_xihm(mass)
    r = radius_hinv
    lnr = np.log(r)
    n = len(r)
    xihm = np.ravel(xihm_all)
    r0 = 1.0
    r2 = 10.0
    rho_mean = 1.e-12*37473017305.5/0.702**2
    r1 = get_break(r, xihm)

    i0 = np.searchsorted(r, r0)
    i1 = np.searchsorted(r, r1)
    if abs(r[i1] - r1) > abs(r[i1+1] - r1):
        i1 = i1 + 1
    elif abs(r[i1] - r1) > abs(r[i1-1] - r1):
        i1 = i1 - 1
    else:
        print("find break index at %4d"%i1)
    i2 = np.searchsorted(r, r2)

    lnr_left_0  = lnr[0:i1]
    lnr_right_0 = lnr[i1:n]

    left  = np.hstack([range(i0), [i1,]])
    right = np.hstack([[i1,], range(i2, n)])

    r_left    = r[left]
    lnr_left  = np.log(r_left)
    xi_left   = xihm[left]
    r_right   = r[right]
    lnr_right = np.log(r_right)
    xi_right  = xihm[right]

    xibreak0 = xihm[i1]

    fraclist = [-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30]
    prop = matplotlib.font_manager.FontProperties(size=10)

    rp, delsig0, xirp0, xirpmean0 = DeltaSigmaR(r, xihm, rho=rho_mean, rp_max = 30)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_axes([0.08, 0.5, 0.40, 0.40])
    ax2 = fig.add_axes([0.55, 0.5, 0.40, 0.40])
    ax3 = fig.add_axes([0.08, 0.1, 0.40, 0.40])
    ax4 = fig.add_axes([0.55, 0.1, 0.40, 0.40])
    for frac in fraclist:
        xibreak = xibreak0*(1.+frac)
        # i0 becomes the index of the break point in the new xi_left array
        xi_left[i0] = xibreak
        # 0 becomes the index of the break point in the new xi_right array
        xi_right[0]  = xibreak

        f_left       = spline1d(lnr_left, np.log(xi_left))
        xi_left_new  = np.exp(f_left(lnr_left_0))
        f_right      = spline1d(lnr_right, np.log(xi_right))
        xi_right_new = np.exp(f_right(lnr_right_0))
    
        xihm_new  = np.hstack([xi_left_new, xi_right_new])
        ax1.plot(r,  xihm_new,  "-", alpha=1.0)

        rp, delsig, xirp, xirpmean = DeltaSigmaR(r, xihm_new, rho=rho_mean, rp_max = 30)
        ax2.plot(rp, delsig,   "-",  label=str(frac*100.0)+"$\%$")

        ax3.plot(r, xihm_new/xihm, "-", alpha=1.0)

        ax4.plot(rp, delsig/delsig0, "-", alpha=1.0)


    ax1.text(0.02, 0.5, str(mass)+" $M_\odot$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(1e-2, 99)
    ax1.set_ylim(1e-2, 2e5)
    ax1.set_ylabel("$\\xi_{hm}$")
    ax2.legend(loc=1, ncol=2, fancybox=True, shadow=True, prop=prop)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1e-2, 99)
    ax2.set_ylim(5e-1, 2e2)
    ax2.set_ylabel("$\Delta\Sigma(R)$")
    ax3.set_xscale("log")
    ax3.set_xlim(1e-2, 99)
    ax3.set_xlabel("$r$ [h$^{-1}$Mpc]")
    ax4.set_xscale("log")
    ax4.set_xlim(1e-2, 99)
    ax4.set_xlabel("$r$ [h$^{-1}$Mpc]")
    ax3.set_ylim(0.6, 1.4)
    ax4.set_ylim(0.6, 1.4)
    ax3.grid(True)
    ax4.grid(True)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.show()



if __name__ == "__main__":
#    profile_demo(5.e14)
#    transition_demo(mass=5.e14)
    decomposition_demo()
