#Last-modified: 24 May 2012 07:31:13 PM

""" Quantities related to cluster mass-richness relationship.
"""

import warnings
import numpy as np
from math import log, sqrt
from scipy.special import erfc

from .nfw import NFW
from .param import MassRichParams, MassRichCurved, MassRichPowerLaw


class ClusterRichness(object):
    """ A class with methods to convolve the mass-richness relationship 
    with halo statistics (as presented in Lima & Hu (2005)).

    .. warning::

        If input as a function, massrich should be a function with 
        :math:`\ln M` as input and :math:`\ln N_{200}` as output.
        If input as a function, scatter should be a function with 
        :math:`\ln M` as input and :math:`\sigma_{\ln N_{200}}` as 
        output.

    .. warning::

        You can not input scatter as a constant when *massrich* is an
        :py:class:`zypy.zycosmo.param.MassRichParams` object but you 
        should set the scatter through *massrich.sigma_r_m*.

    Parameters
    ----------
    massrich: MassRichParams or callable object, optional
        Use a Rozo log normal mass-richness relationship if input 
        is an :py:class:`zypy.zycosmo.param.MassRichParams` object; 
        use a self-defined function 
        if input is a callable object, i.e., function; 
        otherwise use default parameters in Rozo paper if with no 
        input (default: None).

    scatter: float or a function, optional
        Set scatter as a function of :math:`\ln M` if you want the scatter
        in the mass-richness relation to be varying with mass; set scatter
        as a constant if input massrich is a function; 

    """
    def __init__(self, massrich=None, scatter=None, set_poissonsca=False):
        if massrich is None:
            mr = MassRichParams()
        elif isinstance(massrich, MassRichParams):
            mr = massrich
        elif isinstance(massrich, MassRichCurved):
            mr = None
            self.log_richness_mean = massrich.log_richness_mean
        elif isinstance(massrich, MassRichPowerLaw):
            mr = None
            self.log_richness_mean = massrich.log_richness_mean
        elif hasattr(massrich, "__call__"):
            mr = None
            self.log_richness_mean = massrich
        else:
            raise TypeError("massirch needs to be either an MassRichParams object, MassRichCurved or a callable function")

        if mr is None:
            if scatter is None:
                raise RuntimeError("Need an input scatter when massrich is a function or MassRichCurved class")
            elif hasattr(scatter, "__call__"):
                # scatter is a function of ln(M)
                if set_poissonsca :
                    self.scatter = lambda x: np.sqrt(scatter(x)**2 +
                            1./np.exp(self.log_richness_mean(x))) 
                else :
                    self.scatter = scatter
            else:
                # scatter is a concstant function of mass
                if set_poissonsca :
                    self.scatter = lambda x: np.sqrt(scatter**2 +
                            1./np.exp(self.log_richness_mean(x))) 
                else :
                    self.scatter = lambda x: scatter
        else:
            # a simple MassRichParams instance with scatter assigned.
            self.A          = mr.A
            self.alpha      = mr.alpha
            self.Mpivot     = mr.Mpivot
            self.logMpivot  = log(mr.Mpivot)
            self.log_richness_mean = self.log_richness_mean_rozo
            if scatter is None:
                if set_poissonsca :
                    self.scatter = lambda x: np.sqrt(mr.sigma_r_m**2 +
                            1./np.exp(self.log_richness_mean(x))) 
                else :
                    self.scatter = lambda x: mr.sigma_r_m
            elif hasattr(scatter, "__call__"):
                # scatter is a function of ln(M)
                if set_poissonsca :
                    self.scatter = lambda x: np.sqrt(scatter(x)**2 +
                            1./np.exp(self.log_richness_mean(x))) 
                else :
                    self.scatter = scatter
            else:
                # scatter needs to be assign by massrich.sigma_r_m if 
                # as a constant
                raise TypeError("scatter needs to be set via MassRichParams.sigma_r_m if massrich is an MassRichParams object and scatter is a constant of mass")


    def log_richness_mean_rozo(self, log_mass):
        """
        Get :math:`\ln(\\bar{N}_{200})` from :math:`\ln(M)` assuming
        the following relationship defined in Rozo et al. (2009).

        .. math::
            
            \ln(\\bar{N}_{200}) = A + \\alpha \ln(M/M_{pivot})

        Parameters
        ----------
        log_mass: scalar or array_like
            :math:`\ln(M)`

        Returns
        -------
        log_richness: scalar or array_like
            :math:`\ln(\\bar{N}_{200})`

        """
        _log_richness = self.A + self.alpha*(log_mass - self.logMpivot)
        return(_log_richness)


    def log_mass_mean_rozo(self, log_richness):
        """
        Get :math:`\ln(\\bar{M})` from :math:`\ln(N_{200})` assuming
        the following relationship defined in Rozo et al. (2009).

        .. math::
            
            \ln(\\bar{M}) = \\frac{(\ln(N_{200}) - A)}{\\alpha} + \ln(M_{pivot})

        Parameters
        ----------
        log_richness: scalar or array_like
            :math:`\ln(N_{200})`

        Returns
        -------
        log_mass: scalar or array_like
            :math:`\ln(\\bar{M})`

        """
        _log_richness = log_richness
        _log_mass = (_log_richness - self.A)/self.alpha + self.logMpivot
        return(_log_mass)


    def richness_mean(self, mass):
        """
        Get :math:`N_{200}` from cluster mass using *log_richness_mean*.

        Parameters
        ----------
        mass: scalar or array_like
            Cluster mass.

        Returns
        -------
        richness: scalar or array_like
            :math:`\\bar{N}_{200}`

        """
        _log_mass = np.log(mass)
        _log_richness_mean = self.log_richness_mean(_log_mass)
        _richness_mean = np.exp(_log_richness_mean)
        return(_richness_mean)

    def richness_random(self, mass, size=1):
        """
        Get random realizations of :math:`N_{200}` from cluster mass using *log_richness_mean*
        and *scatter*.

        Parameters
        ----------
        mass: scalar or array_like
            Cluster mass.

        size: scalar
            Number of richness measures for the cluster.

        Returns
        -------
        richness: scalar or array_like
            :math:`N_{200}`

        """
        _log_mass = np.log(mass)
        _log_richness_mean = self.log_richness_mean(_log_mass)
        _log_richness_random = np.random.normal(loc=_log_richness_mean,
                scale=self.scatter(_log_mass), size=size)
        _richness_random = np.exp(_log_richness_random)
        return(_richness_random)

    def norm_prob_richness_bin(self, r1, r2, mass):
        """

        .. math::

            \int_{r1}^{r2} p(r|M) dr 
            =
            \int_{\ln(r1)}^{\ln(r2)} p(\ln(r)|\ln\\bar{r}) d\ln(r)

        where 

        .. math::

            p(\ln(r)) = \\frac{e^{-x^2}}{\sqrt{2\pi\sigma^2_{\ln r}}}

        and 

        .. math::
        
            x = \\frac{\ln(r) - \ln(\\bar{r}|M)}{\sqrt{2}\sigma_{\ln r}}

        the above equation evaluates to:

        .. math::

            0.5({\\rm erfc}(x_1) - {\\rm erfc}(x_2))

        .. note::

            x is not the reduced variable in a Normal Distribution, 
            but that divided by :math:`\sqrt{2}`.

        Parameters
        ----------
        r1: scalar
            Lower limit of the richness bin.
        r2: scalar
            Upper limit of the richness bin.
        mass: scalar or array_like
            Masses of halos that contribute to the cluster 
            population.

        Returns
        -------
        p: scalr or array_like
            Weight of input halos in determining the specific cluster 
            statistics.

        """
        if r2 < r1:
            warnings.warn("Warning: reversed input richness bracket?")
            r1, r2 = r2, r1
        if r2 == r1:
            return(0.0)
        else:
            x1 = self.x(r1, mass)
            x2 = self.x(r2, mass)
            p = 0.5*(erfc(x1) - erfc(x2))
            return(p)

    def x(self, r, mass):
        """ Reduced variable x as described in Eqn(3) of Lima & Hu (2005).

        .. math::
        
            x = \\frac{\ln(r) - \ln(\\bar{r}|M)}{\sqrt{2}\sigma_{\ln r}}

        Parameters
        ----------
        r: scalar
            Richness.
        mass: scalar or array_like
            Halo masses

        Returns
        -------
        x: scalar or array_like

        """
        _log_mass   = np.log(mass)
        _log_r_mean = self.log_richness_mean(_log_mass)
        _log_r      = log(r)
#        _x = (_log_r - _log_r_mean)/(sqrt(2.)*self.sigma_r_m)
        _x = (_log_r - _log_r_mean)/(sqrt(2.)*self.scatter(_log_mass))
        return(_x)



def fig01_LimaHu2005():
    from .halostat import HaloStat
    import matplotlib.pyplot as plt
    hs = HaloStat()
    hs.reset_halo_mass(mmin=1.e13, mmax=1.e16, nmbin=500)
    mass    = hs.mass                                                      
    dlogm   = hs.dlogm                                                    
    dndm    = hs.mass_function                                             
    dndlogm = dndm*mass 
    dnm = dndlogm*dlogm

    sigma_r_m = 0.25
    mr = MassRichParams(A=0., Mpivot=1.0, alpha=1.0, sigma_r_m=sigma_r_m)
#    scatter = lambda x:0.25*(x/30.)**-10
#    cr = ClusterRichness(massrich=mr, scatter=scatter)
    cr = ClusterRichness(massrich=mr)

    log10mobs = np.arange(14.2, 15.2, 0.2)
    mobs = np.power(10., log10mobs)

    infty = float("inf")

    dnrs = []
    for mo in mobs:
        wgt = cr.norm_prob_richness_bin(mo, infty, mass)
        dnr = dndlogm*wgt*dlogm
        dnrs.append(dnr)

    m14 = np.searchsorted(mass, 1.e14)

    norm = dnm[m14]

    fig2 = plt.figure(2, figsize=(8,6))
    ax21 = fig2.add_subplot(111)
    ax21.plot(mass, dnm/norm, "k-", lw=3)
    for dnr, mo in zip(dnrs, mobs):
        ax21.plot(mass, dnr/norm, "k-", lw=1)
        ax21.fill_between(mass, dnr/norm, y2=0, where=mass<=mo, color="k", alpha=0.8)
        ax21.fill_between(mass, dnm/norm, y2=dnr/norm, where=mass>=mo, color="k", alpha=0.2)

    tax = ax21.twiny()                                           
    tax.xaxis.tick_top()                                       
    tax.set_xscale("log")
    tax.set_xlim(5.e13, 3.e15)       
    tax.set_xlabel("$\log(M_{obs})$")                   
    tax.set_xticks(mobs)                   
    tax.set_xticklabels([format(x, ".3g") for x in log10mobs])                   

    ax21.text(1e15, 0.5, "$\sigma_{\ln M_{obs}} = 0.25$", ha="left")

    ax21.set_xscale("log")
    ax21.set_xlim(5.e13, 3.e15)
    ax21.set_ylim(0.0001, 0.6)
    ax21.set_xlabel("$M_{true}$")
    ax21.set_ylabel("$dn$ arbitrary unit")
    plt.show()
    

def bintest():
    from .halostat import HaloStat
    import matplotlib.pyplot as plt
    hs = HaloStat()
    hs.reset_halo_mass(mmin=5.e13, mmax=1.e16, nmbin=1000)
    mass    = hs.mass                                                      
    dlogm   = hs.dlogm                                                    
    dndm    = hs.mass_function                                             
    dndlogm = dndm*mass 
    dnm = dndlogm*dlogm

    sigma_r_m = 0.25
    mr = MassRichParams(A=0., Mpivot=1.0, alpha=1.0, sigma_r_m=sigma_r_m)
#    scatter = lambda x:0.25*(x/30.)**-10
#    cr = ClusterRichness(massrich=mr, scatter=scatter)
#    cr = ClusterRichness(massrich=mr)
    cr = ClusterRichness(massrich=mr, set_poissonsca=True)

    log10mobs  = np.arange(14.2, 15.2, 0.2)
    mobs  = np.power(10., log10mobs)
    log10mobs2 = np.arange(14.4, 15.4, 0.2)
    mobs2 = np.power(10., log10mobs2)

    infty = float("inf")

    dnrs = []
    for mo, mo2 in zip(mobs, mobs2):
        wgt = cr.norm_prob_richness_bin(mo, mo2, mass)
        dnr = dndlogm*wgt*dlogm
        dnrs.append(dnr)

    m14 = np.searchsorted(mass, 1.e14)

    norm = dnm[m14]

    fig2 = plt.figure(2, figsize=(8,6))
    ax21 = fig2.add_subplot(111)
    ax21.plot(mass, dnm/norm, "k-", lw=3)
    for dnr, mo, mo2 in zip(dnrs, mobs, mobs2):
        ax21.plot(mass, dnr/norm, "k-", lw=1)
        ax21.fill_between(mass, dnr/norm, y2=0, where=((mass>=mo2) | (mass<=mo)), color="k", alpha=0.8)
#        ax21.fill_between(mass, dnr/norm, y2=0, where=((mass>=mo2) | (mass<=mo)), color="w", alpha=0.5)
        ax21.fill_between(mass, dnm/norm, y2=dnr/norm, where=((mass>=mo) & (mass<=mo2)), color="k", alpha=0.2)
        ax21.axvline(x=mo, lw=1, color="k", ls="-", alpha=0.2)
        ax21.axvline(x=mo2, lw=1, color="k", ls="-", alpha=0.2)

    tax = ax21.twiny()                                           
    tax.xaxis.tick_top()                                       
    tax.set_xscale("log")
    tax.set_xlim(5.e13, 3.e15)       
    tax.set_xlabel("$\log(M_{obs})$")                   
    tax.set_xticks(mobs)                   
    tax.set_xticklabels([format(x, ".3g") for x in log10mobs])                   

    ax21.text(1e15, 0.5, "$\sigma_{\ln M_{obs}} = 0.25$", ha="left")

    ax21.set_xscale("log")
    ax21.set_xlim(5.e13, 3.e15)
    ax21.set_ylim(0.0001, 0.6)
    ax21.set_xlabel("$M_{true}$")
    ax21.set_ylabel("$dn$ arbitrary unit")
    plt.show()
    



if __name__ == "__main__":
    """
    """
#    fig01_LimaHu2005()
    bintest()

