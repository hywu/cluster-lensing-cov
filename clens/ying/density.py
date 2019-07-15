#!/usr/bin/env python
#Last-modified: 02 Jul 2012 10:47:13 PM

""" Mattter density, critical density, and mean density etc. given 
cosmology and redshift.
"""

import numpy as np

from .param import CosmoParams
from . import constants as cc
from . import cex

import unittest

# Heidi: not used here
# try:                                                                    
#     import MassConvert
# except ImportError:
#     print "Error import MassConvert, "
#     print "vir2delta module will not work."

class Density(object):
    """ Calculate various cosmological densities and overdensities.

    Parameters
    ----------
    cosmo: CosmoParams object, optional
        Cosmological parameters.

    set_hinv: bool, optional
        h is set to one if True, when all the results are in units independent of h, otherwise
        h is explicitly included in the calculation.

    """
    def __init__(self,cosmo=None, set_hinv=False):
        self.isflat = True
        if cosmo is None:
            cp = CosmoParams(set_flat=True)
        else:
            cp = cosmo
            if not cp.isflat:
                print("Warning: Non-flat Universe encountered!")
        self.isflat   = cp.isflat()
        self.isEdS    = cp.isEdS()
        self.isopen   = cp.isopen()
        self.isclosed = cp.isclosed()
        if set_hinv:
            self.h  = 1.0
        else:
            self.h  = cp.h
        self.m0 = cp.omega_M_0
        self.q0 = cp.omega_lambda_0
        self.k0 = cp.omega_k_0
        self.s0 = cp.sigma_8

    def e_z(self, z):
        """ The unitless Hubble expansion rate at redshift z.

        Parameters
        ---------
        z: scalar or array_like
            Redshift

        Returns
        -------
        E(z): scalar or array_like
            Dimensionless Hubble parameter at z.

        """
        return((self.m0 * (1+z)**3. +
                self.k0 * (1+z)**2. +
                self.q0)**0.5)

    def h_z(self, z):
        """ h as a function of redshift z.

        .. note ::

            ALL THE h-DEPENDENCES COME FROM HERE.

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        h(z): scalar or array_like
            Hubble parameter h at z.

        """
        _e_z = self.e_z(z)
        return(self.h*_e_z)

    def omega_M_z(self, z):
        """ Matter density :math:`\Omega_m` as a function of redshift z.

        From Lahav et al. (1991, MNRAS 251, 128) equations 11b-c. This 
        is equivalent to equation 10 of Eisenstein & Hu (1999).

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        omega_M(z):
            Matter density at z.

        """
        if self.k0 == 0.0:
            return(1.0/(1.0+self.q0/(self.m0*np.power(1.0 + z, 3.0))))
        else:
            _e_z = self.e_z(z)
            return(self.m0*np.power((1.0+z),3.0)/np.power(_e_z, 2.0))

    def omega_lambda_z(self, z):
        """ Dark Energy density :math:`\Omega_\lambda` as a function of redshift.

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        omega_lambda(z):
            Cosmological constant at z.

        """
        _e_z = self.e_z(z)
        return(self.q0/np.power(_e_z, 2.0))

    def rho_crit_z(self, z):
        """ Critical density of the universe at z.

        .. note ::

            Density unit in solar masses per cubic Mpc.

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        rho_crit(z): scalar or array_like
            Critical density at z.

        """
        return(3.*np.power((self.h_z(z)*cc.H100_s),2.)/
               (8.*cc.pi*cc.G_const_Mpc_Msun_s))

    def rho_mean_z(self, z):
        """ Mean matter density of the universe at z.

        .. note ::

            Density unit in solar masses per cubic Mpc.

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        rho_mean(z): scalar or array_like
            Mean density at z.

        """
        return(self.omega_M_z(z)*self.rho_crit_z(z))

    def growth_factor(self, z, set_normed=True):
        """ Linear growth factor normalized to 1 at z = 0.

        Approximate forumla from Carol, Press, & Turner (1992, ARA&A,
        30, 499). This is proportional to :math:`D_1(z)` from Eisenstein & Hu
        (1999), Eqn 10, but the normalization is different:
        growth_factor = 1 at z = 0 and
        :math:`D_1(z) = \\frac{1+z_{eq}}{1+z}` as z goes to infinity.

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        growth_factor: scalar or array_like
            :math:`D_1(z)/D_1(0)`

        """
        if not self.isflat:
            raise cex.NonFlatUniverseError(self.k0)
        omega = self.omega_M_z(z)
        lamb  = self.omega_lambda_z(z)
        a = 1.0/(1.0 + z)

        if set_normed:
            norm = 1.0/self.growth_factor(0.0, set_normed=False)
        else:
            norm = 1.0
        return(norm*(5./2.)*a*omega/
               (np.power(omega, (4./7.))-lamb+(1.+omega/2.)*(1.+lamb/70.))
              )

    def sigma_8_z(self, z):
        """ :math:`\sigma_8` at redshift z.

        This function calculates :math:`\sigma_8` at any redshift using
        :math:`\sigma_8(z=0)` and assuming a scale independent growth factor.

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        sigma_8(z): scalar or array_like
            :math:`\sigma_8(z)`.

        """
        return(self.s0*self.growth_factor(z))

    def virial_overdensity(self, z):
        """ :math:`\Delta_{vir}`, overdensity factor for virialized, 
        collapsed objects. 

        From Bryan & Norman (1997).

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        Delta_vir: scalar or array_like
            Spherical overdensities at redshift z.

        """
        omega_z = self.omega_M_z(z)
#        print(omega_z)
        x = omega_z - 1.0
        if self.isflat:
            # Eke et al Case
            _d_vir = (18.*cc.pi*cc.pi + x*(82.-39.*x))/omega_z
        elif self.q0 == 0.0:
            _d_vir = (18.*cc.pi*cc.pi + x*(60.-32.*x))/omega_z
        else:
            raise cex.CosmologyUnapplicable(self.m0, self.q0, self.k0)
        return(_d_vir)

    def delta2vir(self, z, mdelta, cdelta, DELTA_HALO):
        _d_vir = self.virial_overdensity(z)
        mvir = MassConvert.mconvert(mdelta*self.h, DELTA_HALO, _d_vir, cdelta)
        cvir = MassConvert.cconvert(mdelta*self.h, DELTA_HALO, _d_vir, cdelta)
        return(mvir/self.h, cvir)

    def vir2delta(self, z, mvir, cvir, DELTA_HALO):
        """ Derived mass and concentration of a halo defined with 
        overdensity DELTA_HALO at redshift z, given its virial
        mass and virial concentration.

        From Hu & Kravtsov (2003).

        Parameters
        ----------
        z: scalar 
            Redshift.
        mvir: scalar
            Halo virial mass (in Msol units, does not matter if set_hinv is on).
        cvir: scalar
            :math:`R_s/R_vir`, Halo concentration defined by virial radius.
        DELTA_HALO: scalar
            Desired spherical overdensity (w.r.t. background density).

        Returns
        -------
        mnew: scalar
            Halo mass if redefined by DELTA_HALO.
        cnew: scalar
            Halo concentration if redefined by DELTA_HALO.

        """
        _d_vir = self.virial_overdensity(z)
#        print(_d_vir)
        # input for MassConvert is in h^-1Msol units.
        mnew = MassConvert.mconvert(mvir*self.h, _d_vir, DELTA_HALO, cvir)
        cnew = MassConvert.cconvert(mvir*self.h, _d_vir, DELTA_HALO, cvir)
        return(mnew/self.h, cnew)

    def delta_crit(self, z):
        """ This function calculates the barrier height :math:`\delta_c` 
        at redshift z using the fitting formulae from Eke et al. (1993) 
        and Lacey & Cole (1993) for different cosmologies. 

        .. warning ::
        
            Taken from Zhao et al. (2009) code, which claims it was 
            copied from A. Jenkins' code. Exactly origin still unknown.

        Parameters
        ----------
        z: scalar or array_like
            Redshift

        Returns
        -------
        delta_c: scalar or array_like
            Critical overdensity.

        """
        if self.isEdS:
            fdeltac = 1.0
        elif self.isflat:
            omega_z = self.omega_M_z(z)
            fdeltac = (1. - 0.0052*(1.-omega_z) - 
                            0.009*np.power(1.-omega_z, 3) -
                            0.01*np.power(1.-omega_z, 18))
        elif self.isopen and (self.q0 == 0.0):
            omega_z = self.omega_M_z(z)
            fdeltac = self._rfac(omega_z)
        else:
            raise cex.CosmologyUnapplicable(self.m0, self.q0, self.k0)

        _delta_crit_z = 1.6864702*fdeltac/self.growth_factor(z)

        return(_delta_crit_z)

    def _rfac(self, omega):
        """ Internal function used by delta_crit.
        """
        delta_c = 1.6864702
        ceta = 2./omega-1                                                 
        seta = np.sqrt(ceta**2-1.)                                           
        eta = np.log(ceta+seta) 
        rfac = (1.5/delta_c*(1.+np.power((2.*cc.pi/(seta-eta)), 2./3.))*              
                (3.*seta*(seta-eta)/(ceta-1.)**2-2.))
        return(rfac)

    def __str__(self):
        return('At z=%g: Omega_m %g, rho_crit %g M_sun/Mpc^3'%
                                       (
                                        0.0, 
                                        self.omega_M_z(0.0), 
                                        self.rho_mean_z(0.0),
                                        ))



class DensityTest(unittest.TestCase):
    def runTest(self):
#        cp  = CosmoParams(omega_M_0=1, set_flat=True)
        cp  = CosmoParams(set_flat=True)
        den = Density(cp)
        print(den)
        z=np.asarray([0.0, 0.1, 0.3, 1.0, 3.0, 5.0])
        rho_crit = den.rho_mean_z(z)
        delta_crit = den.delta_crit(z)
#        print(rho_crit)
#        print(delta_crit)
        for zi in z:
            print("redshift: %5.3f -> crit density: %g M_sun/Mpc^3" % 
                    (zi, den.rho_crit_z(zi)))
        for zi in z:
            print("redshift: %5.3f -> growth factor: %g " % 
                    (zi, den.growth_factor(zi)))
        for zi in z:
            print("redshift: %5.3f -> sigma_8: %g " % 
                    (zi, den.sigma_8_z(zi)))
        mvir = 1.e10
        cvir = 10.0
        DELTA_HALO = 200.
        print("for a halo with mvir: %g, cvir: %g " % 
                    (mvir, cvir))
        for zi in z:
            delta_vir  = den.virial_overdensity(zi)
            mnew, cnew = den.vir2delta(zi, mvir, cvir, DELTA_HALO)
            print(("redshift: %5.3f Delta_vir: %5.3f -> m200: %g, c200: %g") % 
                    (zi, delta_vir, mnew, cnew))


if __name__=='__main__':
    unittest.main()

