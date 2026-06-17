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

        # CPL dark-energy equation of state:
        #     w(a) = w0 + wa * (1 - a)
        # These getattr calls keep the class backward compatible with older
        # CosmoParams objects that only define Lambda-CDM parameters.
        self.w0 = self._get_cosmo_param(cp,
                                        ['w0', 'w_0', 'omega_w_0',
                                         'dark_energy_w0'],
                                        -1.0)
        self.wa = self._get_cosmo_param(cp,
                                        ['wa', 'w_a', 'omega_w_a',
                                         'dark_energy_wa'],
                                        0.0)

    def _get_cosmo_param(self, cosmo, names, default):
        """Return the first available cosmological parameter in names."""
        for name in names:
            if hasattr(cosmo, name):
                return(getattr(cosmo, name))
        return(default)

    def de_density_factor_z(self, z):
        """Dark-energy density evolution relative to z=0.

        For the CPL form w(a) = w0 + wa * (1 - a),

            rho_de(z) / rho_de(0)
              = (1 + z) ** [3 * (1 + w0 + wa)]
                * exp[-3 * wa * z / (1 + z)].

        This is 1 for Lambda-CDM, i.e. w0 = -1 and wa = 0.
        """
        z = np.asarray(z)
        return(np.power(1.0 + z, 3.0 * (1.0 + self.w0 + self.wa)) *
               np.exp(-3.0 * self.wa * z / (1.0 + z)))

    def _e2_z(self, z):
        """E(z)^2 for matter + curvature + CPL dark energy."""
        return(self.m0 * np.power(1.0 + z, 3.0) +
               self.k0 * np.power(1.0 + z, 2.0) +
               self.q0 * self.de_density_factor_z(z))

    def _e2_a(self, a):
        """E(a)^2 for matter + curvature + CPL dark energy."""
        de = (np.power(a, -3.0 * (1.0 + self.w0 + self.wa)) *
              np.exp(3.0 * self.wa * (a - 1.0)))
        return(self.m0 * np.power(a, -3.0) +
               self.k0 * np.power(a, -2.0) +
               self.q0 * de)

    def _dlnh_dlna(self, a):
        """d ln H / d ln a for matter + curvature + CPL dark energy."""
        de = (np.power(a, -3.0 * (1.0 + self.w0 + self.wa)) *
              np.exp(3.0 * self.wa * (a - 1.0)))
        d_de_dlna = de * (-3.0 * (1.0 + self.w0 + self.wa) +
                          3.0 * self.wa * a)
        e2 = self._e2_a(a)
        d_e2_dlna = (-3.0 * self.m0 * np.power(a, -3.0) -
                     2.0 * self.k0 * np.power(a, -2.0) +
                     self.q0 * d_de_dlna)
        return(0.5 * d_e2_dlna / e2)

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
        return(np.sqrt(self._e2_z(z)))

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
        return(self.q0*self.de_density_factor_z(z)/np.power(_e_z, 2.0))

    def omega_de_z(self, z):
        """Dark-energy density parameter as a function of redshift."""
        return(self.omega_lambda_z(z))

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

    def _growth_rhs(self, x, y):
        """Right-hand side for growth ODE in x = ln(a).

        y[0] = D and y[1] = dD/dln(a).  The ODE is

            D'' + [2 + dlnH/dlna] D' - 3/2 Omega_m(a) D = 0,

        valid for smooth, non-clustering dark energy.
        """
        a = np.exp(x)
        e2 = self._e2_a(a)
        omega_m_a = self.m0 * np.power(a, -3.0) / e2
        dydx0 = y[1]
        dydx1 = (1.5 * omega_m_a * y[0] -
                 (2.0 + self._dlnh_dlna(a)) * y[1])
        return(np.asarray([dydx0, dydx1]))

    def _growth_unnorm_scalar(self, z, a_init=1.0e-4, nstep_per_lna=256):
        """Unnormalized linear growth factor for one redshift."""
        a_target = 1.0/(1.0 + float(z))
        if a_target <= 0.0:
            raise ValueError('Redshift must satisfy z > -1.')

        # Deep in matter domination, D(a) is proportional to a.  If the
        # requested redshift is even earlier than a_init, this asymptotic
        # value is already the desired unnormalized growth to high accuracy.
        if a_target <= a_init:
            return(a_target)

        x0 = np.log(a_init)
        x1 = np.log(a_target)
        nstep = max(8, int(np.ceil((x1 - x0) * nstep_per_lna)))
        h = (x1 - x0)/float(nstep)

        y = np.asarray([a_init, a_init], dtype=float)
        x = x0
        for i in range(nstep):
            k1 = self._growth_rhs(x, y)
            k2 = self._growth_rhs(x + 0.5*h, y + 0.5*h*k1)
            k3 = self._growth_rhs(x + 0.5*h, y + 0.5*h*k2)
            k4 = self._growth_rhs(x + h, y + h*k3)
            y = y + h*(k1 + 2.0*k2 + 2.0*k3 + k4)/6.0
            x = x + h
        return(y[0])

    def growth_factor(self, z, set_normed=True):
        """Linear growth factor.

        For Lambda-CDM this replaces the older Carroll, Press, & Turner
        approximation with a direct ODE solution.  For w0waCDM it solves the
        same linear-growth equation with the CPL dark-energy equation of state

            w(a) = w0 + wa * (1 - a),

        assuming smooth, non-clustering dark energy.  With set_normed=True,
        the result is normalized to D(z=0) = 1.

        Parameters
        ----------
        z: scalar or array_like
            Redshift.
        set_normed: bool, optional
            If True, return D(z)/D(0).  If False, return the unnormalized
            solution with early-time initial condition D(a_init)=a_init.

        Returns
        -------
        growth_factor: scalar or array_like
            Linear growth factor.
        """
        if not self.isflat:
            raise cex.NonFlatUniverseError(self.k0)

        zarr = np.asarray(z)
        zflat = np.atleast_1d(zarr).astype(float).ravel()
        dflat = np.asarray([self._growth_unnorm_scalar(zi) for zi in zflat])

        if set_normed:
            dflat = dflat/self._growth_unnorm_scalar(0.0)

        dout = dflat.reshape(np.atleast_1d(zarr).shape)
        if np.isscalar(z):
            return(float(dout[0]))
        return(dout)

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

