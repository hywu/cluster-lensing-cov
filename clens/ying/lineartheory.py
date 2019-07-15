#!/usr/bin/env python
#Last-modified: 13 Nov 2013 11:01:17 PM


""" Routines related to the Linear Density Field.
"""

#---------------------------------------------------------------------
import warnings
import unittest
#---------------------------------------------------------------------
import numpy as np
import scipy.integrate as si
                                                                        

from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
# Heidi: not used here 
# try :                                                                    
#     from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
# except ImportError :                                                     
#     from zypy.contrib.cubicSpline import NaturalCubicSpline as spline1d

#from math import log, log10, sqrt, exp
#---------------------------------------------------------------------
from .param import CosmoParams
from .density import Density
from . import constants as cc
from . import cex
#---------------------------------------------------------------------
# Heidi: not used here
# try :
#     import vecNumSBT
# except ImportError:
#     print "Error import vecNumSBT, "
#     print "Spherical Bessel Transformation will not work."
#     print "You may want to update your LD_LIBRARY_PATH."
#---------------------------------------------------------------------


#---------------------------------------------------------------------
class LinearTheory(object):
    """ Linear density field statistics for given cosmology at 
    certain redshift.

    Parameters
    ----------
    cosmo : CosmoParams class, optional
        Input cosmology (default: None, i.e., a flat WMAP CDM)
    
    z : float, optional
        Redshift to be extrapolated (default: 0.0).

    den : Density class, optional
        Input densities (default: None, i.e., Density(cosmo))

    set_warnsig8err : bool, optional
        Whether to warn large errors in :math:`\sigma_8` (default: True)

    """
    def __init__(self, cosmo=None, z=0.0, den=None, set_warnsig8err=True):
        """ Initialize the LinearTheory class.
        """
        self.set_warnsig8err = set_warnsig8err
        if cosmo is None:
            cp = CosmoParams(set_flat=True)
        else:
            cp = cosmo
            if not cp.isflat:
                warnings.warn("Non-flat Universe encountered!")
        # redshift
        self.z   = z
        # copy the cosmological parameters
        self.m0  = cp.omega_M_0
        self.q0  = cp.omega_lambda_0
        self.k0  = cp.omega_k_0
        self.b0  = cp.omega_b_0
        self.h   = cp.h
        self.n0  = cp.omega_n_0
        self.nnu = cp.N_nu
        self.n   = cp.n
        self.s0  = cp.sigma_8
        if int(self.nnu) != self.nnu: 
            raise TypeError('N_nu must be an integer.')
        if (self.nnu < 1):
            self.nnu = 1
        if (self.b0 <= 0):
            self.b0 = 1e-5
        if (self.n0 <= 0):
            self.n0 = 1e-5
        # FIXME Density class is initiated multiple times everywhere, any idea to 
        # avoid recreating them?
        if den is None:
            den = Density(cp)
        # density and growth factor at redshift=z
        self.mz       = den.omega_M_z(self.z)
        self.qz       = den.omega_lambda_z(self.z)
        self.fgrowth  = den.growth_factor(self.z)
        self.sz       = den.sigma_8_z(self.z)
        # initialize the EH transfer function routine
        # assuming T_cmb = 2.728 K 
        self.theta_cmb = 2.728/2.7  
        (self.omhh, self.f_hdm, self.f_cb, self.growth_k0, 
         self.p_cb, self.alpha_gamma,      
         self.sound_horizon_fit, self.beta_c) = self._TFmdm_set_cosm()
        deltaSqr = self.norm_power
        self.deltaSqr = deltaSqr
        # vectorized version of _sigmasq_r_scalar
        self._sigmasq_r_vec = np.vectorize(self._sigmasq_r_scalar)


    def _TFmdm_set_cosm(self):
        """ Initialize Eisenstein & Hu (1999) transfer function 
        calculation.

        This function calculates all the nesessary quantities that are 
        independent of the wavenumber k.
        
        Parameters
        ----------
        None

        Returns
        -------
        omhh: 
            $\Omega_m h^2$
        f_hdm: 
            density fraction of massive neutrinos.
        f_cb: 
            density fraction of CDM+baryons.
        growth_k0: 
            growth factor, different normalization from zycosmo.Density.
        p_cb: 
            growth rate modified by the massive neutrinos.
        alpha_gamma: 
            baryonic suppression factor.
        sound_horizon_fit: 
            sound horizon scale.
        beta_c: 
            paramter related to the Baryon+Neutrino suppression.

        """
        omega_lambda   = self.q0
        omega_matter   = self.m0
        omega_lambda_z = self.qz
        omega_matter_z = self.mz
        theta_cmb      = self.theta_cmb
        redshift       = self.z
        h2         = self.h*self.h
        omhh       = self.m0*h2
        obhh       = self.b0*h2
        onhh       = self.n0*h2
        f_baryon   = self.b0/self.m0
        f_hdm      = self.n0/self.m0
        f_cdm      = 1.0 - f_baryon - f_hdm
        f_cb       = f_cdm + f_baryon
        f_bnu      = f_baryon + f_hdm
        num_degen_hdm = self.nnu
#       equality scale (Eqn 1 in EH99)
        z_equality = 25000.0*omhh/theta_cmb**4 # Actually 1+z_eq 
        k_equality = 0.0746*omhh/theta_cmb**2
#       drag epoch and sound horizon (Eqn 2)
        z_drag_b1 = 0.313*pow(omhh,-0.419)*(1+0.607*pow(omhh,0.674))
        z_drag_b2 = 0.238*pow(omhh,0.223)
        z_drag = 1291*pow(omhh,0.251)/(1.0+0.659*pow(omhh,0.828))*\
                (1.0+z_drag_b1*pow(obhh,z_drag_b2))
#       (Eqn 3)
        y_drag = z_equality/(1.0+z_drag)
#       (Eqn 4)
        sound_horizon_fit = 44.5*np.log(9.83/omhh)/np.sqrt(1.0+10.0*np.power(obhh,0.75))
#       Set up for the free-streaming & infall growth function */
#       (Eqn 11)
        p_c = 0.25*(5.0-np.sqrt(1+24.0*f_cdm))
        p_cb = 0.25*(5.0-np.sqrt(1+24.0*f_cb))
#       (Eqn 10)
#       growth_k0 is used for calculating the free streaming scale
        growth_k0 = (z_equality/(1.0+redshift)*2.5*
                     omega_matter_z/(pow(omega_matter_z,4.0/7.0)-
                     omega_lambda_z+(1.0+omega_matter_z/2.0)*
                     (1.0+omega_lambda_z/70.0)))
#       for scale-independent growth of power spectrum 
#       (same as Density.growth_factor)
#        growth_to_z0 = (z_equality*2.5*omega_matter/
#                        (pow(omega_matter,4.0/7.0)-
#                        omega_lambda +
#                        (1.0+omega_matter/2.0)*(1.0+omega_lambda/70.0)))
#        growth_to_z0 = growth_k0/growth_to_z0

#       Compute small-scale suppression 
#       (Eqn 15)
        alpha_nu = (f_cdm/f_cb*(5.0-2.*(p_c+p_cb))/(5.-4.*p_cb)*
                    pow(1+y_drag,p_cb-p_c)*
                    (1+f_bnu*(-0.553+0.126*f_bnu*f_bnu))/
                    (1-0.193*np.sqrt(f_hdm*num_degen_hdm)+
                    0.169*f_hdm*pow(num_degen_hdm,0.2))*
                    (1+(p_c-p_cb)/2*(1+1/(3.-4.*p_c)/(7.-4.*p_cb))/(1+y_drag))
                   )
        alpha_gamma = np.sqrt(alpha_nu)
#       (Eqn 21)
        beta_c = 1/(1-0.949*f_bnu)
#       Done setting scalar variables
        return(omhh, f_hdm, f_cb, growth_k0, p_cb, 
               alpha_gamma, sound_horizon_fit, beta_c)


    def _TFmdm_k_mpc(self, k):
        """ Internal transfer Function for scalar k in 1/Mpc unit.

        This function calculates the transfer function at wavenumber k.
        Modified from Eisenstain & Hu (1999).

        Parameters
        ----------
        k: scalar or array_like
            wavenumbers in 1/Mpc unit.

        Returns
        -------
        tf_cb: scalar or array_like
            transfer function value with CDM+Baryon contribution.
        tf_cbnu: scalar or array_like
            transfer function value with CDM+Baryon+Neutrino contribution.
        """
        (omhh, f_hdm, f_cb, growth_k0, p_cb,
         alpha_gamma, sound_horizon_fit, 
         beta_c) = (self.omhh, self.f_hdm, self.f_cb, self.growth_k0, 
                    self.p_cb,
                    self.alpha_gamma, self.sound_horizon_fit,
                    self.beta_c)
        num_degen_hdm = self.nnu
#       (Eqn 5)
        qq = k/omhh*(self.theta_cmb*self.theta_cmb)
#       Compute the scale-dependent growth function
#       (Eqn 14)
        y_freestream = (17.2*f_hdm*(1+0.488*pow(f_hdm,-7.0/6.0))*
                        (num_degen_hdm*qq/f_hdm)**2)
        temp1 = pow(growth_k0, 1.0-p_cb)
        temp2 = np.power(growth_k0/(1+y_freestream),0.7)
#       (Eqn 12)
        growth_cb = np.power(1.0+temp2, p_cb/0.7)*temp1
#       (Eqn 13)
        growth_cbnu = np.power(pow(f_cb,0.7/p_cb)+temp2, p_cb/0.7)*temp1

#       Compute the master function
#       (Eqn 16)
        gamma_eff =(omhh*(alpha_gamma+(1-alpha_gamma)/
                (1+(k*sound_horizon_fit*0.43)**4)))
#       (Eqn 17)
        qq_eff = qq*omhh/gamma_eff
#       (Eqn 19)
        tf_sup_L = np.log(2.71828+1.84*beta_c*alpha_gamma*qq_eff)
#       (Eqn 20)
        tf_sup_C = 14.4+325/(1+60.5*pow(qq_eff,1.11))
#       (Eqn 18)
        tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*qq_eff**2)
#       (Eqn 23)
        qq_nu = 3.92*qq*np.sqrt(num_degen_hdm/f_hdm)
#       (Eqn 22)
        max_fs_correction = (1+1.2*pow(f_hdm,0.64)*
                             pow(num_degen_hdm,0.3+0.6*f_hdm)/
                             (np.power(qq_nu,-1.6)+np.power(qq_nu,0.8)))
        tf_master = tf_sup*max_fs_correction

#       Now compute the CDM+baryon transfer function
        tf_cb = tf_master*growth_cb/growth_k0
#       and the CDM+HDM+baryon transfer function
        tf_cbnu = tf_master*growth_cbnu/growth_k0
        return(tf_cb, tf_cbnu)


    def transfer_function(self, k, set_nu=False):
        """ Calculate the transfer function. 
        
        This function calculates the transfer function for a scalar or 
        a vector wavenumber k, with or without accounting for the 
        contribution from massive neutrinos.

        Parameters
        ----------
        k: scalar or array_like
            Input wavenumbers, in units of *bare* 1/Mpc.
        set_nu: boolean, optional
            True if including massive neutrinos, otherwise False
            (default: False).

        Returns
        -------
        tf: scalar or array_like
            Transfer function, dimension is w.r.t. k.

        """
        if set_nu:
            retval = 1
        else:
            retval = 0
        return(self._TFmdm_k_mpc(k)[retval])

    @property
    def norm_power(self):
        """ Normalize the power spectrum to the specified sigma_8.

        This function calculates the normalization factor for the 
        current power spectrum. If the power spectrum is for redshifts 
        other than zero, the function scales :math:`\sigma_8` back to the 
        specific redshift assuming scale-independent grow factors, 
        then normalizes the power spectrum to :math:`\sigma_8(z)`.
        
        Returns
        -------
        deltaSqr: scalar
            Normalization factor.

        """
        self.deltaSqr = 1.0
        sigz = self.sz
        deltaSqr = ((sigz/ self.sigma_r(8.0/self.h))**2.0)
        self.deltaSqr = deltaSqr 
        sig8 = self.sigma_r(8.0/self.h)
        sigma_8_error = (sig8 - sigz)/sigz
        if (sigma_8_error > 1e-4) and (self.set_warnsig8err):
            warnings.warn("High sigma_8 fractional error = %.4g" % 
                           sigma_8_error)
        del self.deltaSqr
        return(deltaSqr)

    def power_spectrum(self, k, set_nu=False):
        """ Linear power spectrum.

        Parameters
        ----------
        k: scalar or array_like
            Wavenumber in units of 1/Mpc.
        set_nu: boolean, optional
            True if including massive neutrinos, otherwise False
            (default: False).

        Returns
        -------
        ps: scalar or array_like
            Power spectrum, with the dimension of volume
        """
        n = self.n
        h = self.h
        growthFact = self.fgrowth
        transFunc  = self.transfer_function(k, set_nu=set_nu)
        if hasattr(self, "deltaSqr"):
            deltaSqr   = self.deltaSqr
        else:
            deltaSqr   = 1.0
        ps = (deltaSqr*(2.*cc.pi**2.)*k**n*
              (cc.c_light_Mpc_s/(h*cc.H100_s))**(3. + n)*
              (transFunc * growthFact)**2.
             )
        return(ps)

    def power_spectrum_hinv(self, khinv, set_nu=False):
        """ Power spectrum with k in unit of h/Mpc

        .. warning::
            
            P(k) has the dimension of (length)^3, so the h-free
            P(k) requires not only the rescaling of k by h, but also
            that of P(k) by h^3.

        Parameters
        ----------
        khinv: scalar or array_like
            Wavenumber in units of h/Mpc
        set_nu: boolean, optional
            True if including massive neutrinos, otherwise False
            (default: False).

        Returns
        -------
        ps: scalar or array_like
            Power spectrum shape, with the same dimension as k.
        """
        return(self.power_spectrum(khinv*self.h, set_nu=set_nu)*self.h*self.h*self.h)

    def sigma_r_0(self, r):
        """ R.M.S. of the density field on scale r at redshift zero.

        .. note::

            This mainly serves as the input for the calculation of
            halo concentration, which require the r.m.s value at z=0 
            as a proxy for universal halo mass scale.

        .. warning::

            It integrates the power spectrum directly, so if you are to
            calculate sigma_r_0 multiple times for the same cosmology, the
            interpolation-based version *sigma_r_0_interp* should be used.

        Parameters
        ----------
        r: scalar or array_like
            Scale on which the r.m.s is computed, in units of Mpc.

        Returns
        -------
        sigma: scalar or array_like
            R.M.S. result with the same dimension as r.
        error: scalar or array_like
            Error estimated from the integrator.

        """
        return(self.sigma_r(r)/self.fgrowth)

    def sigma_r_0_hinv(self, rhinv):
        return(self.sigma_r_0(rhinv/self.h))

    def sigma_r(self, r, set_reterr=False):
        """ R.M.S. of the density field on scale r.

        This function calculates the root mean square of the linear 
        density field defined by
        :py:func:`zypy.zycosmo.lineartheory.LinearTheory.power_spectrum`.

        .. warning ::

            This is the density r.m.s. at redshift z, rather than 0.

        .. warning::

            It integrates the power spectrum directly, so if you are to
            calculate sigma_r multiple times for the same cosmology, the
            interpolation-based version *sigma_r_interp* should be used.

        Parameters
        -----------
        r: scalar or array_like
            Scale on which the r.m.s is computed, in units of Mpc.
        set_reterr: boolean, optional
            True if error return is needed (default: False)

        Returns
        -------
        sigma: scalar or array_like
            R.M.S. result with dimension the same as r.
        error: scalar or array_like, optional
            Error estimated from the integrator.

        """

        if hasattr(self, "deltaSqr"):
            deltaSqr   = self.deltaSqr
        else:
            deltaSqr   = 1.0

        if np.isscalar(r):
            sigmasq_0, errorsq_0 = self._sigmasq_r_scalar(r)
        else:
#            _nr = len(r)
#            sigmasq_0 = np.zeros(_nr)
#            errorsq_0 = np.zeros(_nr)
#            for i in xrange(_nr):
#                sigmasq_0[i], errorsq_0[i] = self._sigmasq_r_scalar(r[i])
            sigmasq_0, errorsq_0 = self._sigmasq_r_vec(r)

        sigma = np.sqrt(sigmasq_0)
#       Propagate the error on sigmasq_0 to sigma.
        if set_reterr:
            error = errorsq_0/(2.0*sigmasq_0)
            return(sigma, error)
        else:
            return(sigma)

    def _sigmasq_r_scalar(self, r):
        """ Calculate :math:`\sigma^2(r)`.

        This function calculates the variance of density field at r.

        Parameters
        ----------
        r: scalar
            Scale in unit of Mpc.

        Returns
        -------
        sigma_r^2: scalar
            Density variance.
        error: scalar
            Error estiamted from scipy integrator quad.
        """

#        logk_lim = self._klims(r)
#        integral, error = si.quad(self._sigmasq_integrand_log,
#                logk_lim[0],
#                logk_lim[1],
#                args=(r),
#                limit=10000)

        integral, error =(
                          np.array(si.quad(self._sigmasq_integrand_log,
                                   np.log(0.01/r), np.log(0.1/r),
                                   args=(r),
                                   limit=50, epsabs=1.e-7)) 
                          + 
                          np.array(si.quad(self._sigmasq_integrand_log,
                                   np.log(0.1/r), np.log(1.0/r),
                                   args=(r),
                                   limit=50, epsabs=1.e-7)) 
                          + 
                          np.array(si.quad(self._sigmasq_integrand_log,
                                   np.log(1.0/r), np.log(10.0/r),
                                   args=(r),
                                   limit=50, epsabs=1.e-7)) 
                          + 
                          np.array(si.quad(self._sigmasq_integrand_log,
                                   np.log(10.0/r), np.log(100.0/r),
                                   args=(r),
                                   limit=50, epsabs=1.e-7))
                         )

        return(1.e10*integral/(2.*cc.pi**2.0), 1.e10*error/(2.*cc.pi**2.0))

    def _klims(self, r, min_fraction=1.e-4):
        """ Find the range of r that should be safe to integrate sigmasq_integrand over without loss of precision.
        """
        logk = np.arange(-20., 20., 0.1)
        integrand = self._sigmasq_integrand_log(logk, r)
        maxintegrand = np.max(integrand)
        highmask = integrand > maxintegrand*min_fraction
        while highmask.ndim > logk.ndim:
            highmask = np.logical_or.reduce(highmask)
        mink = np.min(logk[highmask])
        maxk = np.max(logk[highmask])
        return(mink, maxk)

    def _klims_pk(self, power_spectrum=None, min_fraction=1.e-8):
        """ Find the range of r that should be safe to integrate p(k) over without loss of precision.
        """
        logk = np.arange(-20.0, 20.0, 0.2)
        k    = np.exp(logk)
        if power_spectrum is None:
            pk = self.power_spectrum(k)
        else:
            pk = power_spectrum(k)
        maxpk = np.max(pk)
        highmask =pk > maxpk*min_fraction
        while highmask.ndim > k.ndim:
            highmask = np.logical_or.reduce(highmask)
        mink = np.min(k[highmask])
        maxk = np.max(k[highmask])
        return(mink, maxk)
    
    def _sigmasq_integrand_log(self, logk,  r):
        """ Integrand used internally by the sigma_r function.

        This function is the integrand used by sigma_r.

        Parameters
        ----------
        logk: scalar or array_like
            Natural Log of wavenumbers.
        r: scalar or array_like
            Scale of the radius of 3D variance sphere.

        Returns
        -------
        integrand: scalar or array_like
            :math:`(k/2\pi^2) k^2 W_{tophat}(k, r)^2 P(k)`
        """
        k      = np.exp(logk)

#       The 1e-10 factor in the integrand is added to avoid roundoff
#       error warnings. It is divided out later.
#       extra factor of k here is to compensate for that we are integrating over logk
        return (1.e-10*k**3.0* self.w_tophat(k, r)**2.0* self.power_spectrum(k))

    def w_tophat(self, k, r):
        """ The k-space Fourier transform of a spherical tophat.
        
        Parameters
        ----------
        k: scalar or array_like
            Wavenumbers.
        r: scalar or array_like
            Scale of the radius of 3D variance sphere.

        Returns
        -------
        w_tophat:
            :math:`\\frac{3(\sin(k r)-kr\cos(k r)}{(kr)^3}`

        """
        return(3.0*(np.sin(k*r)-k*r*np.cos(k*r))/((k*r)**3.0))

    def correlation_function(self, nexp=9, kmin=1.e-5, kmax=1.e3, 
                             rmax = 1000.0, power_spectrum=None, 
                             set_autoklim=False):
        """ Two-point correlation function by a fast spherical Bessel
        transformation algorithm developed by Talman J.D. (2009).

        .. warning::

            The validity of the method is strongly dependent on the 
            analytic properties of the functions being approximated.
            The ideal input function for the present approach is one 
            analytic on (0, +infty), with exponential decrease for 
            k approaches infinity, and k^l behavior at r = 0 and a small
            number of nodes.


        Parameters
        ----------
        nexp: int, optional
            Base-2 logarithm of the scale array length, needs to be
            less than or equal to 9 (default: 9).
        kmin: float, optional
            Minimum wavenumber (default: 1.e-5).
        kmax: float, optional
            Maximum wavenumber (default: 1.e+3).
        rmax: float, optional
            Maximum scale for the correlation function (default:
            1000).
        power_spectrum: callable object, optional
            Function to calculate power spectrum (default: EW power
            spectrum)
        set_autoklim: boolean, optional
            True if the calculation of density r.m.s has an adaptive
            cutoff in integration limits (default: False).

        Returns
        -------
        rr: ndarray
            Scale array, as determined by nexp, kmin, kmax, and rmax.
        xi: ndarray
            Correlation function value on rr.

        """
        if set_autoklim:
#           this can only make sure the large scale xi is *right*.
            kmin, kmax = self._klims_pk(power_spectrum=power_spectrum)
            print("Integrating from k = %g to %g." % (kmin, kmax))
#       number of mesh grid
        nr   = 2**nexp
#       rho is ln(k)
#       default k range, need to capture the large scale xi behavior.
        rhomin = np.log(kmin)
        rhomax = np.log(kmax)
        dk   = (rhomax - rhomin)/(nr-1)
#        rmax = 200.0
#       kp is ln(r)
        kpmin = np.log(rmax) - rhomax + rhomin

        rr = np.zeros(nr)
        kk = np.zeros(nr)
        cf = np.exp(dk)
        rr[0] = np.exp(kpmin)
        kk[0] = np.exp(rhomin)
        for i in xrange(1, nr):
            rr[i] = cf*rr[i-1]
            kk[i] = cf*kk[i-1]

        if power_spectrum is None:
            pk = self.power_spectrum(kk)
        else:
            if not hasattr(power_spectrum, "__call__"):
                raise cex.NonCallableObject("power_spectrum not callable")
            pk = power_spectrum(kk)

        xi = vecNumSBT.vecnumsbt(pk, kk, rr, 0, 0.0, nexp)
        xi = xi/(2.*cc.pi**2.0)
        return(rr, xi)

    # the following functions try to provide interpolation-based solutions to
    # matter autocorrelation and rms density variance. Several principles are:

    # 1) we want to perform the integration only once for an array of scales
    # that encompasses the need of all other applications.
    #
    # 2) we want to return a function that can serve as input for other classes
    # and functions.

    # for sigma(r) 

    def sigma_r_interp(self, rmin=0.001, rmax=100, nrbin=50, set_sigma0=False,
            set_hinv=False):
        """ Return a function for spline interpolating sigma_r.
        """
        if not hasattr(self, "sigma_r_table"):
            self.setup_sigma_r_table(rmin=rmin, rmax=rmax, nrbin=nrbin)

        def _sigma_r_func(rr):
            if set_hinv:
                r = rr/self.h
            else:
                r = rr
            if np.max(r) > self.sigtbl_rmax or np.min(r) < self.sigtbl_rmin:
                warnings.warn("input scale beyond sigma_r_table, resetting...")
                self.setup_sigma_r_table(
                        rmin=min(np.min(r), self.sigtbl_rmin),
                        rmax=max(np.max(r), self.sigtbl_rmax),
                        nrbin=self.sigtbl_nrbin)
            lgr = np.log10(r)
            if set_sigma0:
                return(self.sigma_r_table(lgr)/self.fgrowth)
            else:
                return(self.sigma_r_table(lgr))
        return(_sigma_r_func)

    def sigma_r_0_interp(self, rmin=0.001, rmax=100, nrbin=50):
        """ Return a function for spline interpolating sigma_r_0.
        """
        return(self.sigma_r_interp(rmin=rmin, rmax=rmax, nrbin=nrbin, set_sigma0=True))

    def sigma_r_0_interp_hinv(self, rmin=0.001, rmax=100, nrbin=50):
        """ Return a function for spline interpolating sigma_r_0 in hinv units.
        """
        return(self.sigma_r_interp(rmin=rmin, rmax=rmax, nrbin=nrbin,
            set_sigma0=True, set_hinv=True))

    def setup_sigma_r_table(self, rmin=0.001, rmax=100, nrbin=50):
        _lgr = np.linspace(np.log10(rmin), np.log10(rmax), num=nrbin)
        _r   = np.power(10.0, _lgr)
        _sig = self.sigma_r(_r)
        self.sigtbl_rmin = rmin
        self.sigtbl_rmax = rmax
        self.sigtbl_nrbin= nrbin
        self.sigma_r_table = spline1d(_lgr, _sig)

    # for correlation function
    def correlation_func_interp(self):
        """ Returns a function for spline interpolating vecNumSBT result.

        Returns
        -------
        f: function
            An cubic spline interpolation realization of the
            correlation function derived by
            :py:func:`zypy.zycosmo.lineartheory.LinearTheory.correlation_function`

        """
        if not hasattr(self, "correlation_func_table"):
            self.setup_correlation_func_table()
        def _correlation_func(r):
            if np.max(r) > self.cortbl_rmax or np.min(r) < self.cortbl_rmin:
                raise cex.ExtrapolationRequired()
            logr = np.log(r)
            return(self.correlation_func_table(logr))
        return(_correlation_func)

    def setup_correlation_func_table(self):
        _r, _x = self.correlation_function()
        self.cortbl_rmin = _r[0]
        self.cortbl_rmax = _r[-1]
        # use natural log here because NumSBT returns a 2^n grid.
        _logr = np.log(_r)
        self.correlation_func_table = spline1d(_logr, _x)

    def _xi_integrand_log(self, logk, r):
        """ This function has been deprecated and should never be used except
        for the example plotintegrand.

        Parameters
        ----------
        logk: scalar or array_like
            Natural log of wavenumbers
        r: scalar or array_like
            Distance scale in two-point statistics, in unit of Mpc.

        Returns
        -------
        integrand: scalar or array_like
            :math:`(k/2pi^2)k^2(\sin(kr)/(kr))P(k)`

        """
        k = np.exp(logk)
        x = k*r
        return(k* (1./(2.*cc.pi**2.0))*k**2.0* (np.sin(x)/x)* self.power_spectrum(k))

    def __str__(self):
        return( ('r = 8 Mpc/h,\n\
                  z = %g,\n\
                  norm: %g,\n\
                  transfer function: %g,\n\
                  power spectrum (without Nu): %g,\n\
                  sigma8 by integrating P(k) at z: %g,\n\
                  sigma8 by integrating P(k) at 0: %g,\n\
                  input sigma8: %g') % 
                (
                 self.z,
                 self.deltaSqr,
                 self.transfer_function(self.h/8.0),
                 self.power_spectrum(self.h/8.0),
                 self.sigma_r(8.0/self.h),
                 self.sigma_r_0(8.0/self.h),
                 self.sz,
                 )
                )


class LinearTheoryTest(unittest.TestCase):
    def testPS0(self):
        cp = CosmoParams(set_flat=True)
        powspec = LinearTheory(cp, z=0.0)
        print(powspec)
    def testPS1(self):
        cp = CosmoParams(set_flat=True)
        powspec = LinearTheory(cp, z=1.0)
        print(powspec)


def compareEHcode():
    """ Compare this code with the original code published by
    `Eisenstein & Hu  <http://background.uchicago.edu/~whu/transfer/transferpage.html>`_ .
    """
    from zypy.contrib.txtio import readtxt
    from zypy.zyutil import mypath
    import matplotlib.pyplot as plt
    import os

    this_dir = mypath(__file__)
    trans_EH = os.path.join(this_dir, "test", "trans_nu0.2.dat")

    k_h, t_master0, t_cb0, t_cbnu0 = readtxt(trans_EH)

    cp = CosmoParams(set_flat=True, omega_n_0=0.2, N_nu=1)
    k = k_h * cp.h

    lin = LinearTheory(cp, z=0.0)
    t_cb   = lin.transfer_function(k, set_nu=False)
    t_cbnu = lin.transfer_function(k, set_nu=True)

    fig1 = plt.figure(1, figsize=(8, 6))
    ax1  = fig1.add_subplot(111)
    ax1.plot(k_h,t_cb0, "g-", label="EH, no $\\nu$", lw=3, alpha=0.5) 
    ax1.plot(k_h,t_cbnu0, "r-", label="EH, $\\nu$", lw=6, alpha=0.5)
    ax1.plot(k_h,t_cb , "g--", label="YZ, no $\\nu$", lw=1, alpha=0.9) 
    ax1.plot(k_h,t_cbnu, "r--", label="YZ, $\\nu$", lw=2, alpha=0.9)
    ax1.legend(loc=1)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("k [h/Mpc]")
    ax1.set_ylabel("T(k)")
    plt.show()

def testCorrelationFunc():
    """ Test vecNumSBT code.
    """
    from zypy.contrib.txtio import readtxt
    from zypy.zyutil import mypath
    import matplotlib.pyplot as plt
    import os

    this_dir = mypath(__file__)
    xi_fid_dat = os.path.join(this_dir, "test", "xi_mat_fid.dat")
    r_hinv, xi0 = readtxt(xi_fid_dat, usecols=(0,1)) 

    cp = CosmoParams(set_flat=True)
    lin = LinearTheory(cp, z=0.0)
    r = r_hinv / cp.h
    xi3d = lin.correlation_func_interp()
    xi = xi3d(r)

    fig2 = plt.figure(2, figsize=(8, 6))
    ax2  = fig2.add_subplot(111)
    ax2.plot(r_hinv,xi0, "r-", label="brute integration", lw=6, alpha=0.5) 
    ax2.plot(r_hinv,xi, "r--", label="NumSBT", lw=2, alpha=0.9) 
    ax2.legend(loc=1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(0.1, 100)
    ax2.set_xlabel("r [Mpc/h]")
    ax2.set_ylabel("$\\xi(r)$")
    plt.show()

def testxisigma8():
    """ sigma8 dependence of xi.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(2, figsize=(8, 6))
    ax  = fig.add_subplot(111)
    ss = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    os = [0.2/r for r in ss]
    cs = ["k", "r", "m", "g",  "c",  "b"]

    lgr = np.linspace(-2.0, 2.0, 30)
    r   = np.power(10.0, lgr)
    for s, o,  c in zip(ss, os,  cs): 
        cp   = CosmoParams(sigma_8=s, omega_M_0=o, set_flat=True)
        lin  = LinearTheory(cp, z=0.0)
        xi3d = lin.correlation_func_interp()
        xi = xi3d(r)
        ax.plot(r, xi, lw=1, ls="-", color=c, label="$\sigma_8$="+str(s)+"$\Omega_m$="+str(o)) 

    ax.legend(loc=2, ncol=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.001, 100)
#    ax.set_ylim(1e-5, 5e10)
    ax.set_xlabel("r [Mpc]")
    ax.set_ylabel("$\\xi(r)$")
    plt.show()

def testpkshape():
    """ Shape of power spectrum.
    """
    import matplotlib.pyplot as plt
    log10k = np.arange(-5, 5, 0.1)
    k = np.power(10.0, log10k)
    
    cp_fid   = CosmoParams(sigma_8=0.80, omega_M_0=0.30, h=0.7, omega_b_0=0.04, n=1.0, set_flat=True)
    lin_fid  = LinearTheory(cp_fid)

    fig = plt.figure(figsize=(8, 8))
    ax  = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # k in absolute units 
    pk_fid  = lin_fid.power_spectrum(k)

    ax.plot(k, pk_fid, lw=2, ls="-")
    ax2.plot(k, pk_fid*0+1, lw=2, ls="-")

    const = 0.30*0.7*0.7

    oms = np.linspace(0.10, 0.40, 5)
#    nss = np.linspace(0.95, 1.50, 5)
    nss = np.ones(5)*1.0

    for om, ns  in zip(oms, nss):
        h = np.sqrt(const/om)
        omega_b_0=(0.04*0.7*0.7)/h/h
#        omega_b_0=0.04
        cp  = CosmoParams(sigma_8=0.80, omega_M_0=om, h=h, omega_b_0=omega_b_0, n=ns, set_flat=True)
        lin = LinearTheory(cp)
        pk  = lin.power_spectrum(k)
        ax.plot(k, pk, lw=1, ls="-", label="$\Omega_m=$"+str(om))
        ax2.plot(k, pk/pk_fid, lw=1, ls="-")

#    ax.legend(loc=1, ncol=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.0001, 10)
    ax.set_ylim(1e-1, 5e5)
    ax.set_xlabel("k [1/Mpc]")
    ax.set_ylabel("$P(k)$")
    ax2.set_xlim(0.0001, 10)
    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel("k [1/Mpc]")
    ax2.set_ylabel("ratio")
    ax2.set_xscale("log")
    plt.show()

def testpkevo():
    """ Evolution of power spectrum.
    """
    import matplotlib.pyplot as plt
    cp = CosmoParams(set_flat=True, omega_n_0=0.2, N_nu=3)
    log10k = np.arange(-3, 1, 0.1)
    k = np.power(10.0, log10k)

    fig = plt.figure(2, figsize=(8, 6))
    ax  = fig.add_subplot(111)

    zs = [0.0, 0.25, 1.0, 4.0, 16.0, 64.0]
    cs = ["k", "r", "m", "g",  "c",  "b"]
    for z, c in zip(zs, cs): 
        lin = LinearTheory(cp, z=z)
        pk1  = lin.power_spectrum(k)
        pk2  = lin.power_spectrum(k, set_nu=True)
        ax.plot(k, pk1, lw=2, ls="-", color=c, label="z="+str(z))
        ax.plot(k, pk2, lw=1, ls="--", color=c, label="z="+str(z)+" $\\nu$") 

    ax.legend(loc=1, ncol=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.001, 10)
    ax.set_ylim(1e-5, 5e10)
    ax.set_xlabel("k [1/Mpc]")
    ax.set_ylabel("$P(k)$")
    plt.show()

def testxiinterp():
    """ Illustrate the validity of interpolating the base-2 gridded
    correlation function.
    """
    import matplotlib.pyplot as plt
    cp = CosmoParams(set_flat=True)
    powspec0 = LinearTheory(cp, z=0.0)
    r0 = np.linspace(-2, 2, 10)
    r0 = np.power(10., r0)
    xi3d =powspec0.correlation_func_interp()
    xi0 = xi3d(r0)
    r1, xi1 = powspec0.correlation_function()
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111)
    ax.plot(r0, xi0, "ro", linewidth=3)
    ax.plot(r1, xi1, "k-", alpha=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 500)
    ax.set_ylim(1e-4, 1000)
    ax.set_xlabel("r [Mpc]")
    ax.set_ylabel("$\\xi(r)$")
    plt.show()

def testxiparams():
    import matplotlib.pyplot as plt
    cp = CosmoParams(set_flat=True)
    lin = LinearTheory(cp, z=0.0)
    r1, xi1 = lin.correlation_function(kmin=1.e-10, kmax=1.e3)
    r2, xi2 = lin.correlation_function(kmin=1.e-3, kmax=1.e10)
    r3, xi3 = lin.correlation_function(kmin=1.e-10, kmax=1.e10)
    r4, xi4 = lin.correlation_function(set_autoklim=True)
    r5, xi5 = lin.correlation_function()

    
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111)
    ax.plot(r1, xi1, "m-", alpha=1.0, label="kmin=1.e-10, kmax=1.e3")
    ax.plot(r2, xi2, "g-", alpha=1.0, label="kmin=1.e-3, kmax=1.e10")
    ax.plot(r3, xi3, "b-", alpha=1.0, label="kmin=1.e-10, kmax=1.e10")
    ax.plot(r4, xi4, "r-", alpha=1.0, label="set_autoklim=True")
    ax.plot(r5, xi5, "k-", alpha=0.5, lw=4, label="kmin=1.e-5, kmax=1.e3, default")
    ax.plot(r4, 2.e5/r4**4, "c--", label="$\\propto r^{-4}$")
    ax.plot(r4, 2.e0*(5./r4)**1.8, "y--", label="$\\propto r^{-1.8}$")
    ax.legend(loc="lower left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 500)
    ax.set_ylim(1e-4, 500)
    ax.set_xlabel("r [Mpc]")
    ax.set_ylabel("$\\xi(r)$")
    plt.show()

def plotintegrand():
    """ Illustrate the oscillation of the integrand in calculating
    correlation functions.
    """
    import matplotlib.pyplot as plt
    cp = CosmoParams(set_flat=True)
    logk = np.arange(-4, 4, 0.05)

    powspec = LinearTheory(cp, z=0.0)
    int0 = powspec._xi_integrand_log(logk, 0.01)
    int1 = powspec._xi_integrand_log(logk, 0.1)
    int2 = powspec._xi_integrand_log(logk, 1.0)
    int3 = powspec._xi_integrand_log(logk, 10.0)
    int4 = powspec._xi_integrand_log(logk, 100.0)
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111)
    ax.plot(logk, int0, "k-", label="r=0.01") 
    ax.plot(logk, int1, "b-", label="r=0.1")  
    ax.plot(logk, int2, "c-", label="r=1")
    ax.plot(logk, int3, "g-", label="r=10")
    ax.plot(logk, int4, "r-", label="r=100")
    ax.legend(loc="upper left")
    ax.set_xlabel("$\ln k$")
    ax.set_ylabel("$k^2 \\frac{\sin(kr)}{kr} P(k)$")
    plt.show()

def plotintegrand_sigmaR():
    """ Illustrate the oscillation of the integrand in calculating
    rms variance of the density field.
    """
    import matplotlib.pyplot as plt
    cp = CosmoParams(set_flat=True)
    logk = np.arange(-20, 20, 0.001)
    powspec = LinearTheory(cp, z=0.0)
    int0 = 1e10*powspec._sigmasq_integrand_log(logk, 0.01)
    int1 = 1e10*powspec._sigmasq_integrand_log(logk, 0.1)
    int2 = 1e10*powspec._sigmasq_integrand_log(logk, 1.0)
    int3 = 1e10*powspec._sigmasq_integrand_log(logk, 10.0)
    int4 = 1e10*powspec._sigmasq_integrand_log(logk, 100.0)
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111)
    ax.plot(logk, int0, "k-", label="r=0.01") 
    ax.plot(logk, int1, "b-", label="r=0.1")  
    ax.plot(logk, int2, "c-", label="r=1")
    ax.plot(logk, int3, "g-", label="r=10")
    ax.plot(logk, int4, "r-", label="r=100")
    ax.legend(loc="lower left")
    ax.set_xlabel("$\ln k$")
    ax.set_ylabel("$\propto\\frac{k^3}{2\pi^2}\\frac{3(\sin(kr)-kr\cos(kr))}{(kr)^3} P(k)$")
    ax.set_yscale("log")
    plt.show()

def plotwk():
    """ Illustrate the oscillation of the window funcion.
    """
    import matplotlib.pyplot as plt
    logk = np.arange(-10, 10, 0.001)
    k    = np.exp(logk)
    powspec = LinearTheory()
    int0 = powspec.w_tophat(k, 0.01)
    int1 = powspec.w_tophat(k, 0.1)
    int2 = powspec.w_tophat(k, 1.0)
    int3 = powspec.w_tophat(k, 10.0)
    int4 = powspec.w_tophat(k, 100.0)
    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111)
    ax.plot(logk, int0, "k-", label="r=0.01") 
    ax.plot(logk, int1, "b-", label="r=0.1")  
    ax.plot(logk, int2, "c-", label="r=1")
    ax.plot(logk, int3, "g-", label="r=10")
    ax.plot(logk, int4, "r-", label="r=100")
    ax.legend(loc="lower left")
    ax.set_xlabel("$\ln k$")
    ax.set_ylabel("$\\frac{3(\sin(kr)-kr\cos(kr))}{(kr)^3}$")
    ax.set_yscale("log")
    plt.show()

def testRMS():
    from zypy.contrib.txtio import readtxt
    from zypy.zyutil import mypath
    import matplotlib.pyplot as plt
    import os

    this_dir = mypath(__file__)
    sigmar_dat = os.path.join(this_dir, "test", "sigma_r_fid.dat")

    cp = CosmoParams(set_flat=True)
    lin = LinearTheory(cp, z=0.0)

    r_hinv, sigfid = readtxt(sigmar_dat)
    r = r_hinv / cp.h

    sig, err = lin.sigma_r(r, set_reterr=True)

    sig_func = lin.sigma_r_interp()
    sig2     = sig_func(r)
#    sig3     = sig_func(r[0:10]*10.0)

    fig = plt.figure(figsize=(8, 6))
    ax  = fig.add_subplot(111)
    ax.plot(r_hinv, sigfid, "r-", lw=6, alpha=0.5, label="fiducial")
    ax.plot(r_hinv, sig2, "go", alpha=0.5, label="spline")
    ax.errorbar(r_hinv, sig, yerr=err, fmt='o', markersize=3, ecolor='r', mfc='r', 
                label="$\sigma(r)$")
    ax.legend(loc="upper right")
    ax.set_xlabel("r [Mpc]")
    ax.set_ylabel("$\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.04, 100)
    ax.set_ylim(0.04, 10)
    plt.show()



if __name__ == '__main__':
#    unittest.main()
#    compareEHcode()
#    testCorrelationFunc()
     testpkevo()
#    plotintegrand()
#    testxiparams()
#    testxiinterp()
#    testRMS()
#    testxisigma8()
#    plotintegrand_sigmaR()
#    plotwk()
    # testpkshape()
