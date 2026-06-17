#Last-modified: 17 Apr 2013 02:07:51 PM



from math import sqrt
import numpy as np
import scipy.integrate as si

from .param import CosmoParams
from . import constants as cc
from . import cex


class FRW(object):
    def __init__(self,cosmo=None, set_hinv=False):                     
        """Calculate various cosmological distance measures. 
        Mostly follows David Hogg's pedagogical paper arXiv:astro-ph/9905116v4.
        
        .. warning ::
        
            Presumably all the parameters are self-evident, but I'll try to add in
            the full documenataion later.
        
        .. note ::
        
            Mostly copied from CosmoloPy directly.
        
        Distance units are Mpc, time units are seconds.
        
        
        Parameters                                                         
        ----------                                                           
        cosmo: CosmoParams object, optional                                 
            Cosmological parameters.                                        
        
        set_hinv: bool, optional
            h is set to one if True, when all the results are in units independent of h, otherwise
            h is explicitly included in the calculation.
        
        """
        if cosmo is None:                                               
            cp = CosmoParams(set_flat=True)                             
        else:                                                           
            cp = cosmo                              
        if set_hinv:
            self.h = 1.0
        else:
            self.h = cp.h
        self.m0 = cp.omega_M_0                                          
        self.q0 = cp.omega_lambda_0                                     
        self.k0 = cp.omega_k_0           

        # CPL dark-energy equation of state:
        #     w(a) = w0 + wa * (1 - a)
        # These getattr calls keep FRW backward compatible with older
        # Lambda-CDM CosmoParams objects, and match density_w0wa.py.
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

    def e_z(self, z):
        """Calculate the unitless Hubble expansion rate at redshift z.
        
        In David Hogg's (arXiv:astro-ph/9905116v4) formalism, this is
        equivalent to E(z), defined in his eq. 14.

        This version allows CPL/w0waCDM dark energy,
        w(a) = w0 + wa * (1 - a).  For w0=-1 and wa=0 it reduces to the
        original Lambda-CDM expression.
        
        """
        return(np.sqrt(self._e2_z(z)))    

    def omega_de_z(self, z):
        """Dark-energy density parameter as a function of redshift."""
        _e_z = self.e_z(z)
        return(self.q0*self.de_density_factor_z(z)/np.power(_e_z, 2.0))

    def omega_lambda_z(self, z):
        """Backward-compatible alias for omega_de_z."""
        return(self.omega_de_z(z))

    def hubble_z(self, z):
        """Calculate the value of the Hubble constant at redshift z.
    
        Units are s^-1
    
        In David Hogg's (arXiv:astro-ph/9905116v4) formalism, this is
        equivalent to H_0 * E(z) (see his eq. 14).
    
        """
        H_0 = self.h * cc.H100_s
        _hubble_z = H_0 * self.e_z(z)
        return(_hubble_z)

    def hubble_distance_z(self, z):
        """Calculate the value of the Hubble distance at redshift z.
        
        Units are Mpc.
        
        In David Hogg's (arXiv:astro-ph/9905116v4) formalism, this is
        equivalent to D_H / E(z) = c / (H_0 E(z)) [see his eq. 14], 
        which appears in the definitions of many other distance 
        measures.
        
        """
        _hubble_distance_z = cc.c_light_Mpc_s/self.hubble_z(z)
        return(_hubble_distance_z)

    def comoving_distance(self, z, z0 = 0.):
        """Calculate the line-of-sight comoving distance (in Mpc) to redshift z.
        
        See equation 15 of David Hogg's arXiv:astro-ph/9905116v4
        
        Units are Mpc.
        
        Optionally calculate the integral from z0 to z.
        
        Returns
        -------
        d_co: ndarray
           Comoving distance in Mpc.
        
        """
        
        dc_func = np.vectorize(lambda z, z0: 
                     si.quad(self.hubble_distance_z, z0, z, limit=1000)
                                 )
        d_co, err = dc_func(z, z0)
        return(d_co)

    def propper_motion_distance(self, z):
        """Returns comoving_distance_transverse."""
        return(self.comoving_distance_transverse(z))

    def comoving_distance_transverse(self, z):
        """Calculate the transverse comoving distance (in Mpc) to redshift z.
        
        This is also called the proper motion distance, D_M.
        
        See equation 16 of David Hogg's arXiv:astro-ph/9905116v4
        
        Units are Mpc.
        
        This is the distance d_m, such that the comoving distance between
        two events at the same redshift, but separated on the sky by some
        angle delta_theta is d_m * delta_theta.
        
        Warning: currently returns the error on the line-of-sight comoving
        distance D_C, not the error on the transverse comoving distance
        D_M.
        
        """

        d_c = self.comoving_distance(z, 0.0)

        if self.k0 == 0.0:
            d_m = d_c
        else:
            d_h = self.hubble_distance_z(0.0)
            sqrt_k0 = sqrt(abs(self.k0))
            if self.k0 < 0.0:
                d_m = d_h*np.sin(sqrt_k0*d_c/d_h)/sqrt_k0
            elif self.k0 > 0.0:
                d_m = d_h*np.sinh(sqrt_k0*d_c/d_h)/sqrt_k0
        return(d_m)

    def angular_diameter_distance(self, z, z0 = 0.0):
        """Calculate the angular-diameter distance (Mpc) to redshift z.
        
        Optionally find the angular diameter distante between objects at
        z0 and z.
        
        See equations 18-19 of David Hogg's arXiv:astro-ph/9905116v4
        
        Units are Mpc.
        """


        dm2  = self.comoving_distance_transverse(z)
        if z0 == 0.0:
            return(dm2/(1.+z))
        else:
            if self.k0 < 0.0:
                raise cex.CosmologyUnapplicable("Not for Omega_k < 0")
            dm1 = self.comoving_distance_transverse(z0)
            d_h_0 = self.hubble_distance_z(0.0)
            term1 = dm1 * np.sqrt(1.+self.k0*(dm2/d_h_0)**2.)
            term2 = dm2 * np.sqrt(1.+self.k0*(dm1/d_h_0)**2.)
            da12 = (term2-term1)/(1.+z) # only for Omega_k > 0
            return(da12)

    def luminosity_distance(self, z):
        """Calculate the luminosity distance to redshift z.
        
        Optionally calculate the integral from z0 to z.
        
        See, for example, David Hogg's arXiv:astro-ph/9905116v4
        
        """
        da = self.angular_diameter_distance(z)
        dl = da*(1.+z)**2.
        return(dl)

    def distance_modulus(self, z):
        """ Calculate the distance modulus DM = 5 log (D_L/10pc)
        assuming luminosity_distance is in Mpc.
        """
        return(5.0*np.log10(self.luminosity_distance(z))+25.0)

    def diff_comoving_volume(self, z):
        """Calculate the differential comoving volume element
        dV_c/dz/dSolidAngle.
        
        See David Hogg's arXiv:astro-ph/9905116v4, equation 28.
        
        """
    
        d_h_0 = self.hubble_distance_z(0.0)
        d_m = self.comoving_distance_transverse(z)
        ez  = self.e_z(z)
        dvc = d_h_0*d_m**2./ez
        return(dvc)

    def comoving_volume(self, z):
        """Calculate the comoving volume out to redshift z.

        See David Hogg's arXiv:astro-ph/9905116v4, equation 29.

        """
        dm = self.comoving_distance_transverse(z)
        flat_volume = 4.*cc.pi*np.power(dm, 3)/3.

        if (self.k0 == 0.0):
            return(flat_volume)
    
        d_h_0 = self.hubble_distance_z(0.0)

        sqrt_k0 = sqrt(abs(self.k0))
        dmdh = dm/d_h_0
        argument = sqrt_k0*dmdh
        f1 = 4.*cc.pi*d_h_0**3./(2.*self.k0)
        f2 = dmdh*np.sqrt(1.+self.k0*(dmdh)**2.)
        f3 = 1./sqrt_k0

        if self.k0 > 0.0:
            return(f1 * (f2 - f3 * np.arcsinh(argument)))
        elif self.k0 < 0.0:
            return(f1 * (f2 - f3 * np.arcsin(argument)))

    def _lookback_integrand(self, z): 
        _lookback = 1./(self.hubble_z(z)*(1.+z))
        return(_lookback)

    def lookback_time(self, z, z0 = 0.0):
        """Calculate the lookback time (in s) to redshift z.
        
        See equation 30 of David Hogg's arXiv:astro-ph/9905116v4
        
        Units are s.
        
        Optionally calculate the integral from z0 to z.
        
        Returns
        -------
        
        t_look: ndarray
           Lookback time in seconds.
        
        """
        lt_func = np.vectorize(lambda z, z0: 
                si.quad(self._lookback_integrand, z0, z, limit=1000)
                              )
        t_look, err = lt_func(z, z0)
        return(t_look)

    def age(self, z):
        """Calculate the age of the universe as seen at redshift z.
        
        Age at z is the integral from z to infinity of
        1 / [(1+z') H(z')] dz'.  This numerical expression works for
        Lambda-CDM, non-flat CDM, and CPL/w0waCDM.
        
        See also: lookback_time.
        
        Units are s.
        
        """
        age_func = np.vectorize(lambda z: 
                si.quad(self._lookback_integrand, z, np.inf, limit=1000)
                              )
        age, err = age_func(z)
        return(age)

    def age_flat(self, z):
        """Calculate the age of the universe for flat Lambda-CDM.
        
        Units are s.
        
        Analytical formula from Peebles, p. 317, eq. 13.2.  For w0waCDM,
        use age(z), which performs the correct numerical integral.
        
        """
        if self.k0 != 0.0:
            raise cex.CosmologyUnapplicable("Not for Omega_k != 0")
        if (self.w0 != -1.0) or (self.wa != 0.0):
            raise cex.CosmologyUnapplicable(
                    "age_flat analytic formula is only for Lambda-CDM; use age(z)")
        
        om = self.m0
        lam = 1. - om
        t_z = (2.*np.arcsinh(sqrt(lam/om)*np.power((1.+z),(-3./2.)))/
               (cc.H100_s*self.h*3.*sqrt(lam))
              )
        return(t_z)

    def __str__(self):
        z = np.asarray([0.1, 0.25, 0.5])
        return(("z  : %s \nage: %s Gyrs \nda : %s Mpc \ndl : %s Mpc\n")%
               (
                ' '.join(format(i, "5g") for i in z), 
                ' '.join(format(i, "5g") for i in self.age(z)/cc.Gyr_s),
                ' '.join(format(i, "5g") for i in self.angular_diameter_distance(z)),
                ' '.join(format(i, "5g") for i in self.luminosity_distance(z)),
               )
              )



if __name__ == "__main__":
    frw = FRW()
    print(frw)
    frw.k0 = 0.1
    print(frw)
    frw.k0 = -0.1
    print(frw)
    cp = CosmoParams(omega_M_0=0.28, h=0.7, set_flat=True)
    frw = FRW(cosmo=cp)
    print(frw.comoving_distance(0.23))
    print((frw.comoving_volume(0.3)-frw.comoving_volume(0.1))*7398/41253)

