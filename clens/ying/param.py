#Last-modified: 12 Nov 2012 03:52:07 PM

""" This module holds the classes for cosmological and other parameters,
and provides default values and an interface for accessing them.  """

import unittest
from . import cex
import numpy as np

try :
    from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
except ImportError :
    from zypy.contrib.cubicSpline import NaturalCubicSpline as spline1d
                                                    
# Heidi: not used
#from zypy.contrib.lininterp import LinInterp as interp1d
from scipy.interpolate import interp1d # add by Heidi


# mean N200-M relationship from Rozo et al. (2010b)
defaultMassRichParamDict = {
    "A"             : 2.34,
    "alpha"         : 0.757,
    "Mpivot"        : 1.09e+14,
    "sigma_r_m"     : 0.357,
    }


# cosmological parameters 
defaultCosmoParamDict = {
   'omega_b_0'      : 0.0458,
   'omega_M_0'      : 0.274,
   'omega_lambda_0' : 0.726,
   'omega_k_0'      : 0.0,
   'sigma_8'        : 0.816,
   'h'              : 0.702,
   'n'              : 0.968,
   'w'              : -1,
   'tau'            : 0.088,
   'z_reion'        : 10.6,
   't_0'            : 13.76,
   'omega_n_0'      : 1.0e-6,
   'N_nu'           : 3,
   'Y_He'           : 0.24,
   }


class MassRichParams(dict):
    """
    Base class for mass-richness relationship parameters.

    Parameters
    ----------
    **pardict: dict_like
        Mass-richness relation parameters with key names as

        #. 'A'         : :math:`A`
        #. 'alpha'     : :math:`\\alpha`
        #. 'Mpivot'    : :math:`M_{pivot}`
        #. 'sigma_r_m' : :math:`\sigma_{\ln N_{200}}`

    """
    def __init__(self, **pardict):
        super(MassRichParams, self).__init__(defaultMassRichParamDict)
        for k in pardict:
            if k in self: self[k] = pardict[k]
            else: raise cex.UnknownParameterError(k)

    def __getattr__(self,name):
        return self[name]

    def __str__(self):
        r = 'parameters in MassRichParams:\n'
        for k in self:
            r += "\t-> %s = %s\n" % (k, str(self[k]))
        return r[:-1]

class MassRichCurved(object):
    """
    Base class for a more flexible mass-richness relationship, no scatteri
    inclued.
    """
    def __init__(self, lnN_1=2.47, lnN_2=4.21, lnN_3=None, M_1=1.3e14, M_2=1.3e15, M_3=None):
        self.lnN_1  = lnN_1
        self.lnN_2  = lnN_2
        self.M_1    = M_1
        self.M_2    = M_2
        if (M_3 is not None) and (lnN_3 is not None):
            if (M_3 > M_2) or (M_3 < M_1):
                raise cex.ParameterOutsideDefaultRange()
            else:
                self.lnMnod = np.log(np.array([M_1, M_3, M_2]))
                self.lnNnod = np.array([lnN_1, lnN_3, lnN_2])
                self.numnod = 3
        elif (M_3 is None) and (lnN_3 is None):
            self.lnMnod = np.log(np.array([M_1, M_2]))
            self.lnNnod = np.array([lnN_1, lnN_2])
            self.numnod = 2
        else:
            raise cex.ParametersNotPairSet()
        self.set_massrich()

    def set_massrich(self):
        if self.numnod == 2:
            self.log_richness_mean = interp1d(self.lnMnod, self.lnNnod)
        if self.numnod > 2:
            self.log_richness_mean = spline1d(self.lnMnod, self.lnNnod, k=self.numnod-1)

    def resetNode3(self, lnN_3, M_3=None):
        if M_3 is None:
            if self.numnod != 3:
                raise cex.ParametersNotPairSet("M_3 needs to be set when numnod is not 3")
            else:
                self.lnNnod[1] = lnN_3
        else:
            if (M_3 > self.M_2) or (M_3 < self.M_1):
                raise cex.ParameterOutsideDefaultRange()
            else:
                self.lnMnod[1] = np.log(M_3)
                self.lnNnod[1] = lnN_3
                self.numnod = 3
        self.set_massrich()


class MassRichPowerLaw(object):
    """
    Base class for a simple powerlaw mass-richness relationship, no scatteri
    inclued.
    """
    def __init__(self, A=2.34, alpha=0.757, Mpivot= 1.09e+14):
        self.A      = A
        self.alpha  = alpha
        self.logMpivot = np.log(Mpivot)

    def log_richness_mean(self, log_mass):
         _log_richness = self.A + self.alpha*(log_mass - self.logMpivot)
         return(_log_richness)


class CosmoParams(dict):
    """
    Base class for cosmological parameters.

    Parameters
    ----------
    **pardict: dict_like
        Cosmological parameters with key names as

        #. 'omega_b_0'      : :math:`\Omega_b`
        #. 'omega_M_0'      : :math:`\Omega_m`  
        #. 'omega_lambda_0' : :math:`\Omega_\lambda`  
        #. 'omega_k_0'      : :math:`\Omega_k` 
        #. 'sigma_8'        : :math:`\sigma_8`
        #. 'h'              : :math:`h` 
        #. 'n'              : :math:`n_s` 
        #. 'w'              : :math:`w`
        #. 'tau'            : :math:`\\tau`
        #. 'z_reion'        : :math:`z_{reionization}`
        #. 't_0'            : :math:`t_0`
        #. 'omega_n_0'      : :math:`\Omega_{\\nu}`
        #. 'N_nu'           : :math:`N_{\\nu}`
        #. 'Y_He'           : :math:`Y_{He}`

    """
    def __init__(self, set_flat=False, **pardict):
        super(CosmoParams, self).__init__(defaultCosmoParamDict)
        for k in pardict:
            if k in self: self[k] = pardict[k]
            else: raise cex.UnknownParameterError(k)

        if set_flat:
            self.set_flat()

        self.tolerance = 0.0001
        self.omega_total = self["omega_lambda_0"] + self["omega_M_0"]

    def __getattr__(self,name):
        """ Retrieve any parameter *name* from current CosmoParams.

        Parameters
        ----------
        name: string
            Name of desired parameter.

        Returns
        -------
        value: float
            Value of Parameter *name*.
        """
        return(self[name])

    def __str__(self):
        """ Print the content of current CosmoParams.
        """
        r = 'parameters in CosmoParams:\n'
        for k in self:
            r += "\t-> %s = %s\n" % (k, str(self[k]))
        return r[:-1]

    def set_flat(self):
        """ Set the cosmology to be flat.

        Force flatness by requring omega_lambda_0 to be
        1 - omega_M_0 and omega_k_0 to be 0.

        """
        self["omega_lambda_0"] = 1.0 - self["omega_M_0"]
        self["omega_k_0"]      = 0.0
        self.omega_total = self["omega_lambda_0"] + self["omega_M_0"]

    def isflat(self):
        """ Tell if the input cosmology is flat.

        Returns
        -------
        isflat: boolean
            True if flat.
        """
        return((abs(self["omega_k_0"]) <= self.tolerance) and
               (abs(self.omega_total - 1.0) <= self.tolerance))

    def isopen(self):
        """ Tell if the input cosmology is open.

        Returns
        -------
        isopen: boolean
            True if open.
        """
        return((self["omega_k_0"] > self.tolerance) and 
               (self.omega_total - 1.0 > self.tolerance))

    def isclosed(self):
        """ Tell if the input cosmology is closed.

        Returns
        -------
        isclosed: boolean
            True if closed.
        """
        return((self["omega_k_0"] < 0.0 - self.tolerance) and 
               (self.omega_total - 1.0 < 0.0 - self.tolerance))

    def isEdS(self):
        """ Tell if the input cosmology is Einstein de-Sitter.

        Returns
        -------
        isEdS: boolean
            True if EdS.
        """
        return(self.isflat and 
               abs(self["omega_lambda_0"]) <= self.tolerance)


class CosmoParamsTests(unittest.TestCase):
    """
    unit tests for cosmological parameters
    """
    def testParams0(self):
        self.cp = CosmoParams(h=1.000)
        print(self.cp) 
        self.failIf(self.cp.h != 1.000)

    def testParams1(self):
        try:
            self.cp = CosmoParams(no_h = 1.000)
            self.failIf(True)
        except cex.UnknownParameterError:
            self.failIf(False)
    def testParams3(self):
        self.cp = CosmoParams(omega_M_0 = 0.5, set_flat=True)
        print(self.cp) 
        self.failIf(self.cp.omega_lambda_0 != 0.5)


class MassRichParamsTests(unittest.TestCase):
    """
    unit tests for mass-richness ralationship parameters
    """
    def testParams0(self):
        self.mp = MassRichParams(alpha=1.000)
        print(self.mp) 
        self.failIf(self.mp.alpha != 1.000)

    def testParams1(self):
        try:
            self.mp = MassRichParams(no_alpha = 1.000)
            self.failIf(True)
        except cex.UnknownParameterError:
            self.failIf(False)


def main():
    unittest.main()


if __name__=='__main__':
    main()
