#!/usr/bin/env python
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

""""
mimicking zycosmo/constants.py
"""

doc = ""

doc += "c: speed of light in km/s"
c = 3.e+5 # km/s


doc += "G: graviational constant in Mpc/Msun (km/s)^2"
G = 4.302e-9 # Mpc/Msun (km/s)^2


doc += "rho_crit_with_h: critical density in h^2 Msun/Mpc^3, remember to *h**2 before use"
rho_crit_with_h = 2.775e11 # h^2 Msun/Mpc^3


doc += "sqarcmin_sterad: square arcminute in steradians"
sqarcmin_sterad = 8.461594994075237e-08

delta_c = 1.686

radian_to_arcmin = 180.*60./np.pi
arcmin_to_radian = 1./radian_to_arcmin


# __doc__ += "\n".join(sorted(doc.split("\n"))) NOT WORKING...
