#!/usr/bin/env python
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import numpy as np
# import matplotlib.pyplot as plt


defaultCosmoParameters = {
    "OmegaM": 0.267,
    "OmegaDE": 0.733,
    "h": 0.710,
    "sigma8":0.806,
    "OmegaB":0.045,
    "ns":0.963,
    "tau":0.088
}


defaultNuisanceParameters = {
    "lgM0":14.34,
    "alpha_M":0.752,
    "lambda0": 40,
    "sigma_lambda": 1e-6, #
    "lnMmin": 32.23619130191664,   # lnMmin=14*np.log(10): used in self-similar model only
    "delta": 0.1
}

defaultNuisanceParametersPiecewise = {
    "norm_mid": -21.1415,
    "alpha_M0":0.752,
    "alpha_M1":0.752,
    "alpha_M2":0.752,
    "scatter0": 0.5,
    "scatter1": 0.5,
    "scatter2": 0.5
}


class CosmoParameters(dict):
    def __init__(self, **pardict):
        super(CosmoParameters, self).__init__(defaultCosmoParameters) # set to default value
        for k in pardict:
            if k in self:
                self[k] = pardict[k]

    def __getattr__(self, attr):
        return self[attr]

    def __str__(self):
        r = 'parameters:\n'
        for k in self:
            r += "\t-> %s = %s\n" % (k, str(self[k]))
        return r[:-1]



class NuisanceParameters(dict):
    def __init__(self, **pardict):
        super(NuisanceParameters, self).__init__(defaultNuisanceParameters) # set to default value
        for k in pardict:
            if k in self:
                self[k] = pardict[k]
            

    def __getattr__(self, attr):
        return self[attr]

    def __str__(self):
        r = 'parameters:\n'
        for k in self:
            r += "\t-> %s = %s\n" % (k, str(self[k]))
        return r[:-1]



class NuisanceParametersPiecewise(dict):
    def __init__(self, **pardict):
        super(NuisanceParametersPiecewise, self).__init__(defaultNuisanceParametersPiecewise) # set to default value
        for k in pardict:
            if k in self:
                self[k] = pardict[k]
            

    def __getattr__(self, attr):
        return self[attr]

    def __str__(self):
        r = 'parameters:\n'
        for k in self:
            r += "\t-> %s = %s\n" % (k, str(self[k]))
        return r[:-1]




        
if __name__ == "__main__":

    print(CosmoParameters())
    print(NuisanceParameters())

    co = CosmoParameters()
    nu = NuisanceParameters(sigma_lambda=0.1)
    print(nu.sigma_lambda)
    print(nu['sigma_lambda'])

    '''
    nu_para_name_list = ['sigma_lambda', 'lgM0', 'alpha_M']
    for i in range(3):
        name = nu_para_name_list[i]
        print(name, nu[name])
        #print(name, co[name])
    '''
    co_new = {'sigma8': 0.9, 'lnM0':20} # it's okay to have irrelevant stuff
    co = CosmoParameters(**co_new)
    print(co)

    # for Mobs = Mtrue (1-1 scaling, no scatter)
    nu = NuisanceParameters(sigma_lambda=1e-5, lgM0=0, alpha_M=1, lambda0=1)

