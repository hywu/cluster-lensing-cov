#!/usr/bin/env python

defaultCosmoParameters = {
    "OmegaM": 0.267,
    "OmegaDE": 0.733,
    "h": 0.710,
    "sigma8":0.806,
    "OmegaB":0.045,
    "ns":0.963,
    "tau":0.088
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


    co_new = {'sigma8': 0.9, 'lnM0':20} # it's okay to have irrelevant stuff
    co = CosmoParameters(**co_new)
    print(co)
