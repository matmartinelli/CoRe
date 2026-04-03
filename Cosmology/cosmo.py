import sys
import numpy  as np
import pandas as pd

from copy import deepcopy

from scipy.interpolate import UnivariateSpline

import camb

#THIS IS ALL TEMPORARY!!!!

class CalcCosmology:

    def __init__(self,params):

        self.zcalc  = np.linspace(0.,5,1000)

        self.params = self.cambify(params)

        self.results = self.get_cosmology()

    def get_cosmology(self):

        zdrag = 1060
        zcamb = np.logspace(-3,np.log10(max(self.zcalc)),250)


        pars = camb.set_params(redshifts=zcamb,silent=True,**self.params)
        results = camb.get_results(pars)

        camb_fsig8 = np.array(results.get_fsigma8())
        camb_sig8  = np.array(results.get_sigma8())
        camb_f     = camb_fsig8/camb_sig8

        hubble = UnivariateSpline(self.zcalc,results.hubble_parameter(self.zcalc),s=0,k=4)
        growth = UnivariateSpline(zcamb,np.flip(camb_f),s=0,k=4)
        dL     = UnivariateSpline(self.zcalc,(1+self.zcalc)**2*results.angular_diameter_distance(self.zcalc),s=0,k=4)

        Om = (self.params['ombh2']+self.params['omch2']+self.params['omnuh2'])/(self.params['H0']/100)**2

        output = {'Hubble': {'f': hubble,
                             'd1': hubble.derivative(n=1),#lambda x: 3*self.params['H0']**2*Om*(1+x)**2/(2*hubble(x)),
                             'd2': hubble.derivative(n=2)},
                  'DM': {'f': lambda x: (1+x)*results.angular_diameter_distance(x)}, #WARNING! CURVATURE!
                  'dL': {'f': dL,
                         'd1': dL.derivative(n=1)},
                  'growth_rate': {'f': growth},
                  'rd': results.sound_horizon(zdrag)}

        return output

    def cambify(self,params):

        new_params = deepcopy(params)

        if 'omegam' in params:
            new_params['ombh2']  = 0.05*(new_params['H0']/100)**2
            new_params['omnuh2'] = 0.0006
            new_params['omch2']  = new_params.pop('omegam')*(new_params['H0']/100)**2-new_params['ombh2']-new_params['omnuh2']

        new_params['WantTransfer'] = True

        return new_params



