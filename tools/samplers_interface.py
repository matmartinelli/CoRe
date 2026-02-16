import sys
import numpy   as np
import pandas  as pd

from nautilus import Prior
from nautilus import Sampler

from getdist import MCSamples

class SamplersInterface:

    def __init__(self,sampler='Nautilus',run_options='poor',chatty=False):

        self.run_options = run_options
        self.chatty      = chatty

        if sampler == 'Nautilus':
            self.run = self.run_nautilus
        else:
            sys.exit('Unknown sampler: {}'.format(sampler))

    def run_nautilus(self,parameters,likelihood):
    
        if self.run_options == 'poor':
            sets = {'num_threads': 1,
                    'pool': 1,
                    'n_live': 500,
                    'n_batch': 64,
                    'n_networks': 2}
        elif self.run_options == 'good':
            sets = {'num_threads': 1,
                    'pool': 1,
                    'n_live': 4000,
                    'n_batch': 512,
                    'n_networks': 16}
    
    
        prior = Prior()
        ndim = 0
    
        for par,val in parameters.items():
            dist_prior = tuple(val['prior'])
            ndim += 1
            prior.add_parameter(par, dist=dist_prior)
            
        if self.chatty:
            print('Loaded prior into Nautilus with dimension',prior.dimensionality())
            print('Prior keys: ',prior.keys)
            print('Starting to sample with Nautilus...')
        nautilus_options = {k:v for k,v in sets.items() if k != 'num_threads'}
        sampler = Sampler(prior,likelihood,**nautilus_options,n_dim=ndim)
        sampler.run(verbose=True)

        points, log_w, log_l = sampler.posterior(equal_weight=True)
        nautilus_dict = {par: parameters[par]['latex'] for par in prior.keys}
    
        results = pd.DataFrame(np.c_[points, np.exp(log_w), -log_l],columns=list(nautilus_dict.keys())+['weight','minuslogpost'])
        results = results[['weight','minuslogpost']+list(nautilus_dict.keys())]

        if self.chatty:
            print('NAUTILUS SAMPLING FINISHED')

        sample = MCSamples(samples=results[list(nautilus_dict.keys())].values,
                           names=list(nautilus_dict.keys()),
                           labels=list(nautilus_dict.values()))
        
        return sample


