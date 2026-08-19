import sys,os
import numpy  as np
import pandas as pd

from bios import read
from time import time

from nautilus import Prior
from nautilus import Sampler

import emcee

from getdist import MCSamples

class SamplersInterface:

    def __init__(self,sampler='nautilus',run_options='poor',outroot=None,chatty=True):

        self.settings = run_options
        self.root     = outroot
        self.chatty   = chatty

        if sampler.lower() == 'nautilus':
            self.run = self.run_nautilus
        elif sampler.lower() == 'emcee':
            self.run = self.run_emcee
        else:
            sys.exit('UNKNOWN SAMPLER: {}'.format(sampling_options['sampler']))

    def run_nautilus(self,parameters,likelihood):

        if self.settings == 'poor':
            sets = {'num_threads': 1,
                    'pool': 1,
                    'n_live': 500,
                    'n_batch': 64,
                    'n_networks': 2}
        elif self.settings == 'good':
            sets = {'num_threads': 1,
                    'pool': 1,
                    'n_live': 4000,
                    'n_batch': 512,
                    'n_networks': 16}
        elif type(self.settings) == dict:
            sets = self.settings
        else:
            sys.exit('UNKNOWN OPTIONS FOR NAUTILUS')


        #SETTING UP FREE AND DERIVED PARAMETERS---------------------
        prior    = Prior()
        derived  = {}
        ndim     = 0
        nderived = 0

        for par,val in parameters.items():
            if type(val) == dict:
                if 'derived' in val:
                    derived[par] = val['latex']
                    nderived += 1
                else:
                    if type(val['prior']) == dict:
                        dist_prior = norm(loc=val['prior']['mean'],scale=val['prior']['error'])
                    else:
                        dist_prior = tuple(val['prior'])
                    ndim += 1
                    prior.add_parameter(par, dist=dist_prior)


        nautilus_options = {k:v for k,v in sets.items() if k != 'num_threads'}
        if self.root != None:
            nautilus_options['filepath'] = self.root+'.hdf5'

        if nderived == 0:
            sampler = Sampler(prior,likelihood,**nautilus_options,n_dim=ndim)
            sampler.run(verbose=self.chatty)
            points, log_w, log_l = sampler.posterior(equal_weight=True)

            nautilus_dict = {par: parameters[par]['latex'] for par in prior.keys}
            results       = pd.DataFrame(np.c_[points, np.exp(log_w), -log_l],columns=list(nautilus_dict.keys())+['weight','minuslogpost'])
        else:
            #INCLUDING DERIVED PARAMETERS
            derived_pars = list(derived.keys())
            blob_vec     = [(par, float) for par in derived_pars]

            sampler = Sampler(prior,likelihood,**nautilus_options,n_dim=ndim,blobs_dtype=blob_vec)
            sampler.run(verbose=self.chatty)
            points, log_w, log_l, derived_points = sampler.posterior(equal_weight=True,return_blobs=True)

            derived_array = np.array([np.array(list(der)) for der in derived_points])
            nautilus_dict = {par: parameters[par]['latex'] for par in prior.keys} | {par: derived[par] for par in derived_pars}
            results       = pd.DataFrame(np.c_[points, derived_array, np.exp(log_w), -log_l],columns=list(nautilus_dict.keys())+['weight','minuslogpost'])

        results = results[['weight','minuslogpost']+list(nautilus_dict.keys())]
        if self.chatty:
            print('NAUTILUS SAMPLING FINISHED')

        sample = MCSamples(samples=results[list(nautilus_dict.keys())].values,
                           names=list(nautilus_dict.keys()),
                           labels=list(nautilus_dict.values()))

        return sample


    def run_emcee(self,parameters,likelihood):

        #TODO: create two settings
        if self.settings == 'poor':
            sets = {'nwalkers': 32,        # Number of ensemble walkers (typically >= 2 * ndim)
                    'nsteps': 3000,        # Total iterations per walker
                    'burn_in': 500,        # Steps to discard from the start
                    'ini_width': 1.e-2,    #TODO
                    'thin': 5}             # Thinning factor to reduce autocorrelation
        elif self.settings == 'good':
            sys.exit('Not available yet') 
        elif type(self.settings) == dict:
            sets = self.settings
        else:
            sys.exit('UNKNOWN OPTIONS FOR EMCEE')

        derived  = {}
        freepars = {}
        ndim     = 0
        nderived = 0

        #Handling prior (Gaussian to be added!!!)
        for par,val in parameters.items():
            if type(val) == dict:
                if 'derived' in val:
                    derived[par] = val['latex']
                    nderived += 1
                else:
                    if type(val['prior']) == dict:
                        dist_prior = norm(loc=val['prior']['mean'],scale=val['prior']['error'])
                    else:
                        dist_prior = tuple(val['prior'])

                    freepars[par] = dist_prior
                    ndim += 1

        def log_prior(params):
            for p, (p_min, p_max) in zip(params,list(freepars.values())):
                if not (p_min <= p <= p_max):
                    return -np.inf

            return 0.0

        def log_probability(params):
            lp = log_prior(params)
            if not np.isfinite(lp):
                return (-np.inf, *([np.nan] * nderived)) if nderived > 0 else -np.inf

            try:
                like_res = likelihood({par: params[i] for i, par in enumerate(freepars.keys())})

                if nderived > 0:
                    if isinstance(like_res, (tuple, list)):
                        raw_like = like_res[0]
                        der_vals = tuple(like_res[1]) if len(like_res) == 2 and isinstance(like_res[1], (tuple, list, np.ndarray)) else tuple(like_res[1:])
                    else:
                        raw_like = like_res
                        der_vals = tuple([np.nan] * nderived)

                    # Squeeze array dimensions to convert 0D/1D JAX arrays cleanly
                    like_arr = np.squeeze(np.asarray(raw_like))

                    if like_arr.ndim != 0:
                        print(f"\n[DEBUG] Non-scalar likelihood detected (shape {like_arr.shape})!")
                        print(f"  Walker Params: {dict(zip(freepars.keys(), params))}")
                        print(f"  Raw Return: {like_res}\n")
                        return -np.inf, *([np.nan] * nderived)

                    like_val = float(like_arr.item())
                    if np.isnan(like_val):
                        return -np.inf, *([np.nan] * nderived)
                    return lp + like_val, *der_vals

                else:
                    # Squeeze 0D/1D JAX or NumPy arrays
                    like_arr = np.squeeze(np.asarray(like_res))

                    if like_arr.ndim != 0:
                        print(f"\n[DEBUG] Non-scalar likelihood detected (shape {like_arr.shape})!")
                        print(f"  Walker Params: {dict(zip(freepars.keys(), params))}")
                        print(f"  Raw Return: {like_res}\n")
                        return -np.inf

                    like_val = float(like_arr.item())
                    if np.isnan(like_val):
                        return -np.inf
                    return lp + like_val

            except Exception as e:
                print(f"\n[DEBUG ERROR] Likelihood evaluation crashed at parameters:")
                print(f"  Walker Params: {dict(zip(freepars.keys(), params))}")
                raise e

        param_names  = list(freepars.keys())
        param_labels = [parameters[par]['latex'] for par in param_names]

        #TODO: better initial guess?
        initial_guess = np.array([np.mean(prior_lims) for prior_lims in freepars.values()])
        pos = initial_guess + sets['ini_width'] * np.random.randn(sets['nwalkers'],ndim)

        sampler = emcee.EnsembleSampler(sets['nwalkers'], ndim, log_probability)
        sampler.run_mcmc(pos,sets['nsteps'],progress=self.chatty)
        if self.chatty:
            print('EMCEE SAMPLING FINISHED')

        raw_chains = sampler.get_chain(discard=sets['burn_in'],thin=sets['thin'])

        if nderived > 0:
            # Retrieve derived parameter blobs: shape (nsteps, nwalkers, nderived) or (nsteps, nwalkers)
            raw_blobs = sampler.get_blobs(discard=sets['burn_in'],thin=sets['thin'])
            raw_blobs = np.asarray(raw_blobs)
            
            # Ensure blobs array is 3D even if nderived == 1
            if nderived == 1 and raw_blobs.ndim == 2:
                raw_blobs = raw_blobs[:, :, np.newaxis]

            # Combine free and derived parameter values along axis 1 for each walker
            walker_chains = [np.hstack([raw_chains[:, i, :], raw_blobs[:, i, :]]) for i in range(sets['nwalkers'])]
            
            param_names  = list(freepars.keys()) + list(derived.keys())
            param_labels = [parameters[par]['latex'] for par in freepars.keys()] + list(derived.values())
        else:
            walker_chains = [raw_chains[:, i, :] for i in range(sets['nwalkers'])]
            param_names  = list(freepars.keys())
            param_labels = [parameters[par]['latex'] for par in param_names]

        sample = MCSamples(samples=walker_chains,names=param_names,labels=param_labels)

        return sample
