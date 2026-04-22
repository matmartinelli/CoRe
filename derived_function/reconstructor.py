import numpy as np
import pandas as pd
import inspect
import json
import sys

from scipy.stats import multivariate_normal
from getdist.gaussian_mixtures import GaussianND

from utils.samplers_interface import SamplersInterface

class DerivedFunction:
    def __init__(self, recon_dict, cov_dict, method_dict, chatty=True):
        """
        recon_dict: {'name1': df_recon1, 'name2': df_recon2}
        cov_dict:   {'name1': df_cov1, 'name2': df_cov2}
        """ 
        self.recon_dict = recon_dict
        self.cov_dict   = cov_dict
        
        if method_dict['type'] == 'sampling':
            self.sampler = SamplersInterface(sampler=method_dict['options']['sampler'], 
                                             run_options=method_dict['options']['run_options'], 
                                             chatty=chatty)
            self.run = self.run_sampling
        elif method_dict['type'] == 'realizations':
            self.Nreals   = method_dict['options']['Nreals']
            self.Nsamples = method_dict['options']['Nsamples']
            self.realizations = self.get_realizations(self.Nreals)
            self.run = self.run_realizations
        else:
            sys.exit(f'UNKNOWN RECONSTRUCTION TYPE: {method_dict["type"]}')
            
        self.chatty = chatty

        recon_lengths = [len(df) for df in recon_dict.values()]
        self.N_recon = recon_lengths[0]
        self.x_recon = next(iter(recon_dict.values()))['x'].values
        
        if not all(x == self.N_recon for x in recon_lengths):
            sys.exit('ALL RECONSTRUCTIONS NEED TO BE DONE AT THE SAME X VALUES!')
        
    def _split_arg(self, arg):
        """Robustly splits 'recName_comp' into (recName, comp)."""
        for recon_key in self.recon_dict.keys():
            if arg.startswith(recon_key + '_'):
                component = arg[len(recon_key)+1:]
                return recon_key, component
        parts = arg.split('_')
        return "_".join(parts[:-1]), parts[-1]

    def _get_required_params(self, func):
        """Introspects logic and identifies GP components vs coordinate 'x'."""
        all_args = inspect.getfullargspec(func).args
        gp_info = []
        needs_x = False

        for arg in all_args:
            if arg == 'x':
                needs_x = True
                continue
            recon_key, component = self._split_arg(arg)
            gp_info.append({'recon_key': recon_key, 'comp': component, 'full_arg': arg})

        return gp_info, needs_x, all_args

    def get_realizations(self, Nreals):
        """Generates realizations for all GP components."""
        realizations = {}
        for key, cov in self.cov_dict.items():
            def parse_col(col):
                parts = col.split('_')
                idx = int(parts[-1])
                comp = "_".join(parts[:-1])
                return comp, idx
            
            mean = [self.recon_dict[key].iloc[parse_col(col)[1]][parse_col(col)[0]] 
                    for col in cov.columns]
            
            realizations[key] = pd.DataFrame(
                multivariate_normal(mean=mean, cov=cov.values, allow_singular=True).rvs(size=Nreals),
                columns=cov.columns
            )
        return realizations

    def run_realizations(self, logic_list, name_list):
        """
        Modified to handle a list of logic statements and compute joint correlations.
        """
        all_reals = []
        
        # Pre-calculate param info for each logic
        logics_info = [self._get_required_params(l) for l in logic_list]

        for ind in range(self.Nreals):
            joint_vector = []
            for logic, (gp_info, needs_x, arg_order) in zip(logic_list, logics_info):
                derived_vals = []
                for i in range(self.N_recon):
                    call_args = []
                    for arg_name in arg_order:
                        if arg_name == 'x':
                            call_args.append(self.x_recon[i])
                        else:
                            recon, comp = self._split_arg(arg_name)
                            call_args.append(self.realizations[recon].iloc[ind][f"{comp}_{i}"])
                    derived_vals.append(logic(*call_args))
                joint_vector.extend(derived_vals)
            all_reals.append(joint_vector)

        data_array = np.array(all_reals)
        mean_func  = np.mean(data_array, axis=0)
        cov_func   = np.cov(data_array, rowvar=False)

        # Generate joint labels: [name1_0...name1_N, name2_0...name2_N]
        all_labels = []
        for name in name_list:
            all_labels += [f"{name}_{i}" for i in range(self.N_recon)]

        sample = GaussianND(mean_func, cov_func, is_inv_cov=False,
                            names=all_labels).MCSamples(self.Nsamples)
        return sample

    def run_sampling(self, logic_list, name_list, sigma_width=5):
        """
        Modified to handle a list of logic statements and compute joint correlations.
        """
        # 1. Aggregate all required parameters from all logics
        all_gp_info = []
        logics_info = []
        for logic in logic_list:
            info, nx, args = self._get_required_params(logic)
            all_gp_info.extend(info)
            logics_info.append((info, nx, args))

        # Unique parameters for priors
        unique_params = {item['full_arg']: item for item in all_gp_info}.values()

        if self.chatty:
            print(f"Sampling joint GP components: {[p['full_arg'] for p in unique_params]}")

        # 2. Prepare Priors
        parameters = {}
        for item in unique_params:
            recon_key, component, arg_name = item['recon_key'], item['comp'], item['full_arg']
            df_recon = self.recon_dict[recon_key]
            df_cov = self.cov_dict[recon_key]
            labels = [f"{component}_{i}" for i in range(self.N_recon)]
            sigmas = np.sqrt(np.diag(df_cov.loc[labels, labels].values))

            for i in range(self.N_recon):
                mean_val = df_recon.iloc[i][component]
                parameters[f"{arg_name}_{i}"] = {
                    'prior': [mean_val - sigma_width * sigmas[i],
                              mean_val + sigma_width * sigmas[i]],
                    'latex': rf'${arg_name}_{{{i}}}$'
                }

        # 3. Likelihood Preparation (includes all components across all logics)
        likeparts = {}
        for recon_key, df_recon in self.recon_dict.items():
            df_cov = self.cov_dict[recon_key]
            # Get components from any logic that belong to this recon
            requested_comps = list(set(g['comp'] for g in all_gp_info if g['recon_key'] == recon_key))
            
            if not requested_comps:
                continue

            keep_labels = [col for col in df_cov.columns if "_".join(col.split('_')[:-1]) in requested_comps]
            reduced_cov = df_cov.loc[keep_labels, keep_labels]

            likeparts[recon_key] = {
                'labels': keep_labels,
                'data_vec': np.array([df_recon.iloc[int(l.split('_')[-1])]["_".join(l.split('_')[:-1])] 
                                      for l in keep_labels])
            }
            try:
                likeparts[recon_key]['inv_cov'] = np.linalg.inv(reduced_cov.values)
            except np.linalg.LinAlgError:
                likeparts[recon_key]['inv_cov'] = np.linalg.pinv(reduced_cov.values)

        def likelihood(param_dict):
            chi2 = 0
            for recon_key, parts in likeparts.items():
                theory_vec = np.array([param_dict[f"{recon_key}_{label}"] for label in parts['labels']])
                diff = theory_vec - parts['data_vec']
                chi2 += np.dot(diff, np.dot(parts['inv_cov'], diff))

            loglike = -0.5 * chi2
            
            all_derived_values = []
            for logic, (_, _, arg_order) in zip(logic_list, logics_info):
                for i in range(self.N_recon):
                    call_args = []
                    for arg_name in arg_order:
                        if arg_name == 'x':
                            call_args.append(self.x_recon[i])
                        else:
                            call_args.append(param_dict[f"{arg_name}_{i}"])
                    all_derived_values.append(logic(*call_args))
                
            return tuple([loglike] + all_derived_values)

        derived_names = {}
        for name in name_list:
            for i in range(self.N_recon):
                derived_names[f"{name}_{i}"] = {'latex': rf'${name}_{{{i}}}$'}

        return self.sampler.run(parameters, likelihood, derived=derived_names)
