import numpy as np
import pandas as pd
import inspect
import json
import sys

from scipy.stats  import multivariate_normal
from scipy.linalg import block_diag

from getdist.gaussian_mixtures import GaussianND

from CoRe.utils.samplers_interface import SamplersInterface

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
            self.Nsamples = method_dict['options']['Nsamples']
            self.sample = self.get_joint_sample()
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

    def get_joint_sample(self):
        """Generates a joint MCSamples object for all GP components."""

        labelled_cov = {}
        for data,cov in self.cov_dict.items():
            labelled_cov[data]          = cov.copy()
            labelled_cov[data].columns  = [data+'_'+col for col in cov.columns]
            labelled_cov[data].index    = labelled_cov[data].columns

        matrices = [cov.values for cov in labelled_cov.values()]

        # 1. Create the block diagonal matrix
        combined_array = block_diag(*matrices)

        # 2. Combine the indices and columns
        new_index = pd.Index(np.concatenate([cov.index for cov in labelled_cov.values()]))
        new_columns = pd.Index(np.concatenate([cov.columns for cov in labelled_cov.values()]))

        # 3. Reconstruct the DataFrame
        full_matrix = pd.DataFrame(combined_array, index=new_index, columns=new_columns)

        label_vec   = [col.split('_') for col in full_matrix.columns]
        mean_vector = [self.recon_dict[lab[0]].iloc[int(lab[2])][lab[1]] for lab in label_vec]

        sample = GaussianND(mean_vector,full_matrix,is_inv_cov=False,names=full_matrix.columns).MCSamples(self.Nsamples)

        return sample

    def run_realizations(self, logic_list, name_list):
        """
        Handles a list of logic statements and compute joint correlations.
        """

        p = self.sample.getParams()
        for name,func in zip(name_list,logic_list):
            gp_info, needs_x, arg_order = self._get_required_params(func)
            for i in range(self.N_recon):
                call_args = []
                for arg_name in arg_order:
                    if arg_name == 'x':
                        call_args.append([self.x_recon[i]]*self.Nsamples)
                    else:
                        recon, comp = self._split_arg(arg_name)
                        call_args.append(getattr(p,recon+'_'+comp+'_'+str(i)))
        
                derpar = []
                for ind in range(self.Nsamples):
                    final_args = [arg[ind] for arg in call_args]
                    derpar.append(func(*final_args))
                self.sample.addDerived(derpar,name=name+'_'+str(i))


        return self.sample

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
        # 2.1 Prepare derived parameters
        #TODO: this is ugly and a repetition, fix it!
        derived = {}
        for name,logic in zip(name_list,logic_list):
            for i in range(self.N_recon):
                derived[f"{name}_{i}"] = {'derived': logic,
                                          'prior': [None,None],
                                          'latex': rf'${name}_{{{i}}}$'}

        all_parameters = parameters | derived

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

        return self.sampler.run(all_parameters, likelihood)#, derived=derived_names)
