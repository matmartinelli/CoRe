import numpy as np
import pandas as pd
import inspect

from utils.samplers_interface import SamplersInterface

class DerivedFunction:
    def __init__(self,recon_dict,cov_dict,sampler='Nautilus',run_options='poor',chatty=True):
        """
        recon_dict: {'name1': df_recon1, 'name2': df_recon2}
        cov_dict:   {'name1': df_cov1, 'name2': df_cov2}
        """ 
        self.recon_dict = recon_dict
        self.cov_dict   = cov_dict
        self.sampler    = SamplersInterface(sampler=sampler,run_options=run_options,chatty=chatty)
        self.chatty     = chatty

        recon_lenghts = [len(df) for df in recon_dict.values()]

        self.N_recon  = recon_lenghts[0]
        self.x_recon = [df['x'].values for df in recon_dict.values()][0]
        if not all(x==self.N_recon for x in recon_lenghts):
            ##MMmod: TODO
            # Include check on the actual x values used by the reconstruction
            # not only on their lenght
            sys.exit('ALL RECONSTRUCTIONS NEED TO BE DONE AT THE SAME X VALUES!')
        
    def _get_required_params(self,func):
        """
        Introspects lambda: e.g., 'lambda x, rec1_f, rec2_d1'
        Distinguishes between coordinate 'x' and sampled GP components.
        """
        all_args = inspect.getfullargspec(func).args
        gp_info = []
        needs_x = False

        for arg in all_args:
            if arg == 'x':
                needs_x = True
                continue

            parts = arg.split('_')
            recon_key = "_".join(parts[:-1])
            component = parts[-1]
            gp_info.append({'recon_key': recon_key, 'comp': component, 'full_arg': arg})

        return gp_info, needs_x, all_args

    def run(self,derived_logic,derived_name,sigma_width=5):
        """
        derived_logic: lambda, e.g., lambda f, d1: d1 / f
        sigma_width: How many sigmas for the prior width
        """

        # 1. Parse required variables
        gp_info, needs_x, arg_order = self._get_required_params(derived_logic)

        if self.chatty:
            print(f"Sampling GP components: {[g['full_arg'] for g in gp_info]}")
            if needs_x:
                print("Detected 'x' coordinate dependency in derived logic.")

        # 2. Prepare Free Parameters (Priors)
        # We only create parameters for GP components, NOT for 'x'
        parameters = {}
        for item in gp_info:
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

        # 3. Derived Parameters setup
        derived_names = {f"{derived_name}_{i}": {'latex': rf'${derived_name}_{{{i}}}$'}
                         for i in range(self.N_recon)}

        #MMmod: TODO 
        #Put some print here for letting the user know parameters that will be used and derived

        # 4. Define Likelihood with Partials
        likeparts = {}
        for recon_key, df_recon in self.recon_dict.items():
            df_cov = self.cov_dict[recon_key]
            requested_functions = [g['comp'] for g in gp_info if g['recon_key'] == recon_key]
            
            if not requested_functions:
                continue

            keep_labels = [col for col in df_cov.columns if col.split('_')[0] in requested_functions]
            reduced_cov = df_cov.loc[keep_labels, keep_labels]

            likeparts[recon_key] = {
                'inv_cov': np.linalg.inv(reduced_cov.values),
                'labels': keep_labels,
                'data_vec': np.array([df_recon.iloc[int(l.split('_')[1])][l.split('_')[0]] for l in keep_labels])
            }

        def likelihood(param_dict):
            chi2 = 0
            for recon_key, parts in likeparts.items():
                # Extract only the parameters belonging to this recon_key
                theory_vec = np.array([param_dict[f"{recon_key}_{label}"] for label in parts['labels']])
                diff = theory_vec - parts['data_vec']
                chi2 += np.dot(diff, np.dot(parts['inv_cov'], diff))

            loglike = -0.5 * chi2
            
            # Compute Derived Values
            derived_values = []
            for i in range(self.N_recon):
                # Build the argument list based on the lambda's original signature
                call_args = []
                for arg_name in arg_order:
                    if arg_name == 'x':
                        call_args.append(self.x_recon[i])
                    else:
                        # Parameter names were defined as f"{arg_name}_{i}" in step 2
                        call_args.append(param_dict[f"{arg_name}_{i}"])
                
                derived_values.append(derived_logic(*call_args))
                
            return tuple([loglike] + derived_values)

        # 4. Execute Sampler
        if self.chatty:
            print(f"Starting sampler for {len(parameters)} parameters...")
            
        sample_results = self.sampler.run(parameters, likelihood, derived=derived_names)

        return sample_results
