import numpy as np
import pandas as pd
import inspect

from tools.samplers_interface import SamplersInterface

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
        Introspects lambda: e.g., 'lambda rec1_f, rec2_d1'
        Returns list of (recon_key, component) tuples
        """
        args = inspect.getfullargspec(func).args
        required = []
        for arg in args:
            parts = arg.split('_')
            # Assuming format: 'reconKey_component'
            # e.g., 'func1_f' -> key='func1', comp='f'
            recon_key = "_".join(parts[:-1])
            component = parts[-1]
            required.append((recon_key, component, arg))
        return required

    def run(self,derived_logic,derived_name,sigma_width=5):
        """
        derived_logic: lambda, e.g., lambda f, d1: d1 / f
        sigma_width: How many sigmas for the prior width
        """
        required_info = self._get_required_params(derived_logic)

        if self.chatty:
            print(f"Detected required variables for derived function: {required_info}")

        # 1. Prepare Free Parameters (Priors)
        parameters = {}
        for recon_key, component, arg_name in required_info:
            df_recon = self.recon_dict[recon_key]
            df_cov = self.cov_dict[recon_key]

            # Map indices for the specific component
            labels = [f"{component}_{i}" for i in range(self.N_recon)]
            sigmas = np.sqrt(np.diag(df_cov.loc[labels, labels].values))

            for i in range(self.N_recon):
                mean_val = df_recon.iloc[i][component]
                parameters[f"{arg_name}_{i}"] = {
                    'prior': [mean_val - sigma_width * sigmas[i],
                              mean_val + sigma_width * sigmas[i]],
                    'latex': f'{recon_key}{component}_{i}'
                }

        # 2. Prepare Derived Parameters for Sampler Output
        # These are the values of D(x) at each reconstruction point
        derived_names = {f"{derived_name}_{i}": {'latex': rf'${derived_name}_{{{i}}}$'} for i in range(self.N_recon)}

        #MMmod: TODO 
        #Put some print here for letting the user know parameters that will be used and derived

        # 3. Define the Likelihood
        likeparts = {}
        #MMmod: this part creates covariance and datavectors separating for key
        for recon_key, df_recon in self.recon_dict.items():
           
            df_cov   = self.cov_dict[recon_key]

            requested_functions = [r[1] for r in required_info if r[0]==recon_key]

            #MMmod: this part marginalizes the unused functions out of the joint covariances
            keep_labels = [col for col in df_cov.columns if col.split('_')[0] in requested_functions]
            reduced_cov = df_cov.drop(columns=[col for col in df_cov.columns if col not in keep_labels])
            reduced_cov = reduced_cov.drop(index=[col for col in reduced_cov.index if col not in keep_labels])

            likeparts[recon_key] = {'inv_cov':  pd.DataFrame(np.linalg.inv(reduced_cov.values),columns=reduced_cov.columns,index=reduced_cov.index),
                                    'data_vec': np.array([df_recon.iloc[int(col.split('_')[1])][col.split('_')[0]] for col in reduced_cov.columns])}


        def likelihood(param_dict):

            #MMmod: WARNING!
            #Here we consider the possibility of correlation only between
            #functions extracted from the same reconstruction.
            #Correlation between separate reconstructions is assumed to be vanishing
            #Introducing this extra correlation seems like an overkill

            chi2 = 0

            for recon_key,parts in likeparts.items():
                # Construct theory vector. 
                theory_vec = [param_dict[recon_key+'_'+label] for label in parts['inv_cov'].columns]
                theory_vec = np.array(theory_vec)
            
                # Mahalanobis Distance (Log-Likelihood)
                diff = theory_vec - parts['data_vec']
                chi2 += np.dot(diff, np.dot(parts['inv_cov'].values, diff))

            loglike = -0.5*chi2
            
            # Compute Derived Values
            # We pass the relevant params for each index i to the lambda
            derived_values = []
            for i in range(self.N_recon):
                args = [param_dict[f"{arg_name}_{i}"] for recon_key, component, arg_name in required_info]
                derived_values.append(derived_logic(*args))
                
            return tuple([loglike] + derived_values)

        # 4. Execute Sampler
        if self.chatty:
            print(f"Starting sampler for {len(parameters)} parameters...")
            
        sample_results = self.sampler.run(parameters, likelihood, derived=derived_names)

        return sample_results
