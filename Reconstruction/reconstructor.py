import numpy as np
import pandas as pd
import inspect

from tools.samplers_interface import SamplersInterface

class DerivedFunction:
    def __init__(self, df_recon, df_joint_cov,sampler='Nautilus',run_options='poor',chatty=True):
        """
        df_recon: DataFrame with columns ['x', 'f', 'd1', etc.]
        df_joint_cov: Labeled DataFrame (f_0, d1_0...)
        sampler_interface: An instance of SamplersInterface (e.g., Nautilus)
        """
        self.df_recon     = df_recon
        self.df_joint_cov = df_joint_cov
        self.sampler      = SamplersInterface(sampler=sampler,run_options=run_options,chatty=chatty)
        self.chatty       = chatty
        self.N_recon      = len(df_recon)
        
        # Pre-compute the precision matrix (Inverse Covariance)
        # We only invert once to keep the likelihood fast
        self.inv_cov = pd.DataFrame(
            np.linalg.pinv(df_joint_cov.values + np.eye(len(df_joint_cov)) * 1e-10),
            index=df_joint_cov.index,
            columns=df_joint_cov.columns
        )

    def _get_required_vars(self,func):
        """Introspects the lambda function to see which GP components are needed."""
        return inspect.getfullargspec(func).args

    def run(self,derived_logic,derived_name,sigma_width=5):
        """
        derived_logic: lambda, e.g., lambda f, d1: d1 / f
        sigma_width: How many sigmas for the prior width
        """
        required_vars = self._get_required_vars(derived_logic)

        if self.chatty:
            print(f"Detected required variables for derived function: {required_vars}")

        # 1. Prepare Free Parameters (Priors)
        parameters = {}
        for var in required_vars:
            # Extract diagonal uncertainty for priors
            labels = [f"{var}_{i}" for i in range(self.N_recon)]
            sigmas = np.sqrt(np.diag(self.df_joint_cov.loc[labels, labels].values))
            
            for i in range(self.N_recon):
                mean_val = self.df_recon.iloc[i][var]
                std_val = sigmas[i]
                
                parameters[f"{var}_{i}"] = {
                    'prior': [mean_val - sigma_width * std_val, 
                              mean_val + sigma_width * std_val],
                    'latex': rf'${var}_{{{i}}}$'
                }

        # 2. Prepare Derived Parameters for Sampler Output
        # These are the values of D(x) at each reconstruction point
        derived_names = {f"{derived_name}_{i}": {'latex': rf'${derived_name}_{{{i}}}$'} for i in range(self.N_recon)}


        # 3. Define the Likelihood
        # Note: We must ensure the vector order matches the covariance matrix
        cov_labels = [col for col in self.df_joint_cov.columns if col.split('_')[0] in required_vars]

        reduced_cov = self.df_joint_cov.drop(columns=[col for col in self.df_joint_cov.columns if col not in cov_labels])
        reduced_cov = reduced_cov.drop(index=[col for col in reduced_cov.index if col not in cov_labels])

        inv_cov = pd.DataFrame(np.linalg.inv(reduced_cov.values),columns=reduced_cov.columns,index=reduced_cov.index)

        data_vec = np.array([
            self.df_recon.iloc[int(label.split('_')[1])][label.split('_')[0]] 
            for label in cov_labels
        ])

        def likelihood(param_dict):
            # Construct theory vector. 
            theory_vec = [param_dict[label] for label in cov_labels]
            theory_vec = np.array(theory_vec)
            
            # Mahalanobis Distance (Log-Likelihood)
            diff = theory_vec - data_vec
            loglike = -0.5 * np.dot(diff, np.dot(inv_cov.values, diff))
            
            # Compute Derived Values
            # We pass the relevant params for each index i to the lambda
            derived_values = []
            for i in range(self.N_recon):
                args = [param_dict[f"{var}_{i}"] for var in required_vars]
                derived_values.append(derived_logic(*args))
                
            return tuple([loglike] + derived_values)

        # 4. Execute Sampler
        if self.chatty:
            print(f"Starting sampler for {len(parameters)} parameters...")
            
        sample_results = self.sampler.run(parameters, likelihood, derived=derived_names)

        return sample_results
