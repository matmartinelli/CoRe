import pandas as pd
import numpy as np
import scipy.optimize as opt
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, value_and_grad
from jax.scipy.linalg import cholesky, cho_solve
from utils.samplers_interface import SamplersInterface
import sys

jax.config.update("jax_enable_x64", True)

class GPCalculator:
    def __init__(self, df_data, df_cov, kernel_type='RBF', chatty=False, savefile=None):
        self.x_train = jnp.array(df_data['x'].values)
        self.noise_cov = jnp.array(df_cov.values)
        self.kernel_type = kernel_type.upper()
        self.chatty = chatty
        self.savefile = savefile

        # Define allowed operators based on kernel differentiability 
        if self.kernel_type == 'RBF':
            self.available_ops = ['f', 'd1', 'd2', 'int']
        elif self.kernel_type == 'MATERN5/2':
            self.available_ops = ['f', 'd1', 'd2', 'int']
        elif self.kernel_type == 'MATERN3/2':
            self.available_ops = ['f', 'd1', 'int'] # d2 is not available 
        else:
            sys.exit('UNKNOWN KERNEL: {}'.format(self.kernel_type))

        # Detect outputs 
        self.func_cols = [c for c in df_data.columns if c not in ['x', 'f_err', 'f1_err', 'f2_err']]
        self.n_out = len(self.func_cols)
        self.y_train = jnp.concatenate([jnp.array(df_data[c].values) for c in self.func_cols])
        
        if self.chatty:
            print(f"Initialized GPCalculator with {self.n_out} output(s): {self.func_cols}")
            print(f"Kernel: {self.kernel_type} | Available Operators: {self.available_ops}")

    def _get_kernel_spatial(self, l):
        """Spatial part of the kernel k(x, x') """
        if self.kernel_type == 'RBF':
            return lambda x1, x2: jnp.exp(-0.5 * ((x1 - x2) / l)**2)
        elif self.kernel_type == 'MATERN3/2':
            return lambda x1, x2: (1.0 + jnp.sqrt(3.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l) * \
                                  jnp.exp(-jnp.sqrt(3.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l)
        elif self.kernel_type == 'MATERN5/2':
            return lambda x1, x2: (1.0 + jnp.sqrt(5.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l + \
                                  (5.0 * ((x1 - x2)**2 + 1e-12)) / (3.0 * l**2)) * \
                                  jnp.exp(-jnp.sqrt(5.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l)
        raise ValueError("Kernel must be RBF, Matern3/2, or Matern5/2")

    def _get_B_matrix(self, sigmas, rho=0.0):
        if self.n_out == 1:
            return jnp.array([[sigmas[0]**2]])
        else:
            v1, v2 = sigmas[0], sigmas[1]
            return jnp.array([[v1**2, rho * v1 * v2],
                             [rho * v1 * v2, v2**2]])

    def _log_marginal_likelihood_pure(self, params):
        l = jnp.exp(params[0])
        sigmas = jnp.exp(params[1:1+self.n_out])
        rho = jnp.tanh(params[-1]) if self.n_out > 1 else 0.0
        
        k_spatial = self._get_kernel_spatial(l)
        B = self._get_B_matrix(sigmas, rho)
        K_s = vmap(lambda x1: vmap(lambda x2: k_spatial(x1, x2))(self.x_train))(self.x_train)
        K = jnp.kron(B, K_s)
        
        try:
            L = cholesky(K + self.noise_cov, lower=True)
            alpha = cho_solve((L, True), self.y_train)
            return -0.5 * jnp.dot(self.y_train, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * len(self.y_train) * jnp.log(2 * jnp.pi)
        except: return -jnp.inf

    def _apply_ops(self, g, x, a, n_steps):
        """Standard calculus operators, filtered by available_ops """
        t = jnp.linspace(a, x, n_steps)
        res = []
        for op in self.available_ops:
            if op == 'f': res.append(g(x))
            elif op == 'd1': res.append(grad(g)(x))
            elif op == 'd2': res.append(grad(grad(g))(x))
            elif op == 'int': res.append(jnp.trapezoid(vmap(g)(t), x=t))
        return jnp.stack(res)

    def predict_at_params(self, x_r, l, sigmas, rho, a, n_steps):
        k_spatial = self._get_kernel_spatial(l)
        B = self._get_B_matrix(sigmas, rho)
        
        K_s_train = vmap(lambda x1: vmap(lambda x2: k_spatial(x1, x2))(self.x_train))(self.x_train)
        K_full_train = jnp.kron(B, K_s_train)
        L = cholesky(K_full_train + self.noise_cov, lower=True)
        alpha = cho_solve((L, True), self.y_train)

        M, N = len(x_r), len(self.x_train)
        n_ops = len(self.available_ops)
        
        # Cross-cov (dynamic reshaping based on n_ops) 
        K_ops_s = vmap(lambda xs: vmap(lambda xt: self._apply_ops(lambda t: k_spatial(t, xt), xs, a, n_steps))(self.x_train))(x_r)
        K_ops_s = jnp.transpose(K_ops_s, (2, 0, 1)).reshape(n_ops * M, N)
        K_cross = jnp.kron(B, K_ops_s)

        # Prior-cov (dynamic matrix size based on n_ops) 
        def prior_matrix(xs1, xs2):
            def get_col(j, t1): return self._apply_ops(lambda y: k_spatial(t1, y), xs2, a, n_steps)[j]
            return jnp.stack([self._apply_ops(lambda t: get_col(j, t), xs1, a, n_steps) for j in range(n_ops)], axis=1)
        
        K_recon_s = vmap(lambda xs1: vmap(lambda xs2: prior_matrix(xs1, xs2))(x_r))(x_r)
        K_recon_s = jnp.transpose(K_recon_s, (2, 0, 3, 1)).reshape(n_ops * M, n_ops * M)
        K_recon = jnp.kron(B, K_recon_s)

        v = jax.scipy.linalg.solve_triangular(L, K_cross.T, lower=True)
        return K_cross @ alpha, K_recon - v.T @ v

    def reconstruct(self, x_r, method='MAP', integral_start=0.0, n_int_steps=50, n_samples=2000):
        x_r = jnp.atleast_1d(x_r)
        info = {}
        n_params = 1 + self.n_out + (1 if self.n_out > 1 else 0)

        # Helper to find MAP (Maximum A Posteriori) hyperparameters
        def find_map():
            if self.chatty: print(f"Optimizing (MAP) to find hyper-parameter centers...")
            nll_g = jit(value_and_grad(lambda p: -self._log_marginal_likelihood_pure(p)))
            # Starting from zero (log-space) is a neutral start
            res = opt.minimize(lambda p: [np.array(x) for x in nll_g(p)],
                               jnp.zeros(n_params), method='L-BFGS-B', jac=True)
            return res

        if method.upper() == 'MAP':
            res = find_map()
            best_p = res.x
            l, sigmas = np.exp(best_p[0]), np.exp(best_p[1:1+self.n_out])
            rho = np.tanh(best_p[-1]) if self.n_out > 1 else 0.0
            mu, cov = self.predict_at_params(x_r, l, sigmas, rho, integral_start, n_int_steps)
            lml = -res.fun
            info['Best-fit'] = {'l': l, 'sigmas': sigmas, 'rho': rho}

        elif method.upper() == 'BAYESIAN':
            # 1. First, perform MAP to determine the prior window
            map_res = find_map()
            map_p = map_res.x

            if self.chatty:
                print(f"MAP found at: {map_p}. Setting informative priors for sampling...")

            # 2. Define Priors centered on MAP results
            # We use a width of +/- 4 in log-space to allow the sampler to explore
            # while staying in the relevant physical regime.
            width = 4.0
            parameters = {'logl': {'prior': [map_p[0] - width, map_p[0] + width], 'latex': r'$\log l$'}}

            for i in range(self.n_out):
                suffix = str(i+1) if self.n_out > 1 else ""
                idx = 1 + i
                parameters[f'logsigma{suffix}'] = {
                    'prior': [map_p[idx] - width, map_p[idx] + width],
                    'latex': rf'$\log \sigma_{{{suffix}}}$'
                }

            if self.n_out > 1:
                # Rho is tanh-space; we give it a broad range but center on MAP
                parameters['atanhrho'] = {
                    'prior': [map_p[-1] - 2.0, map_p[-1] + 2.0],
                    'latex': r'$\text{atanh}(\rho)$'
                }

            # 3. Proceed with Bayesian Sampling
            if self.chatty: print(f"Sampling (MCMC) for {self.n_out} outputs...")
            sampler = SamplersInterface(sampler='Nautilus', run_options='poor',
                                       chatty=self.chatty, savefile=self.savefile)

            def likelihood(param_dict):
                p_array = [param_dict[k] for k in parameters.keys()]
                return self._log_marginal_likelihood_pure(jnp.array(p_array))

            info['sample'] = sampler.run(parameters, likelihood)
            meanpars = info['sample'].getMeans()
            covpars  = info['sample'].getCov()
            
            # Convert samples to jax array for vectorization
            s_samples = jnp.array(np.random.multivariate_normal(meanpars, covpars, size=n_samples))

            # Define a pure helper function for a single prediction
            def single_predict(s):
                l_s = jnp.exp(s[0])
                sigmas_s = jnp.exp(s[1:1+self.n_out])
                rho_s = jnp.tanh(s[-1]) if self.n_out > 1 else 0.0
                return self.predict_at_params(x_r, l_s, sigmas_s, rho_s, integral_start, n_int_steps)

            # Vectorize the helper function and JIT it for maximum performance
            # vmap(func)(samples) will compute all predictions in a single batch
            if self.chatty: print(f"Vectorizing predictions for {n_samples} samples...")
            vectorized_predict = jit(vmap(single_predict))
            
            # Execute the batch (this is where the speedup happens)
            mu_stack, cov_stack = vectorized_predict(s_samples)

            # Calculate the final combined mean and covariance
            mu = jnp.mean(mu_stack, axis=0)
            # Use the Law of Total Variance: E[Var] + Var(E)
            cov = jnp.mean(cov_stack, axis=0) + jnp.cov(mu_stack, rowvar=False)
            lml = None 

        # 2. Unpacking and Labeling (using available_ops)
        results = self._unpack(mu, cov, x_r)
        mean_df = pd.DataFrame({'x': np.array(x_r)})
        all_labels = []
        for out_idx in range(self.n_out):
            suffix = str(out_idx + 1) if self.n_out > 1 else ""
            for op in self.available_ops:
                comp_name = f"{op}{suffix}"
                mean_df[comp_name] = results['means'][comp_name]
                all_labels += [f"{comp_name}_{i}" for i in range(len(x_r))]

        covmat_df = pd.DataFrame(np.array(cov), columns=all_labels, index=all_labels)

        for col in [c for c in mean_df.columns if c != 'x']:
            labels = [f"{col}_{i}" for i in range(len(x_r))]
            mean_df[f"{col}_err"] = np.sqrt(np.diag(covmat_df.loc[labels, labels].values))

        return mean_df, covmat_df, lml, info

    def _unpack(self, mu, cov, x_r):
        """Unpacks results based on available_ops """
        M = len(x_r)
        means = {}
        curr = 0
        for out_idx in range(self.n_out):
            suffix = str(out_idx + 1) if self.n_out > 1 else ""
            for op in self.available_ops:
                means[f"{op}{suffix}"] = mu[curr:curr+M]
                curr += M
        return {"means": means}
