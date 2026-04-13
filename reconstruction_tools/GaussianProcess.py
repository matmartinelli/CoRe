import pandas as pd
import numpy as np
import scipy.optimize as opt
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, value_and_grad
from jax.scipy.linalg import cholesky, cho_solve
from utils.samplers_interface import SamplersInterface

jax.config.update("jax_enable_x64", True)

class GPCalculator:
    def __init__(self, df_data, df_cov, kernel_type='RBF', chatty=False, savefile=None):
        self.x_train = jnp.array(df_data['x'].values)
        self.noise_cov = jnp.array(df_cov.values)
        self.kernel_type = kernel_type.upper()
        self.chatty = chatty
        self.savefile = savefile

        # Detect if we have one or two functions (e.g., 'f' or ['f1', 'f2'])
        self.func_cols = [c for c in df_data.columns if c not in ['x', 'f_err', 'f1_err', 'f2_err']]
        self.n_out = len(self.func_cols)
        
        # Flattened data vector: [f1_0...f1_n, f2_0...f2_n]
        self.y_train = jnp.concatenate([jnp.array(df_data[c].values) for c in self.func_cols])
        
        if self.chatty:
            print(f"Initialized GPCalculator with {self.n_out} output(s): {self.func_cols}")

    def _get_kernel_spatial(self, l):
        """Spatial part of the kernel k(x, x')"""
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
        """Coregionalization matrix B"""
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
        """Standard calculus operators from original code"""
        t = jnp.linspace(a, x, n_steps)
        return jnp.stack([g(x), grad(g)(x), grad(grad(g))(x), jnp.trapezoid(vmap(g)(t), x=t)])

    def predict_at_params(self, x_r, l, sigmas, rho, a, n_steps):
        k_spatial = self._get_kernel_spatial(l)
        B = self._get_B_matrix(sigmas, rho)
        
        K_s_train = vmap(lambda x1: vmap(lambda x2: k_spatial(x1, x2))(self.x_train))(self.x_train)
        K_full_train = jnp.kron(B, K_s_train)
        L = cholesky(K_full_train + self.noise_cov, lower=True)
        alpha = cho_solve((L, True), self.y_train)

        M, N = len(x_r), len(self.x_train)
        
        # Cross-cov (applying calculus ops to the spatial kernel)
        K_ops_s = vmap(lambda xs: vmap(lambda xt: self._apply_ops(lambda t: k_spatial(t, xt), xs, a, n_steps))(self.x_train))(x_r)
        K_ops_s = jnp.transpose(K_ops_s, (2, 0, 1)).reshape(4 * M, N)
        K_cross = jnp.kron(B, K_ops_s)

        # Prior-cov (applying calculus ops to both kernel arguments)
        def prior_matrix(xs1, xs2):
            def get_col(j, t1): return self._apply_ops(lambda y: k_spatial(t1, y), xs2, a, n_steps)[j]
            return jnp.stack([self._apply_ops(lambda t: get_col(j, t), xs1, a, n_steps) for j in range(4)], axis=1)
        
        K_recon_s = vmap(lambda xs1: vmap(lambda xs2: prior_matrix(xs1, xs2))(x_r))(x_r)
        K_recon_s = jnp.transpose(K_recon_s, (2, 0, 3, 1)).reshape(4 * M, 4 * M)
        K_recon = jnp.kron(B, K_recon_s)

        v = jax.scipy.linalg.solve_triangular(L, K_cross.T, lower=True)
        return K_cross @ alpha, K_recon - v.T @ v

    def reconstruct(self, x_r, method='MAP', integral_start=0.0, n_int_steps=50, n_samples=2000):
        x_r = jnp.atleast_1d(x_r)
        info = {}
        
        # 1. Parameter Setup
        # We need: 1 lengthscale + N amplitudes + (1 correlation if N > 1)
        n_params = 1 + self.n_out + (1 if self.n_out > 1 else 0)

        if method.upper() == 'MAP':
            if self.chatty: print(f"Optimizing (MAP) for {self.n_out} outputs...")
            
            # Initial guess: all zeros in log-space/atanh-space
            nll_g = jit(value_and_grad(lambda p: -self._log_marginal_likelihood_pure(p)))
            res = opt.minimize(lambda p: [np.array(x) for x in nll_g(p)], 
                               jnp.zeros(n_params), method='L-BFGS-B', jac=True)
            
            best_p = res.x
            l = np.exp(best_p[0])
            sigmas = np.exp(best_p[1:1+self.n_out])
            rho = np.tanh(best_p[-1]) if self.n_out > 1 else 0.0
            
            mu, cov = self.predict_at_params(x_r, l, sigmas, rho, integral_start, n_int_steps)
            lml = -res.fun
            info['Best-fit'] = {'l': l, 'sigmas': sigmas, 'rho': rho}

        elif method.upper() == 'BAYESIAN':
            if self.chatty: print(f"Sampling (MCMC) for {self.n_out} outputs...")
            sampler = SamplersInterface(sampler='Nautilus', run_options='poor', 
                                       chatty=self.chatty, savefile=self.savefile)
            
            # Dynamically build the parameter dictionary for the sampler
            parameters = {'logl': {'prior': [-3, 5], 'latex': r'$\log l$'}}
            for i in range(self.n_out):
                suffix = str(i+1) if self.n_out > 1 else ""
                parameters[f'logsigma{suffix}'] = {'prior': [-3, 5], 'latex': rf'$\log \sigma_{{{suffix}}}$'}
            
            if self.n_out > 1:
                parameters['atanhrho'] = {'prior': [-3, 3], 'latex': r'$\text{atanh}(\rho)$'}

            def likelihood(param_dict):
                # Convert dict back to the ordered array expected by _log_marginal_likelihood_pure
                p_array = [param_dict[k] for k in parameters.keys()]
                return self._log_marginal_likelihood_pure(jnp.array(p_array))

            info['sample'] = sampler.run(parameters, likelihood)
            
            # Posterior predictive logic
            meanpars = info['sample'].getMeans() # In the order of the dict keys
            covpars  = info['sample'].getCov()
            s_samples = np.random.multivariate_normal(meanpars, covpars, size=n_samples)

            mu_list, cov_list = [], []
            for s in s_samples:
                # Unpack the sample based on n_out
                l_s = np.exp(s[0])
                sigmas_s = np.exp(s[1:1+self.n_out])
                rho_s = np.tanh(s[-1]) if self.n_out > 1 else 0.0
                
                m, c = self.predict_at_params(x_r, l_s, sigmas_s, rho_s, integral_start, n_int_steps)
                mu_list.append(m); cov_list.append(c)
                
            mu_stack, cov_stack = jnp.stack(mu_list), jnp.stack(cov_list)
            mu = jnp.mean(mu_stack, axis=0)
            # Law of Total Variance
            cov = jnp.mean(cov_stack, axis=0) + jnp.cov(mu_stack, rowvar=False)
            lml = None

        # 2. Unpacking and Labeling
        results = self._unpack(mu, cov, x_r)
        mean_df = pd.DataFrame({'x': np.array(x_r)})
        
        # We need a flat list of all labels for the 4N (or 8N) covariance matrix
        all_labels = []
        for out_idx in range(self.n_out):
            suffix = str(out_idx + 1) if self.n_out > 1 else ""
            for op in ['f', 'd1', 'd2', 'int']:
                comp_name = f"{op}{suffix}"
                mean_df[comp_name] = results['means'][comp_name]
                all_labels += [f"{comp_name}_{i}" for i in range(len(x_r))]

        covmat_df = pd.DataFrame(np.array(cov), columns=all_labels, index=all_labels)

        # Add error bars to the mean dataframe
        for col in [c for c in mean_df.columns if c != 'x']:
            labels = [f"{col}_{i}" for i in range(len(x_r))]
            mean_df[f"{col}_err"] = np.sqrt(np.diag(covmat_df.loc[labels, labels].values))

        return mean_df, covmat_df, lml, info

    def _unpack(self, mu, cov, x_r):
        M = len(x_r)
        ops = ['f', 'd1', 'd2', 'int']
        means = {}
        curr = 0
        for out_idx in range(self.n_out):
            suffix = str(out_idx + 1) if self.n_out > 1 else ""
            for op in ops:
                means[f"{op}{suffix}"] = mu[curr:curr+M]
                curr += M
        return {"means": means}
