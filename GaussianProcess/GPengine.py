import pandas as pd
import numpy as np
import scipy.optimize as opt
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, value_and_grad, lax
from jax.scipy.linalg import cholesky, cho_solve
import jax.random as random

from getdist import MCSamples

from tools.samplers_interface import SamplersInterface

jax.config.update("jax_enable_x64", True)

class GPCalculator:
    def __init__(self,df_data,df_cov,kernel_type='RBF'):
        self.x_train = jnp.array(df_data['x'].values)
        self.y_train = jnp.array(df_data['f'].values)
        self.noise_cov = jnp.array(df_cov.values)
        
        self.kernel_type = kernel_type.upper()
        self.l, self.sigma = 1.0, 1.0
        self.alpha, self.L = None, None

    # --- Kernels & Likelihood ---
    def _get_kernel_pure(self, l, sigma):
        if self.kernel_type == 'RBF':
            return lambda x1, x2: sigma**2 * jnp.exp(-0.5 * ((x1 - x2) / l)**2)
        elif self.kernel_type == 'MATERN3/2':
            return lambda x1, x2: sigma**2 * (1.0 + jnp.sqrt(3.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l) * jnp.exp(-jnp.sqrt(3.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l)
        elif self.kernel_type == 'MATERN5/2':
            return lambda x1, x2: sigma**2 * (1.0 + jnp.sqrt(5.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l 
                                              + (5.0 * ((x1 - x2)**2 + 1e-12)) / (3.0 * l**2)) * jnp.exp(-jnp.sqrt(5.0) * jnp.sqrt((x1 - x2)**2 + 1e-12) / l)
        raise ValueError("Kernel must be RBF, Matern3/2, or Matern5/2")

    def _log_marginal_likelihood_pure(self, log_params):
        l, sigma = jnp.exp(log_params[0]), jnp.exp(log_params[1])
        k_fn = self._get_kernel_pure(l, sigma)
        K = vmap(lambda x1: vmap(lambda x2: k_fn(x1, x2))(self.x_train))(self.x_train)
        try:
            L = cholesky(K + self.noise_cov, lower=True)
            alpha = cho_solve((L, True), self.y_train)
            return -0.5 * jnp.dot(self.y_train, alpha) - jnp.sum(jnp.log(jnp.diag(L))) - 0.5 * len(self.y_train) * jnp.log(2 * jnp.pi)
        except: return -jnp.inf

    # --- Core Logic Methods ---
    def _apply_ops(self, g, x, a, n_steps):
        t = jnp.linspace(a, x, n_steps)
        return jnp.stack([g(x), grad(g)(x), grad(grad(g))(x), jnp.trapezoid(vmap(g)(t), x=t)])

    def predict_at_params(self, x_r, l, sigma, a, n_steps):
        k_fn = self._get_kernel_pure(l, sigma)
        K = vmap(lambda x1: vmap(lambda x2: k_fn(x1, x2))(self.x_train))(self.x_train)
        L = cholesky(K + self.noise_cov, lower=True)
        alpha = cho_solve((L, True), self.y_train)
        
        M, N = len(x_r), len(self.x_train)
        
        # Cross-cov
        K_s_X = vmap(lambda xs: vmap(lambda xt: self._apply_ops(lambda t: k_fn(t, xt), xs, a, n_steps))(self.x_train))(x_r)
        K_s_X = jnp.transpose(K_s_X, (2, 0, 1)).reshape(4 * M, N)
        
        # Prior-cov
        def prior_matrix(xs1, xs2):
            def get_col(j, t1): return self._apply_ops(lambda y: k_fn(t1, y), xs2, a, n_steps)[j]
            return jnp.stack([self._apply_ops(lambda t: get_col(j, t), xs1, a, n_steps) for j in range(4)], axis=1)
        
        K_ss = vmap(lambda xs1: vmap(lambda xs2: prior_matrix(xs1, xs2))(x_r))(x_r)
        K_ss = jnp.transpose(K_ss, (2, 0, 3, 1)).reshape(4 * M, 4 * M)
        
        v = jax.scipy.linalg.solve_triangular(L, K_s_X.T, lower=True)
        return K_s_X @ alpha, K_ss - v.T @ v

    # --- User Facing Method ---
    def reconstruct(self, x_r, method='MAP', integral_start=0.0, n_int_steps=50,n_samples=2000,thinning=40,burn_in=0.3):
        """
        method: 'MAP' for optimized params, 'Bayesian' for marginalized.
        kwargs: n_samples, burn_in, thinning for Bayesian.
        """
        x_r = jnp.atleast_1d(x_r)

        info = {}
        
        # 1. Obtain Parameters
        if method.upper() == 'MAP':
            print("Optimizing (MAP)...")
            nll_g = jit(value_and_grad(lambda p: -self._log_marginal_likelihood_pure(p)))
            res = opt.minimize(lambda p: [np.array(x) for x in nll_g(p)], np.zeros(2), method='L-BFGS-B', jac=True)
            self.l, self.sigma = np.exp(res.x)
            info['Best-fit'] = {'l': self.l,'sigma': self.sigma}
            mu, cov = self.predict_at_params(x_r, self.l, self.sigma, integral_start, n_int_steps)
            lml = self._log_marginal_likelihood_pure(jnp.log(jnp.array([self.l, self.sigma])))
            
        elif method.upper() == 'BAYESIAN':
            print("Sampling (MCMC)...")
            #n_samples = kwargs.get('n_samples',2000)
            #thinning = kwargs.get('thinning',40)
           
            #Change to nautilus here?
            sampler = SamplersInterface(sampler='Nautilus',run_options='poor',chatty=False)
            
            parameters = {'logl': {'prior': [-3,3],
                                'latex': '\log l'},
                          'logsigma': {'prior': [-3,3],
                                    'latex': '\log \sigma'}}

            def likelihood(params):

                pars = [p for p in params.values()]

                logl = self._log_marginal_likelihood_pure(pars)

                return logl

            info['sample'] = sampler.run(parameters,likelihood)
            samples = np.exp(info['sample'].samples[::thinning])

            p = info['sample'].getParams()
            info['sample'].addDerived(np.exp(p.logl), name="l", label="l")
            info['sample'].addDerived(np.exp(p.logsigma), name="sigma", label="\sigma")

            # Simplified MCMC call
            #def log_post(p): return self._log_marginal_likelihood_pure(p) - 0.5 * jnp.sum((p/3.0)**2)
            #@jit
            #def step(state, key):
            #    p, lp = state
            #    k1, k2 = random.split(key)
            #    prop = p + random.normal(k1, (2,)) * 0.1
            #    prop_lp = log_post(prop)
            #    accept = jnp.log(random.uniform(k2)) < (prop_lp - lp)
            #    next_p = jnp.where(accept, prop, p)
            #    return (next_p, jnp.where(accept, prop_lp, lp)), next_p
            #
            #_, chain = lax.scan(step, (jnp.zeros(2), log_post(jnp.zeros(2))), random.split(random.PRNGKey(0), n_samples))
            #
            #samples = jnp.exp(chain[int(n_samples*burn_in)::thinning])

            #info['sample'] = MCSamples(samples=jnp.exp(chain),names=['l','sigma'],labels=['l','\sigma'],settings={"ignore_rows": burn_in})


            
            # Marginalize (Law of Total Variance)
            mu_list, cov_list = [], []
            for s in samples:
                m, c = self.predict_at_params(x_r, s[0], s[1], integral_start, n_int_steps)
                mu_list.append(m); cov_list.append(c)
                
            mu_stack, cov_stack = jnp.stack(mu_list), jnp.stack(cov_list)
            mu_marginal = jnp.mean(mu_stack, axis=0)
            # Total Cov = Mean of Covs + Var of Means
            diff = mu_stack - mu_marginal
            cov_marginal = jnp.mean(cov_stack, axis=0) + jnp.einsum('si,sj->ij', diff, diff) / len(samples)
            mu, cov, lml = mu_marginal, cov_marginal, None

        #Packing results in readable formats
        results = self._unpack(mu, cov, x_r)

        mean_df = pd.DataFrame({'x': x_r}|results['means'])

        cols = []
        for func in results['means'].keys():
            cols = cols+['{}_{}'.format(func,ind) for ind in range(len(x_r))]
        covmat_df = pd.DataFrame(cov,columns=cols,index=cols)

        for func in results['means'].keys():
            func_cov = covmat_df[[func+'_{}'.format(ind) for ind in range(len(x_r))]]
            func_cov = func_cov.drop(index=[ind for ind in func_cov.index if ind not in [func+'_{}'.format(ind) for ind in range(len(x_r))]])
            mean_df[func+'_err'] = np.sqrt(np.diag(func_cov))

        return mean_df,covmat_df,lml,info
            
    def _unpack(self, mu, cov, x_r):
        M = len(x_r)
        keys = ['f', 'd1', 'd2', 'int']
        means = {k: mu[i*M:(i+1)*M] for i, k in enumerate(keys)}
        blocks = {}
        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                blocks[f"{k1}_{k2}"] = cov[i*M:(i+1)*M, j*M:(j+1)*M]
        return {"means": means, "blocks": blocks}
