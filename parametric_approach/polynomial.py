import sys
import numpy  as np
import pandas as pd

from numpy.polynomial import Polynomial

class PolynomialFit:

    def __init__(self,dataset,covmat,order=3):
        """
        Initializes the generator for a specific polynomial order.
        An order 'n' polynomial requires n+1 coefficients.
        """
        self.order = order
        self.num_params = order + 1
        self.dataset = dataset
        self.covmat = covmat

    def generate(self,x,coeffs):
        """
        Takes coefficients [a0, a1, ..., an] and returns
        (Function, 1st Deriv, 2nd Deriv, Integral)
        """
        if len(coeffs) != self.num_params:
            raise ValueError(f"Expected {self.num_params} coefficients for order {self.order}.")

        # Create the base polynomial: P(x) = a0 + a1*x + ... + an*x^n
        p = Polynomial(coeffs)

        # Compute derivatives and integral using NumPy's built-in methods
        p_deriv1 = p.deriv(m=1)   # First derivative
        p_deriv2 = p.deriv(m=2)   # Second derivative
        p_integ  = p.integ(m=1)   # Antiderivative (integration constant C=0)

        results = {'f': p(x), 
                   'f_p': p_deriv1(x),
                   'f_pp': p_deriv2(x),
                   'f_i': p_integ(x)}

        return results 

    def likelihood(self,params_dict):

        parvals = [float(par) for par in params_dict.values()]

        vals    = self.generate(self.dataset['x'],parvals)
        diffvec = vals['f']-self.dataset['y']
        loglike = -0.5*np.dot(diffvec,np.dot(np.linalg.inv(self.covmat),diffvec))

        return loglike

    def get_derived(self,sample,x_recon,model_params):

        p = sample.getParams()

        for ind,x in enumerate(x_recon):
            f    = []
            f_p  = []
            f_pp = []
            f_i  = []
            for ci in range(len(getattr(p,list(model_params.keys())[0]))):
                parsloc = [getattr(p,par)[ci] for par in model_params.keys()]
                locres  = self.generate(x,parsloc)
                f.append(locres['f'])
                f_p.append(locres['f_p'])
                f_pp.append(locres['f_pp'])
                f_i.append(locres['f_i'])

            sample.addDerived(f,name='f_x_{}'.format(ind), label='f(x_{})'.format(ind))
            sample.addDerived(f_p,name='f_p_x_{}'.format(ind), label='f_p(x_{})'.format(ind))
            sample.addDerived(f_pp,name='f_pp_x_{}'.format(ind), label='f_pp(x_{})'.format(ind))
            sample.addDerived(f_i,name='f_i_x_{}'.format(ind), label='f_i(x_{})'.format(ind))

        return sample

    def get_reconstruction(self,sample,x_recon):

        par_to_ind = {par:ind for ind,par in enumerate(sample.getParamNames().list())}

        mean = sample.getMeans(pars=[par_to_ind[par] for par in ['f_x_{}'.format(ind) for ind in range(len(x_recon))]])
        covmat = sample.cov(pars=['f_x_{}'.format(ind) for ind in range(len(x_recon))])

        reconstruction = pd.DataFrame({'x': x_recon,
                                       'y': mean,
                                       'y_err': np.sqrt(np.diag(covmat)),
                                       'Type': 'Polynomial (order {})'.format(self.order)})

        return reconstruction

