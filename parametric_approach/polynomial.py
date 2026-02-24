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
                   'd1': p_deriv1(x),
                   'd2': p_deriv2(x),
                   'int': p_integ(x)}

        return results 

    def likelihood(self,params_dict):

        parvals = [float(par) for par in params_dict.values()]

        vals    = self.generate(self.dataset['x'],parvals)
        diffvec = vals['f']-self.dataset['f']
        loglike = -0.5*np.dot(diffvec,np.dot(np.linalg.inv(self.covmat),diffvec))

        return loglike

    def get_derived(self,sample,x_recon,model_params):

        p = sample.getParams()

        for ind,x in enumerate(x_recon):
            f    = []
            d1   = []
            d2   = []
            inte = []
            for ci in range(len(getattr(p,list(model_params.keys())[0]))):
                parsloc = [getattr(p,par)[ci] for par in model_params.keys()]
                locres  = self.generate(x,parsloc)
                f.append(locres['f'])
                d1.append(locres['d1'])
                d2.append(locres['d2'])
                inte.append(locres['int'])

            sample.addDerived(f,name='f_x_{}'.format(ind), label='f(x_{})'.format(ind))
            sample.addDerived(d1,name='d1_x_{}'.format(ind), label='d1(x_{})'.format(ind))
            sample.addDerived(d2,name='d2_x_{}'.format(ind), label='d2(x_{})'.format(ind))
            sample.addDerived(inte,name='int_x_{}'.format(ind), label='int(x_{})'.format(ind))

        return sample

    def get_reconstruction(self,sample,x_recon):

        Npred = len(x_recon)
        recons  = {'x': x_recon}
        covmats = {}

        par_to_ind = {par:ind for ind,par in enumerate(sample.getParamNames().list())}

        recons['f']  = sample.getMeans(pars=[par_to_ind[par] for par in ['f_x_{}'.format(ind) for ind in range(Npred)]])
        covmats['f'] = pd.DataFrame(sample.cov(pars=['f_x_{}'.format(ind) for ind in range(Npred)]),
                              columns=['f_{}'.format(ind) for ind in range(Npred)],
                              index=['f_{}'.format(ind) for ind in range(Npred)])

        recons['f_err'] = np.sqrt(np.diag(covmats['f'].values))

        recons['d1']  = sample.getMeans(pars=[par_to_ind[par] for par in ['d1_x_{}'.format(ind) for ind in range(Npred)]])
        covmats['d1'] = pd.DataFrame(sample.cov(pars=['d1_x_{}'.format(ind) for ind in range(Npred)]),
                              columns=['d1_{}'.format(ind) for ind in range(Npred)],
                              index=['d1_{}'.format(ind) for ind in range(Npred)])

        recons['d1_err'] = np.sqrt(np.diag(covmats['d1'].values))

        recons['d2']  = sample.getMeans(pars=[par_to_ind[par] for par in ['d2_x_{}'.format(ind) for ind in range(Npred)]])
        covmats['d2'] = pd.DataFrame(sample.cov(pars=['d2_x_{}'.format(ind) for ind in range(Npred)]),
                              columns=['d2_{}'.format(ind) for ind in range(Npred)],
                              index=['d2_{}'.format(ind) for ind in range(Npred)])

        recons['d2_err'] = np.sqrt(np.diag(covmats['d2'].values))

        recons['int']  = sample.getMeans(pars=[par_to_ind[par] for par in ['int_x_{}'.format(ind) for ind in range(Npred)]])
        covmats['int'] = pd.DataFrame(sample.cov(pars=['int_x_{}'.format(ind) for ind in range(Npred)]),
                              columns=['int_{}'.format(ind) for ind in range(Npred)],
                              index=['int_{}'.format(ind) for ind in range(Npred)])

        recons['int_err'] = np.sqrt(np.diag(covmats['int'].values))

        return recons,covmats

