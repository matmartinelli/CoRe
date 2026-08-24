import os
import numpy as np
import pandas as pd
import pytest

from CoRe.reconstruction_tools.GaussianProcess import GPCalculator
from CoRe.derived_function.reconstructor import DerivedFunction

from getdist import MCSamples

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def load_test_data():
    """Loads dataset and covariance matrix from the tests directory."""
    test_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(test_dir, "test_recon_dataset.txt")
    covmat_path = os.path.join(test_dir, "test_recon_covmat.txt")

    dataset = pd.read_csv(dataset_path, sep='\t', header=0)
    covmat = pd.read_csv(covmat_path, sep=r'\s+', header=0, index_col=0)
    covmat.index = covmat.columns

    return dataset, covmat


@pytest.fixture
def gp_reconstruction(load_test_data):
    """Pre-runs GP MAP reconstruction to supply inputs for the derived function tests."""
    dataset, covmat = load_test_data
    x_recon = np.linspace(dataset['x'].min(), dataset['x'].max(), 10)

    gp = GPCalculator(dataset, covmat, kernel_type='RBF', chatty=False)
    means, joint_cov, _, _ = gp.reconstruct(x_recon, method='MAP')

    return x_recon, means, joint_cov


# -----------------------------------------------------------------------------
# Test Functions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("method_dict", [
    {'type': 'realizations', 'options': {'Nsamples': 1000}},
    {'type': 'sampling', 'options': {'sampler': 'emcee', 'run_options': {'nwalkers': 100,        # Number of ensemble walkers (typically >= 2 * ndim)
                                                                         'nsteps': 3000,        # Total iterations per walker
                                                                         'burn_in': 500,        # Steps to discard from the start
                                                                         'ini_width': 1.e-2,    #TODO
                                                                         'thin': 5},'sigma_width': 5}}
])
def test_derived_function_execution(gp_reconstruction, method_dict):
    """Test derived function reconstruction across realizations and emcee sampling."""
    x_recon, means, joint_cov = gp_reconstruction

    recon_dict = {'funcs': means}
    cov_dict = {'funcs': joint_cov}

    reconstructor = DerivedFunction(recon_dict, cov_dict, method_dict, chatty=False)

    derived_logic = lambda x, funcs_f1, funcs_d12: (1 + x) * funcs_d12 / funcs_f1
    derived_name = 'D'
    
    results = reconstructor.run([derived_logic], [derived_name])

    # Basic validity checks
    assert results is not None
    assert isinstance(results, MCSamples)
