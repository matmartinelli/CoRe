import os
import numpy as np
import pandas as pd
import pytest
import warnings

from CoRe.reconstruction_tools.GaussianProcess import GPCalculator
from CoRe.utils.diagnostics                    import Scorer

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def load_test_data():
    """Loads the dataset and covariance matrix from the tests directory."""
    test_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(test_dir, "test_dataset.txt")
    covmat_path = os.path.join(test_dir, "test_covmat.txt")

    dataset = pd.read_csv(dataset_path,sep='\t',header=0)
    covmat  = pd.read_csv(covmat_path,sep=r'\s+',header=0,index_col=0)
    covmat.index = covmat.columns
    
    return dataset, covmat


@pytest.fixture
def x_recon(load_test_data):
    """Generates evaluation grid based on input dataset domain."""
    dataset, _ = load_test_data
    x_min, x_max = dataset['x'].min(), dataset['x'].max()
    return np.linspace(x_min, x_max, 50)


# -----------------------------------------------------------------------------
# Test Functions
# -----------------------------------------------------------------------------

def test_gp_initialization(load_test_data):
    """Test initializing the GPCalculator object."""
    dataset, covmat = load_test_data
    gp = GPCalculator(dataset, covmat, kernel_type='RBF', chatty=False)
    
    assert gp is not None
    assert gp.kernel_type == 'RBF'


def test_gp_map_reconstruction(load_test_data, x_recon):
    """Test Maximum A Posteriori (MAP) reconstruction and best-fit transformations."""
    dataset, covmat = load_test_data
    gp = GPCalculator(dataset, covmat, kernel_type='RBF', chatty=False)
    
    means, joint_cov, lml, info = gp.reconstruct(x_recon, method='MAP')

    # Shape and type checks
    assert len(means) == len(x_recon)
    assert not means.isna().values.any()
    assert not joint_cov.isna().values.any()
    
    assert isinstance(lml, (float, np.floating))
    assert 'Best-fit' in info

    # Verify best-fit log-transformations
    best_fit = info['Best-fit']
    best_fit['logl'] = np.log(best_fit['l'])
    best_fit['logsigma'] = np.log(best_fit['sigmas'][0])

    assert np.isfinite(best_fit['logl'])
    assert np.isfinite(best_fit['logsigma'])


def test_gp_bayesian_reconstruction_and_scoring(load_test_data, x_recon):
    """Test Bayesian marginalization and subsequent diagnostic scoring against data."""
    dataset, covmat = load_test_data
    gp = GPCalculator(dataset, covmat, kernel_type='RBF', chatty=False)

    sampler = 'nautilus'
    sampler_options = 'poor'

    means, joint_cov, lml, info = gp.reconstruct(
        x_recon, 
        method='BAYESIAN', 
        sampler_name=sampler, 
        sampler_options=sampler_options, 
        n_samples=50
    )

    assert not means.isna().values.any()
    assert not joint_cov.isna().values.any()

    # Test Scorer module
    scorer = Scorer(means, joint_cov, chatty=False)
    datasets = [{'type': 'f', 'df': dataset, 'cov': covmat}]
    
    res = scorer.score_against_data(datasets)

    assert res is not None
    assert 'p_value' in res

    p_val = res['p_value']

    # Fail test if p_value < 0.01
    if p_val < 0.01:
        pytest.fail(f"Nautilus reconstruction rejected! p-value is critically low: {p_val:.4f} (< 0.01)")

    # Issue warning if 0.01 <= p_value <= 0.05
    if 0.01 <= p_val <= 0.05:
        warnings.warn(
            UserWarning(f"Marginal p-value detected: {p_val:.4f} (0.01 <= p_value <= 0.05)")
        )
