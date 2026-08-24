import os
import numpy as np
import pandas as pd
import pytest
import warnings
from numpy.polynomial import Polynomial

from CoRe.reconstruction_tools.Binning import BinnedCalculator
from CoRe.utils.diagnostics import Scorer

# -----------------------------------------------------------------------------
# Helpers & Fixtures
# -----------------------------------------------------------------------------

def get_theory(coeffs):
    """Generates derivative and integral evaluation functions for polynomial coefficients."""
    p = Polynomial(coeffs)
    p_deriv1 = p.deriv(m=1)
    p_deriv2 = p.deriv(m=2)
    p_integ = p.integ(m=1)
    
    return {
        'f': lambda x: p(x),
        'd1': lambda x: p_deriv1(x),
        'd2': lambda x: p_deriv2(x),
        'int': lambda x: p_integ(x)
    }


@pytest.fixture
def load_test_data():
    """Loads the dataset and covariance matrix from the tests directory."""
    test_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(test_dir, "test_dataset.txt")
    covmat_path = os.path.join(test_dir, "test_covmat.txt")

    dataset = pd.read_csv(dataset_path, sep='\t', header=0)
    covmat = pd.read_csv(covmat_path, sep=r'\s+', header=0, index_col=0)
    covmat.index = covmat.columns

    return dataset, covmat


@pytest.fixture
def x_recon(load_test_data):
    """Generates evaluation grid based on input dataset domain."""
    N_recon    = 5
    xmin_recon = 0.1
    xmax_recon = 1.8
    x_recon = np.linspace(xmin_recon,xmax_recon,N_recon) 
    return x_recon


@pytest.fixture
def theory_model():
    """Generates fiducial polynomial theory model."""
    fiducial = {'a0': 3, 'a1': -1, 'a2': 5, 'a3': -0.1}
    return get_theory(list(fiducial.values()))


# -----------------------------------------------------------------------------
# Test Functions
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["FLAT", "GLS"])
def test_binning_initialization(load_test_data, method):
    """Test initializing the BinnedCalculator object."""
    dataset, covmat = load_test_data
    bin_engine = BinnedCalculator(dataset, covmat, method=method)

    assert bin_engine is not None


@pytest.mark.parametrize("method", ["FLAT", "GLS"])
def test_binning_reconstruction(load_test_data, x_recon, theory_model, method):
    """Test reconstruction for FLAT and GLS binning methods."""
    dataset, covmat = load_test_data
    bin_engine = BinnedCalculator(dataset, covmat, method=method)

    means, joint_cov, mask = bin_engine.reconstruct(x_recon, fiducial=theory_model['f'])

    # Shape and type checks
    assert len(means) == len(x_recon)
    assert joint_cov.shape == (4*len(x_recon), 4*len(x_recon))

    # Check for NaNs
    assert not means.isna().values.any()
    assert not joint_cov.isna().values.any()


@pytest.mark.parametrize("method", ["FLAT", "GLS"])
def test_binning_reconstruction_and_scoring(load_test_data, x_recon, theory_model, method):
    """Test binning reconstruction and diagnostic scoring against input data."""
    dataset, covmat = load_test_data
    bin_engine = BinnedCalculator(dataset, covmat, method=method)

    means, joint_cov, mask = bin_engine.reconstruct(x_recon, fiducial=theory_model['f'])

    # Test Scorer module
    scorer = Scorer(means, joint_cov, chatty=False,eigen_trunc_factor=1.e-5)
    datasets = [{'type': 'f', 'df': dataset, 'cov': covmat}]

    res = scorer.score_against_data(datasets)

    assert res is not None
    assert 'p_value' in res

    p_val = res['p_value']

    # Fail test if p_value < 0.01
    if p_val < 0.01:
        pytest.fail(f"Binned ({method}) reconstruction rejected! p-value is critically low: {p_val:.4f} (< 0.01)")

    # Issue warning if 0.01 <= p_value <= 0.05
    if 0.01 <= p_val <= 0.05:
        warnings.warn(
            UserWarning(f"Marginal p-value detected for Binned ({method}): {p_val:.4f} (0.01 <= p_value <= 0.05)")
        )
