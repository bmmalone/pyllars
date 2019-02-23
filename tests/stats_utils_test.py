""" Tests for the `pyllars.stats_utils` module.
"""
import pytest
import pyllars.stats_utils as stats_utils

import numpy as np

def test_symmetric_entropy():
    """ Test the calculation of symmetric KL divergence
    for categorical distributions.
    
    The example calculation is taked from wikipedia: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    """
    p = (0.36, 0.48, 0.16)
    q = (0.333, 0.333, 0.333)
    
    expected_output = (0.0852996+0.097455)/2
    actual_output = stats_utils.symmetric_entropy(p,q)
    
    assert pytest.approx(expected_output, actual_output)
    
def test_calculate_univariate_gaussian_kl():
    """ Test the calculation of univariate Gaussian KL divergence
    """
    p = (0,2)
    q = (1,1)
    
    # expected derived by hand
    expected_output = 1 - np.log(np.sqrt(2))
    actual_output = stats_utils.calculate_univariate_gaussian_kl(p,q)
    
    assert pytest.approx(expected_output, actual_output)
    
def test_symmetric_gaussian_kl():
    """ Test the calculation of symmetric KL divergence for
    univariate Gaussian distributions.
    
    Implicitly, this also tests calculating the non-symmetric KL
    for univariate Gaussians.
    """
    p = (0,2)
    q = (1,1)
    
    kl_pq = np.log(np.sqrt(2))
    kl_qp = 1 - np.log(np.sqrt(2))
    
    expected_output = (kl_pq + kl_qp) / 2
    actual_output = stats_utils.symmetric_gaussian_kl(p,q)
    
    assert pytest.approx(expected_output, actual_output)