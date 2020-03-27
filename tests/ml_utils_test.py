""" Tests for the `pyllars.ml_utils module.
"""
import pytest
import pyllars.ml_utils as ml_utils

import numpy as np
import pandas as pd

@pytest.fixture
def colors():
    """ Create an array with different color names
    """
    colors = np.array([
        'Green',
        'Green',
        'Yellow',
        'Green',
        'Yellow',
        'Green',
        'Red',
        'Red',
        'Red'
    ])
    
    return colors

def test_get_cv_folds_array(colors):
    """ Test splitting into folds with a numpy array
    """
    
    expected_output = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0])
    actual_output = ml_utils.get_cv_folds(colors, num_splits=2)    
    np.testing.assert_array_equal(expected_output, actual_output)

def test_get_cv_folds_series(colors):
    """ Test splitting into folds with a pandas series
    """
    
    # this change from the test above is to make the tests work
    expected_output = pd.Series([1, 1, 0, 0, 1, 0, 0, 1, 0])
    actual_output = ml_utils.get_cv_folds(colors, num_splits=2)
    np.testing.assert_array_equal(expected_output, actual_output)
