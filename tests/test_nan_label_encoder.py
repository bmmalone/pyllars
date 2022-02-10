""" Tests for the `pyllars.sklearn_transformers.nan_label_encoder` module.
"""
import pytest
from pyllars.sklearn_transformers.nan_label_encoder import NaNLabelEncoder

import numpy as np

def test_string_values_with_nans():

    string_values_with_nans = np.array([
        1.0, "np.nan", 0.0
    ], dtype=object)
    
    expected = [
        0,np.nan,1,
    ]

    le = NaNLabelEncoder()
    actual = le.fit_transform(string_values_with_nans)

    assert (expected == actual)


def main():
    test_string_values_with_nans()

if __name__ == '__main__':
    main()