"""
This module contains utilities which simplify working with the numpy stack:
    * numpy
    * pandas (but see the note below about pandas_utils)
    * scipy
    * sklearn

It also contains other math helpers.

This module differs from pandas_utils because this module treats pandas
data frames as data matrices (in a statistical/machine learning sense),
while that module considers data frames more like database tables which
hold various types of records.
"""
import logging
logger = logging.getLogger(__name__)

import collections
import itertools
from enum import Enum

import more_itertools
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn.model_selection

import pyllars.validation_utils as validation_utils



def random_pick(probs):
    """ Select an item according to the specified categorical distribution
    """
    '''
    >>> probs = [.3, .7]
    >>> random_pick(probs)
    '''
    cutoffs = np.cumsum(probs)
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))
    return idx


def has_nans(X):
    """ Check if `X` has any np.nan values

    Parameters
    ----------
    X: np.array
        The array

    Returns
    -------
    has_nans: bool
        True if any np.nan's are in `X`, False otherwise
    """
    # please see: https://stackoverflow.com/questions/6736590 for details
    return np.isnan(np.sum(X))



def remove_negatives(x): 
    """ Remove all negative and NaN values from x

    Parameters
    ----------
    x: np.array
        An array

    Returns
    -------
    non_negative_x: np.array
        A copy of x which does not contain any negative (or NaN) values. The
        shape of non_negative_x depends on the number of negative/NaN values
        in x.
    """
    x = x[x >= 0]
    return x



def is_monotonic(x, increasing=True):
    """ This function checks whether a given list is monotonically increasing
        (or decreasing). By definition, "monotonically increasing" means the
        same as "strictly non-decreasing".

        Adapted from: http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity

        Args:
            x (sequence) : a 1-d list of numbers

            increasing (bool) : whether to check for increasing monotonicity
    """
    import numpy as np

    dx = np.diff(x)

    if increasing:
        return np.all(dx >= 0)
    else:
        return np.all(dx <= 0)





def l1_distance(p, q):
    """ Calculate the l1 distance between the two vectors.

    Parameters
    ----------
    p, q: np.arrays of the same shape
        The vectors (or matrices) for which the distance will be calculated

    Returns
    -------
    l1_distance: float
        The l1 distance (sum of absolute differences) between p and q

    Raises
    ------
    ValueError: if p and q do not have the same shape
    """
    if p.shape != q.shape:
        msg = "[math_utils.l1_distance]: p and q must have the same shape"
        raise ValueError(msg)

    diff = np.abs(p - q)
    return np.sum(diff)


