"""
This module contains helpers for typical validation routines.
"""

import logging
logger = logging.getLogger(__name__)

import numpy as np
import scipy.sparse

import misc.utils as utils

def _raise_value_error(msg, name="array", caller=None):
    """ Raise a ValueError with the given message

    This internal helper formats the message nicely using `name` and `caller`
    appropriately.

    Parameters
    ----------
    msg: string
        The basic error message. It should include template locations for
        `name` and `caller`.

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """

    s = ""
    if caller is not None:
        s = "[{}]: ".format(caller)

    msg = msg.format(caller=s, name=name)
    raise ValueError(msg)




def validate_1d(array, name="array", caller=None):
    """ Ensure that `array` is 1-dimensional

    If `array` is not 1d, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if len(array.shape) != 1:
        _raise_value_error("{caller}{name} must be a 1-d data structure",
            name, caller)

def validate_2d(array, name="array", caller=None):
    """ Ensure that `array` is 2-dimensional

    If `array` is not 2d, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if len(array.shape) != 2:
        _raise_value_error("{caller}{name} must be a 2-d data structure",
            name, caller)
            
def validate_all_between(array, min_, max_, name="array", caller=None):
    """ Ensure that all values in `array` are (inclusive) in the range
    [min_, max_]

    If any values in `array` are outside the range, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array

    min_, max_: numbers
        The min and max for the (inclusive) range

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """

    valid_vals = [
        math_utils.check_range(v, min_, max_, raise_on_invalid=False)
            for v in array
    ]

    if not np.all(valid_vals):
        msg = ''.join([
            "{caller}all values in {name} must be between ",
            str(min_),
            " and ",
            str(max_),
            ", inclusive"
        ])        
        _raise_value_error(msg, name, caller)

def validate_all_finite(array, name="array", caller=None):
    """ Ensure that all values in `array` are finite

    If any values in `array` are np.nan, np.inf, etc., then raise a ValueError.

    N.B. See `np.isfinite` for details

    Parameters
    ----------
    array: np.array
        A numpy array

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if not np.isfinite(array).all():
        _raise_value_error("{caller}{name} is not allowed to include "
            "non-finite values", name, caller)

              
def validate_all_non_negative(array, name="array", caller=None):
    """ Ensure that all values in `array` are non-negative.

    If any values in `array` are negative, then raise a ValueError.
    
    So, for example, "0"s cause this validation to fail.

    Parameters
    ----------
    array: np.array or scipy.sparse.matrix
        A numpy array or sparse matrix

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if scipy.sparse.issparse(array):
        min_val = array.min()
        if min_val < 0:
            _raise_value_error("{caller}{name} is not allowed to include "
                "negative values", name, caller)
    else:
        if np.any(array < 0):
            _raise_value_error("{caller}{name} is not allowed to include "
                "negative values", name, caller)
            
def validate_all_positive(array, name="array", caller=None):
    """ Ensure that all values in `array` are positive

    If any values in `array` are non-positive, then raise a ValueError.
    
    So, for example, "0"s cause this validation to fail.

    Parameters
    ----------
    array: np.array
        A numpy array

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    
    if np.any(array <= 0):
        _raise_value_error("{caller}{name} is not allowed to include "
            "non-positive values", name, caller)

def validate_equal_length(array1, array2, name1="array1", name2="array2",
        caller=None):
    """ Ensure that the arrays have the same length

    If the lengths are not equal, then raise a ValueError.
    
    Lengths are taken to be the first dimension of `array.shape`.

    Parameters
    ----------
    array1: np.array
        A numpy array

    array2: np.array
        A numpy array
    
    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if array1.shape[0] != array2.shape[0]:
        # a bit of a hack to make the error message look correct
        name = "{},{}".format(name1, name2)
        _raise_value_error("{caller}{name} must have the same length", name,
            caller)
            
def validate_equal_shape(array1, array2, name1="array1", name2="array2",
        caller=None):
    """ Ensure that the arrays have the same shape

    If the shapes are not equal, then raise a ValueError.

    Parameters
    ----------
    array1: np.array
        A numpy array

    array2: np.array
        A numpy array
    
    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if array1.shape != array2.shape:
        # a bit of a hack to make the error message look correct
        name = "{},{}".format(name1, name2)
        _raise_value_error("{caller}{name} must have the same shape", name,
            caller)


def validate_integral(array, name="array", caller=None, array_int=None):
    """ Ensure that all values in `array` are integral

    If any values in `array` are not integral, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message

    array_int: np.array or None
        Optionally, if the array has already been converted to an integer
        array, that can be passed to avoid re-creating the int array
    """
    if array_int is None:
        array_int = np.array(array, dtype=int)

    if not np.all(array_int == array):
        _raise_value_error("{caller}{name} is not allowed to include "
            "non-integral values", name, caller)
           
def validate_is_sequence_of_integers(sequence, name="sequence", caller=None):
    """ Ensure that all base elements of `sequence` are integral

    This function handles sequences which are np.arrays, scipy.sparse matrices,
    lists and other iterable types (please see `misc.utils.is_sequence`), as
    well as base types.

    N.B. The difference between this function and `validate_integral` is that
    this function ensures that the *types* of the elements of the np.array are
    integral, while `validate_integral` ensures that the *values* can be cast
    as integers.

    If any values in `array` are not integral, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message


    """
    msg = ("validate_is_integral. sequence: {}".format(sequence))
    logger.info(msg)

    if isinstance(sequence, np.ndarray) or scipy.sparse.issparse(sequence):
        if sequence.dtype == object:
            # then assume this is a ragged array
            for c in sequence:
                validate_is_sequence_of_integers(c)
        elif not np.issubdtype(sequence.dtype, np.integer):
            msg = ("invalid dtype for integral sequences. found: {}".format(
                sequence.dtype))
            _raise_value_error(msg, name, caller)
        else:
            # then this validates
            return

    elif np.issubdtype(type(sequence), np.integer):
        # then this validates
        return
    elif utils.is_sequence(sequence):
        for c in sequence:
            validate_is_sequence_of_integers(c)
    else:
        msg = ("unrecognized data type for (sequence of) integers: value: {}. "
            "type: {}".format(sequence, type(sequence)))
        _raise_value_error(msg, name, caller)
    
def validate_numeric(array, name="array", caller=None):
    """ Ensure that the array has some numeric dtype

    If the shapes are not equal, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array
    
    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if not np.issubdtype(array.dtype, np.number):
        msg = ("invalid dtype for numeric sequences. found: {}".format(
            array.dtype))
        _raise_value_error(msg, name, caller)
            
def validate_unique(array, name="array", caller=None):
    """ Ensure that all values in `array` are unique

    If any values in `array` appear more than once, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    if len(array) != len(set(array)):
        _raise_value_error("{caller}{name} is not allowed to include "
            "duplicate values", name, caller)

###
# sklearn-style helpers
###
def check_is_fitted(estimator, attributes, msg=None, all_or_any=all):
    """Perform is_fitted validation for estimator.

    N.B. This is essentially the same as the function in
    `sklearn.utils.validation`; however, it *does not* check that `estimator`
    has a `fit` method.

    Checks if the estimator is fitted by verifying the presence of
    "all_or_any" of the passed attributes and raises a NotFittedError with the
    given message.
    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.
    attributes : attribute name(s) given as string or a list/tuple of strings
        Eg.:
            ``["coef_", "estimator_", ...], "coef_"``

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this method."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.

    Returns
    -------
    None

    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    import sklearn.exceptions
    
    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call the appropriate "
               "initialization, fit, etc., methods before using this method.")

    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        raise sklearn.exceptions.NotFittedError(msg % {'name': type(estimator).__name__})


