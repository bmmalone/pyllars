"""
This module contains helpers for typical validation routines.
"""

import logging
logger = logging.getLogger(__name__)

import pyllars.utils as utils

import collections
import importlib
import numpy as np
import operator
import scipy.sparse
import sklearn
import typing

def _raise_value_error(msg, name="array", caller=None, error_type=ValueError):
    """ Raise an `error_type` error with the given message

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
        
    error_type: Error type
        The type of error to raise
    """

    s = ""
    if caller is not None:
        s = "[{}]: ".format(caller)

    msg = msg.format(caller=s, name=name)
    raise error_type(msg)

    
def check_range(val, min_, max_, min_inclusive=True, max_inclusive=True, 
        variable_name=None, raise_on_invalid=True, logger=logger):

    """ Checks whether `val` falls within the specified range
    
    If not, either raise a ValueError or log a warning depending on the value
    of `raise_on_invalid`.

    Parameters
    ----------
    val (number): the value to check

    min_, max_ (numbers): the acceptable range

    min_inclusive, max_inclusive (bools): whether the end
        points are included in the acceptable range

    variable_name (string): for the exception/warning, the
        name to use in the message

    raise_on_invalid (bool): whether to raise an exception (True)
        or issue a warning (False) when the value is invalid

    logger (logging.Logger): the logger to use in case a
        warning is issued

    Returns
    -------
    in_range: bool
        Whether `val` is in the allowed range
    """

    in_range = True

    # first, check min
    min_op = operator.le
    if min_inclusive:
        min_op = operator.lt

    if min_op(val, min_):
        in_range = False
        msg = ("Variable: {}. The given value ({}) was less than the "
            "acceptable minimum ({})".format(variable_name, val, min_))

        if raise_on_invalid:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # now max
    max_op = operator.ge
    if max_inclusive:
        max_op = operator.gt

    if max_op(val, max_):
        in_range = False
        msg = ("Variable: {}. The given value ({}) was greater than the "
            "acceptable maximum ({})".format(variable_name, val, max_))

        if raise_on_invalid:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    return in_range

def check_keys_exist(d, keys, name="array", caller=None):
    """ Ensures the given keys are present in the dictionary

    It does not other validate the type, value, etc., of the keys or their
    values. If a key is not present, a KeyError is raised.

    The motivation behind this function is to verify that a config dictionary
    read in at the beginning of a program contains all of the required values.
    Thus, the program will immediately detect when a required config value is
    not present and quit.

    Parameters
    ----------
    d: dictionary
        The dictionary

    keys: iterable
        A list of keys to check

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
        
    Returns
    -------
    missing_keys: list of strings
        The keys which were not present in the dictionary. However, since a
        KeyError is raised, it must be caught for this to be used.            
    """
    missing_keys = [k for k in keys if k not in d]

    if len(missing_keys) > 0:
        missing_keys = ' '.join(missing_keys)
        msg = "{caller}{name} the following keys were not found: " + missing_keys
        _raise_value_error(msg, name, caller, KeyError)

    return missing_keys

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
            
def validate_all_between(array, min_val, max_val, name="array", caller=None):
    """ Ensure that all values in `array` are (inclusive) in the range
    [min_val, max_val]

    If any values in `array` are outside the range, then raise a ValueError.

    Parameters
    ----------
    array: np.array
        A numpy array

    min_val, max_val: numbers
        The min and max for the (inclusive) range

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """

    valid_vals = [
        check_range(v, min_val, max_val, raise_on_invalid=False)
            for v in array
    ]

    if not np.all(valid_vals):
        msg = ''.join([
            "{caller}all values in {name} must be between ",
            str(min_val),
            " and ",
            str(max_val),
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
        try:
            array_int = np.array(array, dtype=int)
        except:
            # for some reason, we could not cast to integer at all
            # definitely fail.
            _raise_value_error("{caller}{name} is not allowed to include "
                "non-integral values", name, caller)

    if not np.all(array_int == array):
        _raise_value_error("{caller}{name} is not allowed to include "
            "non-integral values", name, caller)
           


def validate_in_set(value, valid_values, name="value", caller=None):
    """ Ensures `value` is one of `valid_values`

    Parameters
    ----------
    value: object
        The value to check

    valid_values: set-like
        The permissible values

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message                    
    """

    if value not in valid_values:
        vv = ",".join(valid_values)
        msg = ("{caller}{name}. the given value is not valid. found: '" + value +
            "'. allowed: " + str(vv))
        _raise_value_error(msg, name, caller, ValueError)
           
def validate_is_sequence_of_integers(sequence, name="sequence", caller=None):
    """ Ensure that all base elements of `sequence` are integral

    This function handles sequences which are np.arrays, scipy.sparse matrices,
    lists and other iterable types (please see `is_sequence`), as
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
    elif validate_is_sequence(sequence, raise_on_invalid=False):
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
    array : np.array
        A numpy array
    
    name : string
        A name for the variable in the error message

    caller : string
        A name for the caller in the error message
    """
    if not np.issubdtype(array.dtype, np.number):
        msg = ("{caller}{name} invalid dtype for numeric sequences. found: " +
            array.dtype)
        _raise_value_error(msg, name, caller)
        
def validate_packages_installed(packages:typing.Iterable[str], caller:str=None) -> None:
    """ Ensure `packages` are installed and can be imported
    
    Parameters
    ----------
    packages : iterable of strings
        The package names

    caller : string
        A name for the caller in the error message
    """
    
    missing = [
        (importlib.util.find_spec(p) is None)
            for p in packages
    ]
    
    if len(missing) > 0:
        s = ','.join(missing)
        msg = ("{caller} could not find the following packages: {}".format(s))
        _raise_value_error(msg, caller=caller, error_type=ImportError)
        
    
        
def validate_type(var, allowed_types, name="var", caller=None):
    """ Ensure `var` has one of the allowed types
    
    If var is not an instance of one of the allowed types, then raise a TypeError.
    
    Parameters
    ----------
    var: object
        A variable
        
    allowed_types: iterable of types
        The allowed types for `var`
    
    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message
    """
    
    is_valid = any(isinstance(var,t) for t in allowed_types)
    if not is_valid:
        at = ",".join(str(t) for t in allowed_types)
        msg = ("{caller}{name} invalid type. found: " + str(type(var)) + 
            ". allowed: " + at)
        _raise_value_error(msg, name, caller, TypeError)
            
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
def check_is_fitted(estimator, attributes, name="estimator", caller=None, all_or_any=all):
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

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message

    all_or_any : callable, {all, any}, default all
        Specify whether all or any of the given attributes must exist.
    """
   
    if not isinstance(attributes, (list, tuple)):
        attributes = [attributes]

    if not all_or_any([hasattr(estimator, attr) for attr in attributes]):
        msg = ("{caller}{name} is not fitted yet. Call the appropriate "
            "initialization, fit, etc., methods before using this method.")
        _raise_value_error(msg, name, caller, sklearn.exceptions.NotFittedError)
        

def validate_is_sequence(
        maybe_sequence:typing.Any,
        raise_on_invalid:bool=True,
        name:str="array",
        caller=None) -> bool:
    """ Check whether `maybe_sequence` is a `collections.Sequence` or `np.ndarray`

    The function specifically checks is maybe_sequence is an instance of a
    string and returns False in that case.

    Parameters
    ----------
    maybe_sequence : object
        An object which may be a sequence
        
    raise_on_invalid : bool
        Whether to raise an error if `maybe_sequence` is not a sequence

    name: string
        A name for the variable in the error message

    caller: string
        A name for the caller in the error message

    Returns
    -------
    is_sequence : bool
        Whether the object is a sequence (as described above)
    """

    if isinstance(maybe_sequence, str):
        return False

    is_sequence = isinstance(maybe_sequence, collections.Sequence)
    is_ndarray = isinstance(maybe_sequence, np.ndarray)
    validated =  is_sequence or is_ndarray
    
    if (not validated) and raise_on_invalid:
        msg = ("{caller}{name} invalid type. found: " + str(type(maybe_sequence)) + 
            ". allowed: sequence types.")
        _raise_value_error(msg, name, caller, TypeError)
        
    return validated
    