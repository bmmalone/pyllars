""" Utilities for working with iterable objects

In some cases, the iterable is restricted to a particular type, such as a list.
"""

import logging
logger = logging.getLogger(__name__)

import collections
import itertools
import numpy as np
import pandas as pd
import tqdm
import typing

import misc.validation_utils as validation_utils

def is_iterator_exhausted(iterator:typing.Iterable, return_element:bool=False):
    """ Check if the iterator is exhausted

    This method is primarily intended to check if an iterator is empty.

    N.B. THIS CONSUMES THE NEXT ELEMENT OF THE ITERATOR! The `return_element`
    parameter can change this behavior.

    This method is adapted from this SO question:
        https://stackoverflow.com/questions/661603

    Parameters
    ----------
    iterator : an iterator

    return_element : bool
        Whether to return the next element of the iterator

    Returns
    -------
    is_exhausted : bool
        Whether there was a next element in the iterator

    [optional] next_element : object
        It `return_element` is `True`, then the consumed element is also
        returned.
    """

    # create a flag as the default value for "next"
    flag = object()

    # grab the next thing in the iterator, or our flag if there is nothing
    n = next(iterator, flag)

    # check if we saw the flag or some real value
    is_exhausted = (n == flag)

    # build up the return
    ret = is_exhausted

    if return_element:
        ret = (is_exhausted, n)

    return ret


def apply_no_return(items:typing.Iterable, func:typing.Callable, *args,
        progress_bar:bool=False, total_items=None, **kwargs) -> None:
    """ Apply func to each item in the list
    
    Unlike `map`, this function does not return anything.
    
    Parameters
    ----------
    items : iterable
        An iterable
        
    func : function pointer
        The function to apply to each item

    args, kwargs
        The other arguments to pass to `func`

    progress_bar : bool
        Whether to show a progress bar when waiting for results.

    total_items : int
        The number of items in `items`. If not given, `len` is used.
        Presumably, this is used when `items` is a generator and `len` does
        not work.

    Returns
    -------
    None
        If a return value is expected, use list comprehension instead
    """    
    if progress_bar:

        if total_items is None:
            items = tqdm.tqdm(items, total=total_items)
        else:
            items = tqdm.tqdm(items, total=len(items))

    for i in items:
        func(*(i, *args), **kwargs)
    
    return None

###
# Iterable helpers
###
def list_to_dict(l:typing.List, f=None) -> typing.Dict:
    """ Convert the list to a dictionary in which keys and values are adjacent
    in the list. Optionally, a function can be passed to apply to each value
    before adding it to the dictionary.

    Example: 

        list = ["key1", "value1", "key2", "value2"]
        dict = {"key1": "value1", "key2": "value2"}

    Parameters
    ----------
    l: sequence
        The list of items

    f: function
        A function to apply to each value before inserting it into the list.
        For example, "float" could be passed to convert each value to a float.

    Returns
    -------
    dict: dictionary
        The dictionary, defined as described above
    """
    if len(l) % 2 != 0:
        msg = ("[iter_utils.list_to_dict]: the list must contain an even number"
            "of elements")
        raise ValueError(msg)

    if f is None:
        f = lambda x: x

    keys = l[::2]
    values = l[1::2]
    d = {k:f(v) for k, v in zip(keys, values)}
    return d


def wrap_string_in_list(maybe_string:typing.Any) -> typing.Sequence:
    """ If `maybe_string` is a string, then wrap it in a list.

    The motivation for this function is that some functions return either a
    single string or multiple strings as a list. The return value of this
    function can be iterated over safely.
    
    This function will fail if `maybe_string` is not a string and it not
    a sequence.

    Parameters
    ----------
    maybe_string : object
        An object which may be a string

    Returns
    -------
    list
        Either the original object, or maybe_string wrapped in a list, if
            it was a string}
    """
    
    ret = maybe_string
    if isinstance(maybe_string, str):
        ret = [ret]
        
    validation_utils.validate_is_sequence(ret)
    
    return ret


def wrap_in_list(maybe_sequence:typing.Any) -> typing.Sequence:
    """ If `maybe_sequence` is not a sequence, then wrap it in a list
    
    See `is_sequence` for more details about what counts as a sequence.

    Parameters
    ----------
    maybe_sequence : object
        An object which may be a sequence

    Returns
    -------
    list : iterable
        Either the original object, or maybe_sequence wrapped in a list, if
            it was not already a sequence
    """
    ret = maybe_sequence
    
    is_sequence = validation_utils.validate_is_sequence(ret, raise_on_fail=False)
    
    if not is_sequence:
        ret = [ret]
        
    return ret


def flatten_lists(list_of_lists:typing.Iterable) -> typing.List:
    """ Flatten a list of iterables into a single list
    
    This function does not further flatten inner iterables.

    Parameters
    ----------
    list_of_lists : iterable
        The iterable to flatten

    Returns
    -------
    flattened_list: list
        The flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]


def list_remove_list(l:typing.Iterable, to_remove:typing.Container) -> typing.List:
    """ Remove items in `to_remove` from `l`
    
    Note that  "not in" is used to match items in `to_remove`. Additionally,
    the return *is not* lazy.
    
    Parameters
    ----------
    l : iterable
        An iterable of items
        
    to_remove : container
        The set of items to remove from `l`
        
    Returns
    -------
    copy_of_l : list
        A shallow copy of `l` without the items in `to_remove`.
    """
    ret = [i for i in l if i not in to_remove]
    return ret


def list_insert_list(l:typing.Sequence, to_insert:typing.Sequence, index:int) -> typing.List:
    """ Insert `to_insert` into `l` at position `index`.
    
    This function returns a shallow copy.
    
    This function is adapted from: http://stackoverflow.com/questions/7376019/
    
    Parameters
    ----------
    l : iterable
        An iterable 
    
    to_insert : iterable
        The items to insert
        
    index : int
        The location to begin the insertion
        
    Returns
    -------
    updated_l : list
        A list with `to_insert` inserted into `l` at position `index`
    """

    ret = list(l)
    ret[index:index] = list(to_insert)
    return ret

def remove_nones(l:typing.Iterable, return_np_array:bool=False) -> typing.List:
    """ Remove `None`s from `l`
    
    Compared to other single-function tests, this uses "is" and avoids strange
    behavior with data frames, lists of bools, etc.
    
    This function returns a shallow copy and is not lazy.

    N.B. This does not test nested lists. So, for example, a list of lists
    of `None`s would be unchanged by this function.
    
    Parameters
    ----------
    l : iterable
        The iterable
        
    return_np_array : bool
        If true, the filtered list will be wrapped in an np.array.

    Returns
    -------
    l_no_nones : list
        A list or np.array with the `None`s removed from `l`

    """
    ret = [i for i in l if i is not None]

    if return_np_array:
        ret = np.array(ret)

    return ret

def replace_none_with_empty_iter(i:typing.Any) -> typing.Iterator:
    """ Return an empty iterator if `i` is `None`. Otherwise, return `i`.

    The purpose of this function is to make iterating over results from
    functions which return either an iterator or None cleaner.

    Parameters
    ----------
    i: None or some object

    Returns
    -------
    empty_iterator: list of size 0
        If iterator is None
    --- OR ---
    iterator:
        The original iterator, if it was not None
    """
    if i is None:
        return []
    return i

###
# Set helpers
###

def merge_sets(*set_args) -> typing.Set:
    """ Given any number of sets, merge them into a single set

    N.B. This function only performs a "shallow" merge. It does not handle
    nested containers within the "outer" sets.

    Parameters
    ----------
    set_args: iterable of sets
        The sets to merge

    Returns
    -------
    merged_set: set
        A single set containing unique elements from each of the input sets
    """
    ret = {item for s in set_args for item in s}
    return ret


def get_set_pairwise_intersections(
        dict_of_sets:typing.Dict,
        return_intersections:bool=True) -> pd.DataFrame:
    """ Find the pairwise intersections among sets in `dict_of_sets`
    
    Parameters
    ----------
    dict_of_sets : mapping from set names to the sets
        A dictionary in which the keys are the "names" of the sets and the values
        are the actual sets
        
    return_intersections : bool
        Whether to include the actual set intersections in the return. If `False`,
        then only the intersection size will be included.
        
    Returns
    -------
    df_pairswise_intersections : pd.DataFrame
        A dataframe with the following columns:
        * set1 : the name of one set in the pair
        * set2 : the name of the second set in the pair
        * len(set1) : the size of set1
        * len(set2) : the size of set2
        * len(intersection) : the size of the intersection
        * coverage_small : the fraction of the smaller of set1 or set2 in the intersection
        * coverage_large : the fraction of the larger of set1 or set2 in the intersection
        * intersection : the intersection set. Only included if `return_intersections`
            is True.
    """
    all_intersection_sizes = []

    it = itertools.combinations(dict_of_sets, 2)

    for i in it:
        s1 = i[0]
        s2 = i[1]
        set1 = dict_of_sets[s1]
        set2 = dict_of_sets[s2]

        intersection = set1 & set2
        
        # determine the coverage of both sets
        coverage_set1 = len(intersection) / len(set1)
        coverage_set2 = len(intersection) / len(set2)
        
        # and set the appropriate "coverage" variables
        if len(set1) > len(set2):
            coverage_small = coverage_set2
            coverage_large = coverage_set1
        else:
            coverage_small = coverage_set1
            coverage_large = coverage_set2
            

        intersection_size = {
            'set1': s1,
            'set2': s2,
            'len(set1)': len(set1),
            'len(set2)': len(set2),
            'len(intersection)': len(intersection),
            'coverage_small': coverage_small,
            'coverage_large': coverage_large
        }

        if return_intersections:
            intersection_size['intersection'] = intersection

        all_intersection_sizes.append(intersection_size)

    df_intersection_sizes = pd.DataFrame(all_intersection_sizes)
    return df_intersection_sizes

###
# Dictionary helpers
###
def reverse_dict(d:typing.Dict) -> typing.Dict:
    """ Create a new dictionary in which the keys and values of d are switched
    
    In the case of duplicate values, it is arbitrary which will be retained.
    
    Parameters
    ----------
    d : dictionary
        The dictionary
        
    Returns
    -------
    reversed_d : dictionary
        A dictionary in which the values of `d` now map to the keys
    """
    reverse_d = {v:k for k,v in d.items()}    
    return reverse_d


def sort_dict_keys_by_value(d:typing.Dict) -> typing.List:
    """ Sort the keys in the dictionary by their value and return as a list

    This function uses `sorted`, so the values should be able to be sorted
    appropriately by that builtin function.
    
    Parameters
    ----------
    d : dictionary
        The dictionary
        
    Returns
    -------
    sorted_keys : list
        The keys sorted by the associated values
    """
    ret = sorted(d, key=d.get)
    return ret