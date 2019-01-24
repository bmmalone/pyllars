"""
This module implements helpers for working with collections. In some cases,
the iterable is restricted to a particular type, such as a list or set.
"""

import logging
logger = logging.getLogger(__name__)

from typing import Any, Callable, Container, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import collections
import itertools
import numpy as np
import pandas as pd
import tqdm
import typing

import pyllars.validation_utils as validation_utils


def apply_no_return(items:Iterable, func:Callable, *args,
        progress_bar:bool=False, total_items:Optional[int]=None, **kwargs) -> None:
    """ Apply `func` to each item in `items`
    
    Unlike :py:func:`map`, this function does not return anything.
    
    Parameters
    ----------
    items : typing.Iterable
        An iterable
        
    func : typing.Callable
        The function to apply to each item

    args
        Positional arguments for `func`.
    
    kwargs
        Keyword arguments to pass to `func`

    progress_bar : bool
        Whether to show a progress bar when waiting for results.

    total_items : int or None
        The number of items in `items`. If not given, `len` is used.
        Presumably, this is used when `items` is a generator and `len` does
        not work.

    Returns
    -------
    None : None
        If a return value is expected, use list comprehension instead.
    """    
    if progress_bar:

        if total_items is not None:
            items = tqdm.tqdm(items, total=total_items)
        else:
            items = tqdm.tqdm(items, total=len(items))

    for i in items:
        func(*(i, *args), **kwargs)
    
    return None



def flatten_lists(list_of_lists:Iterable) -> List:
    """ Flatten a list of iterables into a single list
    
    This function does not further flatten inner iterables.

    Parameters
    ----------
    list_of_lists : typing.Iterable
        The iterable to flatten

    Returns
    -------
    flattened_list: typing.List
        The flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]

def is_iterator_exhausted(iterator:Iterable, return_element:bool=False) -> Tuple[bool, Optional[object]]:
    """ Check if the iterator is exhausted

    N.B. THIS CONSUMES THE NEXT ELEMENT OF THE ITERATOR! The `return_element`
    parameter can change this behavior.

    This method is adapted from this SO question: https://stackoverflow.com/questions/661603

    Parameters
    ----------
    iterator : typing.Iterable
        The iterator

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



def list_insert_list(l:Sequence, to_insert:Sequence, index:int) -> List:
    """ Insert `to_insert` into a shallow copy of `l` at position `index`.
    
    This function is adapted from: http://stackoverflow.com/questions/7376019/
    
    Parameters
    ----------
    l : typing.Sequence
        An iterable 
    
    to_insert : typing.Sequence
        The items to insert
        
    index : int
        The location to begin the insertion
        
    Returns
    -------
    updated_l : typing.List
        A list with `to_insert` inserted into `l` at position `index`
    """

    ret = list(l)
    ret[index:index] = list(to_insert)
    return ret

def list_remove_list(l:Iterable, to_remove:Container) -> List:
    """ Remove items in `to_remove` from `l`
    
    Note that  "not in" is used to match items in `to_remove`. Additionally,
    the return *is not* lazy.
    
    Parameters
    ----------
    l : typing.Iterable
        An iterable of items
        
    to_remove : typing.Container
        The set of items to remove from `l`
        
    Returns
    -------
    copy_of_l : typing.List
        A shallow copy of `l` without the items in `to_remove`.
    """
    ret = [i for i in l if i not in to_remove]
    return ret

def list_to_dict(l:Sequence, f:Optional[Callable]=None) -> Dict:
    """ Convert the list to a dictionary in which keys and values are adjacent
    in the list. Optionally, a function `f` can be passed to apply to each value
    before adding it to the dictionary.


    Parameters
    ----------
    l: typing.Sequence
        The list of items

    f: typing.Callable
        A function to apply to each value before inserting it into the list.
        For example, `float` could be passed to convert each value to a float.

    Returns
    -------
    d: typing.Dict
        The dictionary, defined as described above
        
    
    Examples
    --------
    
    .. code-block:: python
    
        l = ["key1", "value1", "key2", "value2"]
        list_to_dict(l, f) == {"key1": f("value1"), "key2": f("value2")}
    """
    if len(l) % 2 != 0:
        msg = ("[collection_utils.list_to_dict]: the list must contain an even number"
            "of elements")
        raise ValueError(msg)

    if f is None:
        f = lambda x: x

    keys = l[::2]
    values = l[1::2]
    d = {k:f(v) for k, v in zip(keys, values)}
    return d


def remove_nones(l:Iterable, return_np_array:bool=False) -> List:
    """ Remove `None`s from `l`
    
    Compared to other single-function tests, this uses "is" and avoids strange
    behavior with data frames, lists of bools, etc.
    
    This function returns a shallow copy and is not lazy.
    
    N.B. This does not test nested lists. So, for example, a list of lists
    of `None` values would be unchanged by this function.
    
    Parameters
    ----------
    l : typing.Iterable
        The iterable
        
    return_np_array : bool
        If true, the filtered list will be wrapped in an np.array.

    Returns
    -------
    l_no_nones : typing.List
        A list or np.array with the `None`s removed from `l`
    """
    ret = [i for i in l if i is not None]

    if return_np_array:
        ret = np.array(ret)

    return ret

def replace_none_with_empty_iter(i:Optional[Iterable]) -> Iterable:
    """ Return an empty iterator if `i` is `None`. Otherwise, return `i`.

    The purpose of this function is to make iterating over results from
    functions which return either an iterator or `None` cleaner. This function
    does not verify that `i` is actually an iterator.

    Parameters
    ----------
    i: None or typing.Iterable
        The possibly-empty iterator

    Returns
    -------
    i: typing.Iterable
        An empty list if iterator is None, or the original iterator otherwise
    """
    if i is None:
        return []
    return i



def wrap_in_list(maybe_sequence:Any) -> Sequence:
    """ If `maybe_sequence` is not a sequence, then wrap it in a list
    
    See :func:`pyllars.validation_utils.is_sequence` for more details about
    what counts as a sequence.

    Parameters
    ----------
    maybe_sequence : typing.Any
        An object which may be a sequence

    Returns
    -------
    list : typing.Sequence
        Either the original object, or `maybe_sequence` wrapped in a list, if
        it was not already a sequence
    """
    ret = maybe_sequence
    
    is_sequence = validation_utils.validate_is_sequence(ret, raise_on_invalid=False)
    
    if not is_sequence:
        ret = [ret]
        
    return ret

def wrap_string_in_list(maybe_string:Any) -> Sequence:
    """ If `maybe_string` is a string, then wrap it in a list.

    The motivation for this function is that some functions return either a
    single string or multiple strings as a list. The return value of this
    function can be iterated over safely.
    
    This function will fail if `maybe_string` is not a string and it not
    a sequence.

    Parameters
    ----------
    maybe_string : typing.Any
        An object which may be a string

    Returns
    -------
    l : typing.Sequence
        Either the original object, or `maybe_string` wrapped in a list, if
        it was a string}
    """
    
    ret = maybe_string
    if isinstance(maybe_string, str):
        ret = [ret]
        
    validation_utils.validate_is_sequence(ret)
    
    return ret


###
# Set helpers
###


def wrap_in_set(maybe_set:Optional[Any], wrap_string:bool=True) -> Set:
    """ If `maybe_set` is not a set, then wrap it in a set.
    
    Parameters
    ----------
    maybe_set : typing.Optional[typing.Any]
        An object which may be a set
        
    wrap_string : bool
        Whether to wrap `maybe_set` as a singleton if it is a string.
        Otherwise, the string will be converted into a set of individual
        characters.
        
    Returns
    -------
    s : typing.Set
        Either the original object, or `maybe_set` wrapped in a set, if
        it was not already a set. If `maybe_set` was `None`, then an
        empty set is returned.
    """
    ret = maybe_set
    
    if ret is None:
        ret = set()
        
    # handle strings explicitly
    if isinstance(ret, str):
        if wrap_string:
            ret = set([ret])

    # check if we already have a set-like object
    if not isinstance(ret, collections.abc.Set):

        # if not and it is an iterable
        if isinstance(ret, collections.abc.Iterable):
            # then we can just directly wrap it
            ret = set(ret)
            
        else:
            # otherwise, we must first wrap the object in a list, and
            # then wrap it in the set
            ret = set([ret])
            
    return ret

def get_set_pairwise_intersections(
        dict_of_sets:Mapping[str,Set],
        return_intersections:bool=True) -> pd.DataFrame:
    """ Find the pairwise intersections among sets in `dict_of_sets`
    
    Parameters
    ----------
    dict_of_sets : typing.Mapping[str,typing.Set]
        A mapping in which the keys are the "names" of the sets and the values
        are the actual sets
        
    return_intersections : bool
        Whether to include the actual set intersections in the return. If `False`,
        then only the intersection size will be included.
        
    Returns
    -------
    df_pairswise_intersections : pandas.DataFrame
        A dataframe with the following columns:
        
        * `set1` : the name of one set in the pair
        * `set2` : the name of the second set in the pair
        * `len(set1)` : the size of set1
        * `len(set2)` : the size of set2
        * `len(intersection)` : the size of the intersection
        * `coverage_small` : the fraction of the smaller of set1 or set2 in the intersection
        * `coverage_large` : the fraction of the larger of set1 or set2 in the intersection
        * `intersection` : the intersection set. Only included if `return_intersections` is `True`.
        
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


def merge_sets(*set_args:Iterable[Container]) -> Set:
    """ Given any number of sets, merge them into a single set

    N.B. This function only performs a "shallow" merge. It does not handle
    nested containers within the "outer" sets.

    Parameters
    ----------
    set_args: typing.Iterable[typing.Container]
        The sets to merge

    Returns
    -------
    merged_set: typing.Set
        A single set containing unique elements from each of the input sets
    """
    ret = {item for s in set_args for item in s}
    return ret

###
# Dictionary helpers
###
def reverse_dict(d:Mapping) -> Dict:
    """ Create a new dictionary in which the keys and values of `d` are switched
    
    In the case of duplicate values, it is arbitrary which will be retained.
    
    Parameters
    ----------
    d : typing.Mapping
        The mapping
        
    Returns
    -------
    reversed_d : typing.Dict
        A dictionary in which the values of `d` now map to the keys
    """
    reverse_d = {v:k for k,v in d.items()}    
    return reverse_d


def sort_dict_keys_by_value(d:Mapping) -> List:
    """ Sort the keys in `d` by their value and return as a list

    This function uses `sorted`, so the values should be able to be sorted
    appropriately by that builtin function.
    
    Parameters
    ----------
    d : typing.Mapping
        The dictionary
        
    Returns
    -------
    sorted_keys : typing.List
        The keys sorted by the associated values
    """
    ret = sorted(d, key=d.get)
    return ret