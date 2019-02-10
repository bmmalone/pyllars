Collection utilities
**************************

.. automodule:: pyllars.collection_utils
    :noindex:
   
.. currentmodule::  pyllars.collection_utils
    
Many of the function names mention specific data structures, such as "list"s or "dict"s,
in the names for historical reasons. In most cases, these functions work with any instance
of the more general type (such as `Iterable` or `Mapping`). Please see the specific
documentation for more details, though.

Iterable helpers
----------------

.. autosummary::
    apply_no_return
    flatten_lists
    is_iterator_exhausted
    list_insert_list
    list_remove_list
    list_to_dict
    remove_nones
    replace_none_with_empty_iter
    wrap_in_list
    wrap_string_in_list


Set helpers
-------------

.. autosummary::
    wrap_in_set
    get_set_pairwise_intersections
    merge_sets

Mapping helpers
------------------

.. autosummary::
    reverse_dict
    sort_dict_keys_by_value

Definitions
----------------

.. automodule:: pyllars.collection_utils
    :members:
