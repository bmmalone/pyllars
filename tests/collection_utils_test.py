""" Tests for the `pyllars.collection_utils` module.
"""

import pyllars.collection_utils as collection_utils

def test_flatten_lists():
    """ Test for flattening nested lists.
    """
    list_of_lists = [
        [0,1,2],
        [3,4,5],
        [6,7]
    ]
    
    expected_output = [0,1,2,3,4,5,6,7]    
    actual_output = collection_utils.flatten_lists(list_of_lists)
    
    assert expected_output == actual_output
    
def test_is_iterator_exhausted():
    """ Test for checking if iterator has additional items.
    """
    
    it = iter([1])
    expected_output = (False, 1)
    actual_output = collection_utils.is_iterator_exhausted(it, return_element=True)
    assert expected_output == actual_output
    
    # we should see that the iterator is now empty
    expected_output = True
    actual_output = collection_utils.is_iterator_exhausted(it)
    assert expected_output == actual_output
   