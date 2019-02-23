""" Tests for the `pyllars.nlp_utils module.
"""
import pytest
import pyllars.nlp_utils as nlp_utils

def test_clean_doc():
    """ Test for stemming, etc., a document.
    """
    input_str = ("He was running pretty quickly to a dock "
        "which was fairly far (3.14159 miles) from where he was "
        "currently located.")
    
    expected_output = "run pretti quick dock fair far mile current locat"    
    actual_output = nlp_utils.clean_doc(input_str)
    
    assert expected_output == actual_output
                 
                