""" Tests for the `pyllars.pandas_utils` module.
"""
import pytest
import pyllars.pandas_utils as pd_utils

import pandas as pd

@pytest.fixture
def patient_diagnosis_dict():
    """ Create a dictionary mapping from patient identifiers to a diagnosis string
    """
    
    patient_diagnosis_dict = {
        'A': 'BENZODIAZEPINE OVERDOSE',
        'B': 'CORONARY ARTERY DISEASE\\CORONARY ARTERY BYPASS GRAFT/SDA',
        'C': 'BRAIN MASS',
        'D': 'INTERIOR MYOCARDIAL INFARCTION',
        'E': 'ACUTE CORONARY SYNDROME',
        'F': 'V-TACH',
        'G': 'CORONARY ARTERY DISEASE\\CORONARY ARTERY BYPASS GRAFT/SDA',
        'H': 'UNSTABLE ANGINA\\CATH',
        'I': 'TRACHEAL STENOSIS/SDA'
    }
    
    return patient_diagnosis_dict
    

@pytest.fixture
def simple_patient_df(patient_diagnosis_dict):
    """ Create a simple data frame for testing.
    """
    
    gender = {
        'A': 'Male',
        'B': 'Female',
        'C': 'Male',
        'D': 'Male',
        'E': 'Female',
        'F': 'Female',
        'G': 'Female',
        'H': 'Female',
        'I': 'Male'
    }

    race = {
        'A': 'White',
        'B': 'White',
        'C': 'Asian',
        'D': 'White',
        'E': 'Asian',
        'F': 'White',
        'G': 'Black',
        'H': 'Black',
        'I': 'Black'
    }

    blood_pressure = {
        'A': 81,
        'B': 90,
        'C': 83,
        'D': 88,
        'E': 75,
        'F': 95,
        'G': 85,
        'H': 97,
        'I': 83
    }

    heart_rate = {
        'A': 93,
        'B': 75,
        'C': 67,
        'D': 80,
        'E': 63,
        'F': 99,
        'G': 76,
        'H': 65,
        'I': 75
    }
    
    weight = {
        'A': 40.5,
        'B': 75.3,
        'C': 67.6,
        'D': 80.1,
        'E': 63.3,
        'F': 30.6,
        'G': 76.5,
        'H': 65.2,
        'I': 75.1
    }

    # cm: cardiomyopathy
    has_cm = {
        'A': True,
        'B': False,
        'C': False,
        'D': False,
        'E': False,
        'F': True,
        'G': True,
        'H': False,
        'I': False
    }
    

    # and create the data frame
    patient_df = pd.DataFrame.from_dict({
        'gender': gender,
        'race': race,
        'blood_pressure': blood_pressure,
        'heart_rate': heart_rate,
        'weight': weight,
        'has_cm': has_cm,
        'diagnosis' : patient_diagnosis_dict
    })
    patient_df = patient_df.reset_index().rename(columns={'index':'identity'})
    
    return patient_df


def test_dict_to_dataframe(patient_diagnosis_dict):
    """ Test converting a dictionary to a data frame
    """
    
    expected_keys = [
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I'
    ]
    
    expected_values = [
        'BENZODIAZEPINE OVERDOSE',
        'CORONARY ARTERY DISEASE\\CORONARY ARTERY BYPASS GRAFT/SDA',
        'BRAIN MASS',
        'INTERIOR MYOCARDIAL INFARCTION',
        'ACUTE CORONARY SYNDROME',
        'V-TACH',
        'CORONARY ARTERY DISEASE\\CORONARY ARTERY BYPASS GRAFT/SDA',
        'UNSTABLE ANGINA\\CATH',
        'TRACHEAL STENOSIS/SDA'
    ]
    
    expected_out = pd.DataFrame()
    expected_out['k'] = expected_keys
    expected_out['v'] = expected_values
    
    actual_out = pd_utils.dict_to_dataframe(patient_diagnosis_dict, key_name='k', value_name='v')
    
    pd.testing.assert_frame_equal(expected_out, actual_out)

def test_dataframe_to_dict(patient_diagnosis_dict, simple_patient_df):
    """ Test converting columns of a data frame to a dictionary
    """
    
    expected_out = patient_diagnosis_dict
    
    actual_out = pd_utils.dataframe_to_dict(
        simple_patient_df,
        key_field='identity',
        value_field='diagnosis'
    )

    assert expected_out == actual_out