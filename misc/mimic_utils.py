###
#   This module contains functions for working with the MIMIC dataset.
###
import os

import dask
import pandas as pd
from dask import dataframe as dd

import misc.pandas_utils as pd_utils
import misc.parallel as parallel

def _fix_mimic_icd(icd):
    """ Add the decimal to the correct location for the ICD code
    
    From the mimic documentation (https://mimic.physionet.org/mimictables/diagnoses_icd/):
    
        > The code field for the ICD-9-CM Principal and Other Diagnosis Codes
        > is six characters in length, with the decimal point implied between
        > the third and fourth digit for all diagnosis codes other than the V
        > codes. The decimal is implied for V codes between the second and third
        > digit.

    """
    
    icd = str(icd)
    
    if len(icd) == 3:
        # then we just have the major chapter
        icd = icd
    else:
        icd = [
            icd[:3],
            ".",
            icd[3:]
        ]
        icd = "".join(icd)
        
    return icd

def fix_mimic_icds(icds, num_cpus=1):
    """ Add the decimal to the correct location for the given ICD codes

    Since adding the decimals is a string-based operation, it can be somewhat
    slow. Thus, it may make sense to perform any filtering before fixing the
    (possibly much smaller number of) ICD codes.

    Parameters
    ----------
    icds: list-like of MIMIC icd codes
        ICDs from the various mimic ICD columns

    Returns
    -------
    fixed_icds: list of strings
        The ICD codes with decimals in the correct location
    """
    fixed_icds = parallel.apply_parallel_iter(
        icds,
        num_cpus,
        _fix_mimic_icd
    )

    return fixed_icds



def get_diagnosis_icds(mimic_base, to_pandas=True, fix_icds=False):
    """ Load the DIAGNOSES_ICDS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    fix_icds: bool
        Whether to add the decimal point in the correct position for the
        ICD codes

    Returns
    -------
    diagnosis_icds: pd.DataFrame or dd.DataFrame
        The diagnosis ICDs table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    diagnosis_icds = os.path.join(mimic_base, "DIAGNOSES_ICD.csv.gz")

    if to_pandas:
        diagnosis_icds = pd_utils.read_df(diagnosis_icds)
    else:
        diagnosis_icds = dd.read_csv(diagnosis_icds)

    if fix_icds:
        msg = "[mimic_utils]: Adding decimals to ICD codes"
        logger.debug(msg)

        fixed_icds = fix_mimic_icds(diagnosis_icds['ICD9_CODE'])
        diagnosis_icds['ICD9_CODE'] = fixed_icd

    return diagnosis_icds
        

def get_procedure_icds(mimic_base, to_pandas=True):
    """ Load the PROCEDURES_ICD table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    Returns
    -------
    procedure_icds: pd.DataFrame or dd.DataFrame
        The procedure ICDs table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    procedure_icds = os.path.join(mimic_base, "PROCEDURES_ICD.csv.gz")

    if to_pandas:
        procedure_icds = pd_utils.read_df(procedure_icds)
    else:
        procedure_icds = dd.read_csv(procedure_icds)

    return procedure_icds


