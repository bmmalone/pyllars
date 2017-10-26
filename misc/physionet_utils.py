###
#   This module contains functions for working with datasets from physionet,
#   including MIMIC and the Computing in Cardiology Challenge datasets.
###
import datetime
import os

import dask
import numpy as np
import pandas as pd
from dask import dataframe as dd

import misc.pandas_utils as pd_utils
import misc.parallel as parallel

###
# MIMIC
###

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


def get_admissions(mimic_base, to_pandas=True):
    """ Load the ADMISSIONS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    Returns
    -------
    admissions: pd.DataFrame or dd.DataFrame
        The admissions table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    admissions = os.path.join(mimic_base, "ADMISSIONS.csv.gz")

    if to_pandas:
        admissions = pd_utils.read_df(admissions)
    else:
        admissions = dd.read_csv(admissions)

    return admissions


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


def get_patients(mimic_base, to_pandas=True):
    """ Load the PATIENTS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    Returns
    -------
    patients: pd.DataFrame or dd.DataFrame
        The patients table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    patients = os.path.join(mimic_base, "PATIENTS.csv.gz")

    if to_pandas:
        patients = pd_utils.read_df(patients)
    else:
        patients = dd.read_csv(patients)

    return patients

        

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

###
# CinC 2012: https://physionet.org/challenge/2012/
#
#   "cinc_2012"
###

CinC_2012_DESCRIPTOR_FIELDS = {
    'RecordID',
    'Age',
    'Gender',
    'Height',
    'ICUType',
    'Weight'
}

CinC_2012_GENDER_MAP = {
    0: 'female',
    1: 'male',
    2: np.nan,
    None: np.nan,
    np.nan: np.nan
}

CinC_2012_ICU_TYPE_MAP = {
    1: "coronary_care_unit", 
    2: "cardiac_surgery_recovery_unit",
    3: "medical_icu",
    4: "surgical_icu",
    None: np.nan,
    np.nan: np.nan
}


def get_cinc_2012_outcomes(cinc_2012_base, to_pandas=True):
    """ Load the Outcomes-a.txt file.

    N.B. This file is assumed to be named "Outcomes-a.txt" and located
    directly in the cinc_2012_base directory

    Parameters
    ----------
    cinc_2012_base: path-like
        The path to the main folder for this CinC challenge

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    Returns
    -------
    cinc_2012_base: pd.DataFrame or dd.DataFrame
        The "Outcomes-a" table as either a pandas or dask data frame,
        depending on the value of to_pandas. It comtains the following columns:

            * HADM_ID: string, a key into the record table
            * SAPS-I: integer, the SAPS-I score
            * SOFA: integer, the SOFA score
            * ADMISSION_ELAPSED_TIME: pd.timedelta, time in the hospital, in days
            * SURVIVAL: time between ICU admission and observed death.
                If the patient survived (or the death was not recorded), then
                the value is np.nan.
            * EXPIRED: bool, whether the patient died in the hospital
    """

    outcomes_df = os.path.join(cinc_2012_base, "Outcomes-a.txt")

    if to_pandas:
        outcomes_df = pd_utils.read_df(outcomes_df)
    else:
        outcomes_df = dd.read_csv(outcomes_df)

    new_columns = {
        'RecordID': 'HADM_ID',
        'Length_of_stay': 'ADMISSION_ELAPSED_TIME',
        'Survival': 'SURVIVAL',
        'In-hospital_death': 'EXPIRED'
    }

    outcomes_df = outcomes_df.rename(columns=new_columns)

    # convert expired to a boolean
    outcomes_df['EXPIRED'] = (outcomes_df['EXPIRED'] == 1)
    
    # and survival to an elapsed time
    outcomes_df['SURVIVAL'] = outcomes_df['SURVIVAL'].replace(-1, np.nan)
    s = pd.to_timedelta(outcomes_df['SURVIVAL'], unit='d')
    outcomes_df['SURVIVAL'] = s

    # and the time of the episode to a timedelta
    e = pd.to_timedelta(outcomes_df['ADMISSION_ELAPSED_TIME'], unit='d')
    outcomes_df['ADMISSION_ELAPSED_TIME'] = e

    return outcomes_df

def _get_cinc_2012_record_descriptor(record_file_df):
    """ Given the record file data frame, use the first six rows to extract
    the descriptor information. See the documentation (https://physionet.org/challenge/2012/,
    "General descriptors") for more details.
    """

    # first, only look for the specified fields
    m_icu_fields = record_file_df['Parameter'].isin(CinC_2012_DESCRIPTOR_FIELDS)

    # and at time "00:00"
    m_time = record_file_df['Time'] == "00:00"

    m_descriptors = m_icu_fields & m_time


    record_descriptor = record_file_df[m_descriptors]
    record_descriptor = pd_utils.dataframe_to_dict(record_descriptor, "Parameter", "Value")

    # handle Gender as a special case
    if np.isnan(record_descriptor.get('Gender')):
        record_descriptor['Gender'] = 2

    # now, fix the data types
    record_descriptor = {
        "HADM_ID": int(record_descriptor.get('RecordID', 0)),
        "ICU_TYPE": CinC_2012_ICU_TYPE_MAP[record_descriptor.get('ICUType')],
        "GENDER": CinC_2012_GENDER_MAP[record_descriptor.get('Gender')],
        "AGE": record_descriptor.get('Age'),
        "HEIGHT": record_descriptor.get('Height'),
        "WEIGHT": record_descriptor.get('Weight')
    }

    return record_descriptor

def get_cinc_2012_record(cinc_2012_base, record_id, wide=True):
    """ Load the record file for the given id.

    N.B. This file is assumed to be named "<record_id>.txt" and located
    in the "<cinc_2012_base>/set-a" directory.

    Parameters
    ----------
    cinc_2012_base: path-like
        The path to the main folder for this CinC challenge

    record_id: string-like
        The identifier for this record, e.g., "132539"

    wide: bool
        Whether to return a "long" or "wide" data frame

    N.B. According to the specification (https://physionet.org/challenge/2012/,
    "General descriptors"), six descriptors are recorded only when the patients
    are admitted to the ICU and are included only once at the beginning of the
    record. 

    Returns
    -------
    record_descriptors: dictionary
        The six descriptors:
        
            * HADM_ID: string, the record id. We call it "HADM_ID" to
                keep the nomenclature consistent with the MIMIC data

            * ICU_TYPE: string ["coronary_care_unit", 
                "cardiac_surgery_recovery_unit", "medical_icu","surgical_icu"]

            * GENDER: string ['female', 'male']
            * AGE: float (or np.nan for missing)
            * WEIGHT: float (or np.nan for missing)
            * HEIGHT: float (or np.nan for missing)


    observations: pd.DataFrame
        The remaining time series entries for this record. This is returned as
        either a "long" or "wide" data frame with columns:

            * HADM_ID: string (added for easy joining, etc.)
            * ELAPSED_TIME: timedelta64[ns]
            * MEASUREMENT: the name of the measurement
            * VALUE: the value of the measurement

        For a wide data frame, there is instead one column for each
        measurement.
    """
    record_file = "{}.txt".format(record_id)
    record_file = os.path.join(cinc_2012_base, "set-a", record_file)

    observations = pd.read_csv(record_file)

    # from documentation:
    #
    # "A value of -1 indicates missing or unknown data"
    observations = observations.replace(-1, np.nan)

    # first, get the descriptor
    descriptor = _get_cinc_2012_record_descriptor(observations)

    # we do not want to expose this function, but we know the elapsed times
    # are of the form "HH:MM"
    def parse_timedelta(hhmm):
        hours, minutes = hhmm.split(":")
        delta = datetime.timedelta(hours=int(hours), minutes=int(minutes))
        return delta

    # and now the observations

    # first, discard the descriptor fields
    m_d = observations['Parameter'].isin(CinC_2012_DESCRIPTOR_FIELDS)
    observations = observations[~m_d]

    # It is possible we actually do not have any observations besides the
    # descriptors
    if len(observations) == 0:
        msg = "Did not have any observations for record: {}".format(record_id)
        raise ValueError(msg)

    observations['Time'] = observations['Time'].apply(parse_timedelta)

    # rename the columns to match the MIMIC columns a bit better
    new_columns = {
        'Time': 'ELAPSED_TIME',
        'Parameter': 'MEASUREMENT',
        'Value': 'VALUE'
    }

    observations = observations.rename(columns=new_columns)

    # pivot, if needed
    if wide:
        observations = pd.pivot_table(
            observations,
            values='VALUE',
            index='ELAPSED_TIME',
            columns='MEASUREMENT',
            fill_value=np.nan
        )

        observations = observations.reset_index()
        observations.columns.name = None

    # either way, add the episode id
    observations['HADM_ID'] = descriptor['HADM_ID']

    # and return the descriptor and observations
    return descriptor, observations
