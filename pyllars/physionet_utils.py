"""
This module contains functions for working with datasets from physionet,
including MIMIC and the Computing in Cardiology Challenge 2012 datasets.

In the future, this module may be renamed and updated to also work with
the eICU dataset.

Please see the respective documentation for more details:

* MIMIC-III: https://mimic.physionet.org/about/mimic/
* CINC 2012: https://physionet.org/challenge/2012/
* eICU: https://eicu-crd.mit.edu/about/eicu/
"""

import logging
logger = logging.getLogger(__name__)

import datetime
import os

import dask
import joblib
import numpy as np
import pandas as pd
from dask import dataframe as dd

import pyllars.pandas_utils as pd_utils
import pyllars.utils as utils

import typing
from typing import Iterable, List

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

def fix_mimic_icds(icds:Iterable[str]) -> List[str]:
    """ Add the decimal to the correct location for the given ICD codes

    Since adding the decimals is a string-based operation, it can be somewhat
    slow. Thus, it may make sense to perform any filtering before fixing the
    (possibly much smaller number of) ICD codes.

    Parameters
    ----------
    icds : typing.Iterable[str]
        ICDs from the various mimic ICD columns

    Returns
    -------
    fixed_icds: List[str]
        The ICD codes with decimals in the correct location
    """
    fixed_icds = [
        _fix_mimic_icd(icd) for icd in icds
    ]

    return fixed_icds


def get_admissions(mimic_base:str, **kwargs) -> pd.DataFrame:
    """ Load the ADMISSIONS table
    
    This function automatically treats the following columns as date-times:
    
    * `ADMITTIME`
    * `DISCHTIME`
    * `DEATHTIME`
    * `EDREGTIME`
    * `EDOUTTIME`

    Parameters
    ----------
    mimic_base : str
        The path to the main MIMIC folder

    kwargs: <key>=<value> pairs
        Additional key words to pass to `read_df`

    Returns
    -------
    admissions : pandas.DataFrame
        The admissions table as a pandas data frame
    """
    date_cols = [
        'ADMITTIME',
        'DISCHTIME',
        'DEATHTIME',
        'EDREGTIME',
        'EDOUTTIME'
    ]
    
    admissions = os.path.join(mimic_base, "ADMISSIONS.csv.gz")
    admissions = pd_utils.read_df(admissions, parse_dates=date_cols, **kwargs)

    return admissions


def get_diagnosis_icds(
        mimic_base:str,
        drop_incomplete_records:bool=False,
        fix_icds:bool=False) -> pd.DataFrame:
    """ Load the `DIAGNOSES_ICDS` table

    Parameters
    ----------
    mimic_base : str
        The path to the main MIMIC folder

    drop_incomplete_records : bool
        Some of the ICD codes are missing. If this flag is `True`, then those
        records will be removed.

    fix_icds : bool
        Whether to add the decimal point in the correct position for the
        ICD codes

    Returns
    -------
    diagnosis_icds : pandas.DataFrame
        The diagnosis ICDs table as a pandas data frame
    """
    diagnosis_icds = os.path.join(mimic_base, "DIAGNOSES_ICD.csv.gz")
    diagnosis_icds = pd_utils.read_df(diagnosis_icds)

    if fix_icds:
        msg = "[mimic_utils]: Adding decimals to ICD codes"
        logger.debug(msg)

        fixed_icds = fix_mimic_icds(diagnosis_icds['ICD9_CODE'])
        diagnosis_icds['ICD9_CODE'] = fixed_icd

    if drop_incomplete_records:
        msg = "[mimic_utils]: removing incomplete diagnosis ICD records"
        logger.debug(msg)

        diagnosis_icds = diagnosis_icds.dropna()

    return diagnosis_icds

def get_followups(mimic_base:str) -> pd.DataFrame:
    """ Load the (constructed) FOLLOWUPS table

    Parameters
    ----------
    mimic_base : str
        The path to the main MIMIC folder

    Returns
    -------
    df_followups : pandas.DataFrame
        A data frame containing the followup information
    """
    followups = os.path.join(mimic_base, 'FOLLOWUPS.jpkl.gz')
    df_followups = joblib.load(followups)
    return df_followups

def get_icu_stays(mimic_base, to_pandas=True, **kwargs):
    """ Load the ICUSTAYS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    kwargs: key=value pairs
        Additional keywords to pass to the appropriate `read` function

    Returns
    -------
    patients: pd.DataFrame or dd.DataFrame
        The patients table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    date_cols = [
        'INTIME',
        'OUTTIME'
    ]
    icu_stays = os.path.join(mimic_base, "ICUSTAYS.csv.gz")

    if to_pandas:
        icu_stays = pd_utils.read_df(icu_stays, parse_dates=date_cols, **kwargs)
    else:
        icu_stays = dd.read_csv(icu_stays, parse_dates=date_cols, **kwargs)

    return icu_stays


def get_lab_events(mimic_base, to_pandas=True, drop_missing_admission=False,
        parse_dates=True, **kwargs):
    """ Load the LABEVENTS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    drop_missing_admission: bool
        About 20% of the lab events do not have an associated HADM_ID. If this
        flag is True, then those will be removed.

    parse_dates: bool
        Whether to directly parse `CHARTTIME` as a date. The main reason to
        skip this (when `parse_dates` is `False`) is if the `CHARTTIME` column
        is skipped (using the `usecols` parameter).

    kwargs: key=value pairs
        Additional keywords to pass to the appropriate `read` function

    Returns
    -------
    lab_events: pd.DataFrame or dd.DataFrame
        The notes table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    date_cols = []
    if parse_dates:
        date_cols = ['CHARTTIME']

    lab_events = os.path.join(mimic_base, "LABEVENTS.csv.gz")

    if to_pandas:
        lab_events = pd_utils.read_df(lab_events, parse_dates=date_cols, **kwargs)
    else:
        lab_events = dd.read_csv(lab_events, parse_dates=date_cols, **kwargs)

    if drop_missing_admission:
        msg = ("[physionet.get_lab_events] removing lab events with no "
            "associated hospital admission")
        logger.debug(msg)
        lab_events = lab_events.dropna(subset=['HADM_ID'])

    return lab_events


def get_lab_items(mimic_base, to_pandas=True, **kwargs):
    """ Load the D_LABITEMS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    kwargs: key=value pairs
        Additional keywords to pass to the appropriate `read` function

    Returns
    -------
    diagnosis_icds: pd.DataFrame or dd.DataFrame
        The notes table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    lab_items = os.path.join(mimic_base, "D_LABITEMS.csv.gz")

    if to_pandas:
        lab_items = pd_utils.read_df(lab_items, **kwargs)
    else:
        lab_items = dd.read_csv(lab_items, **kwargs)

    return lab_items


def get_notes(mimic_base, to_pandas=True, **kwargs):
    """ Load the NOTEEVENTS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    kwargs: key=value pairs
        Additional keywords to pass to the appropriate `read` function

    Returns
    -------
    diagnosis_icds: pd.DataFrame or dd.DataFrame
        The notes table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    note_events = os.path.join(mimic_base, "NOTEEVENTS.csv.gz")

    if to_pandas:
        note_events = pd_utils.read_df(note_events, **kwargs)
    else:
        note_events = dd.read_csv(note_events, **kwargs)

    return note_events



def get_patients(mimic_base, to_pandas=True, **kwargs):
    """ Load the PATIENTS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame

    kwargs: key=value pairs
        Additional keywords to pass to the appropriate `read` function

    Returns
    -------
    patients: pd.DataFrame or dd.DataFrame
        The patients table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    date_cols = [
        "DOB",
        "DOD",
        "DOD_HOSP",
        "DOD_SSN"
    ]
    patients = os.path.join(mimic_base, "PATIENTS.csv.gz")

    if to_pandas:
        patients = pd_utils.read_df(patients, parse_dates=date_cols, **kwargs)
    else:
        patients = dd.read_csv(patients, parse_dates=date_cols, **kwargs)

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
    
def get_transfers(mimic_base, to_pandas=True, **kwargs):
    """ Load the TRANSFERS table

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    to_pandas: bool
        Whether to read the table as a pandas (True) or dask (False) data frame


    kwargs: key=value pairs
        Additional keywords to pass to the appropriate `read` function
        
    Returns
    -------
    transfers: pd.DataFrame or dd.DataFrame
        The transfers table as either a pandas or dask data frame,
            depending on the value of to_pandas
    """
    date_cols = [
        "INTIME",
        "OUTTIME"
    ]
    transfers = os.path.join(mimic_base, "TRANSFERS.csv.gz")

    if to_pandas:
        transfers = pd_utils.read_df(transfers, parse_dates=date_cols, **kwargs)
    else:
        transfers = dd.read_csv(transfers, parse_dates=date_cols, **kwargs)

    return transfers

###
# Creating the FOLLOWUPS table
###
def _get_followups(g):
    subject_id = g.iloc[0]['SUBJECT_ID']
    g = g.sort_values('ADMITTIME')

    # just move the id's up one
    followup_hadm_ids = g['HADM_ID'].shift(-1)

    # set the followup of the last admission to -1
    followup_hadm_ids = followup_hadm_ids.fillna(-1)

    # and make them integral
    followup_hadm_ids = followup_hadm_ids.astype(int)


    df_followups = pd.DataFrame()
    df_followups['HADM_ID'] = g['HADM_ID'].copy()
    df_followups['FOLLOWUP_HADM_ID'] = followup_hadm_ids
    df_followups['FOLLOWUP_TIME'] = g['ADMITTIME'].shift(-1) - g['DISCHTIME']
    df_followups['SUBJECT_ID'] = subject_id
    
    return df_followups

def create_followups_table(mimic_base, progress_bar=True):
    """ Create the FOLLOWUPS table, based on the admissions

    In particular, the table has the following columns:
        * HADM_ID
        * FOLLOWUP_HADM_ID
        * FOLLOWUP_TIME: the difference between the discharge time of the
            first admission and the admit time of the second admission
        * SUBJECT_ID

    Parameters
    ----------
    mimic_base: path-like
        The path to the main MIMIC folder

    progress_bar: bool
        Whether to show a progress bar for creating the table

    Returns
    -------
    df_followups: pd.DataFrame
        The data frame constructed as described above. Currently, there is
        no need to create this table more than once. It can just be written
        to disk and loaded using `get_followups` after the initial creation.
    """
    df_admissions = physionet_utils.get_admissions(mimic_basepath)
    g_admissions = df_admissions.groupby('SUBJECT_ID')
    all_followup_dfs = pd_utils.apply_groups(
        g_admissions,
        _get_followups,
        progress_bar=progress_bar
    )
    
    df_followups = pd.concat(all_followup_dfs)
    return df_followups
    
###
# waveform database
###

def parse_rdsamp_datetime(fname, version=2):
    """ Extract the identifying information from the filename of the MIMIC-III
    header (\*hea) files

    In this project, we refer to each of these files as an "episode".

    Parameters
    ----------
    fname: string
        The name of the file. It should be of the form:
            
            version 1:
                /path/to/my/s09870-2111-11-04-12-36.hea
                /path/to/my/s09870-2111-11-04-12-36n.hea
                
            version 2:
                /path/to/my/p000020-2183-04-28-17-47.hea
                /path/to/my/p000020-2183-04-28-17-47n.hea

    Returns
    -------
    episode_timestap: dict
        A dictionary containing the time stamp and subject id for this episode.
        Specifically, it includes the following keys:

        * SUBJECT_ID: the patient identifier
        * EPISODE_ID: the identifier for this episode
        * EPISODE_BEGIN_TIME: the beginning time for this episode
    """
    if version == 1:
        end = 6
    elif version == 2:
        end = 7
    else:
        msg = "[parse_rdsamp_datetime] unknown version: {}".format(version)
        raise ValueError(msg)
        
    dt_fmt = "%Y-%m-%d-%H-%M"
    basename = utils.get_basename(fname)
    episode_id = basename
    subject_id = basename[1:end]
    subject_id = int(subject_id)
    dt = basename[end+1:]
    
    # in case dt still has the "n" at the end, remove it
    dt = dt.replace("n", "")
    dt_p = datetime.datetime.strptime(dt, dt_fmt)
    ret = {
        "SUBJECT_ID": subject_id,
        "EPISODE_BEGIN_TIME": dt_p,
        "EPISODE_ID": episode_id
    }
    
    return ret

###
# CinC 2012: https://physionet.org/challenge/2012/
#
#   "cinc_2012"
###

CinC_2012_BOOKKEEPING_FIELDS = {
    'HADM_ID'
}

_CinC_2012_DESCRIPTOR_FIELDS = {
    'RecordID',
    'Age',
    'Gender',
    'Height',
    'ICUType',
    'Weight'
}

CinC_2012_DESCRIPTOR_FIELDS = {
    'AGE',
    'GENDER',
    'HEIGHT',
    'ICU_TYPE',
    'WEIGHT'
}

CinC_2012_GENDER_MAP = {
    0: 'female',
    1: 'male',
    2: np.nan,
    None: np.nan,
    np.nan: np.nan,
    'female': 0,
    'FEMALE': 0,
    'male': 1,
    'MALE': 1
}

CinC_2012_ICU_TYPE_MAP = {
    1: "coronary_care_unit", 
    2: "cardiac_surgery_recovery_unit",
    3: "medical_icu",
    4: "surgical_icu",
    None: np.nan,
    np.nan: np.nan,
    "coronary_care_unit": 1,
    "cardiac_surgery_recovery_unit": 2,
    "medical_icu": 3,
    "surgical_icu": 4
}

CinC_2012_TIME_SERIES_MEASUREMENTS = [
    'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
    'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose', 'HCO3', 'HCT', 'HR',
    'K', 'Lactate', 'MAP', 'MechVent', 'Mg', 'NIDiasABP', 'NIMAP',
    'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets', 'SaO2', 'SysABP',
    'Temp', 'Urine', 'WBC', 'pH'
]

CinC_2012_OUTCOME_FIELDS = {
    'ADMISSION_ELAPSED_TIME',
    'EXPIRED',
    'SAPS-I',
    'SOFA',
    'SURVIVAL'
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
    m_icu_fields = record_file_df['Parameter'].isin(_CinC_2012_DESCRIPTOR_FIELDS)

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
    m_d = observations['Parameter'].isin(_CinC_2012_DESCRIPTOR_FIELDS)
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
