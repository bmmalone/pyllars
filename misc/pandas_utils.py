###
#   This module contains utilities for data frame manipulation. 
# 
#   This module differs from math_utils because that module treats pandas
#   data frames as data matrices (in a statistical/machine learning sense),
#   while this module considers data frames more like database tables which
#   hold various types of records.
###

import numpy as np
import pandas as pd

def split_df(df:pd.DataFrame, num_groups:int):
    """ Split the df into num_groups roughly equal-sized groups. The groups are
    contiguous rows in the data frame.

    Parameters
    ----------
    df: pd.DataFrame
        the data frame

    num_groups: int
        the number of groups        
    """
    parallel_indices = np.arange(len(df)) // (len(df) / num_groups)
    split_groups = df.groupby(parallel_indices)
    return split_groups

def get_group_extreme(df:pd.DataFrame, ex_field:str, ex_type:str="max",
        group_fields=None, groups:pd.core.groupby.GroupBy=None):
    """ Find the row in each group of df with an extreme value for ex_field.

    "ex_type" must be either "max" or "min" and indicated which type of extreme
    to consider. Either the "group_field" or "groups" must be given.

    Parameters
    ----------
    df: pd.DataFrame
        The original data frame. Even if the groups are created externally, the
        original data frame must be given.

    ex_field: str
        The field to find for which to find the extreme values

    ex_type: str {"max" or "min"}, case-insensitive
        The type of extreme to consider.
    
    groups: None or pd.core.groupby.GroupBy
        If not None, then these groups will be used to find the maximum values.

    group_fields: None or str or list of strings
        If not None, then the field(s) by which to group the data frame. This
        value must be something which can be interpreted by
        pd.DataFrame.groupby.

    Returns
    -------
    ex_df: pd.DataFrame
        A data frame with rows which contain the extreme values for the
        indicated groups.
    """
    
    # make sure we were given something by which to group
    if (group_fields is None) and (groups is None):
        msg = ("[pandas_utils.get_group_extreme]: No groups or group field "
            "provided")
        raise ValueError(msg)

    # we also can't have both
    if (group_fields is not None) and (groups is not None):
        msg = ("[pandas_utils.get_group_extreme]: Both groups and group field "
            "provided")
        raise ValueError(msg)

    # and that we have a valid exteme op
    is_max = ex_type.lower() == "max"
    is_min = ex_type.lower() == "min"

    if not (is_max or is_min):
        msg = ("[pandas_utils.get_group_extreme]: Invalid ex_type given. "
            "Choices: \"max\" or \"min\"")
        raise ValueError(msg)

    # so we either have groups or group_field
    if group_fields is not None:
        groups = df.groupby(group_fields)

    if is_max:
        ex_vals = groups[ex_field].idxmax()
    elif is_min:
        ex_vals = groups[ex_field].idxmin()

    ex_rows = df.loc[ex_vals]
    return ex_rows

def groupby_to_generator(groups:pd.core.groupby.GroupBy):
    """ Convert the groupby object to a generator of data frames """
    for k, g in groups:
        yield g

