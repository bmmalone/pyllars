"""
This module contains utilities for data frame manipulation. 
 
This module differs from `ml_utils` and others because this module
treats pandas data frames more like database tables which hold various
types of records. The other modules tend to treat data frames as data
matrices (in a statistical/machine learning sense).
"""
import logging
logger = logging.getLogger(__name__)

import functools
import gzip
import os
import shutil

import numpy as np
import pandas as pd
import tqdm

import openpyxl

import pyllars.utils as utils
import pyllars.validation_utils as validation_utils

import typing

from typing import Callable, Dict, Generator, Iterable, List, Set
StrOrList = typing.Union[str,typing.List[str]]


def apply(df:pd.DataFrame, func:Callable, *args, progress_bar:bool=False,
        **kwargs) -> List:
    """ Apply func to each row in the data frame
    
    Unlike :py:meth:`pandas.DataFrame.apply`, this function does not attempt to
    "interpret" the results and cast them back to a data frame, etc.
    
    Parameters
    ----------
    df: pandas.DataFrame
        the data frame
        
    func: typing.Callable
        The function to apply to each row in `data_frame`

    args
        Positional arguments to pass to `func`
    
    kwargs
        Keyword arguments to pass to `func`

    progress_bar: bool
        Whether to show a progress bar when waiting for results.

    Returns
    -------
    results: typing.List
        The result of each function call
    """    
    it = df.iterrows()
    if progress_bar:
        it = tqdm.tqdm(it, total=len(df))

    ret_list = [
        func(*(row[1], *args), **kwargs) for row in it
    ]
    
    return ret_list


def apply_groups(groups:pd.core.groupby.DataFrameGroupBy, func:Callable,
        *args, progress_bar:bool=False, **kwargs) -> List:
    """ Apply `func` to each group in `groups`
    
    Unlike :py:meth:`pandas.core.groupby.GroupBy.apply`, this function does not attempt to
    "interpret" the results by casting to a data frame, etc.
    
    Parameters
    ----------
    groups: pandas.core.groupby.GroupBy
        The result of a call to `groupby` on a data frame
        
    func: function pointer
        The function to apply to each group in `groups`

    args
        Positional arguments to pass to `func`
    
    kwargs
        Keyword arguments to pass to `func`

    progress_bar: bool
        Whether to show a progress bar when waiting for results.

    Returns
    -------
    results: typing.List
        The result of each function call
    """    
    it = groups
    if progress_bar:
        it = tqdm.tqdm(it, total=len(groups))

    ret_list = [
        func(*(group, *args), **kwargs) for name, group in it
    ]
    
    return ret_list




def dict_to_dataframe(dic:Dict, key_name:str='key', value_name:str='value') -> pd.DataFrame:
    """ Convert a dictionary into a two-column data frame using the given
    column names. Each entry in the data frame corresponds to one row.

    Parameters
    ----------
    dic: typing.Dict
        A dictionary

    key_name: str
        The name to use for the column for the keys

    value_name: str
        The name to use for the column for the values

    Returns
    -------
    df: pandas.DataFrame
        A data frame in which each row corresponds to one entry in dic
    """
    df = pd.Series(dic, name=value_name)
    df.index.name = key_name
    df = df.reset_index()
    return df


def dataframe_to_dict(df:pd.DataFrame, key_field:str, value_field:str) -> Dict:
    """ Convert two columns of a data frame into a dictionary

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame

    key_field : str
        The field to use as the keys in the dictionary

    value_field : str
        The field to use as the values

    Returns
    -------
    the_dict: typing.Dict
        A dictionary which has one entry for each row in the data
        frame, with the keys and values as indicated by the fields
        
    """
    dic = dict(zip(df[key_field], df[value_field]))
    return dic

def get_series_union(*pd_series:Iterable[pd.Series]) -> Set:
    """ Take the union of values from the list of series
    
    Parameters
    ----------
    pd_series : typing.Iterable[pandas.Series]
        The list of pandas series
        
    Returns
    -------
    set_union : typing.Set
        The union of the values in all series
    """
    res = set.union(
        *[set(s) for s in pd_series]
    )
    
    return res

excel_extensions = ('xls', 'xlsx')
hdf5_extensions = ('hdf', 'hdf5', 'h5', 'he5')
parquet_extensions = ('parq', )

def _guess_df_filetype(filename:str):
    """ Guess the filetype of `filename`.
    
    The supported types and extensions used for guessing are:

        * excel: xls, xlsx
        * hdf5: hdf, hdf5, h5, he5
        * parquet: parq
        * csv: all other extensions

    Additionally, if filename is a pd.ExcelWriter object, then the guessed
    filetype will be 'excel_writer'

    Parameters
    ----------
    filename :str
        The name of the file (including extension) one which we will guess

    Returns
    -------
    guessed_type : str
        The guessed file type. See above for the supported types
            and extensions.
    """
    msg = "Attempting to guess the extension. Filename: {}".format(filename)
    logger.debug(msg)

    if isinstance(filename, pd.ExcelWriter):
        filetype = 'excel_writer'
    elif filename.endswith(excel_extensions):
        filetype = 'excel'
    elif filename.endswith(hdf5_extensions):
        filetype= 'hdf5'
    elif filename.endswith(parquet_extensions):
        filetype= 'parquet'
    else:
        filetype = 'csv'

    msg = "The guessed filetype was: {}".format(filetype)
    logger.debug(msg)

    return filetype

def read_df(filename:str, filetype:str='AUTO', sheet:str=None, **kwargs) -> pd.DataFrame:
    """ Read a data frame from a file
    
    By default, this function attempts to guess the type of the file based
    on its extension. Alternatively, the filetype can be exlicitly specified.
    The supported types and extensions used for guessing are:

        * excel: xls, xlsx
        * hdf5: hdf, hdf5, h5, he5
        * parquet: parq
        * csv: all other extensions

    **N.B.** In principle, matlab data files are hdf5, so this function should
    be able to read them. This has not been thoroughly tested, though.

    Parameters
    ----------
    filename : str
        The input file

    filetype : str
        The type of file, which determines which pandas read function will
        be called. If `AUTO`, the function uses the extensions mentioned above
        to guess the filetype.

    sheet: str
        For excel or hdf5 files, this will be passed to extract the desired
        information from the file. Please see :py:func:`pandas.read_excel` or
        :py:func:`pandas.read_hdf` for more information on how values are
        interpreted.
    
    kwargs
        Keyword arguments to pass to the appropriate `read` function.

    Returns
    -------
    df : pandas.DataFrame
        The data frame
    """
    # first, see if we want to guess the filetype
    if filetype == 'AUTO':
        filetype = _guess_df_filetype(filename)
        
    # now, parse the file
    if filetype == 'csv':
        df = pd.read_csv(filename, **kwargs)
    elif filetype == 'excel':
        df = pd.read_excel(filename, sheetname=sheet, **kwargs)
    elif filetype == 'hdf5':
        df = pd.read_hdf(filename, key=sheet, **kwargs)
    elif filetype == "parquet":
        
        caller = "pandas_utils.read_df"
        validation_utils.validate_packages_installed(['fastparquet'], caller)
        
        import fastparquet
        pf = fastparquet.ParquetFile(filename, **kwargs)

        # multi-indices are not yet supported, so we always have to turn
        # off indices to avoid a NotImplementedError
        df = pf.to_pandas(index=False)
    else:
        msg = "Could not read dataframe. Invalid filetype: {}".format(filetype)
        raise ValueError(msg)

    return df

def write_df(df:pd.DataFrame, out, create_path:bool=False, filetype:str='AUTO',
        sheet:str='Sheet_1', compress:bool=True,  **kwargs) -> None:
    """ Writes a data frame to a file of the specified type
    
    Unless otherwise specified, csv files are gzipped when written. By
    default, the filetype will be guessed based on the extension. The 
    supported types and  extensions used for guessing are:

        * excel: xls, xlsx
        * hdf5: hdf, hdf5, h5, he5
        * parquet: parq
        * csv: all other extensions (e.g., "gz" or "bed")

    Additionally, the filetype can be specified as 'excel_writer'. In this
    case, the out object is taken to be a pd.ExcelWriter, and the df is
    appended to the writer. AUTO will also guess this correctly.

    **N.B.** The hdf5 filetype has not been thoroughly tested.

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame

    out : str or pandas.ExcelWriter
        The (complete) path to the file.

        The file name WILL NOT be modified. In particular, ".gz" WILL 
        NOT be added if the file is to be zipped. As mentioned above,
        if the filetype is passed as 'excel_writer', then this is taken
        to be a pd.ExcelWriter object.

    create_path : bool
        Whether to create the path directory structure to the file if it
        does not already exist.

        N.B. This will not attempt to create the path to an excel_writer
        since it is possible that it does not yet have one specified.

    filetype : str
        The type of output file to write.  If `AUTO`, the function uses the
        extensions mentioned above to guess the filetype.

    sheet : str
        The name of the sheet (excel) or key (hdf5) to use when writing the
        file. This argument is not used for csv. For excel, the sheet is
        limited to 31 characters. It will be trimmed if necessary.

    compress : bool
        Whether to compress the output. This is only used for csv files.
    
    kwargs
        Keyword arguments to pass to the appropriate "write" function.

    Returns
    -------
        None : None
            The file is created as specified
    """
    
    # check for a deprecated keyword
    if 'do_not_compress' in kwargs:
        msg = ("[pd_utils.write_df] `do_not_compress` keyword has been removed. "
            "Please use `compress` instead.")
        raise DeprecationWarning(msg)
    
    # first, see if we want to guess the filetype
    if filetype == 'AUTO':
        filetype = _guess_df_filetype(out)

    # check if we want to and can create the path
    if create_path:
        if filetype != 'excel_writer':
            utils.ensure_path_to_file_exists(out)
        else:
            msg = ("[utils.write_df]: create_path was passed as True, but the "
                "filetype is 'excel_writer'. This combination does not work. "
                "The path to the writer will not be created.")
            logger.warning(msg)

    
    if filetype == 'csv':
        if compress:
            with gzip.open(out, 'wt') as out:
                df.to_csv(out, **kwargs)
        else:
            df.to_csv(out, **kwargs)

    elif filetype == 'excel':
        with pd.ExcelWriter(out) as out:
            df.to_excel(out, sheet[:31], **kwargs)

    elif filetype == 'excel_writer':
        df.to_excel(out, sheet[:31], **kwargs)

    elif filetype == 'hdf5':
        df.to_hdf(out, sheet, **kwargs)

    elif filetype == 'parquet':
        if compress:
            kwargs['compression'] = 'GZIP'

        # handle "index=False" kwarg
        if 'index' in kwargs:
            index = kwargs.pop('index')

            # if index is true, then no need to do anything
            # that is the default
            if not index:
                kwargs['write_index'] = False

        # if a parquet "file" exists, delete it
        if os.path.exists(out):
            # it could be either a folder or a file
            if os.path.isfile(out):
                # delete file
                os.remove(out)
            else:
                # delete directory
                shutil.rmtree(out)
        
        caller = "pandas_utils.read_df"
        validation_utils.validate_packages_installed(['fastparquet'], caller)
        
        import fastparquet
        fastparquet.write(out, df, **kwargs)

    else:
        msg = ("Could not write the dataframe. Invalid filetype: {}".format(
            filetype))
        raise ValueError(msg)

def append_to_xlsx(df:pd.DataFrame, xlsx:str, sheet='Sheet_1', **kwargs) -> None:
    """ Append `df` to `xlsx`
    
    If the sheet already exists, it will be overwritten. If the file does
    not exist, it will be created.

    **N.B.** This *will not* work with an open file handle! The xlsx argument
        *must be* the path to the file.

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame to write

    xlsx : str
        The path to the excel file

    sheet :str
        The name of the sheet, which will be truncated to 31 characters
    
    kwargs
        Keyword arguments to pass to the appropriate "write" function.

    Returns
    -------
    None : None
        The sheet is appended to the excel file
    """

    # check if the file already exists
    if os.path.exists(xlsx):
        book = openpyxl.load_workbook(xlsx)
        with pd.ExcelWriter(xlsx, engine='openpyxl') as writer:
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            write_df(df, writer, sheet=sheet, **kwargs)
    else:
        # then we can just create it fresh
        write_df(df, xlsx, sheet=sheet, **kwargs)



def split_df(df:pd.DataFrame, num_groups:int=None, chunk_size:int=None) -> pd.core.groupby.DataFrameGroupBy:
    """ Split `df` into roughly equal-sized groups
    
    The size of the groups can be specified by either giving the number
    of groups (`num_groups`) or the size of each group (`chunk_size`).
    
    The groups are contiguous rows in the data frame. 

    Parameters
    ----------
    df : pandas.DataFrame
        The data frame

    num_groups : int
        The number of groups

    chunk_size : int
        The size of each group. If given, `num_groups` groups has precedence
        over chunk_size
        
    Returns
    --------
    groups : pandas.core.groupby.GroupBy
        The groups
    """

    if num_groups is None:
        if chunk_size is not None:
            num_groups = int(df.shape[0] / chunk_size)
        else:
            msg = ("[pd_utils.split_df] one of `num_groups` and `chunk_size` "
                "must be provided")
            raise ValueError(msg)

    parallel_indices = np.arange(len(df)) // (len(df) / num_groups)
    split_groups = df.groupby(parallel_indices)
    return split_groups

def group_and_chunk_df(df:pd.DataFrame, groupby_field:str, chunk_size:int) -> pd.core.groupby.DataFrameGroupBy:
    """ Group `df` using then given field, and then create "groups of groups"
    with `chunk_size` groups in each outer group
    
    Parameters
    ----------
    df: pandas.DataFrame
        The data frame
        
    groupby_field: str
        The field for creating the initial grouping

    chunk_size: int
        The size of each outer group
        
    Returns
    --------
    groups : pandas.core.groupby.GroupBy
        The groups
    """
    
    # first, pull out the unique values for the groupby field
    df_chunks = pd.DataFrame(columns=[groupby_field],data=df[groupby_field].unique())
    
    # now, create a map from each unique groupby value to its chunk
    chunk_indices = np.arange(len(df_chunks)) // chunk_size
    df_chunks['chunk'] = chunk_indices
    stays_chunk_map = dataframe_to_dict(
        df_chunks,
        key_field=groupby_field,
        value_field='chunk'
    )
    
    # finally, determine the chunk of each row in the original data frame
    group_chunks = df[groupby_field].map(stays_chunk_map)
    
    # and create the group chunks
    group_chunks = df.groupby(group_chunks)
    
    return group_chunks

def get_group_extreme(df:pd.DataFrame, ex_field:str, ex_type:str="max",
        group_fields=None, groups:pd.core.groupby.GroupBy=None) -> pd.DataFrame:
    """ Find the row in each group of `df` with an extreme value for `ex_field`

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
    
    groups: None or pandas.core.groupby.GroupBy
        If not None, then these groups will be used to find the maximum values.

    group_fields: None or str or typing.List[str]
        If not `None`, then the field(s) by which to group the data frame. This
        value must be something which can be interpreted by
        pd.DataFrame.groupby.

    Returns
    -------
    ex_df: pandas.DataFrame
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

def groupby_to_generator(groups:pd.core.groupby.GroupBy) -> Generator:
    """ Convert the groupby object to a generator of data frames
    
    Parameters
    -----------
    groups : pandas.core.groupby.GroupBy
        The groups
        
    Returns
    --------
    group_generator : typing.Generator
        A generator over the data frames in `groups`
    """
    for k, g in groups:
        yield g

def join_df_list(dfs:List[pd.DataFrame], join_col:StrOrList, *args,
        **kwargs) -> pd.DataFrame:
    """ Join a list of data frames on a common column

    Parameters
    ----------
    dfs: typing.Iterable[pandas.DataFrame]
        The data frames

    join_col: str or typing.List[str]
        The name of the column(s) to use for joining. All of the data frames in
        `dfs` must have this column (or all columns in the list).

    args
        Positional arguments to pass to :py:func:`pandas.merge`
    
    kwargs
        Keyword arguments to pass to :py:func:`pandas.merge`


    Returns
    -------
    joined_df: pandas.DataFrame
        The data frame from joining all of those in the list on `join_col`.
        This function does not especially handle other columns which appear
        in all data frames, and their names in the joined data frame will be
        adjusted according to the standard pandas suffix approach.
    """
    joined_df = functools.reduce(
        lambda left,right: pd.merge(left,right,on=join_col, *args, **kwargs), dfs)

    return joined_df


def filter_rows(
        df_filter:pd.DataFrame,
        df_to_keep:pd.DataFrame,
        filter_on:List[str],
        to_keep_on:List[str],
        drop_duplicates:bool=True) -> pd.DataFrame:
    """ Filter rows from `df_to_keep` which have matches in `df_filter`
    
    **N.B.** The order of the the columns in `filter_on` and `to_keep_on`
    *must* match.
    
    This is adapted from: https://stackoverflow.com/questions/44706485.
    
    Parameters
    ----------
    df_filter : pandas.DataFrame
        The rows which will be used *as the filter*
        
    df_to_keep : pandas.DataFrame
        The rows which will be kept, unless they appear in `df_filter`
        
    filter_on : typing.List[str]
        The columns from `df_filter` to use for matching
        
    to_keep_on : typing.List[str]
        The columns from `df_to_keep` to use for matching
        
    drop_duplicates : bool
        Whether to remove duplicate rows from the filtered data frame
        
    Returns
    -------
    df_filtered : pandas.DataFrame
        The rows of `df_to_keep` which do not appear in `df_filter` (considering
        only the given columns)
    """
    
    d = df_filter.merge(
        df_to_keep,
        left_on=filter_on,
        right_on=to_keep_on,
        indicator=True,
        how='outer'
    )
    
    m_right_only = d['_merge'] == 'right_only'
    d = d[m_right_only]
    
    if drop_duplicates:
        d = d.drop_duplicates()
        d = d.reset_index(drop=True)
        
    return d