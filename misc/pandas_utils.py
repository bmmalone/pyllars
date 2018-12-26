"""
This module contains utilities for data frame manipulation. 
 
This module differs from math_utils because that module treats pandas
data frames as data matrices (in a statistical/machine learning sense),
while this module considers data frames more like database tables which
hold various types of records.
"""

import functools
import gzip
import os
import shutil

import numpy as np
import pandas as pd
import tqdm

import openpyxl

import misc.utils as utils
import misc.validation_utils as validation_utils

import typing

import logging
logger = logging.getLogger(__name__)

def dict_to_dataframe(dic, key_name='key', value_name='value'):
    """ Convert a dictionary into a two-column data frame using the given
    column names. Each entry in the data frame corresponds to one row.

    Parameters
    ----------
    dic: dictionary
        a dictionary

    key_name: string
        the name to use for the column for the keys

    value_name: string
        the name to use for the column for the values

    Returns
    -------
    df: pd.DataFrame
        a data frame in which each row corresponds to one entry in dic
    """
    df = pd.Series(dic, name=value_name)
    df.index.name = key_name
    df = df.reset_index()
    return df


def dataframe_to_dict(df, key_field, value_field):
    """ Convert two columns of a data frame into a dictionary

    Parameters
    ----------
    df : pd.DataFrame
        The data frame

    key_field : string
        The field to use as the keys in the dictionary

    value_field : string
        The field to use as the values

    Returns
    -------
    the_dict: dictionary
        A dictionary which has one entry for each row in the data
        frame, with the keys and values as indicated by the fields
        
    """
    dic = dict(zip(df[key_field], df[value_field]))
    return dic

def get_series_union(*pd_series):
    """ Take the union of values from the list of series
    
    Parameters
    ----------
    pd_series : list of pandas series
        The list of pandas series
        
    Returns
    -------
    set_union : set
        The union of the values in all series
    """
    res = set.union(
        *[set(s) for s in pd_series]
    )
    
    return res

excel_extensions = ('xls', 'xlsx')
hdf5_extensions = ('hdf', 'hdf5', 'h5', 'he5')
parquet_extensions = ('parq', )

def _guess_df_filetype(filename):
    """ This function attempts to guess the filetype given a filename. It is
        primarily intended for internal use, namely, for reading and writing
        dataframes. The supported types and extensions used for guessing are:

            excel: xls, xlsx
            hdf5: hdf, hdf5, h5, he5
            parquet: parq
            csv: all other extensions

        Additionally, if filename is a pd.ExcelWriter object, then the guessed
        filetype will be 'excel_writer'

        Args:
            filename (string): the name of the file for which we will guess

        Returns:
            string: the guessed file type. See above for the supported types
                and extensions.

        Imports:
            pandas
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

def read_df(filename, filetype='AUTO', sheet=None, **kwargs):
    """ This function reads a data frame from a file. By default it attempts
        to guess the type of the file based on its extension. Alternatively,
        the filetype can be exlicitly specified.  The supported types and
        extensions used for guessing are:

            excel: xls, xlsx
            hdf5: hdf, hdf5, h5, he5
            parquet: parq
            csv: all other extensions

        N.B. In principle, matlab data files are hdf5, so this function should
            be able to read them. This has not been tested, though.

        Args:
            filename (string): the input file

            filetype (string): the type of file, which determines which pandas
                read function will be called. If AUTO, the function uses the 
                extensions mentioned above to guess the filetype.

            sheet (string): for excel or hdf5 files, this will be passed
                to extract the desired information from the file. Please see
                pandas.read_excel and pandas.read_hdf for more information on
                how values are interpreted.

            kwards: these will be passed unchanged to the read function

        Returns:
            pd.DataFrame: a data frame

        Raises:
            ValueError: if the filetype is not 'AUTO' or one of the values
                mentioned above ('excel', 'hdf5', 'csv')

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
        sheet:str='Sheet_1', compress:bool=True,  **kwargs):
    """ This function writes a data frame to a file of the specified type. 
        Unless otherwise specified, csv files are gzipped when written. By
        default, the filetype will be guessed based on the extension. The 
        supported types and  extensions used for guessing are:

            excel: xls, xlsx
            hdf5: hdf, hdf5, h5, he5
            parquet: parq
            csv: all other extensions (e.g., "gz" or "bed")

        Additionally, the filetype can be specified as 'excel_writer'. In this
        case, the out object is taken to be a pd.ExcelWriter, and the df is
        appended to the writer. AUTO will also guess this correctly.

        N.B. The hdf5 filetype has not been thoroughly tested!!!

        Parameters
        ----------
        df: pd.DataFrame
            The data frame

        out: string or pd.ExcelWriter
            The (complete) path to the file.

            The file name WILL NOT be modified. In particular, ".gz" WILL 
            NOT be added if the file is to be zipped. As mentioned above,
            if the filetype is passed as 'excel_writer', then this is taken
            to be a pd.ExcelWriter object.

        create_path: bool
            Whether to create the path directory structure to the file if it
            does not already exist.

            N.B. This will not attempt to create the path to an excel_writer
            since it is possible that it does not yet have one specified.

        filetype: string
            The type of output file to write.  If AUTO, the function uses the
            extensions mentioned above to guess the filetype.

        sheet : string
            The name of the sheet (excel) or key (hdf5) to use when writing the
            file. This argument is not used for csv. For excel, the sheet is
            limited to 31 characters. It will be trimmed if necessary.

        compress : bool
            Whether to compress the output. This is only used for csv files.

        **kwargs : other keyword arguments to pass to the df.to_XXX method

        Returns
        -------
        None, but the file is created
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

def append_to_xlsx(df, xlsx, sheet='Sheet_1', **kwargs):
    """ This function appends the given dataframe to the excel file if it
        already exists. If the file does not exist, it will be created.

        N.B. This *will not* work with an open file handle! The xlsx argument
            *must be* the path to the file.

        Args:
            df (pd.DataFrame): the data frame to write

            xlsx (string): the path to the excel file.

            sheet (string): the name of the sheet, which will be truncated to
                31 characters

            **kwargs : other keyword arguments to pass to the df.to_XXX method

        Returns:
            None

        Imports:
            pandas
            openpyxl
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

def apply(df:pd.DataFrame, func:typing.Callable, *args, progress_bar:bool=False,
        **kwargs) -> typing.List:
    """ Apply func to each row in the data frame
    
    Unlike pd.DataFrame.apply, this function does not attempt to
    "interpret" the results.
    
    Parameters
    ----------
    df: pd.DataFrame
        the data frame
        
    func: function pointer
        The function to apply to each row in `data_frame`

    args, kwargs
        The other arguments to pass to `func`

    progress_bar: bool
        Whether to show a progress bar when waiting for results.

    Returns
    -------
    results: list
        The result of each function call
    """    
    it = df.iterrows()
    if progress_bar:
        it = tqdm.tqdm(it, total=len(df))

    ret_list = [
        func(*(row[1], *args), **kwargs) for row in it
    ]
    
    return ret_list


def apply_groups(groups:pd.core.groupby.DataFrameGroupBy, func:typing.Callable,
        *args, progress_bar:bool=False, **kwargs) -> typing.List:
    """ Apply func to each group
    
    Unlike pd.GroupBy.apply, this function does not attempt to
    "interpret" the results.
    
    Parameters
    ----------
    groups: pd.DataFrameGroupBy
        the groups
        
    func: function pointer
        The function to apply to each row in `data_frame`

    args, kwargs
        The other arguments to pass to `func`

    progress_bar: bool
        Whether to show a progress bar when waiting for results.

    Returns
    -------
    results: list
        The result of each function call
    """    
    it = groups
    if progress_bar:
        it = tqdm.tqdm(it, total=len(groups))

    ret_list = [
        func(*(group, *args), **kwargs) for name, group in it
    ]
    
    return ret_list



def split_df(df:pd.DataFrame, num_groups:int=None, chunk_size:int=None):
    """ Split the df into num_groups roughly equal-sized groups. The groups are
    contiguous rows in the data frame.

    Parameters
    ----------
    df: pd.DataFrame
        the data frame

    num_groups: int
        the number of groups

    chunk_size: int
        the size of each group. If given, `num_groups` groups has precedence
        over chunk_size
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

def group_and_chunk_df(df, groupby_field, chunk_size):
    """ Group `df` using then given field, and then create "groups of groups"
    with `chunk_size` groups in each outer group
    
    Parameters
    ----------
    df: pd.DataFrame
        the data frame
        
    groupby_field: string
        the field for creating the initial grouping

    chunk_size: int
        the number of groups in each outer group     
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

StrOrList = typing.Union[str,typing.List[str]]
def join_df_list(dfs:typing.List[pd.DataFrame], join_col:StrOrList, *args,
        **kwargs) -> pd.DataFrame:
    """ Join a list of data frames on a common column

    Parameters
    ----------
    dfs: list of pd.DataFrames
        The data frames

    join_col: string or list of strings
        The name of the column(s) to use for joining. All of the data frames in
        `dfs` must have this column (or all columns in the list).

    args, kwargs
        These are passed through to pd.merge

    Returns
    -------
    joined_df: pd.DataFrame
        The data frame from joining all of those in the list on `join_col`.
        This function does not especially handle other columns which appear
        in all data frames, and their names in the joined data frame will be
        adjusted according to the standard pandas suffix approach.
    """
    joined_df = functools.reduce(
        lambda left,right: pd.merge(left,right,on=join_col, *args, **kwargs), dfs)

    return joined_df

