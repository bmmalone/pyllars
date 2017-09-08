import logging
logger = logging.getLogger(__name__)

import itertools
import os
import shutil
import subprocess
import sys

import misc.shell_utils as shell_utils


def raise_deprecation_warning(function, new_module, final_version=None,
        old_module="misc"):
    """ Ths function raises a deprecation about a function that has been
    moved to a new module.

    Parameters
    ----------
    function : string
        The name of the (existing) function

    new_module : string
        The name of the new module containing the function

    old_module: string
        The name of the old module for the function. Default: misc

    final_version: string
        The name of the last version for which the function will be available
        in the old_module, or None.

    Returns
    -------
    None, but prints a "warn" message. If final_version is not None, then the
    message will include a bit about when the method will be removed from the
    current module.
            
    """
    
    msg = ("[{}]: This function is deprecated. Please use the version in {} "
        "instead.".format(function, new_module))

    if final_version is not None:
        msg_2 = (" The function will be removed from the module {} in version "
        "{}".format(old_module, final_version))
        msg = msg + msg_2
        
    logger.warn(msg)


### Parsing and writing utilities

trueStrings = ['true', 'yes', 't', 'y', '1']

def str2bool(string):
    return (string.lower() in trueStrings)

def try_parse_int(string):
    try:
        return int(string)
    except ValueError:
        return None

def try_parse_float(string):
    try:
        return float(string)
    except ValueError:
        return None

def is_int(s):
    """ This function checks whether the provided string represents and integer.

        This code was adapted from: http://stackoverflow.com/questions/1265665/

        Args:
            s (string) : the string

        Returns:
            bool : whether the string can be interpretted as an integer
    """

    if s[0] in ('-', '+'):
    	return s[1:].isdigit()
    return s.isdigit()

def check_keys_exist(d, keys):
    """ This function ensures the given keys are present in the dictionary. It
        does not other validate the type, value, etc., of the keys or their
        values. If a key is not present, a KeyError is raised.

        The motivation behind this function is to verify that a config dictionary
        read in at the beginning of a program contains all of the required values.
        Thus, the program will immediately detect when a required config value is
        not present and quit.

        Input:
            d (dict) : the dictionary

            keys (list) : a list of keys to check
        Returns:
            list of string: a list of all programs which are not found

        Raises:
            KeyError: if any of the keys are not in the dictionary
    """
    missing_keys = [k for k in keys if k not in d]

    
    if len(missing_keys) > 0:
        missing_keys = ' '.join(missing_keys)
        msg = "The following keys were not found: " + missing_keys
        raise KeyError(msg)

    return missing_keys

# http://goo.gl/zeJZl
def bytes2human(n, format="%(value)i%(symbol)s"):
    """
    >>> bytes2human(10000)
    '9K'
    >>> bytes2human(100001221)
    '95M'
    """
    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    for symbol in reversed(symbols[1:]):
        if n >= prefix[symbol]:
            value = float(n) / prefix[symbol]
            return format % locals()
    return format % dict(symbol=symbols[0], value=n)

# http://goo.gl/zeJZl
def human2bytes(s):
    """
    >>> human2bytes('1M')
    1048576
    >>> human2bytes('1G')
    1073741824
    """
    # first, check if s is already a number
    if is_int(s):
        return s

    symbols = ('B', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    letter = s[-1:].strip().upper()
    num = s[:-1]
    assert num.isdigit() and letter in symbols
    num = float(num)
    prefix = {symbols[0]:1}
    for i, s in enumerate(symbols[1:]):
        prefix[s] = 1 << (i+1)*10
    return int(num * prefix[letter])

def simple_fill(text, width=60):
    """ This is a simplified version of textwrap.fill. It splits the string
        into exactly equal-sized chuncks on length <width>. This avoids the
        pathological case of one long string (e.g., when splitting long DNA
        sequences).

        The code is adapted from: http://stackoverflow.com/questions/11781261

        Args:
            text (string) : the text to split

            width (int) : the (exact) length of each line after splitting

        Returns:
            string : a single string with lines of length width (except 
                possibly the last line)
    """
    return '\n'.join(text[i:i+width] 
                        for i in range(0, len(text), width))


def split(delimiters, string, maxsplit=0):
    """ This function splits the given string using all delimiters in the list.
        
        The code is taken from: http://stackoverflow.com/questions/4998629/

        Args:
            delimiters (list of strings): the strings to use as delimiters
            string (string): the string to split
            maxsplit (int): the maximum number of splits (or 0 for no limit)

        Returns:
            list of strings: the split string
    """
    import re
    regex_pattern = '|'.join(map(re.escape, delimiters))
    return re.split(regex_pattern, string, maxsplit)

def read_commented_file(filename):
    f = open(filename)
    lines = []
    for line in f:
        line = line.partition("#")[0].strip()
        if len(line) > 0:
            lines.append(line)
    return lines

def get_vars_to_save(to_save, to_remove=['parser', 'args']):
    import types

    # remove the system variables, modules and functions
    for (var_name,value) in to_save.items():
        if var_name.startswith('__'):
            to_remove.append(var_name)

        elif (
            isinstance(value, types.FunctionType) or 
            isinstance(value, types.ModuleType)):
            
            to_remove.append(var_name)

    for var_name in to_remove:
        if var_name in to_save:
            del to_save[var_name]

    return to_save


def command_line_option_to_keyword(option):
    """ Convert the command line version of the option to a keyword.

    Parameters
    ----------
    option: string
        The "long" command line option version

    Returns
    -------
    keyword: string
        The "keyword" version of the option. Namely, the initial "--" is
        removed and all internal "-"s are replaced with "_"s.
    """
    # first, remove the initial "--"
    option = option[2:]
    
    # and replace "-" with "_"
    option = option.replace("-", "_")
    
    return option


def get_config_argument(config, var_name, argument_name=None, default=None):
    """ This function checks to see if the config dictionary contains the given
        variable. If so, it constructs a command line argument based on the type
        of the variable. If a default is given, then that value is used if the
        variable is not present in the config dictionary.

        Args:
            config (dict): a dictionary, presumably containing configuration
            options

            var_name (string): the name of the variable to look up

            argument_name (string): if present, then the command line argument
            will be "--<argument_name>". Otherwise, the command line switch
            will be: "--<var_name.replace(_,-)"

            default (string or list): if present, then this value is used if 
                the variable is not in the dictionary

        Returns:
            string: either the empty string if var_name is not in config, or a
                properly formatted command line switch, based on whether the
                variable is a string or list
    """
    import shlex
    argument = ""

    if (var_name in config) or (default is not None):
        # check if we have a string
        var = config.get(var_name, default)

        # we could have included the variable in the config with a 'None' value
        if var is None:
            return argument
    
        if isinstance(var, (str, )) and (len(str(var)) > 0):
            argument = shlex.quote(var)
        elif isinstance(var, (int, float)) and (len(str(var)) > 0):
            argument = shlex.quote(str(var))
        elif len(var) > 0:
            # assume this is a list
            argument = " ".join(shlex.quote(str(v)) for v in var)

        if argument_name is None:
            argument_name = var_name.replace('_', '-')

        if len(argument) > 0:
            argument = "--{} {}".format(argument_name, argument)
    return argument

def get_config_args_value(default_value, config_value, args_value):
    """ This helper function selects which value to use based on the precedence
        order: args, config, default (that is, the args value is chosen if
        present, etc.)

        N.B. This seems like a common pattern; there may be a better way to do
        this. https://pypi.python.org/pypi/ConfigArgParse, for example.

        Args:
            default_value: the default value to use if neither the config nor
                the args value is given

            config_value: the value to use (presumably from a config file) to
                use if args value is not given

            args_value: the value to use, if present

        Returns:
            obj: the selected value, according to the precedence order
    """

    if args_value is not None:
        return args_value

    if config_value is not None:
        return config_value

    return default_value

def concatenate_files(in_files, out_file, call=True):
    """ Concatenate the input files to the output file.

    Parameters
    ----------
    in_files: list of strings
        The paths to the input files, which will be opened in binary mode

    out_file: string
        The path to the output file. This *should not* be the same as one of
        the input files.

    call: bool
        Whether to actually perform the action
    """
    in_files_str = ",".join(in_files)
    msg = ("Concatenating files. Output file: {}; Input files: {}".format(
        out_file, in_files_str))
    logger.info(msg)

    if not call:
        msg = "Skipping concatenation due to --call value"
        logger.info(msg)

        return

    with open(out_file, 'wb') as out:
        for in_file in in_files:
            with open(in_file, 'rb') as in_f:
                shutil.copyfileobj(in_f, out)

def check_gzip_file(filename, has_tar=False, raise_on_error=True, logger=logger):
    """ This function wraps a call to "gunzip -t". Optionally, it 
        raises an exception if the return code is not 0. Otherwise, it writes
        a "critical" warning message.

        This function can also test that a tar insize the gzipped file is valid.

        This code is adapted from: http://stackoverflow.com/questions/2001709/

        Args:
            filename (str): a path to the bam file

            has_tar (bool): whether to check for a valid tar inside the
                gzipped file

            raise_on_error (bool): whether to raise an OSError (if True) or log
                a "critical" message (if false)

            logger (logging.Logger): a logger for writing the message if an
                error is not raised

        Returns:
            bool: whether the file was valid

        Raises:
            OSError: if gunzip does not return 0 and raise_on_error is True
    """
    
    programs = ['gunzip', 'tar']
    shell_utils.check_programs_exist(programs)

    if has_tar:
        cmd = "gunzip -c {} | tar t > /dev/null".format(filename)
    else:
        cmd = "gunzip -t {}".format(filename)

    ret = shell_utils.check_call_step(cmd, raise_on_error=False)

    if ret != 0:
        msg = "The gzip file does not appear to be valid: {}".format(filename)

        if raise_on_error:
            raise OSError(msg)

        logger.critical(msg)
        return False

    # then the file was okay
    return True

def ensure_path_to_file_exists(f):
    """ If the base path to f does not exist, create it. """

    out_dir = os.path.dirname(f)

    # if we are given just a filename, do not do anything
    if len(out_dir) > 0:
        msg = "Ensuring directory exists: {}".format(out_dir)
        logger.info(msg)
        os.makedirs(out_dir, exist_ok=True)

def check_files_exist(files, raise_on_error=True, logger=logger, 
        msg="The following files were missing: ", source=None):
    """ This function ensures that all of the files in the list exists. If any
        do not, it will either raise an exception or print a warning, depending
        on the value of raise_on_error.

        Parameters
        ----------
        files: list of strings
            the file paths to check

        raise_on_error: bool
            whether to raise an error if any of the files are missing

        logger: logging.Logger
            a logger to use for writing the warning if an error is not raised

        msg: string
            a message to write before the list of missing files

        source: string
            a description of where the check is made, such as a module name. If
            this is not None, it will be prepended in brackets before msg.

        Returns
        -------
        all_exist: bool
            True if all of the files existed, False otherwise

        Raises
        ------
        FileNotFoundError, if raise_on_error is True and any of the files
                do not exist.
    """
    missing_files = []

    for f in files:
        if not os.path.exists(f):
            missing_files.append(f)

    if len(missing_files) == 0:
        return True

    missing_files_str = ",".join(missing_files)
    source_str = ""
    if source is not None:
        source_str = "[{}]: ".format(source)
    msg = "{}{}{}".format(source_str, msg, missing_files_str)
    if raise_on_error:
        raise FileNotFoundError(msg)
    else:
        logger.warn(msg)

    return False

def remove_file(filename):
    """Remove the file, if it exists. Ignore FileNotFound errors.""" 
    import contextlib
    import os

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)

def count_lines(filename):
    """ This function counts the number of lines in filename.

    Parameters
    ----------
    filename : string 
        The path to the file. gzipped files are handled transparently

    Returns
    -------
    num_lines : int
        The number of lines in the file
    """

    with open(filename) as f:
        i = -1
        for i, l in enumerate(f):
            pass
    return i + 1

### Path utilities

def abspath(*fn):
    return os.path.abspath(os.path.join(os.sep, *fn))

def add_home_dir(*fn):
    return os.path.join(os.path.expanduser('~'), *fn)

def listdir_full(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def list_subdirs(path):
    """ List all subdirectories directly under path
    """
    subdirs = [
        d for d in listdir_full(path) if os.path.isdir(d)
    ]
    return subdirs

def get_basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def create_symlink(src, dst, remove=True, create=False, call=True):
    """ Creates or updates a symlink at dst which points to src.

    Parameters
    ----------
    src: string
        the path to the original file

    dst: string
        the path to the symlink

    remove: bool
        whether to remove any existing file at dst

    create: bool
        whether to create the directory structure necessary for dst

    call: bool
        whether to actually do anything

    Returns
    -------
    None, but the symlink is created

    Raises
    ------
    FileExistsError, if a file already exists at dst and the remove flag is
        False
    """
    import logging

    raise_deprecation_warning("misc.utils.create_symlink", "misc.shell_utils")

    if not call:
        return

    if os.path.lexists(dst):
        if remove:
            msg = ("[utils.create_symlink]: file already exists at: '{}'. It "
                "will be removed".format(dst))
            logging.warning(msg)
            os.remove(dst)
        else:
            msg = "A file already exists at: '{}'".format(dst)
            raise FileExistsError(msg)

    if create:
        os.makedirs(os.path.dirname(dst), exist_ok=True)

    os.symlink(src, dst)


### numpy stack helpers


def to_dense(data, row, dtype=float, length=-1):
    import numpy as np
    d = data.getrow(row).todense()
    d = np.squeeze(np.asarray(d, dtype=dtype))

    if length > 0:
        d = d[:length]

    # make sure we do not return a scalar
    if isinstance(d, dtype):
        d = np.array([d])

    return d

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
    raise_deprecation_warning("dict_to_dataframe", "misc.pandas_utils",
        "0.3.0", "misc")
    import pandas as pd

    df = pd.Series(dic, name=value_name)
    df.index.name = key_name
    df = df.reset_index()
    return df

def dataframe_to_dict(df, key_field, value_field):
    """ This function converts two columns of a data frame into a dictionary.

        Args:
            df (pd.DataFrame): the data frame

            key_field (string): the field to use as the keys in the dictionary

            value_field (string): the field to use as the values

        Returns:
            dict: a dictionary which has one entry for each row in the data
                frame, with the keys and values as indicated by the fields
        
    """
    raise_deprecation_warning("dataframe_to_dict", "misc.pandas_utils",
        "0.3.0", "misc")
    dic = dict(zip(df[key_field], df[value_field]))
    return dic

def pandas_join_string_list(row, field, sep=";"):
    """ This function checks if the value for field in the row is a list. If so,
        it is replaced by a string in which each value is separated by the
        given separator.

        Args:
            row (pd.Series or similar): the row to check
            field (string): the name of the field
            sep (string): the separator to use in joining the values
    """
    raise_deprecation_warning("pandas_join_string_list", "misc.pandas_utils",
        "0.3.0", "misc")
    s = wrap_string_in_list(row[field])
    return sep.join(s)

excel_extensions = ('xls', 'xlsx')
hdf5_extensions = ('hdf', 'hdf5', 'h5', 'he5')

def _guess_df_filetype(filename):
    """ This function attempts to guess the filetype given a filename. It is
        primarily intended for internal use, namely, for reading and writing
        dataframes. The supported types and extensions used for guessing are:

            excel: xls, xlsx
            hdf5: hdf, hdf5, h5, he5
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
    raise_deprecation_warning("_guess_df_filetype", "misc.pandas_utils",
        "0.3.0", "misc")
    import pandas as pd

    msg = "Attempting to guess the extension. Filename: {}".format(filename)
    logger.debug(msg)

    if isinstance(filename, pd.ExcelWriter):
        filetype = 'excel_writer'
    elif filename.endswith(excel_extensions):
        filetype = 'excel'
    elif filename.endswith(hdf5_extensions):
        filetype= 'hdf5'
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

        Imports:
            pandas
    """
    raise_deprecation_warning("read_df", "misc.pandas_utils",
        "0.3.0", "misc")
    import pandas as pd

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
    else:
        msg = "Could not read dataframe. Invalid filetype: {}".format(filetype)
        raise ValueError(msg)

    return df

def write_df(df, out, create_path=False, filetype='AUTO', sheet='Sheet_1',
        do_not_compress=False, **kwargs):
    """ This function writes a data frame to a file of the specified type. 
        Unless otherwise specified, csv files are gzipped when written. By
        default, the filetype will be guessed based on the extension. The 
        supported types and  extensions used for guessing are:

            excel: xls, xlsx
            hdf5: hdf, hdf5, h5, he5
            csv: all other extensions (e.g., "gz" or "bed")

        Additionally, the filetype can be specified as 'excel_writer'. In this
        case, the out object is taken to be a pd.ExcelWriter, and the df is
        appended to the writer. AUTO will also guess this correctly.

        N.B. The hdf5 filetype has not been tested!!!

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

        sheet: string
            The name of the sheet (excel) or key (hdf5) to use when writing the
            file. This argument is not used for csv. For excel, the sheet is
            limited to 31 characters. It will be trimmed if necessary.

        do_not_compress: bool
            Whether to compress the output. This is only used for csv files.

        **kwargs : other keyword arguments to pass to the df.to_XXX method

        Returns
        -------
        None, but the file is created
    """
    raise_deprecation_warning("write_df", "misc.pandas_utils",
        "0.3.0", "misc")
    import gzip
    import pandas as pd
    
    # first, see if we want to guess the filetype
    if filetype == 'AUTO':
        filetype = _guess_df_filetype(out)

    # check if we want to and can create the path
    if create_path:
        if filetype != 'excel_writer':
            ensure_path_to_file_exists(out)
        else:
            msg = ("[utils.write_df]: create_path was passed as True, but the "
                "filetype is 'excel_writer'. This combination does not work. "
                "The path to the writer will not be created.")
            logger.warning(msg)

    
    if filetype == 'csv':
        if do_not_compress:
            df.to_csv(out, **kwargs)
        else:
            with gzip.open(out, 'wt') as out:
                df.to_csv(out, **kwargs)

    elif filetype == 'excel':
        with pd.ExcelWriter(out) as out:
            df.to_excel(out, sheet[:31], **kwargs)

    elif filetype == 'excel_writer':
        df.to_excel(out, sheet[:31], **kwargs)

    elif filetype == 'hdf5':
        df.to_hdf(out, sheet, **kwargs)
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
    raise_deprecation_warning("append_to_xlsx", "misc.pandas_utils",
        "0.3.0", "misc")
    import os
    import pandas as pd
    import openpyxl

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

###
#   Functions to help with built-in (ish) data structures
###

def list_to_dict(l, f=None):
    """ Convert the list to a dictionary in which keys and values are adjacent
    in the list. Optionally, a function can be passed to apply to each value
    before adding it to the dictionary.

    Example: 

        list = ["key1", "value1", "key2", "value2"]
        dict = {"key1": "value1", "key2": "value2"}

    Parameters
    ----------
    l: sequence
        The list of items

    f: function
        A function to apply to each value before inserting it into the list.
        For example, "float" could be passed to convert each value to a float.

    Returns
    -------
    dict: dictionary
        The dictionary, defined as described above
    """
    if len(l) % 2 != 0:
        msg = ("[utils.list_to_dict]: the list must contain an even number"
            "of elements")
        raise ValueError(msg)

    if f is None:
        f = lambda x: x

    keys = l[::2]
    values = l[1::2]
    d = {k:f(v) for k, v in zip(keys, values)}
    return d

def merge_dicts(*dict_args):
    """ Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.

    This is exactly taken from: http://stackoverflow.com/questions/38987
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def get_type(type_string):
    """ Find the type object corresponding to the fully qualified class

    Parameters
    ----------
    type_string : string
        A fully qualified class name. 
        
        Example: sklearn.neighbors.regression.KNeighborsRegressor

    Returns
    -------
    type : type
        The type object specified by the string. For example, this can be used
        for calls to "isinstance"
    """
    import importlib

    class_ = None
    try:
        module, class_ = type_string.rsplit(".", 1)
        module = importlib.import_module(module)
        class_ = getattr(module, class_)
    except Exception as e:
        msg = "[utils.get_type]: could not parse type: {}".format(type_string)
        logger.debug(msg)

    return class_

def is_sequence(maybe_sequence):
    """ This function is a light wrapper around collections.Sequence to check
        if the provided object is a sequence-like object. It also checks for
        numpy arrays.

        The function specifically checks is maybe_sequence is an instance of a
        string and returns False if it is a string.

        Args:
            maybe_sequence (object) an object which may be a list-like

        Returns:
            bool: whether the object is recognized as an instance of
                collections.Sequence or numpy.ndarray

        Imports:
            collections
            numpy
    """
    import collections
    import numpy

    if isinstance(maybe_sequence, str):
        return False

    is_sequence = isinstance(maybe_sequence, collections.Sequence)
    is_ndarray = isinstance(maybe_sequence, numpy.ndarray)
    return  is_sequence or is_ndarray

def wrap_in_list(maybe_list):
    """ This function checks if maybe_list is a list (or anything derived
        from list). If not, it wraps it in a list.

        The motivation for this function is that some functions return either
        a single object (e.g., a dictionary) or a list of those objects. The
        return value of this function can be iterated over safely.

        N.B. This function would not be helpful for ensuring something is a
        list of lists, for example.

        Args:
            maybe_list (obj): an object which may be a list

        Returns:
            either maybe_list if it is a list, or maybe_list wrapped in a list
    """
    if isinstance(maybe_list, list):
        return maybe_list
    return [maybe_list]

def wrap_string_in_list(maybe_string):
    """ This function checks if maybe_string is a string (or anything derived
        from str). If so, it wraps it in a list.

        The motivation for this function is that some functions return either a
        single string or multiple strings as a list. The return value of this
        function can be iterated over safely.

        Args:
            maybe_string (obj): an object which may be a string

        Returns:
            either the original object, or maybe_string wrapped in a list, if
                it was a string
    """
    if isinstance(maybe_string, str):
        return [maybe_string]
    return maybe_string

def flatten_lists(list_of_lists):
    """ This function flattens a list of lists into a single list.

        Args:
            list_of_lists (list): the list to flatten

        Returns:
            list: the flattened list
    """
    return [item for sublist in list_of_lists for item in sublist]

def list_remove_list(l, to_remove):
    """ This function removes items in to_remove from the list l. Note that 
        "not in" is used to match items in the list.

        Args:
            l (list): a list

            to_remove (list): a list of things to remove from l

        Returns:
            list: a copy of l, excluding items in to_remove
    """
    ret = [i for i in l if i not in to_remove]
    return ret

def list_insert_list(l, to_insert, index):
    """ This function inserts items from one list into another list at the
        specified index. This function returns a copy; it does not alter the
        original list.


        This function is adapted from: http://stackoverflow.com/questions/7376019/

        Example:

        a_list = [ "I", "rad", "list" ]
        b_list = [ "am", "a" ]
        c_list = list_insert_list(a_list, b_list, 1)

        print( c_list ) # outputs: ['I', 'am', 'a', 'rad', 'list']
    """

    ret = list(l)
    ret[index:index] = list(to_insert)
    return ret

def remove_keys(d, to_remove):
    """ This function removes the given keys from the dictionary d. N.B.,
        "not in" is used to match the keys.

        Args:
            d (dict): a dictionary

            to_remove (list): a list of keys to remove from d

        Returns:
            dict: a copy of d, excluding keys in to_remove
    """
    ret = {
        k:v for k,v in d.items() if k not in to_remove
    }
    return ret

def remove_nones(l, return_np_array=False):
    """ This function removes "None" values from the given list. Importantly,
        compared to other single-function tests, this uses "is" and avoids
        strange behavior with data frames, lists of bools, etc.

        Optionally, the filtered list can be returned as an np array.

        This function returns a copy of the list (but not a deep copy).

        N.B. This does not test nested lists. So, for example, a list of lists
        of Nones would be unchanged by this function.

        Args:
            l (list-like): a list which may contain Nones

            return_np_array (bool): if true, the filtered list will be wrapped
                in an np.array.

        Returns:
            list: a list or np.array with the Nones removed

        Imports:
            numpy
    """
    import numpy as np

    ret = [i for i in l if i is not None]

    if return_np_array:
        ret = np.array(ret)

    return ret

def replace_none_with_empty_iter(iterator):
    """ If it is "None", return an empty iterator; otherwise, return iterator.

    The purpose of this function is to make iterating over results from
    functions which return either an iterator or None cleaner.

    Parameters
    ----------
    it: None or some object

    Returns
    -------
    empty_iterator: list of size 0
        If iterator is None
    --- OR ---
    iterator:
        The original iterator, if it was not None
    """
    if iterator is None:
        return []
    return iterator

def open(filename, mode='r', compress=False, is_text=True):
    """ This function returns a file handle to the given file. The only
        difference between this and the standard open command is that this
        function transparently opens zip files, if specified. If a gzipped
        file is to be opened, the mode is adjusted according to the "is_text"
        flag.

        Args:
            filename (str): the file to open

            mode (str): the mode to open the file. This *should not* include
                "t" for opening gzipped text files. That is handled by the
                "is_text" flag.

            compress (bool): whether to open the file as a gzipped file

            is_text (bool): for gzip files, whether to open in text (True) or
                binary (False) mode

        Returns:
            file_handle: the file handle to the file

        Imports:
            gzip, if compress is True
    """
    import builtins

    if compress:
        import gzip

        if is_text:
            mode = mode + "t"
        out = gzip.open(filename, mode)
    else:
        out = builtins.open(filename, mode)

    return out


def grouper(n, iterable):
    """ This function returns lists of size n of elements from the iterator. It
        does not pad the last group.

        The code was directly take from stackoverflow:
            http://stackoverflow.com/questions/3992735/

    """
    iterable = iter(iterable)
    return iter(lambda: list(itertools.islice(iterable, n)), [])


def nth(iterable, n, default=None):
    """ Returns the nth item or a default value.

    This code is mildly adapted from the documentation.

    N.B. This returns the *base-0* nth item in the iterator. For example, 
    nth(range(10), 1) returns 1.
    """
    return next(itertools.islice(iterable, n, None), default)


def dict_product(dicts):
    """ Create an iterator from a GridSearchCV-like dictionary

    This code is directly take from stackoverflow:
        http://stackoverflow.com/a/40623158/621449
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
