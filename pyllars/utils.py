"""
A hodgepodge of utilities, most of which concern working with basic types.
"""

import logging
logger = logging.getLogger(__name__)

import collections
import itertools
import os
import shutil
import subprocess
import sys
import typing
import yaml

import numpy as np
import pandas as pd

from pyllars.deprecated_decorator import deprecated


### Parsing and writing utilities



def load_config(config, required_keys=None):
    """ Read in the config file, print a logging (INFO) statement and verify
    that the required keys are present
    """
    import pyllars.validation_utils as validation_utils

    msg = "Reading config file"
    logger.info(msg)

    try:
        config = yaml.full_load(open(config))
    except OSError as ex:
        logger.warning(ex)
        raise ex


    if required_keys is not None:
        validation_utils.check_keys_exist(config, required_keys)

    return config

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


_gzip_extensions = ('gz',)
_bzip2_extensions = ('bz2',)

def _guess_compression(filename):
    """ Guess the compression type of `fname` based on its extension.
    
    If not matching compression extensions are found, then this function
    guesses that the file name does not correspond to a compressed file.
    
    Compression type and extensions:
    
        * gzip: gz
        * bzip2 : bz2
        * no_compression: everything else
    
    Parameters
    ----------
    filename : string
        The name of the file
        
    Returns
    -------
    compression_type : string
        The compression type. See above for details about the value.
    """
    
    compression_type = 'no_compression'
    
    if filename.endswith(_gzip_extensions):
        compression_type = "gzip"
    elif filename.endswith(_bzip2_extensions):
        compression_type = "bzip2"
        
    return compression_type
    

def open_file(
        filename,
        mode='r',
        guess_compression=True,
        compression_type=None,
        is_text=True,
        *args, **kwargs):
    """ Return a file handle to the given file. 
    
    The main difference between this and the standard open command is that this
    function transparently opens zip files, if specified. If a gzipped file is
    to be opened, the mode is adjusted according to the "is_text" flag.

    Parameters
    ---------
    filename : string
        the file to open

    mode : string
        the mode to open the file. This *should not* include
        "t" for opening gzipped text files. That is handled by the
        "is_text" flag.
        
    guess_compression : bool
        Whether to guess the compression mode of the file.

    compression_type : string or None
        If given, then the file will be opened using the specified
        compression type. This overrides the `guess_compression`
        flag.Valid options are:
        
        * no_compression
        * gzip
        * zip2

    is_text : bool
        For zip files, whether to open in text (True) or binary
        (False) mode

    args, kwargs
        Additional arguments are passed to the call to open

    Returns
    -------
    file_handle: the file handle to the file
    """
    
    if compression_type is None:
        compression_type = _guess_compression(filename)
        
    if is_text:
        mode = mode + "t"
        
    if compression_type == 'gzip':        
        import gzip
        out = gzip.open(filename, mode, *args, **kwargs)
        
    elif compression_type == 'bzip2':        
        msg = "bzip2 file handling has not been tested."
        logger.warning(msg)
        
        import bz2            
        out = gzip.open(filename, mode, *args, **kwargs)
        
    elif compression_type == 'no_compression':
        out = open(filename, mode, *args, **kwargs)

    return out
