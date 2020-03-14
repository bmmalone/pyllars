"""
Utilities for interacting with the python logging module. Mostly, this module
provides functions for easily adding command line options to an
`argparse.ArgumentParser` and then setting logging parameters accordingly.

More details and examples for logging are given in the python documentation:

* **Introduction**: https://docs.python.org/3/howto/logging.html
* **Format string options**: https://docs.python.org/3/library/logging.html#logrecord-attributes
"""

import argparse
import logging
import shlex
import sys

LOGGING_LEVELS = ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

_DEFAULT_LOG_FILE = ""
_DEFAULT_LOGGING_LEVEL = 'WARNING'
_DEFAULT_SPECIFIC_LOGGING_LEVEL = 'NOTSET'

def add_logging_options(parser:argparse.ArgumentParser,
        default_log_file:str=_DEFAULT_LOG_FILE,
        default_logging_level:str=_DEFAULT_LOGGING_LEVEL,
        default_specific_logging_level:str=_DEFAULT_SPECIFIC_LOGGING_LEVEL)->None:
    """ Add options for controlling logging to an argument parser.
    
    In particular, it adds options for logging to a file, stdout and
    stderr. In addition, it adds options for controlling the logging
    level of each of the loggers, and a general option for controlling
    all of the loggers.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        An argument parser
        
    default_log_file : str
        The default for the `--log-file` flag

    default_logging_level : str
        The default for the `--logging-level` flag

    default_specific_logging_level : str
        The default for the `--{file,stdout,stderr}-logging-level` flags
        
    Returns
    -------
    None : None
        The parser has the additional options added
    """

    logging_options = parser.add_argument_group("logging options")

    logging_options.add_argument('--log-file', help="This option specifies a file to "
        "which logging statements will be written (in addition to stdout and "
        "stderr, if specified)", default=default_log_file)
    logging_options.add_argument('--log-stdout', help="If this flag is present, then "
        "logging statements will be written to stdout (in addition to a file "
        "and stderr, if specified)", action='store_true')
    logging_options.add_argument('--no-log-stderr', help="Unless this flag is present, then "
        "logging statements will be written to stderr (in addition to a file "
        "and stdout, if specified)", action='store_true')

    logging_options.add_argument('--logging-level', help="If this value is specified, "
        "then it will be used for all logs", choices=LOGGING_LEVELS,
        default=default_logging_level)
    logging_options.add_argument('--file-logging-level', help="The logging level to be "
        "used for the log file, if specified. This option overrides "
        "--logging-level.", choices=LOGGING_LEVELS, 
        default=default_specific_logging_level)
    logging_options.add_argument('--stdout-logging-level', help="The logging level to be "
        "used for the stdout log, if specified. This option overrides "
        "--logging-level.", choices=LOGGING_LEVELS, 
        default=default_specific_logging_level)
    logging_options.add_argument('--stderr-logging-level', help="The logging level to be "
        "used for the stderr log, if specified. This option overrides "
        "--logging-level.", choices=LOGGING_LEVELS, 
        default=default_specific_logging_level)

def add_logging_values_to_args(
        args:argparse.Namespace,
        log_file:str=_DEFAULT_LOG_FILE,
        log_stdout:bool=False,
        no_log_stderr:bool=False,
        logging_level:str=_DEFAULT_LOGGING_LEVEL,
        file_logging_level:str=_DEFAULT_SPECIFIC_LOGGING_LEVEL,
        stdout_logging_level:str=_DEFAULT_SPECIFIC_LOGGING_LEVEL,
        stderr_logging_level:str=_DEFAULT_SPECIFIC_LOGGING_LEVEL) -> None:
    """ Add the options from `add_logging_options` to `args`

    This is intended for use in notebooks or other settings where the logging
    option functionality is required, but a command line interface is not
    used.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace to which the options will be added

    log_file : str
        The path to a log file. If this is the empty string, then a log file
        will not be used.

    log_stdout : bool
        Whether to log to stdout

    no_log_stderr : bool
        Whether to _not_ log to stderr. So, if this is `True`, then logging
        statements _will_ be written to stderr. (The negative is used because
        that is more natural for the command line arguments.)

    logging_level : str
        The logging level for all loggers

    {file,stdout,stderr}_logging_level : str
        The logging level for the specific loggers. This overrides
        `logging_level` for the respective logger when given.

    Returns
    -------
    None : None
        The respective options will be set on the namespace
    """
    args.log_file = log_file
    args.log_stdout = log_stdout
    args.no_log_stderr = no_log_stderr
    args.logging_level = logging_level
    args.file_logging_level = file_logging_level
    args.stdout_logging_level = stdout_logging_level
    args.stderr_logging_level = stderr_logging_level


def get_logging_options_string(args:argparse.Namespace) -> str:
    """ Extract the flags and options specified for logging from
    the parsed arguments and join them as a string.
    
    Presumably, these were added with `add_logging_options`. Compared
    to `get_logging_cmd_options`, this function returns the arguments
    as a single long string. Thus, they are suitable for use when
    building single strings to pass to the command line (such as with
    `subprocess.run` when `shell` is `True`).
    
    Parameters
    -----------
    args : argparse.Namespace
        The parsed arguments
        
    Returns
    -------
    logging_options_str : str
        A string containing all logging flags and options
    """        
    logging_options = get_logging_cmd_options(args)
    logging_options_str = ' '.join(logging_options)

    return logging_options_str


def get_logging_cmd_options(args:argparse.Namespace) -> str:
    """ Extract the flags and options specified for logging from
    the parsed arguments.
    
    Presumably, these were added with `add_logging_options`. Compared
    to `get_logging_options_string`, this function returns the arguments
    as an array. Thus, they are suitable for use with `subprocess.run`
    and similar functions.
    
    Parameters
    -----------
    args : argparse.Namespace
        The parsed arguments
        
    Returns
    -------
    logging_options : typing.List[str]
        The list of logging options and their values.
    """
    
    args_dict = vars(args)

    # first, pull out the text arguments
    logging_options = ['log_file', 'logging_level', 'file_logging_level',
        'stdout_logging_level', 'stderr_logging_level']

    # create a list of command line arguments
    ret = []
    
    for o in logging_options:
        if len(args_dict[o]) > 0:
            ret.append('--{}'.format(o.replace('_', '-')))
            ret.append(args_dict[o])
    
    if args.log_stdout:
        ret.append("--log-stdout")
        
    if args.no_log_stderr:
        ret.append("--no-log-stderr")
        
    ret = [shlex.quote(c) for c in ret]
    
    return ret

    

def set_logging_values(**kwargs) -> None:
    """ Set the logging options for the default logger as given

    This is intended for use in tests or other cases where a CLI is not
    easily available.

    Parameters
    ----------
    kwargs : key=value pairs
        These are passed unaltered to `add_logging_values_to_args`. Please see
        that documentation for details on valid options and their effect.

    Returns
    -------
    None : None
        The respective options will be set for the default logger
    """
    args = argparse.Namespace()
    add_logging_values_to_args(args, **kwargs)
    update_logging(args)

def update_logging(args, logger=None, 
        format_str='%(levelname)-8s %(name)-8s %(asctime)s : %(message)s'):
    """ Update `logger` to use the settings in `args`
    
    Presumably, the logging options were added with `add_logging_options`.

    Parameters
    ----------
    args: argparse.Namespace
        A namespace with the arguments added by add_logging_options

    logger: typing.Optional[logging.Logger]
        The logger which will be updated. If `None` is given, then the default
        logger will be updated.

    format_str: str
        The logging format string. Please see the python logging documentation
        for examples and more description.

    Returns
    -------
    None, but the default (or given) logger is updated to take into account
        the specified logging options
    """

    # find the root logger if another logger is not specified
    if logger is None:
        logger = logging.getLogger('')
            
    logger.handlers = []

    # set the base logging level
    level = logging.getLevelName(args.logging_level)
    logger.setLevel(level)

    # now, check the specific loggers

    if len(args.log_file) > 0:
        h = logging.FileHandler(args.log_file)
        formatter = logging.Formatter(format_str)
        h.setFormatter(formatter)
        if args.file_logging_level != 'NOTSET':
            l = logging.getLevelName(args.file_logging_level)
            h.setLevel(l)
        logger.addHandler(h)

    if args.log_stdout:
        h = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(format_str)
        h.setFormatter(formatter)
        if args.stdout_logging_level != 'NOTSET':
            l = logging.getLevelName(args.stdout_logging_level)
            h.setLevel(l)
        logger.addHandler(h)

    log_stderr = not args.no_log_stderr
    if log_stderr:
        h = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(format_str)
        h.setFormatter(formatter)
        if args.stderr_logging_level != 'NOTSET':
            l = logging.getLevelName(args.stderr_logging_level)
            h.setLevel(l)
        logger.addHandler(h)

def get_ipython_logger(logging_level='DEBUG', format_str='%(levelname)-8s : %(message)s'):
    """ Get a logger for use in jupyter notebooks
    
    This function is useful because the default logger in notebooks
    has a number of handlers by default. This function removes those,
    so the logger behaves as expected.
    
    Parameters
    ----------
    logging_level : str
        The logging level for the logger. This can be updated later.
        
    format_str : str
        The logging format string. Please see the python logging documentation
        for examples and more description.
        
    Returns
    -------
    logger : logging.Logger
        A logger suitable for use in a notebook
    """

    level = logging.getLevelName(logging_level)
    formatter = logging.Formatter(format_str)
    
    logger = logging.getLogger()
    logger.setLevel(level)

    # clear whatever handlers were there
    while len(logger.handlers):
        logger.removeHandler(logger.handlers[0])
    
    #h_out = logging.StreamHandler(sys.stdout)
    h_err = logging.StreamHandler(sys.stderr)

    #logger.addHandler(h_out)
    logger.addHandler(h_err)

    for h in logger.handlers:
        h.setFormatter(formatter)

    return logger


