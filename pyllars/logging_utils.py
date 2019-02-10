"""
Utilities for interacting with the python logging module. Mostly, this module
provides functions for easily adding command line options to an
`argparse.ArgumentParser` and then setting logging parameters accordingly.
"""

import logging
import sys


def add_logging_options(parser, default_log_file=""):
    """ This function add options for logging to an argument parser. In 
        particular, it adds options for logging to a file, stdout and stderr.
        In addition, it adds options for controlling the logging level of each
        of the loggers, and a general option for controlling all of the loggers.

        Args:
            parser (argparse.ArgumentParser): an argument parser

        Returns:
            None, but the parser has the additional options added
    """

    logging_options = parser.add_argument_group("logging options")

    default_log_file = ""
    logging_level_choices = ['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    default_logging_level = 'WARNING'
    default_specific_logging_level = 'NOTSET'

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
        "then it will be used for all logs", choices=logging_level_choices,
        default=default_logging_level)
    logging_options.add_argument('--file-logging-level', help="The logging level to be "
        "used for the log file, if specified. This option overrides "
        "--logging-level.", choices=logging_level_choices, 
        default=default_specific_logging_level)
    logging_options.add_argument('--stdout-logging-level', help="The logging level to be "
        "used for the stdout log, if specified. This option overrides "
        "--logging-level.", choices=logging_level_choices, 
        default=default_specific_logging_level)
    logging_options.add_argument('--stderr-logging-level', help="The logging level to be "
        "used for the stderr log, if specified. This option overrides "
        "--logging-level.", choices=logging_level_choices, 
        default=default_specific_logging_level)

def get_logging_options_string(args):
    """ This function extracts the flags and options specified for logging options
        added with add_logging_options. Presumably, this is used in "process-all"
        scripts where we need to pass the logging options to the "process" script.

        Args:
            args (namespace): a namespace with the arguments added by add_logging_options

        Returns:
            string: a string containing all logging flags and options

    """

    args_dict = vars(args)

    # first, pull out the text arguments
    logging_options = ['log_file', 'logging_level', 'file_logging_level',
        'stdout_logging_level', 'stderr_logging_level']

    # create a new dictionary mapping from the flag to the value
    logging_flags_and_vals = {'--{}'.format(o.replace('_', '-')) : args_dict[o] 
        for o in logging_options if len(args_dict[o]) > 0}

    s = ' '.join("{} {}".format(k,v) for k,v in logging_flags_and_vals.items())

    # and check the flags
    if args.log_stdout:
        s = "--log-stdout {}".format(s)

    if args.no_log_stderr:
        s = "--no-log-stderr {}".format(s)

    return s

def update_logging(args, logger=None, 
        format_str='%(levelname)-8s %(name)-8s %(asctime)s : %(message)s'):

    """ This function interprets the logging options in args. Presumably, these
        were added to an argument parser using add_logging_options.

    Parameters
    ----------
    args: argparse.Namespace
        a namespace with the arguments added by add_logging_options

    logger: logging.Logger or None
        a logger which will be updated. If None is given, then the default
        logger will be updated.

    format_str: string
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


