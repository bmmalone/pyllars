"""
This module contains helper functions for interacting with the shell and
file system.
"""
import logging
logger = logging.getLogger(__name__)

import contextlib
import os
import requests
import shlex
import shutil
import subprocess
import sys


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



def download_file(url, local_filename=None, chunk_size=1024, overwrite=False):
    """ Download the file at the given URL to the specified local location.

    This function is adapted from: https://stackoverflow.com/questions/16694907

    Parameters
    ----------
    url: string
        The location of the file to download, including the protocol

    local_filename: string path
        The *complete* path (including filename) to the local file. Default:
        the current working directory, plus the filename of the downloaded file

    chunk_size: int
        The chunk size to use for donwload. Please see requests.iter_content
        for more details. It is unlikely this value should be changed.

    overwrite: bool
        Whether to overwrite an existing file at local_filename

    Returns
    -------
    local_filename: string
        The name of the local file

    Raises
    ------
    FileExistsError, if a file exists at local_filename and overwrite is
        False.
    """
    if local_filename is None:
        local_filename = url.split('/')[-1]
    
    # check on the local file
    if os.path.exists(local_filename):
        msg = ("[shell_utils.download_file]: The local file exists. {}".format(
            local_filename))
        if overwrite:
            # we don't care, but write a message anyway
            logger.warning(msg)
        else:
            raise FileExistsError(msg)

    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=chunk_size): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return local_filename

def check_programs_exist(programs, raise_on_error=True, package_name=None, 
            logger=logger):

    """ This function checks that all of the programs in the list cam be
        called from python. After checking all of the programs, an exception
        is raised if any of them are not callable. Optionally, only a warning
        is raised. The name of the package from which the programs are
        available can also be included in the message.

        Internally, this program uses shutil.which, so see the documentation
        for more information about the semantics of calling.

        Arguments:
            programs (list of string): a list of programs to check

        Returns:
            list of string: a list of all programs which are not found

        Raises:
            EnvironmentError: if any programs are not callable, then
                an error is raised listing all uncallable programs.
    """

    missing_programs = []
    for program in programs:
        exe_path = shutil.which(program)

        if exe_path is None:
            missing_programs.append(program)

    if len(missing_programs) > 0:
        missing_programs = ' '.join(missing_programs)
        msg = "The following programs were not found: " + missing_programs

        if package_name is not None:
            msg = msg + ("\nPlease ensure the {} package is installed."
                .format(package_name))

        if raise_on_error:
            raise EnvironmentError(msg)
        else:
            logger.warning(msg)

    return missing_programs


def check_call_step(cmd, current_step = -1, init_step = -1, call=True, 
        raise_on_error=True):
    
    logging.info(cmd)
    ret_code = 0

    if current_step >= init_step:
        if call:
            #logging.info(cmd)
            logging.info("calling")
            ret_code = subprocess.call(cmd, shell=True)

            if raise_on_error and (ret_code != 0):
                raise subprocess.CalledProcessError(ret_code, cmd)
            elif (ret_code != 0):
                msg = ("The command returned a non-zero return code\n\t{}\n\t"
                    "Return code: {}".format(cmd, ret_code))
                logger.warning(msg)
        else:
            msg = "skipping due to --do-not-call flag"
            logging.info(msg)
    else:
        msg = "skipping due to --init-step; {}, {}".format(current_step, init_step)
        logging.info(msg)

    return ret_code



def check_call(cmd, call=True, raise_on_error=True):
    return check_call_step(cmd, call=call, raise_on_error=raise_on_error)

def check_output_step(cmd, current_step = 0, init_step = 0, raise_on_error=True):

    logging.info(cmd)
    if current_step >= init_step:
        logging.info("calling")

        try:
            out = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError as exc:
            if raise_on_error:
                raise exc
        return out.decode()

def check_output(cmd, call=True, raise_on_error=True):
    current_step = 1
    init_step = 1
    if not call:
        init_step = 2
    return check_output_step(cmd, current_step, init_step, 
        raise_on_error=raise_on_error)

def call_if_not_exists(cmd, out_files, in_files=[], overwrite=False, call=True,
            raise_on_error=True, file_checkers=None, num_attempts=1, 
            to_delete=[], keep_delete_files=False):

    """ This function checks if out_file exists. If it does not, or if overwrite
        is true, then the command is executed, according to the call flag.
        Otherwise, a warning is issued stating that the file already exists
        and that the cmd will be skipped.

        Additionally, a list of input files can be given. If given, they must
        all exist before the call will be executed. Otherwise, a warning is 
        issued and the call is not made.

        However, if call is False, the check for input files is still made,
        but the function will continue rather than quitting. The command will
        be printed to the screen.

        The return code from the called program is returned.

        By default, if the called program returns a non-zero exit code, an
        exception is raised.

        Furthermore, a dictionary can be given which maps from a file name to
        a function which check the integrity of that file. If any of these
        function calls return False, then the relevant file(s) will be deleted
        and the call made again. The number of attempts to succeed is given as
        a parameter to the function.

        Args:
            cmd (string): the command to execute

            out_files (string or list of strings): path to the files whose existence 
                to check. If they do not exist, then the path to them will be 
                created, if necessary.

            in_files (list of strings): paths to files whose existence to check
                before executing the command

            overwrite (bool): whether to overwrite the file (i.e., execute the 
                command, even if the file exists)

            call (bool): whether to call the command, regardless of whether the
                file exists

            raise_on_error (bool): whether to raise an exception on non-zero 
                return codes

            file_checkers (dict-like): a mapping from a file name to a function
                which is used to verify that file. The function should return
                True to indicate the file is okay or False if it is corrupt. The
                functions must also accept "raise_on_error" and "logger" 
                keyword arguments.

            num_attempts (int): the number of times to attempt to create the
                output files such that all of the verifications return True.

            to_delete (list of strings): paths to files to delete if the command
                is executed successfully

            keep_delete_files (bool): if this value is True, then the to_delete
                files will not be deleted, regardless of whether the command
                succeeded

        Returns:
            int: the return code from the called program

        Warnings:
            warnings.warn if the out_file already exists and overwrite is False
            warnings.warn if the in_files do not exist

        Raises:
            subprocess.CalledProcessError: if the called program returns a
                non-zero exit code and raise_on_error is True
                
            OSError: if the maximum number of attempts is exceeded and the 
                file_checkers do not all return true and raise_on_error is True

        Imports:
            os
            shell
    """

    ret_code = 0

    # check if the input files exist
    missing_in_files = []
    for in_f in in_files:
        # we need to use shlex to ensure that we remove surrounding quotes in
        # case the file name has a space, and we are using the quotes to pass
        # it through shell
        in_f = shlex.split(in_f)[0]

        if not os.path.exists(in_f):
            missing_in_files.append(in_f)

    if len(missing_in_files) > 0:
        msg = "Some input files {} are missing. Skipping call: \n{}".format(missing_in_files, cmd)
        logger.warn(msg)
        return ret_code

        # This is here to create a directory structue using "do_not_call". In
        # hindsight, that does not seem the best way to do this, so it has been
        # removed.
        #if call:
        #    return


    # make sure we are working with a list
    if isinstance(out_files, str):
        out_files = [out_files]

    # check if the output files exist
    all_out_exists = False
    if out_files is not None:
        all_out_exists = all([os.path.exists(of) for of in out_files])

    all_valid = True
    if overwrite or not all_out_exists:
        attempt = 0
        while attempt < num_attempts:
            attempt += 1

            # create necessary paths
            if out_files is not None:
                [os.makedirs(os.path.dirname(x), exist_ok=True) for x in out_files]
            
            # make the call
            ret_code = check_call(cmd, call=call, raise_on_error=raise_on_error)

            # do not check the files if we are not calling anything
            if (not call) or (file_checkers is None):
                break

            # now check the files
            all_valid = True
            for filename, checker_function in file_checkers.items():
                msg = "Checking file for validity: {}".format(filename)
                logger.debug(msg)

                is_valid = checker_function(filename, logger=logger, 
                                raise_on_error=False)

                # if the file is not valid, then rename it
                if not is_valid:
                    all_valid = False
                    invalid_filename = "{}.invalid".format(filename)
                    msg = "Rename invalid file: {} to {}".format(filename, invalid_filename)
                    logger.warning(msg)

                    os.rename(filename, invalid_filename)

            # if they were all valid, then we are done
            if all_valid:
                break


    else:
        msg = "All output files {} already exist. Skipping call: \n{}".format(out_files, cmd)
        logger.warn(msg)

    # now, check if we succeeded in creating the output files
    if not all_valid:
        msg = ("Exceeded maximum number of attempts for cmd. The output files do "
            "not appear to be valid: {}".format(cmd))

        if raise_on_error:
            raise OSError(msg)
        else:
            logger.critical(msg)

    elif (not keep_delete_files):
        # the command succeeded, so delete the specified files
        for filename in to_delete:
            if os.path.exists(filename):
                msg = "Removing file: {}".format(filename)
                logger.info(cmd)
                
                os.remove(filename)

    return ret_code



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
    check_programs_exist(programs)

    if has_tar:
        cmd = "gunzip -c {} | tar t > /dev/null".format(filename)
    else:
        cmd = "gunzip -t {}".format(filename)

    ret = check_call_step(cmd, raise_on_error=False)

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
        logger.debug(msg)
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

    with contextlib.suppress(FileNotFoundError):
        os.remove(filename)