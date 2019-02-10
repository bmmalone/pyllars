"""
This module contains functions for distribution jobs to clusters over ssh,
and related ssh utilities.

These functions are largely wrappers around spur: 
    https://pypi.python.org/pypi/spur
"""

import logging
logger = logging.getLogger(__name__)

import os
import subprocess
import sys

import random
import socket
import spur
import paramiko.ssh_exception
import time

import pyllars.utils as utils


default_connection_timeout = 5
default_max_tries = 10

def distribute(cmd, node=None, 
                    node_list=None, 
                    connection_timeout=5, 
                    max_tries=10):

    """ This function assigns the given command to a node on a cluster.

    The node is either taken as the given node, or is one randomly selected out
    of the list. Passwordless ssh must be set up between the node from which the
    command is executed and the target node.

    Parameters
    ----------

    cmd : string
        The command to execute on the remote node

    node : string
        The name of a remote node

    node_list : string, or list of strings
        The path to a file containing node names, where each line contains one
        remote node name (and only that).

        Alternatively, this can be a list containing the node names.

    connection_timeout : int
        The number of seconds to wait to establish a connection

    max_tries : int
        The number of times to try to establish a connection

    Returns
    -------

    node : string
        The name of the node where the command is executing

    proc : spur.SshProcess
        A handle to the remotely-running process

    Raises
    ------
    spur.ssh.ConnectionEror
        If no connections can be made after the given number of tries.
        
    """
    cwd = os.getcwd()
    env = os.environ

    # read in possible nodes to use
    if not utils.is_sequence(node_list):
        with open(node_list) as node_list_file:
            node_list = [l.strip() for l in node_list_file]

    # if a node was not specified, pick one at random
    if node is None:
        node = random.choice(node_list)

    # now, try to distribute the work somewhere
    try_num = 0
    while try_num < max_tries:
        try:
            msg = "{}: {}".format(node, cmd)
            logger.info(msg)
            
            shell = spur.SshShell(hostname=node, 
                missing_host_key=spur.ssh.MissingHostKey.accept, 
                connect_timeout=connection_timeout)

            out = shell.spawn(cmd.split(), update_env=env, cwd=cwd)
            break
        except (spur.ssh.ConnectionError, paramiko.ssh_exception.SSHException, socket.error) as e:
            # if there is a connection problem, just skip it and try again
            msg = "{}: connection problem. skipping this node...".format(node)
            logger.warning(msg)
            logger.warning(e)
            node = random.choice(node_list)
            try_num += 1

    # check if we succeeded
    if try_num == max_tries:
        msg = "Unable to connect to any node after {} attempts".format(max_tries)
        raise spur.ssh.ConnectionError(msg)

    return (node, out)

def distribute_all(cmd_list, 
                    node_list=None, 
                    connection_timeout=5, 
                    max_tries=10,
                    ignore_errors=False,
                    pause=0):

    """ This function distributes all of the given commands across the cluster.

    This is just a light wrapper around the distribute function for sending many
    jobs to the cluster at once.

    Parameters
    ----------
    cmd_list : list-like of strings
        The commands to distribute

    ignore_errors: bool
        Whether to ignore (print a warning) or raise distribute errors

    pause: int
        A number of seconds to wait between issuing each call

    Other parameters are the same as in distribute

    Returns
    -------
    cmd_node_list : list-like of strings
        The node names to which each command was assigned

    proc_list : list-like of spur.SshProcess
        A list of the handles to the remotely-running processes

    Raises
    ------
    spur.ssh.ConnectionEror
        If no connections can be made after the given number of tries for any of the commands.

    """
    # if the node list is a file, read it
    if not utils.is_sequence(node_list):
        with open(node_list) as node_list_file:
            node_list = [l.strip() for l in node_list_file]

    cmd_node_list = []
    proc_list = []
    for cmd in cmd_list:
        time.sleep(pause)
        try:
            (node, proc) = distribute(cmd, 
                        node_list=node_list, 
                        connection_timeout=connection_timeout,
                        max_tries=max_tries)

            cmd_node_list.append(node)
            proc_list.append(proc)
        except spur.ssh.ConnectionError as e:
            if ignore_errors:
                logger.warning(e)
            else:
                raise e

    return (cmd_node_list, proc_list)

def wait_for_all_results(cmd_list, cmd_node_list, proc_list, log_results=True):
    """ Wait on a list of spur.SshProcesses to finish.

    This function waits until each of the spur.SshProcesses in the list finish.
    Optionally (by default), it logs the return code, stdout and stderr of
    each process. It just ignores any errors.

    Presumably, this is used in conjunction with distribute_all.

    Parameters
    ----------
    cmd_list : list-like of strings
        The commands to distribute

    cmd_node_list : list-like of strings
        The node names to which each command was assigned

    proc_list : list-like of spur.SshProcess
        A list of the handles to the remotely-running processes

    log_results : bool
        Whether to log (level INFO) the results for each command as it is received

    Returns
    -------

    return_codes : list of ints
        The return code of each command

    stdouts: list of strings
        The stdout of each command

    stderrs: list of strings
        The stderr of each command

    """
    import spur.results

    return_codes = []
    stdouts = []
    stderrs = []

    for i in range(len(proc_list)):
        try:
            msg = "Waiting on: {}: {}".format(cmd_node_list[i], cmd_list[i])
            logger.info(msg)

            res = proc_list[i].wait_for_result()

            return_codes.append(res.return_code)
            stdouts.append(res.output)
            stderrs.append(res.stderr_output)

            if  log_results:
                msg = "return code: '{}'".format(res.return_code)
                logger.info(msg)

                msg = "stdout: '{}'".format(res.output)
                logger.info(msg)

                msg = "stderr: '{}'".format(res.stderr_output)
                logger.info(msg)

        except spur.results.RunProcessError as err:

            msg = "There was an error\n\tNode: {}\n\t{}".format(cmd_node_list[i], 
                cmd_list[i])
            logger.warning(msg)
            
            return_codes.append(err.return_code)
            stdouts.append(err.output)
            stderrs.append(err.stderr_output)

            if  log_results:
                msg = "return code: '{}'".format(err.return_code)
                logger.info(msg)

                msg = "stdout: '{}'".format(err.output)
                logger.info(msg)

                msg = "stderr: '{}'".format(err.stderr_output)
                logger.info(msg)

    return (return_codes, stdouts, stderrs)


def write_results_to_file(out_filename, commands, node_list, return_codes, stdouts, stderrs):
    """ Print the output of wait_for_all_results to a file.

    Parameters
    ----------
    out_filename: string
        Path to the file

    commands: list of strings
        The commands distributed to the cluster

    nodel_list: list of strings
        The nodes to which each command was sent, given by 
        ssh_utils.distribute_all

    return_codes, stdouts, stderrs: lists of strings
        The strings to write to disk. Presumably, the values returned from 
        wait_for_all_results.

    Returns
    -------
    None, but the file is (overwritten) with the text
    """
    with open(out_filename, 'w') as out:
        for i in range(len(return_codes)):
            out.write(commands[i])
            out.write("\n")
            out.write(node_list[i])
            out.write("\n")
            out.write("return code: {}".format(return_codes[i]))
            out.write("\n")
            out.write("stdout: {}".format(stdouts[i]))
            out.write("\n")
            out.write("stderr: {}".format(stderrs[i]))
            out.write("\n")


def add_ssh_options(parser,
    default_node=None, 
    default_node_list=None, 
    default_connection_timeout=5, 
    default_max_tries=10):

    """ This function adds standard ssh command line options to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the argparse parser

    all others
        the default options to use in the parser

    Returns
    -------
    None, but the ssh options are added to the parser
    """

    ssh_options = parser.add_argument_group("SSH options")

    ssh_options.add_argument('--node', help="The node to which ssh will distribute "
        "jobs by default", default=default_node)

    ssh_options.add_argument('--node-list', help="A file containing a list of node "
        "names (one on each line, and only the node name) to which jobs will be "
        "randomly distributed", default=default_node_list)

    ssh_options.add_argument('--connection-timeout', help="The number of seconds to "
        "wait to establish a connection", type=int, 
        default=default_connection_timeout)

    ssh_options.add_argument('--max-tries', help="The number of times to try to "
        "establish a connection", type=int, default=default_max_tries)

