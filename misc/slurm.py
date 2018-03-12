"""
Utilities to make submitting scripts to slurm easier. These functions mostly
add standard slurm options to an `argparse.ArgumentParser`, parse them, and
direct the execution to slurm as specified.
"""

import logging

import os
import subprocess
import sys

import misc.utils as utils
import misc.shell_utils as shell_utils

logger = logging.getLogger(__name__)

def check_srun(cmd, call=True):
    cmd = 'srun {}'.format(cmd)
    shell_utils.check_call(cmd, call)

def check_sbatch(cmd, call=True, num_cpus=1, mem="2G", time=None, 
        partitions=None, dependencies=None, no_output=False, no_error=False, 
        use_slurm=False, mail_type=['FAIL', 'TIME_LIMIT'], mail_user=None,
        stdout_file=None, stderr_file=None,
        args=None):

    """ This function wraps calls to sbatch. It adds the relevant command line 
        options based on the parameters (either specified or extracted from 
        args, if args is not None).

        The 'ntasks' option is always 1 with the function.

        Args:

            cmd (str): The command to execute

            call (bool): If this flag is false, then the commands will not be
                executed (but will be logged).
            
            num_cpus (int): The number of CPUs to use. This will be translated into 
                an sbatch request like: "--ntasks 1 --cpus-per-task <num-cpus>". 
                default: 1

            mem (str): This will be translated into an sbatch request like: 
                "--mem=<mem>". default: 10G

            time (str): The amount of time to request. This will be translated 
                into an sbatch request like: "--time <time>". default: 0-05:59

            partitions (str): The partitions to request. This will be translated 
                into an sbatch request like: "-p <partitions>". default: general 
                (N.B. This value should be a comma-separated list with no spaces, 
                for example: partitions="general,long")

            dependencies (list of int-likes): A list of all of the job ids to
                use as dependencies for this call. This will be translated into
                an sbatch request like: "--dependency=afterok:<dependencies>".
                default: None (i.e., no dependencies)

                N.B. This IS NOT overwritten by args.
           
            no_output (bool): If this flag is True, stdout will be redirected 
                to /dev/null. This will be translated into an sbatch request 
                like: "--output=/dev/null". default: If the flag is not present, 
                then stdout will be directed to a log file with the job number. 
                This corresponds to "--output=slurm-%J.out" in the sbatch call.

            stdout_file (str): If this value is given and no_output is False,
                then this filename will be used for stdout rather than 
                slurm-%J.out. This corresponds to "--output=<stdout_file>" in 
                the sbatch call.

            no_error (bool): If this flag is True, stderr will be redirected 
                to /dev/null. This will be translated into an sbatch request 
                like: "--error=/dev/null". default: If the flag is not present, 
                then stderr will be directed to a log file with the job number. 
                This corresponds to "--error=slurm-%J.err" in the sbatch call.

            stderr_file (str): If this value is given and no_output is False,
                then this filename will be used for stderr rather than 
                slurm-%J.err. This corresponds to "--output=<stdout_file>" in 
                the sbatch call.


            use_slurm (bool): If this flag is True, then the commands will be 
                submitted to SLURM via sbatch. default: By default, each command 
                is executed sequentially within the current terminal.

            mail_type (list of strings): A list of the types of mail to send.
                This will be translated into an sbatch request like: 
                "--mail-type type_1,type_2,...". default: ['FAIL', 'TIME_LIMIT']

            mail_user (string): The email address (or user name if that is 
                configured) of the recipient of the mails. This is translated
                into an sbatch request like: "--mail-user <user>"

            args (namespace): A namespace which contains values for all of the 
                options (i.e., created from an argparse parser after calling
                add_sbatch_options on the parser)

        Returns:
            If use_slurm is False, None

            If use_slurm is True, the slurm job id
    """

    # use args if they are present
    if args is not None:
        call = not args.do_not_call
        num_cpus = args.num_cpus
        mem = args.mem
        time = args.time
        partitions = args.partitions
        no_output = args.no_output
        no_error = args.no_error
        use_slurm = args.use_slurm
        mail_type = args.mail_type
        mail_user = args.mail_user
        stdout_file = args.stdout_file
        stderr_file = args.stderr_file

    output_str = "--output=slurm-%J.out"
    if stdout_file is not None:
        output_str = "--output={}".format(stdout_file)
    if no_output:
        output_str = "--output=/dev/null"

    error_str = "--error=slurm-%J.err" 
    if stderr_file is not None:
        error_str = "--error={}".format(stderr_file)
    if no_error:
        error_str = "--error=/dev/null"

    dependencies_str = ""
    if dependencies is not None:
        dependencies_str = ':'.join(str(d) for d in dependencies)
        dependencies_str = "--dependency=afterok:{}".format(dependencies_str)

    # check if we actually want to use SLURM
    msg = "slurm.check_sbatch.use_slurm: {}, call: {}".format(use_slurm, call)
    logger.debug(msg)

    # anyway, make sure to remove the --use-slurm option
    cmd = cmd.replace("--use-slurm", "")
    
    if use_slurm:
        time_str = ""
        if time is not None:
            time_str = "--time {}".format(time)

        mem_str = ""
        if mem is not None:
            mem_str = "--mem={}".format(mem)

        partitions_str = ""
        if partitions is not None:
            partitions_str = "-p {}".format(partitions)

        num_cpus_str = ""
        if num_cpus is not None:
            num_cpus_str = "--cpus-per-task {}".format(num_cpus)

        mail_type_str = ""
        if mail_type is not None:
            mail_type_str = "--mail-type {}".format(','.join(mail_type))

        mail_user_str = ""
        if mail_user is not None:
            mail_user_str = "--mail-user {}".format(mail_user)
        else:
            # if we did not give a mail user, then do not specify the mail types
            mail_type_str = ""

        cmd = ("sbatch {} {} --ntasks 1 {}  {} "
            "{} {} {} {} {} {}".format(time_str, mem_str, partitions_str, num_cpus_str, dependencies_str, 
            output_str, error_str, mail_type_str, mail_user_str, cmd))

        output = shell_utils.check_output(cmd, call=call)

        # and parse out the job id
        if call:
            job_id = output.strip().split()[-1]
        else:
            job_id = None
        return job_id
    else:
        shell_utils.check_call(cmd, call=call)
        return None

mail_type_choices = [
    'NONE', 
    'BEGIN', 
    'END', 
    'FAIL', 
    'REQUEUE', 
    'ALL',
    'STAGE_OUT',
    'TIME_LIMIT', 
    'TIME_LIMIT_90',
    'TIME_LIMIT_80',
    'TIME_LIMIT_50',
    'ARRAY_TASKS'
]

def add_sbatch_options(parser, num_cpus=1, mem="2G", time=None, 
            stdout_file=None, stderr_file=None,
            partitions=None, mail_type=['FAIL', 'TIME_LIMIT'], mail_user=None):

    """ This function adds the options for calling sbatch to the given parser.
        The provided arguments are used as defaults for the options.

        Args:
            parser (argparse.ArgumentParser): the parser to which the
                options will be added.

            other arguments: the defaults for the sbatch options

        Returns:
            None, but parser is updated
    """
    slurm_options = parser.add_argument_group("slurm options")

    slurm_options.add_argument('--num-cpus', help="The number of CPUs to use", type=int, 
        default=num_cpus)
    slurm_options.add_argument('--mem', help="The amount of RAM to request", default=mem)
    slurm_options.add_argument('--time', help="The amount of time to request", default=time)
    slurm_options.add_argument('--partitions', help="The partitions to request", default=partitions)
    slurm_options.add_argument('--no-output', help="If this flag is present, stdout "
        "will be redirected to /dev/null", action='store_true')
    slurm_options.add_argument('--no-error', help="If this flag is present, stderr "
        "will be redirected to /dev/null", action='store_true')
    slurm_options.add_argument('--stdout-file', help="If this is present and the "
        "--no-output flag is not given, then stdout will be directed to this "
        "file rather than slurm-<job>.out", default=stdout_file)
    slurm_options.add_argument('--stderr-file', help="If this is present and the "
        "--no-error flag is not given, then stderr will be directed to this "
        "file rather than slurm-<job>.err", default=stderr_file)
    slurm_options.add_argument('--do-not-call', help="If this flag is present, then the commands "
        "will not be executed (but will be printed).", action='store_true')
    slurm_options.add_argument('--use-slurm', help="If this flag is present, the program calls "
        "will be submitted to SLURM.", action='store_true')
    slurm_options.add_argument('--mail-type', help="When to send an email notifcation "
        "of the job status. See official documentation for a description of the "
        "values. If a mail-user is not specified, this will revert to 'None'.", 
        nargs='*', choices=mail_type_choices, default=mail_type)
    slurm_options.add_argument('--mail-user', help="To whom an email will be sent, in "
        "accordance with mail-type", default=mail_user)

def get_slurm_options_string(args):
    """ This function extracts the flags and options specified for slurm/sbatch options
        added with add_sbatch_options. Presumably, this is used in a high-level "process-all"
        script where we need to pass the slurm options to the "process-all" script.

        Args:
            args (namespace): a namespace with the arguments added by add_sbatch_options

        Returns:
            string: a string containing all sbatch flags and options

    """

    args_dict = vars(args)

    # first, pull out the text arguments
    sbatch_options = ['num_cpus', 'mem', 'time', 'partitions', 'mail_user', 
        'stdout_file', 'stderr_file']

    # create a new dictionary mapping from the flag to the value
    sbatch_flags_and_vals = {'--{}'.format(o.replace('_', '-')) : args_dict[o] 
        for o in sbatch_options if args_dict[o] is not None}

    # and we need to handle mail-type
    mail_types = " ".join(mt for mt in args_dict['mail_type'])
    mail_types = "--mail-type {}".format(mail_types)

    s = ' '.join("{} '{}'".format(k,v) for k,v in sbatch_flags_and_vals.items())

    s = "{} {}".format(mail_types, s)

    # and check the flags
    if args.no_output:
        s = "--no-output {}".format(s)

    if args.no_error:
        s = "--no-error {}".format(s)

    if args.do_not_call:
        s = "--do-not-call {}".format(s)

    if args.use_slurm:
        s = "--use-slurm {}".format(s)

    return s
