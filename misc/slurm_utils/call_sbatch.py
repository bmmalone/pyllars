#! /usr/bin/env python3

import argparse
import misc.slurm_utils.slurm_utils as slurm

default_mem = "10G"
default_num_cpus = 1
default_time = "0-05:59" # "1-00:00"
default_partitions = "general" # "hugemem,himem,blade"

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script wraps calls to binary executables. This is necessary for "
        "calling sbatch on binary executables from the command line.")
    
    parser.add_argument('cmd', help="The command to execute", nargs=argparse.REMAINDER)
    slurm.add_sbatch_options(parser, num_cpus=default_num_cpus, mem=default_mem, 
        time=default_time, partitions=default_partitions)

    args = parser.parse_args()

    cmd = ' '.join(args.cmd)

    slurm.check_sbatch(cmd, args=args)

if __name__ == '__main__':
    main()

