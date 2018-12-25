#! /usr/bin/env python3

import argparse
import misc.shell_utils as shell_utils

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script calls scancel for a given range of jobids (inclusive)")
    parser.add_argument('start', type=int, help="The beginning jobid")
    parser.add_argument('end', type=int, help="The ending jobid")
    args = parser.parse_args()

    for jobid in range(args.start, args.end+1):
        cmd = "scancel {}".format(jobid)
        shell_utils.check_call(cmd)

if __name__ == '__main__':
    main()
