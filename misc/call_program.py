#! /usr/bin/env python3

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Call a script. This is mostly used to submit binaries and "
        "other difficult programs to slurm.")

    parser.add_argument('cmd', nargs='+')
    
    args, unknown = parser.parse_known_args()

    cmd = ' '.join(sys.argv[1:])
    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()
