#! /usr/bin/env python3

import argparse
import pickle
from pystan import StanModel

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script compiles a stan model and pickles it to disc")
    parser.add_argument("stan", help="The stan model file")
    parser.add_argument("out", help="The python3 pickle output file")
    args = parser.parse_args()

    sm = StanModel(file=args.stan)
    with open(args.out, 'wb') as f:
        pickle.dump(sm, f)

if __name__ == '__main__':
    main()
