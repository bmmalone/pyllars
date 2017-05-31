#! /usr/bin/env python3

import argparse
import gzip
import hashlib
import os

import tqdm
import misc.utils as utils

import logging
import misc.logging_utils as logging_utils
logger = logging.getLogger(__name__)

default_blocksize = '4k'
default_num_iterations = 30

default_read_file = "read-me.gz"
default_write_file = "write-me.gz"

def write_gzip_file(fname, blocksize_bytes, num_blocks):
    with gzip.open(fname, 'wb') as fout:
        for b in tqdm.trange(num_blocks):
            fout.write(os.urandom(blocksize_bytes))

def get_gzip_checksum(fname, hasher, blocksize_bytes):    
    with gzip.open(fname, 'rb') as fin:

        buf = fin.read(blocksize_bytes)
        while len(buf) > 0:
            hasher.update(buf)
            buf = fin.read(blocksize_bytes)
        return hasher.digest()

def copy_gzip_file(fname_in, fname_out, blocksize_bytes):    
    with gzip.open(fname_in, 'rb') as fin, gzip.open(fname_out, 'wb') as fout:

        buf = fin.read(blocksize_bytes)
        while len(buf) > 0:
            fout.write(buf)
            buf = fin.read(blocksize_bytes)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This script reads and writes a random gzipped file many "
        "times to disk. After each iteration, it calculates the checksum of "
        "the new file. It raises an OSError if the checksums do not match.")

    parser.add_argument('size', help="The size of the random binary file")

    parser.add_argument('--blocksize', help="The size of blocks for reading "
        "and writing", default=default_blocksize)
    parser.add_argument('--num-iterations', help="The number of iterations",
        type=int, default=default_num_iterations)
    parser.add_argument('--read-file', default=default_read_file)
    parser.add_argument('--write-file', default=default_write_file)
    
    logging_utils.add_logging_options(parser)
    args = parser.parse_args()
    logging_utils.update_logging(args)

    msg = "Converting human-readable sizes to bytes"
    logger.info(msg)

    size_bytes = utils.human2bytes(args.size)
    blocksize_bytes = utils.human2bytes(args.blocksize)
    num_blocks = size_bytes // blocksize_bytes

    msg = "Creating random gzipped file"
    logger.info(msg)
    write_gzip_file(args.read_file, blocksize_bytes, num_blocks)

    msg = "Calculating its checksum"
    logger.info(msg)
    checksum = get_gzip_checksum(args.read_file, hashlib.sha256(), blocksize_bytes)

    # now, repeatedly copy and get the checksum
    for attempt in tqdm.trange(args.num_iterations):
        new_checksum = get_gzip_checksum(args.read_file, hashlib.sha256(), blocksize_bytes)
        
        # check that it did not change
        if checksum != new_checksum:
            msg = "The old ({}) and new ({}) checksums are not the same".format(checksum, new_checksum)
            raise OSError(msg)
        else:
            checksum = new_checksum
            
        # now, copy the "in" file to "out"
        copy_gzip_file(args.read_file, args.write_file, blocksize_bytes)
            
        # rename "out" to "in" and start again
        os.rename(args.write_file, args.read_file)

    msg = "The checksums were always the same"
    logger.info(msg)

if __name__ == '__main__':
    main()
