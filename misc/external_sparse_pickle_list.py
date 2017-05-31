#! /usr/bin/env python3

###
#   This class stores a dense list of sparse matrices. For external memory 
#   usage, it writes each sparse matrix to a matrix market format file in a
#   directory. Then, that directory is gzipped to make the file a bit portable.
###

import os
import pickle
import shutil
import tarfile

import misc.utils as utils

# tolerance value for checking equality
tolerance = 1e-10

class ExternalSparsePickleList:
    def __init__(self, size, tarfile_contents=None):
        self.sparse_pkl_list = [None] * size
        if tarfile_contents is not None:
            self.tarfile_contents = tarfile_contents
            self.names = tarfile_contents.getnames()

    def __getitem__(self, key):
        if self.sparse_pkl_list[key] is None:
            fn = '{}.pkl'.format(key)
            if fn in self.names:
                f = self.tarfile_contents.extractfile(fn)
                m = pickle.load(f)
                self[key] = m

        return self.sparse_pkl_list[key]

    def __setitem__(self, key, item):
        self.sparse_pkl_list[key] = item

    def write(self,filename, overwrite=False):
        # first, strip the compression ending, if present
        filename = filename.replace('.tar.gz', '')
        filename = filename.replace('.tgz', '')

        # check that the path is safe
        if os.path.exists(filename) and not overwrite:
            raise OSError("Attempting to overwrite existing file: '{}'".format(filename))

        if os.path.exists(filename):
            shutil.rmtree(filename)

        # create the folder
        os.makedirs(filename)

        # first, write the metadata (just the size)
        fn = os.path.join(filename, "meta.txt")
        with open(fn, 'w') as f:
            f.write(str(len(self)))

        # write each matrix in MM format
        for i in range(len(self.sparse_pkl_list)):
            if self.sparse_pkl_list[i] is not None:
                fn = "{}.pkl".format(i)
                fn = os.path.join(filename, fn)
                with open(fn, 'wb') as f:
                    pickle.dump(self.sparse_pkl_list[i], f)

        # create the tgz file
        fn = '{}.tar.gz'.format(filename)
        tar = tarfile.open(fn, "w:gz")
        tar.add(filename, arcname='')
        tar.close()

        # finally, remove the folder
        shutil.rmtree(filename)

    def __len__(self):
        return len(self.sparse_pkl_list)

def lazy_read_external_sparse_pickle_list(filename):
    # check the extension
    if not filename.endswith('.tar.gz'):
        filename = '{}.tar.gz'.format(filename)

    # open the gzipped tar file
    contents = tarfile.open(filename, "r:gz")

    # read the meta file
    fn = 'meta.txt'
    f = contents.extractfile(fn)
    size = int(f.readline())
    f.close()

    # create the container
    espl = ExternalSparsePickleList(size, tarfile_contents = contents)
    return espl

def read_external_sparse_pickle_list(filename):
    # check the extension
    if not filename.endswith('.tar.gz'):
        filename = '{}.tar.gz'.format(filename)

    # open the gzipped tar file
    contents = tarfile.open(filename, "r:gz")

    # read the meta file
    fn = 'meta.txt'
    f = contents.extractfile(fn)
    size = int(f.readline())
    f.close()

    # create the container
    espl = ExternalSparsePickleList(size)

    # read in each pickle item
    for i in range(size):
        fn = '{}.pkl'.format(i)
        f = contents.extractfile(fn)
        m = pickle.load(f)
        espl[i] = m 
    return espl

def concatenate(lists):
    total_size = 0
    for l in lists:
        total_size += len(l)

    joined_list = ExternalSparsePickleList(total_size)
    index = 0
    for i in range(len(lists)):
        l = lists[i]
        for j in range(len(l)):
            # convert just to make absolutely sure this is a sparse matrix
            joined_list[index] = l[j]
            index += 1

    return joined_list

def main():
    # create a random sparse matrix list
    espl = ExternalSparsePickleList(3)

    espl[0] = "Hello, disk"
    espl[1] = 42
    espl[2] = ["Hello, disk", 42]

    # write it to disk
    espl.write('test_espl', True)

    # read it back
    espl_read = read_external_sparse_pickle_list('test_espl')
    
    # make sure they are equal
    assert(espl[0] == "Hello, disk")
    assert(espl[1] == 42)
    assert(espl[2][0] == "Hello, disk")
    assert(espl[2][1] == 42)

    print("TEST SUCCEEDED: The created sparse pickle list and the one read "
        "from disk are equal.")

if __name__ == '__main__':
    main()





