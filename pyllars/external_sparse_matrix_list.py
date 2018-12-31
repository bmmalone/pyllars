"""
This class stores a dense list of sparse matrices. For external memory 
usage, it writes each sparse matrix to a matrix market format file in a
directory. Then, that directory is gzipped to make the file a bit portable.
"""

import os
import shutil
import tarfile

import pyllars.utils as utils

import numpy as np
import scipy.io
import scipy.sparse

# tolerance value for checking equality
tolerance = 1e-10

class ExternalSparseMatrixList:
    def __init__(self, size, tarfile_contents=None, tarfile_names=None):
        self.sparse_matrix_list = [None] * size
        self.tarfile_contents = tarfile_contents
        self.tarfile_names = tarfile_names

    def __getitem__(self, key):
        # check if we should try to fetch this
        if (self.sparse_matrix_list[key] is None) and (self.tarfile_contents is not None):
            fn = '{}.mtx'.format(key)
            if fn in self.tarfile_names:
                f = self.tarfile_contents.extractfile(fn)
                m = scipy.io.mmread(f)
                self[key] = scipy.sparse.csr_matrix(m)
            else:
                self[key] = scipy.sparse.csr_matrix((1,0))

        return self.sparse_matrix_list[key]

    def __setitem__(self, key, item):
        self.sparse_matrix_list[key] = item

    def write(self,filename, overwrite=False):
        # first, strip the compression ending, if present
        filename = filename.replace('.tar.gz', '')
        filename = filename.replace('.tgz', '')

        # check that the path is safe
        if os.path.exists(filename) and not overwrite:
            raise OSError("Attempting to overwrite existing file: '{}'".format(filename))

        if os.path.exists(filename):
            if os.path.isfile(filename):
                os.remove(filename)
            else:
                shutil.rmtree(filename)

        # create the folder
        os.makedirs(filename)

        # first, write the metadata (just the size)
        fn = os.path.join(filename, "meta.txt")
        with open(fn, 'w') as f:
            f.write(str(len(self)))

        # write each matrix in MM format
        for i in range(len(self.sparse_matrix_list)):
            sm = self.sparse_matrix_list[i]
            if sm is None:
                continue
            fn = os.path.join(filename, str(i))
            scipy.io.mmwrite(fn, sm)

        # create the tgz file
        fn = '{}.tar.gz'.format(filename)
        tar = tarfile.open(fn, "w:gz")
        tar.add(filename, arcname='')
        tar.close()

        # remove the folder
        shutil.rmtree(filename)

        # and rename the tgz file
        os.rename(fn, filename)

    def __len__(self):
        return len(self.sparse_matrix_list)
    
    def __eq__(self, other):
        # make sure they are the same size
        if len(self) != len(other):
            return False
        
        # check that each element is equal
        for i in range(len(self)):
            diff = self[i] - other[i]
            if diff.nnz != 0:
                # check if the nonzero values exceed the tolerance
                nonzeros = diff[diff.nonzero()]
                nonzeros = np.squeeze(np.asarray(nonzeros))
                for nz in nonzeros:
                    if np.abs(nz) > tolerance:
                        return False
                        
        # the all elements are equal
        return True

    def sum(self):
        """ This method calculates the sum of all matrices in the list.

        Returns:
            dtype: the sum of all matrices
        """
        s = sum(matrix.sum() for matrix in self.sparse_matrix_list if matrix is not None)
        return s


    def max_len(self, axis=1):
        """ This method finds the maximum size of any matrix along the given 
            axis. Presumably, this is for use by the to_sparse_matrix method,
            but it could have other uses.
            
            Args:
                axis (int): 0 for rows, 1 for columns (i.e., matrix.shape[axis])

            Returns:
                int: the maximum length along the given axis of any matrix.
        """
        l = list(matrix.shape[axis] for matrix in self.sparse_matrix_list if matrix is not None)
        if len(l) == 0:
            return 0

        max_len = max(l)
        return max_len

    def to_sparse_matrix(self, min_cols=None):
        """ This method attempts to convert the sparse matrix list into a
            single sparse matrix. This operation iterates over the columns
            of the first row of each matrix. So it is only sensible if the
            matrix list consists of a set of column vectors.

            For some cases, the number of columns in the matrix should exceed
            the number of columns in the longest matrix. If this is the case,
            the number of columns can be specified.
            
            NOTE: This method does not do any type checking.

            Returns:
                scipy.sparse matrix: A sparse matrix in which each item in
                    the list is treated as a row in the matrix

                    OR

                    None, if the list contains no items
        """
        # create a 2d numpy array that contains all of the rows
        max_len = self.max_len()

        if (min_cols is not None) and (max_len < min_cols):
            max_len = min_cols

        # determine the dtype
        dtype = None
        for i in range(len(self)):
            if self[i] is not None:
                dtype = self[i].dtype
                break

        # make sure we found something
        if dtype is None:
            return None

        sparse_matrix = scipy.sparse.lil_matrix((len(self), max_len), dtype=dtype)
        
        # transfer all of the values over to the sparse array
        for i in range(len(self)):
            sv = self[i]
            if sv is None:
                continue

            sv = sv.tocoo()

            for j,k,v in zip(sv.row, sv.col, sv.data):
                sparse_matrix[i,k] = v

        return sparse_matrix 

def lazy_read_external_sparse_matrix_list(filename):

    # open the gzipped tar file
    contents = tarfile.open(filename, "r:gz")
    names = contents.getnames()

    # read the meta file
    fn = 'meta.txt'
    f = contents.extractfile(fn)
    size = int(f.readline())
    f.close()

    # create the container
    esml = ExternalSparseMatrixList(size, tarfile_contents=contents, tarfile_names=names)
    return esml

def read_external_sparse_matrix_list(filename):
    # open the gzipped tar file
    contents = tarfile.open(filename, "r:gz")
    names = contents.getnames()

    # read the meta file
    fn = 'meta.txt'
    f = contents.extractfile(fn)
    size = int(f.readline())
    f.close()

    # create the container
    esml = ExternalSparseMatrixList(size)

    # read in each sparse matrix
    for i in range(size):
        fn = '{}.mtx'.format(i)

        if fn in names:
            f = contents.extractfile(fn)
            m = scipy.io.mmread(f)
            esml[i] = scipy.sparse.csr_matrix(m)
        else:
            esml[i] = scipy.sparse.csr_matrix((1,0))

    return esml

def concatenate(lists):
    total_size = 0
    for l in lists:
        total_size += len(l)

    joined_list = ExternalSparseMatrixList(total_size)
    index = 0
    for i in range(len(lists)):
        l = lists[i]
        for j in range(len(l)):
            # convert just to make absolutely sure this is a sparse matrix
            joined_list[index] = scipy.sparse.lil_matrix(l[j])
            index += 1

    return joined_list

def to_sparse_matrix(list_of_sparse_row_vectors):
    """ This function converts a list of sparse row vectors, i.e., sparse
        matrices with a shape like (1, X), into a single sparse matrix. Each
        row of the resulting matrix corresponds to the respective sparse row
        vector in the input list.

        Args:
            list_of_sparse_row_matrices (list of scipy.sparse): a list of scipy
                sparse matrices with shape (1,X).

        Returns:
            scipy.sparse.lil_matrix: a sparse matrix in which the rows
                correspond to the vectors in the list. The dimension of the
                matrix is (N, Y), where N is the number of items in the list
                and Y is the largest size of any input matrix.

        Imports:
            logging

        Raises:
            ValueError: if any of the input matrices do not have a shape of
                the form (1, X)
    """
    import logging

    esml = ExternalSparseMatrixList(len(list_of_sparse_row_vectors))

    msg = "Copying sparse row vectors to sparse matrix list"
    logging.info(msg)

    for i, sparse_row_vector in enumerate(list_of_sparse_row_vectors):
        if sparse_row_vector.shape[0] != 1:
            msg = ("list_of_sparse_row_vectors[{}] did not have the correct shape. "
                "Expected something of the form (1,X); found: ({},{})".format(i,
                sparse_row_vector.shape[0], sparse_row_vector.shape[1]))
            raise ValueError(msg)

        esml[i] = sparse_row_vector

    msg = "Converting matrix list to single sparse matrix"
    logging.info(msg)

    sparse_matrix = esml.to_sparse_matrix()

    return sparse_matrix



def main():
    # create a random sparse matrix list
    esml = ExternalSparseMatrixList(3)
    for i in range(3):
        esml[i] = scipy.sparse.rand(8, 4, density=0.4)

    # write it to disk
    esml.write('test_esml', True)

    # read it back
    esml_read = read_external_sparse_matrix_list('test_esml')
    
    # make sure they are equal
    assert(esml_read == esml)

    print("TEST SUCCEEDED: The created sparse matrix list and the one read "
        "from disk are equal.")

if __name__ == '__main__':
    main()





