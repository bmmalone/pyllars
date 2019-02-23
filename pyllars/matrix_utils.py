""" Helpers for working with (sparse) 2d matrices
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np

import scipy.io
import scipy.sparse

from typing import List, Optional, Tuple

###
# Matrix operation helpers
###

def col_op(m, op):
    """Apply op to each column in the matrix."""
    return op(m, axis=0)

def col_sum(m):
    """Calculate the sum of each column in the matrix."""
    return  np.sum(m, axis=0)

def col_sum_mean(m:np.ndarray, return_var:bool=False) -> float:
    """ Calculate the mean of the sum of each column in the matrix.
    
    Optionally, the variances of the column sums can also be calculated.

    Parameters
    ----------
    m : numpy.ndarray
        The (2d) matrix

    var : bool
        Whether to calculate the variances

    Returns
    -------
    mean : float
        The mean of the column sums in the matrix

    variance : float
        If `return_var` is True, then the variance of the column sums
    """
    col_sums = col_sum(m)
    mean = np.mean(col_sums)

    ret = mean
    if return_var:
        var = np.var(col_sums)
        ret = mean, var

    return ret

def normalize_columns(matrix:np.ndarray) -> np.ndarray:
    """ Normalize the columns of the given (dense) matrix

    Parameters
    ----------
    m : numpy.ndarray
        The (2d) matrix

    Returns
    -------
    normalized_matrix : numpy.ndarray
        The matrix normalized such that all column sums are 1
    """
    col_sums = np.sum(np.abs(matrix), axis=0)
    matrix = np.divide(m, col_sums)
    return matrix

def row_op(m, op):
    """Apply op to each row in the matrix."""
    return op(m, axis=1)

def row_sum(m):
    """Calculate the sum of each row in the matrix."""
    return  np.sum(m, axis=1)

def row_sum_mean(m:np.ndarray, var:bool=False) -> float:
    """ Calculate the mean of the sum of each row in the matrix.
    
    Optionally, the variances of the row sums can also be calculated.

    Parameters
    ----------
    m : numpy.ndarray
        The (2d) matrix

    return_var : bool
        Whether to calculate the variances

    Returns
    -------
    mean : float
        The mean of the row sums in the matrix

    variance : float
        If `return_var` is `True`, then the variance of the row sums
    """
    row_sums = row_sum(m)
    mean = np.mean(row_sums)

    ret = mean
    if return_var:
        var = np.var(row_sums)
        ret = mean, var

    return ret

def normalize_rows(matrix:np.ndarray) -> np.ndarray:
    """ Normalize the rows of the given (dense) matrix

    Parameters
    ----------
    matrix : numpy.ndarray
        The (2d) matrix

    Returns
    -------
    normalized_matrix : numpy.ndarray
        The matrix normalized such that all row sums are 1
    """
    coef = np.abs(matrix).sum(axis=1)
    coef = coef[:,np.newaxis]
    matrix = np.divide(matrix, coef)
    return matrix

###
# Sparse matrix helpers
###
def get_dense_row(
        matrix:scipy.sparse.spmatrix,
        row:int,
        dtype=float,
        max_length:Optional[int]=None) -> np.ndarray:
    """ Extract `row` from the sparse `matrix`
    
    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        The sparse matrix
        
    row : int
        The 0-based row index
        
    dtype : type
        The base type of elements of `matrix`. This is used for
        the corner case where `matrix` is essentially a sparse
        column vector.
        
    max_length : typing.Optional[int]
        The maximum number of columns to include in the returned
        row.
        
    Returns
    -------
    row : numpy.ndarray
        The specified row (as a 1d numpy array)
    """
    d = matrix.getrow(row).todense()
    d = np.squeeze(np.asarray(d, dtype=dtype))

    if max_length is not None:
        d = d[:max_length]

    # make sure we do not return a scalar
    if isinstance(d, dtype):
        d = np.array([d])

    return d

def sparse_matrix_to_dense(sparse_matrix:scipy.sparse.spmatrix) -> np.ndarray:
    """ Convert `sparse_matrix` to a dense numpy array

    Parameters
    ----------
    sparse_matrix : scipy.sparse.spmatrix
        The sparse scipy matrix

    Returns
    -------
    dense_matrix: numpy.ndarray
        The dense (2d) numpy array
    """
    dense = np.array(sparse_matrix.todense())
    return dense

def sparse_matrix_to_list(sparse_matrix:scipy.sparse.spmatrix) -> List:
    """ Convert `sparse_matrix` to a list of "sparse row vectors".
    
    In this context, a "sparse row vector" is simply a sparse matrix
    with dimensionality (1, sparse_matrix.shape[1]).
    
    Parameters
    ----------
    sparse_matrix: scipy.sparse.spmatrix
        The sparse scipy matrix

    Returns
    -------
    list_of_sparse_row_vectors : typing.List[scipy.sparse.spmatrix]
        The list of sparse row vectors
    """
    list_of_sparse_row_vectors = [
        sparse_matrix[i] for i in range(sparse_matrix.shape[0])
    ]
    
    return list_of_sparse_row_vectors

def write_sparse_matrix(
        target:str,
        a:scipy.sparse.spmatrix,
        compress:bool=True,
        **kwargs) -> None:
    """ Write `a` to the file `target` in matrix market format
    
    This function is a drop-in replacement for scipy.io.mmwrite. The only
    difference is that it gzip compresses the output by default. It *does not*
    alter the file extension, which should likely end in "mtx.gz" except in
    special circumstances.
    
    If `compress` is `True`, then this function imports `gzip`.
    
    Parameters
    ----------
    target : str
        The complete path to the output file, including file extension
        
    a : scipy.sparse.spmatrix
        The sparse matrix
        
    compress : bool
        Whether to compress the output
        
    **kwargs : <key>=<value> pairs
        These are passed through to :func:`scipy.io.mmwrite`. Please see the
        scipy documentation for more details.

    Returns
    --------
    None, but the matrix is written to disk
    """

    if compress:
        import gzip
        
        with gzip.open(target, 'wb') as target_gz:
            scipy.io.mmwrite(target_gz, a, **kwargs)
    else:
        scipy.io.mmwrite(target, a, **kwargs)

###
# Other helpers
###
def matrix_multiply(
        m1:np.ndarray,
        m2:np.ndarray,
        m3:np.ndarray) -> np.ndarray:
    """ Multiply the three matrices
    
    This function performs the multiplications in an order such that the
    size of the intermediate matrix created by the first matrix multiplication
    is as small as possible.
    
    Parameters
    ----------
    m{1,2,3} : numpy.ndarray
        The (2d) matrices
        
    Returns
    -------
    product_matrix : numpy.ndarray
        The product of the three input matrices
    """

    # check the dimensions
    if m1.shape[1] != m2.shape[0]:
        msg = ("The inner dimension between the first and second matrices do "
            "not match. {} and {}".format(m1.shape, m2.shape))
        raise ValueError(msg)

    if m2.shape[1] != m3.shape[0]:
        msg = ("The inner dimensions between the second and third matrices do "
            "not match. {} and {}".format(m2.shape, m3.shape))
        raise ValueError(msg)

    # now, check which order to perform the multiplications
    m1_first_size = m1.shape[0] * m2.shape[1]
    m3_first_size = m2.shape[0] * m3.shape[1]

    if m1_first_size < m3_first_size:
        res = np.dot(m1, m2)
        res = np.dot(res, m3)
    else:
        res = np.dot(m2, m3)
        res = np.dot(m1, res)

    return res

def permute_matrix(
        m:np.ndarray,
        is_flat:bool=False,
        shape:Optional[Tuple[int]]=None) -> np.ndarray:
    """ Randomly permute the entries of the matrix. The matrix is first 
    flattened.
    
    For reproducibility, the random seed of numpy should be set **before**
    calling this function.

    Parameters
    ----------
    m : numpy.ndarray
        The matrix (tensor, etc.)

    is_flat : bool
        Whether the matrix values have already been flattened. If they have
        been, then the desired shape must be passed.

    shape : typing.Optional[typing.Tuple]
        The shape of the output matrix, if `m` is already flattened

    Returns
    -------
    permuted_m: numpy.ndarray
        A copy of m (with the same shape as m) with the values randomly 
        permuted. 
    """

    if shape is None:
        shape = m.shape

    if not is_flat:
        m = np.reshape(m, -1)

    # shuffle the actual values
    permuted_m = np.random.permutation(m)

    # and put them back in the correct shape
    permuted_m = np.reshape(permuted_m, shape)

    return permuted_m