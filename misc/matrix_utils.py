""" Helpers for working with 2-d matrices
"""

def permute_matrix(m, is_flat=False, shape=None):
    """ Randomly permute the entries of the matrix. The matrix is first 
    flattened.

    Parameters
    ----------
    m: np.array
        The matrix (tensor, etc.)

    is_flat: bool
        Whether the matrix values have already been flattened. If they have
        been, then the desired shape must be passed.

    shape: tuple of ints
        The shape of the output matrix, if m is already flattened

    Returns
    -------
    permuted_m: np.array
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

def write_sparse_matrix(target, a, compress=True, **kwargs):
    """ This function is a drop-in replacement for scipy.io.mmwrite. The only
        difference is that is compresses the output by default. It *does not*
        alter the file extension, which should likely end in "mtx.gz" except
        in special circumstances.

        Args:
            compress (bool): whether to compress the output.

            All other arguments are exactly the same as for scipy.io.mmwrite.
            Please see the scipy documentation for more details.
        
        Returns:
            None

        Imports:
            gzip
            scipy.io
    """
    import gzip
    import scipy.io

    if compress:
        with gzip.open(target, 'wb') as target_gz:
            scipy.io.mmwrite(target_gz, a, **kwargs)
    else:
        scipy.io.mmwrite(target_gz, a, **kwargs)

def row_op(m, op):
    """Apply op to each row in the matrix."""
    return op(m, axis=1)

def col_op(m, op):
    """Apply op to each column in the matrix."""
    return op(m, axis=0)

def row_sum(m):
    """Calculate the sum of each row in the matrix."""
    return  np.sum(m, axis=1)

def col_sum(m):
    """Calculate the sum of each column in the matrix."""
    return  np.sum(m, axis=0)

def row_sum_mean(m, var=False):
    """ Calculate the mean of the sum of each row in the matrix. Optionally,
    the variances of the row sums can also be calculated.

    Parameters
    ----------
    m: 2-dimensional np.array
        The matrix

    var: bool
        Whether to calculate the variances

    Returns
    -------
    mean: float
        The mean of the row sums in the matrix

    variance: float
        If var is True, then the variance of the row sums
    """
    row_sums = row_sum(m)
    mean = np.mean(row_sums)

    if var:
        var = np.var(row_sums)
        return mean, var

    return mean

def col_sum_mean(m, var=False):
    """ Calculate the mean of the sum of each column in the matrix.
    
    Optionally, the variances of the column sums can also be calculated.

    Parameters
    ----------
    m: 2-dimensional np.array
        The matrix

    var: bool
        Whether to calculate the variances

    Returns
    -------
    mean: float
        The mean of the column sums in the matrix

    variance: float
        If var is True, then the variance of the column sums
    """
    col_sums = col_sum(m)
    mean = np.mean(col_sums)

    if var:
        var = np.var(col_sums)
        return mean, var

    return mean

def normalize_rows(matrix):
    """ Normalize the rows of the given (dense) matrix

    Parameters
    ----------
    matrix: 2-dimensional np.array
        The matrix

    Returns
    -------
    normalized_matrix: 2-dimensional np.array
        The matrix normalized s.t. all row sums are 1
    """
    coef = np.abs(matrix).sum(axis=1)
    coef = coef[:,np.newaxis]
    matrix = np.divide(matrix, coef)
    return matrix

def normalize_columns(m):
    """ Normalize the columns of the given (dense) matrix

    Parameters
    ----------
    m: 2-dimensional np.array
        The matrix

    Returns
    -------
    normalized_matrix: np.array with the same shape as m
        The matrix normalized s.t. all column sums are 1
    """
    col_sums = np.sum(np.abs(m), axis=0)
    m = np.divide(m, col_sums)
    return m


def get_dense_row(data, row, dtype=float, length=-1):
    d = data.getrow(row).todense()
    d = np.squeeze(np.asarray(d, dtype=dtype))

    if length > 0:
        d = d[:length]

    # make sure we do not return a scalar
    if isinstance(d, dtype):
        d = np.array([d])

    return d

def sparse_matrix_to_dense(sparse_matrix):
    """ Convert the scipy.sparse matrix to a dense np.array

    Parameters
    ----------
    sparse_matrix: scipy.sparse
        The sparse scipy matrix

    Returns
    -------
    dense_matrix: 2-dimensional np.array
        The dense np.array
    """
    dense = np.array(sparse_matrix.todense())
    return dense

def matrix_multiply(m1, m2, m3):
    """ This helper function multiplies the three matrices. It minimizes the
        size of the intermediate matrix created by the first multiplication.

        Args:
            m1, m2, m3 (2-dimensional np.arrays): the matrices to multiply

        Returns:
            np.array, with shape m1.shape[0] x m3.shape[1]

        Imports:
            numpy

        Raises:
            ValueError: if the dimensions do not match
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