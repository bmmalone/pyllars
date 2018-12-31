"""
This class is a thin wrapper around scipy.sparse.lil_matrix to reduce
the notational burden when dealing with sparse vectors. Internally, they
are simply stored as sparse matrices.

By default, the sparse vectors are created as integer row matrices. The
scipy.sparse.lil_matrix representation is used.

THIS CLASS HAS NOT BEEN TESTED EXTENSIVELY.
"""

import scipy.sparse

class lil_sparse_vector(scipy.sparse.lil_matrix):
    def __init__(self, size, dtype=int):
        super(lil_sparse_vector, self).__init__((size,1), dtype=dtype)
        
    def __getitem__(self, index):
        """ This getter grabs the value stored in the vector at index.

            Args:
                index (int): The index

            Returns:
                self.dtype: The value stored at index
        """
        return super().__getitem__((index, 0))

    def __setitem__(self, index, value):
        """ This setter puts value into the stored vector at index. 
            Args:
                index (int): The index

                value (self.dtype): The value to set. Type checking IS NOT performed

            Returns:
                None
        """
        super().__setitem__((index, 0), value)

    def __len__(self):
        """ This property returns the length of the first dimension of the
            stored matrix.
        """
        return super().shape[0]

