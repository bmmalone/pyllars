import sklearn.preprocessing
import sklearn.base

from pyllars.validation_utils import check_is_fitted

import numpy as np
import pandas as pd

from pyllars.sklearn_transformers.nan_label_encoder import NaNLabelEncoder

class MultiColumnLabelEncoder(sklearn.base.TransformerMixin):
    """ Encode multiple columns using independent label encoders

    Optionally, the columns to encode can be specified; if they are not given,
    then all columns in the matrix are encoded (independently).

    A number of similar implementations are available online; this
    implementation keeps the encoders around so that they can be used later,
    for example, on test data. It also provides an inverse_transform operation.

    It also attempts with both data frames (with named columns) and simpler
    np.arrays (which only have indices).

    Parameters
    ----------
    columns: list-like of column identifiers, or None
        A list of the columns for imputation. These should be integer indices
        for np.arrays or string column names for pd.DataFrames.

    """

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, *_, **__):
        """ Fit the encoders for all of the specified columns
        """

        # first, check if we have an np array or a data frame
        if isinstance(X, np.ndarray):
            self.is_np_array_ = True
        elif isinstance(X, pd.DataFrame):
            self.is_np_array_ = False
        else:
            msg = ("[multicolumn_le]: attepting to encode an object which is "
                "neither an np.array nor a pd.DataFrame. type: {}".format(
                type(X)))
            raise ValueError(msg)

        # select the columns, if necessary
        if self.columns is None:
            if self.is_np_array_:
                self.columns = list(range(X.shape[1]))
            else:
                self.columns = X.columns

        # keep around the label encoders
        self.le_ = {}

        for c in self.columns:

            # make sure we actually grab a column
            if self.is_np_array_:
                y = X[:,c]
            else:
                y = X[c]

            label_encoder = NaNLabelEncoder()
            label_encoder.fit(y)
            self.le_[c] = label_encoder

        return self

    def transform(self, X, *_, **__):
        """ Encode the respective columns of X
        """
        check_is_fitted(self, "le_")

        # make a copy to keep around everything we do not encode
        X = X.copy()

        for c in self.columns:
            le = self.le_[c]

            # make sure we actually grab a column
            if self.is_np_array_:
                # so np.array
                y = X[:,c]
                y = le.transform(y)
                X[:,c] = y
            else:
                # then pd.DataFrame
                y = X[c]
                y = le.transform(y)
                X[c] = y

        return X

    def inverse_transform(self, X, *_, **__):
        """ Transform labels back to the original encoding
        """
        check_is_fitted(self, "le_")

        # make a copy to keep around everything we do not encode
        X = X.copy()

        for c in self.columns:
            le = self.le_[c]

            # make sure we actually grab a column
            if self.is_np_array_:
                # so np.array
                y = X[:,c]
                y = le.inverse_transform(y)
                X[:,c] = y
            else:
                # then pd.DataFrame
                y = X[c]
                y = le.inverse_transform(y)
                X[c] = y

        return X


