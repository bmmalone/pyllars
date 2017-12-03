import sklearn_pandas.categorical_imputer
import sklearn.base

from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd

class MultiColumnCategoricalImputer(sklearn.base.TransformerMixin):
    """ Replace missing values with the mode using independent categorical
    imputers (from the sklearn-pandas package)

    Optionally, the columns for imputation can be specified; if they are not
    given, then all missing values in all columns in the data matrix are
    replaced with the mode (independently).

    A number of similar implementations are available online; this
    implementation keeps the imputers around so that they can be used later,
    for example, on test data.

    It also attempts with both data frames (with named columns) and simpler
    np.arrays (which only have indices).

    Parameters
    ----------
    columns: list-like of column identifiers, or None
        A list of the columns for imputation. These should be integer indices
        for np.arrays or string column names for pd.DataFrames.

    missing_values: string or "NaN"
        The placeholder for missing values. Please see the sklearn-pandas
        documentation for more details.
    """

    def __init__(self, columns=None, missing_values='NaN'):
        self.columns = columns
        self.missing_values = missing_values


    def fit(self, X, *_, **__):
        """ Fit the imputers for all of the specified columns
        """

        # first, check if we have an np array or a data frame
        if isinstance(X, np.ndarray):
            self.is_np_array_ = True
        elif isinstance(X, pd.DataFrame):
            self.is_np_array_ = False
        else:
            msg = ("[multicolumn_ci]: attepting to impute an object which is "
                "neither an np.array nor a pd.DataFrame. type: {}".format(
                type(X)))
            raise ValueError(msg)

        # select the columns, if necessary
        if self.columns is None:
            if self.is_np_array_:
                self.columns = list(range(X.shape[1]))
            else:
                self.columns = X.columns

        # keep around the imputers
        self.imputers_ = {}

        for c in self.columns:

            # make sure we actually grab a column
            if self.is_np_array_:
                y = X[:,c]
            else:
                y = X[c]

            imputer = sklearn_pandas.categorical_imputer.CategoricalImputer(
                missing_values=self.missing_values
            )

            imputer.fit(y)
            self.imputers_[c] = imputer

        return self

    def transform(self, X, *_, **__):
        """ Impute the respective columns of X
        """
        check_is_fitted(self, "imputers_")

        # make a copy to keep around everything we do not impute
        X = X.copy()

        # we will also keep around missing masks so we can inverse_transform
        self.missing_masks_ = {}

        for c in self.columns:
            imputer = self.imputers_[c]

            # make sure we actually grab a column
            if self.is_np_array_:
                # so np.array
                y = X[:,c]
                y = imputer.transform(y)
                X[:,c] = y
            else:
                # then pd.DataFrame
                y = X[c]
                y = imputer.transform(y)
                X[c] = y

        return X

    




