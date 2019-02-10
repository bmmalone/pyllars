import sklearn.base

from pyllars.validation_utils import check_is_fitted

import copy
import numpy as np
import pandas as pd

class MultiColumnImputer(sklearn.base.TransformerMixin):
    """ Replace missing values with the mode using independent imputers

    Optionally, the columns for imputation can be specified; if they are not
    given, then all missing values in all columns in the data matrix are
    replaced with the given imputation strategy.

    A number of similar implementations are available online; this
    implementation keeps the imputers around so that they can be used later,
    for example, on test data.

    It also attempts with both data frames (with named columns) and simpler
    np.arrays (which only have indices).

    Parameters
    ----------
    imputer_template: sklearn.transformer
        A class which implements the desired imputation strategy. In
        particular, it must implement the standard sklearn transformer
        interface and behave as expected with `copy.copy`.

    columns: list-like of column identifiers, or None
        A list of the columns for imputation. These should be integer indices
        for np.arrays or string column names for pd.DataFrames.
    """

    def __init__(self,
            imputer_template,
            columns=None):

        self.columns = columns
        self.imputer_template = imputer_template


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

            imputer = copy.copy(self.imputer_template)
            imputer.fit(y)
            self.imputers_[c] = imputer

        return self

    def transform(self, X, *_, **__):
        """ Impute the respective columns of X
        """
        check_is_fitted(self, "imputers_")

        # make a copy to keep around everything we do not impute
        X = X.copy()
        
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

    




