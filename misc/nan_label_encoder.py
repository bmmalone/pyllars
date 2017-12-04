import numpy as np
import pandas as pd
import sklearn.base
import sklearn.utils

from sklearn.utils.validation import check_is_fitted

import logging
logger = logging.getLogger(__name__)

class NaNLabelEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """ A label encoder which handles missing values

    In the data, `np.nan` will be taken as a missing value.

    Parameters
    ----------
    missing_values: string
        A flag value which will be used internally for missing values

    labels: list-like of values
        Optionally, a list of values can be given; any values which do not
        appear in the training data, but present in the list, will still be
        accounted for.
    """
    def __init__(self, missing_values='NaN', labels=None):
        self.missing_values = missing_values
        self.labels = labels
        

    def fit(self, y):
        """Fit label encoder
        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.
        Returns
        -------
        self : returns an instance of self.
        """
        y = sklearn.utils.column_or_1d(y, warn=True)
        y = y.copy()

        # use our marker for any NaNs
        m_nan = pd.isnull(y)
        y[m_nan] = self.missing_values
        self.classes_ = np.unique(y)

        # and make sure to include the labels we specified
        if self.labels is not None:
            self.classes_ = np.unique(list(self.classes_) + self.labels)

        return self

    def transform(self, y):
        """Transform labels to normalized encoding.
        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = sklearn.utils.column_or_1d(y, warn=True)
        y = y.copy()

        # use our marker for NaNs
        m_nan = pd.isnull(y)
        y[m_nan] = self.missing_values

        classes = np.unique(y)
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            raise ValueError("y contains new labels: %s" % str(diff))

        # however, put the NaNs back in
        ret = np.searchsorted(self.classes_, y).astype(float)
        ret[m_nan] = np.nan
        return ret

    def inverse_transform(self, y):
        """Transform labels back to original encoding.
        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.
        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if diff:
            raise ValueError("y contains new labels: %s" % str(diff))
        y = np.asarray(y)
        return self.classes_[y]
