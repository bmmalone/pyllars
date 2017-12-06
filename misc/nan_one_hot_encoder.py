import numpy as np
import pandas as pd
import sklearn.base
from sklearn.utils.validation import check_is_fitted

from sklearn.preprocessing import OneHotEncoder

import logging
logger = logging.getLogger(__name__)

class NaNOneHotEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self,
            n_values='auto',
            categorical_features='all',
            sparse=True,
            handle_unknown='error'):

        self.n_values = n_values
        self.categorical_features = categorical_features
        self.sparse = sparse
        self.handle_unknown = handle_unknown

    def fit(self, X, *_):
        
        self.enc_ = OneHotEncoder(
            n_values = self.n_values,
            categorical_features = self.categorical_features,
            sparse = self.sparse,
            handle_unknown = self.handle_unknown
        )

        # just replace all np.nan's with 0
        m = pd.isnull(X)

        X = X.copy()
        X[m] = 0

        self.enc_.fit(X)

        return self

    def transform(self, X, *_):
        check_is_fitted(self, "enc_")

        # first, grab the missing value positions
        m_missing = pd.isnull(X)


    
