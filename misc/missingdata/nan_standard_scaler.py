import numpy as np
import pandas as pd
import sklearn.base

import logging
logger = logging.getLogger(__name__)

class NaNStandardScaler(sklearn.base.TransformerMixin):
    
    def __init__(self, columns=None):
        self.col_mean_ = None
        self.col_std_ = None
        self.columns = columns
    
    def fit(self, X, *_):
        
        if self.columns is None:

            if isinstance(X, pd.DataFrame):
                self.columns_ = X.columns
            elif isinstance(X, np.ndarray):
                if len(X.shape) == 1:
                    X = X.reshape(-1, 1)
                self.columns_ = np.arange(X.shape[1])
        else:
            self.columns_ = self.columns

        # now, actually grab the columns depending on the type of X
        if isinstance(X, pd.DataFrame):
            X_cols = X[self.columns_]
        elif isinstance(X, np.ndarray):
            X_cols = X[:,self.columns_]
        else:
            msg = ("[NanStandardScaler.fit]: unrecognized data type: {}".
                format(type(X)))
            raise ValueError(msg)

        ###
        # make sure we have numpy floats. we might not in cases where
        # the original data matrix contains mixed types (categorical and
        # numeric types)
        #
        # See this SO comment for more details:
        #   https://stackoverflow.com/questions/18557337/numpy-attributeerror-float-object-has-no-attribute-exp#comment27300354_18557337
        ###
        X_cols = X_cols.astype(float)
        self.col_mean_ = np.nanmean(X_cols, axis=0)
        self.col_std_ = np.nanstd(X_cols, axis=0)
        
        # we will not do anything with observations we see less than twice
        m_zero = self.col_std_ == 0
        m_nan = np.isnan(self.col_std_)
        self.col_ignore_ = m_zero | m_nan

        # do this to avoid divide by zero warnings later
        self.col_mean_ = np.nan_to_num(self.col_mean_)
        self.col_std_[self.col_ignore_] = 1

        return self
        
    def transform(self, X, *_):
        # if we did not see a column in the training, or if it had only one
        # value, we cannot really do anything with it

        # so ignore those

        # do not overwrite our original information
        X = X.copy()


        # now, actually grab the columns depending on the type of X
        if isinstance(X, pd.DataFrame):
            X_cols = X[self.columns_].copy()
            X_cols.iloc[:, self.col_ignore_] = 0
 
        elif isinstance(X, np.ndarray):
            # check if we have a single vector
            if len(X.shape) == 1:
                #X[self.col_ignore_] = 0
                X = X.reshape(-1, 1)

            X_cols = X[:,self.columns_]
            X_cols[:,self.col_ignore_] = 0
        else:
            msg = ("[NanStandardScaler.transform]: unrecognized data type: {}".
                format(type(X)))
            raise ValueError(msg)

        X_transform = ((X_cols - self.col_mean_) / self.col_std_)

        # and stick the columns back
        if isinstance(X, pd.DataFrame):
            X[self.columns_] = X_transform
        else:
            X[:,self.columns_] = X_transform

        return X
