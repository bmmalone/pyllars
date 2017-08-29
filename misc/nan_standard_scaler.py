import numpy as np
import sklearn.base

import logging
logger = logging.getLogger(__name__)

class NaNStandardScaler(sklearn.base.TransformerMixin):
    
    def __init__(self):
        self.col_mean_ = None
        self.col_std_ = None
    
    def fit(self, X, *_):
        self.col_mean_ = np.nanmean(X, axis=0)
        self.col_std_ = np.nanstd(X, axis=0)
        
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
        X[:,self.col_ignore_] = 0

        X = ((X - self.col_mean_) / self.col_std_)

        num_nan = np.isnan(X).sum()
        num_inf = np.isinf(X).sum()

        msg = "[NaNStandardScaler.transform]: num_nans: {}".format(num_nan)
        logger.debug(msg)
        
        msg = "[NaNStandardScaler.transform]: num_infs: {}".format(num_inf)
        logger.debug(msg)

        return X
