import numpy as np
import sklearn.base

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

        # make sure anything that is complete missing does not introduce nans
        #self.col_std_ = np.nan_to_num(self.col_std_)

        # finally, make sure col_std_ does not contain actual 0s
        #small_val = np.nextafter(0, 1, dtype=X.dtype)
        #m = self.col_std_ == 0
        #self.col_std_[m] = small_val

        return self
        
    def transform(self, X, *_):
        # if we did not see a column in the training, or if it had only one
        # value, we cannot really do anything with it

        # so ignore those
        X[:,self.col_ignore_] = 0


        X = ((X - self.col_mean_) / self.col_std_)
        return X
