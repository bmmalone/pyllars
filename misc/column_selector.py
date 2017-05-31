
class ColumnSelector(object):
    """ 
    A feature selector for scikit-learn's Pipeline class that returns
    specified columns from a pandas data frame.
    
    """

    def __init__(self, cols, transform_contiguous=False):
        self.cols = cols
        self.transform_contiguous = transform_contiguous

    def transform(self, X, y=None):
        import numpy as np

        ret = X[self.cols]

        if self.transform_contiguous:
            ret = ret.values
            ret = np.ascontiguousarray(ret)

        return ret

    def fit(self, X, y=None):
        return self

class ColumnTransformer(object):
    """ A transformer for an sklearn Pipeline which applies a function to a
    specified set of columns from a pandas data frame.
    """
    def __init__(self, cols, f, transform_contiguous=False):
        self.cols = cols
        self.f = f
        self.transform_contiguous = transform_contiguous

    def transform(self, X, y=None):
        import numpy as np

        X_copy = X.copy()

        vals = X_copy[self.cols]
        vals = self.f(vals)
        X_copy[self.cols] = vals

        if self.transform_contiguous:
            ret = X_copy.values
            vals = np.ascontiguousarray(X_copy)

        return X_copy

    def fit(self, X, y=None):
        return self

