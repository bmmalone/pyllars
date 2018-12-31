import numpy as np
import pandas as pd
import sklearn.base
from pyllars.validation_utils import check_is_fitted

from sklearn.preprocessing import OneHotEncoder

import logging
logger = logging.getLogger(__name__)

# thank you, sklearn/preprocessing/data.py
import six
from scipy import sparse

def _encode_selected(X, encoder, selected="all", copy=True):
    """Apply a transform function to portion of selected features
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape [n_samples, n_features]
        Dense array or sparse matrix.
    encoder : OneHotEncoder
        A fit OneHotEncoder
    copy : boolean, optional
        Copy X even if it could be avoided.
    selected: "all" or array of indices or mask
        Specify which features to apply the transform to.
    Returns
    -------
    X : array or sparse matrix, shape=(n_samples, n_features_new)
    """
    
    if isinstance(selected, six.string_types) and selected == "all":
        return encoder.transform(X)

    if len(selected) == 0:
        return X

    n_features = X.shape[1]
    ind = np.arange(n_features)
    sel = np.zeros(n_features, dtype=bool)
    sel[np.asarray(selected)] = True
    not_sel = np.logical_not(sel)
    n_selected = np.sum(sel)

    if n_selected == 0:
        # No features selected.
        return X
    elif n_selected == n_features:
        # All features selected.
        return encoder.transform(X)
    else:
        X_sel = encoder.transform(X[:, ind[sel]])
        X_not_sel = X[:, ind[not_sel]]

        if sparse.issparse(X_sel) or sparse.issparse(X_not_sel):
            return sparse.hstack((X_sel, X_not_sel))
        else:
            return np.hstack((X_sel, X_not_sel))

class NaNOneHotEncoder(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """ One-hot encode the specified categorical variables, treating np.nan's
    as missing.

    In particular, this class will encode missing values as all 0's. Otherwise,
    it follows the same semantics as a normal sklearn.preprocessing.OneHotEncoder.

    N.B. This class is not particularly "copy-efficient".
    """

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
            categorical_features = "all", # manually handle this
            sparse = self.sparse,
            handle_unknown = self.handle_unknown
        )
        
        X = X.copy()

        # make sure we have a data matrix
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # just replace all np.nan's with 0
        m = pd.isnull(X)
        X[m] = 0

        # only do something if we have some features
        if len(self.categorical_features) > 0:
            # check if we should use everything
            if self.categorical_features == 'all':
                self.categorical_features = np.array(range(X.shape[1]))

            # and encode whatever we had
            self.enc_.fit(X[:,self.categorical_features])

        return self

    def transform(self, X, *_):
        check_is_fitted(self, "enc_")

        X = X.copy()

        # make sure we have a matrix rather than a vector
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # first, replace the missing categorical values with 0
        masks = {}
        for f in self.categorical_features:
            m = pd.isnull(X[:,f])
            masks[f] = m

            # this overwrites data in the passed array
            X[m,f] = 0

        # now, encode the categorical values (ignoring whatever is in the
        # other fields)
        Xt = _encode_selected(
            X,
            self.enc_,
            selected=self.categorical_features,
            copy=True
        )

        if self.sparse:
            Xt = Xt.tocsr()
            
        # and clear out the missing values
        for i, f in enumerate(self.categorical_features):
            m = masks[f]
            indices = self.enc_.feature_indices_[i:i+1]            
            Xt[m,indices] = 0


        return Xt


    
