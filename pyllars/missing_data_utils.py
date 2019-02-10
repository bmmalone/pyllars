"""
This module contains utilities for removing data according to different
missingness mechanisms, including missing at random (MAR), missing completely
at random (MCAR), and not missing at random (NMAR).

Example::

    import functools
    import numpy as np
    import sklearn.datasets
    import pyllars.missing_data_utils as missing_data_utils
    
    X_complete, y = sklearn.datasets.load_iris(return_X_y=True)

    # mcar

    # all observations have a 20% chance of being missing
    missing_likelihood = 0.2
    X_mcar_incomplete = missing_data_utils.get_mcar_incomplete_data(
        X_complete,
        missing_likelihood
    )

    # nmar

    # remove all `x[1]` values greater than 4
    # and all `x[3]` values greater than 0.3
    missing_likelihood = [
        None,
        functools.partial(remove_large_values, threshold=4),
        None,
        functools.partial(remove_large_values, threshold=0.3)
    ]

    X_nmar_incomplete = missing_data_utils.get_nmar_incomplete_data(
        X_complete,
        missing_likelihood
    )

    # mar

    # remove `x[3]` when `x[0]*x[1] > 18`
    missing_likelihood = functools.partial(
        missing_data_utils.remove_y_when_z_is_large,
        y=3, z=[0,1], threshold=18, combination_operator=np.product
    )
    X_mar_incomplete = missing_data_utils.get_mar_incomplete_data(
        X_complete,
        missing_likelihood
    )

    # get training, testing splits suitable for use in sklearn
    mcar_data = missing_data_utils.get_incomplete_data_splits(
        X_complete,
        X_mcar_incomplete,
        y
    )
    nmar_data = missing_data_utils.get_incomplete_data_splits(
        X_complete,
        X_mcar_incomplete,
        y
    )
    mar_data = missing_data_utils.get_incomplete_data_splits(
        X_complete,
        X_mcar_incomplete,
        y
    )
"""
import logging
logger = logging.getLogger(__name__)

import collections
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.model_selection

import pyllars.math_utils as math_utils
import more_itertools

# nan preprocessing
from pyllars.sklearn_transformers.nan_standard_scaler import NaNStandardScaler
from pyllars.sklearn_transformers.nan_one_hot_encoder import NaNOneHotEncoder

from pyllars.sklearn_transformers.multicolumn_imputer import MultiColumnImputer
from sklearn_pandas.categorical_imputer import CategoricalImputer
from sklearn.preprocessing import FunctionTransformer 

###
#   Functions for creating incomplete datasets according to different
#   missingness mechanisms
###

def get_mcar_incomplete_data(X, missing_likelihood=0.1, random_state=8675309):
    """ Remove some of the observations
    
    Internally, this function uses an MCAR mechanism to remove the data. That
    is, the likelihood that an opbservation is independent of both the value
    itself and other values in the rows.
    
    Parameters
    ----------
    X: data matrix
        A data matrix suitable for sklearn
        
    missing_likelihood: float
        The likelihood each observation in the training data will be missing
        
    random_state: int
        An attempt to make things reproducible
        
    Returns
    -------
    X_incomplete: data matrix
        The incomplete training data for this fold
    """
    
    X_incomplete, missing_mask = math_utils.mask_random_values(
        X,
        likelihood=missing_likelihood,
        return_mask=True,
        random_state=random_state
    )

    return X_incomplete


def get_nmar_incomplete_data(X, missing_likelihood=None, random_state=8675309):
    """ Remove some of the observations
    
    Internally, this function uses an NMAR mechanism. That is, the likelihood
    that an observation is missing depends on the value.
    
    Parameters
    ----------
    X: data matrix
        A data matrix suitable for sklearn
        
    missing_likelihood: list of callables
        Callables which determine whether a given feature observation
        is missing, giving the value of that feature. The indices within the
        list should match the columns in `X`.

        Specifically, the callables should take as input a 1D `np.array` of
        floats and return a 1D `np.array` of floats with the same shape. The
        array gives all values for the respective feature. See
        `remove_large_values` for a simple example.
        
        `None` values can be given if all observations for the respective feature
        are present.
        
    random_state: int
        An attempt to make things reproducible
        
    Returns
    -------
    X_incomplete: data matrix
        The incomplete data matrix
    """
    np.random.seed(random_state)

    # make sure we have an appropriate number of functions
    if len(missing_likelihood) != X.shape[1]:
        msg = ("[get_nmar_complete_data]: the number of functions for "
            "missing data does not match the number of dimensions of the "
            "data.")
        raise ValueError(msg)
    
    X_ret = X.copy()
    
    for i in range(X.shape[1]):
        if not missing_likelihood[i] is None:
            X_ret[:,i] = missing_likelihood[i](X[:,i])

    return X_ret

def remove_large_values(X, threshold, return_mask=False):
    """ Remove values above the threshold in `X`

    This function is suitable for use with `get_nmar_incomplete_data`.
    
    All of the shapes are the same as X.
    
    Parameters
    ----------
    X: np.array of floats
        The values
        
    threshold: float
        The threshold to remove values
        
    Returns
    -------
    X_masked: np.array of floats
        A copy of the input array, with the large values replaced with
        np.nan.
        
    mask: np.array of bools
        A mask of the high values which were removed. Returned only if
        `return_mask` was True
    """
    X_small = X.copy()
    
    m = X > threshold
    X_small[m] = np.nan
    
    ret = X_small 
    if return_mask:
        ret = ret, m
        
    return ret


def get_mar_incomplete_data(X, missing_likelihood=None, random_state=8675309):
    """ Remove some of the observations
    
    Internally, this function uses an MAR mechanism. That is, the likelihood
    that an observation is missing depends on the other values in that
    instance.

    Parameters
    ----------
    X: data matrix
        A data matrix suitable for sklearn
        
    missing_likelihood: callable
        A callable which determines the missing values for an instance
        `x`. The callable should take one argument: `x`, the instance,
        and it should return a copy with missing values replaced with `np.nan`.
        
    random_state: int
        An attempt to make things reproducible
        
    Returns
    -------
    X_incomplete: data matrix
        The incomplete data matrix
    """
    np.random.seed(random_state)
    X_ret = np.full_like(X, np.nan)
    
    # determine the missingness patterns in each row
    for i in range(X.shape[0]):
        X_ret[i] = missing_likelihood(X[i])

    return X_ret

def remove_y_when_z_is_large(x, y, z, threshold,
        combination_operator=np.product):
    """ Remove values of `y` when the combined values of `z` exceed `threshold`

    This function is suitable for use with `get_mar_incomplete_data`.
    
    Parameters
    ----------
    x: 1D `np.array` of floats
        The instance
        
    y: int
        The index of the value to consider removing
        
    z: int or iterable of ints
        The indices of the "condition" values
        
    threshold: float
        The threshold to consider a combined value as "large"
        
    combination_operator: callable with one parameter
        The operation to combine the `z` variables. The callable should take
        an array of size :math:`|z|`. 
        
    Returns
    -------
    x_missing: 1D np.array of floats
        A copy of `x` with the value at index `y` replaced with `np.nan` if
        appropriate based on the combined values of `z` and `threshold`.
    """
    val = combination_operator(x[z])
    
    x_missing = np.copy(x)
    if val > threshold:
        x_missing[y] = np.nan
        
    return x_missing

def get_incomplete_data(X, mechanism, missing_likelihood, random_state=8675309):
    """ Remove some observations according to the specified missing  mechanism

    This is a simple wrapper around the respective functions. In principle, it
    provides a consistent interface for all of them.

    Parameters
    ----------
    X: data matrix
        A data matrix suitable for sklearn

    mechanism: string in "mcar", "mar", "nmar" (case-insensitive)
        The missing data mechanism to use
        
    missing_likelihood: object
        The `missing_likelihood` parameter for the respective
        `get_XXX_incomplete_data` functions. Please see the documentation for
        those functions for more details.

    random_state: int
        An attempt to make things reproducible

    Returns
    -------
    X_incomplete: data matrix
        The incomplete data matrix
    """

    mechanism = mechanims.lower()

    if mechanism == "mcar":
        X_incomplete = get_mcar_incomplete_data(X, missing_likelihood, random_state)
    elif mechanism == "mar":
        X_incomplete = get_mar_incomplete_data(X, missing_likelihood, random_state)
    elif mechanism == "nmar":
        X_incomplete = get_nmar_incomplete_data(X, missing_likelihood, random_state)
    else:
        valid_mechanisms = ["mcar", "mar", "nmar"]
        valid_mechanisms = ' '.join(valid_mechanisms)
        msg = ("[get_incomplete_data]: unknown missing data mechansim: {}. Must "
            "be one of: {}".format(mechanism, valid_mechanisms))
        raise ValueError(msg)
        
    return X_incomplete


###
#   Helpers for evaluation with incomplete datasets
###

_incomplete_dataset_fields = (
    "X_train_complete",
    "X_train_incomplete",
    "X_test_complete",
    "X_test_incomplete",
    "y_train",
    "y_test"
)
_incomplete_dataset_fields = ' '.join(_incomplete_dataset_fields)
IncompleteDataset = collections.namedtuple(
    "IncompleteDataset",
    _incomplete_dataset_fields
)

def get_incomplete_data_splits(
        X_complete,
        X_incomplete,
        y,
        fold=0,
        num_folds=10,
        stratify=True,
        random_state=8675309,
        identities=None):

    """ Split the datasets for use with cross-validation
    
    Parameters
    ----------
    X_complete: data matrix
        A data matrix suitable for sklearn, without missing values
        
    X_incomplete: data matrix
        A data matrix suitable for sklearn, with missing values represented
        as `np.nan`
        
    y: target variables
        The target variables corresponding to X
        
    fold: int
        The cv fold to return
        
    num_folds: int
        The number of cv folds to create

    stratify: bool
        Whether to use stratified splits (True) or random (False). In
        particular, stratified splits will not work for regression.
        
    random_state: int
        An attempt to make things reproducible

    identities: list-like of ints or None
        Optionally, the identities of the nodes. If present, they will be
        added as the first column of all of the respective X matrices.
        
    Returns (as a named tuple)
    -------
    X_train_complete: data matrix
        The complete training data for this fold
        
    X_train_incomplete: data matrix
        The incomplete training data for this fold
        
    X_test_complete: data matrix
        The complete testing data for this fold
        
    X_test_incomplete: data matrix
        The incomplete testing data for this fold
        
    y_train: target variables
        The (complete) training target data for this fold
        
    y_test: target variables
        The (complete) testing target data for this fold
    """
    
    if stratify:
        cv = sklearn.model_selection.StratifiedKFold(
            num_folds, random_state=random_state
        )
    else:
        cv = sklearn.model_selection.KFold(
            num_folds, random_state=random_state, shuffle=True
        )

    splits = cv.split(X_complete, y)
    train, test = more_itertools.nth(splits, fold)

    X_train_complete, y_train = X_complete[train], y[train]
    X_test_complete, y_test = X_complete[test], y[test]
    
    X_train_incomplete = X_incomplete[train]
    X_test_incomplete = X_incomplete[test]

    # check on our node identities
    if identities is not None:
        train_identities = identities[train]
        test_identities = identities[test]

        X_train_complete = add_identities(X_train_complete, train_identities)
        X_train_incomplete = add_identities(X_train_incomplete, train_identities)
        X_test_complete = add_identities(X_test_complete, test_identities)
        X_test_incomplete = add_identities(X_test_incomplete, test_identities)

    ret = IncompleteDataset(
        X_train_complete,
        X_train_incomplete,
        X_test_complete,
        X_test_incomplete,
        y_train,
        y_test
    )
    
    return ret

def add_identities(X, node_identities=None):
    """ Add the identity as the first column of X
    
    The identity column is just the integer index of the corresponding
    row of `X`. For example, this could give the index of a node within
    an associated graph structure.
    
    Parameters
    ----------
    X: 2D np.array or scipy.sparse_matrix
        The data matrix
        
    node_identities: iterable of ints or None
        The identities of the nodes. If `None` is given, then the identity
        of the node in row `i` is the integer `i`.
        
    Returns
    -------
    X_with_identity: data matrix
        Either an np.array or scipy.sparse_matrix which is a copy of `X`
        with the identity column prepended.
        
        In case `X` was a sparse matrix, it will always be returned as a
        csr sparse matrix.
    """
    if node_identities is None:
        node_identities = np.arange(X.shape[0], dtype=int)
    
    if scipy.sparse.issparse(X):
        X_with_identity = scipy.sparse.hstack((node_identities[:,np.newaxis], X))
        X_with_identity = X_with_identity.tocsr()
        
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            msg = ("[add_node_identities]: attempting to add identities to "
                "an np.array with invalid number of dimensions: {}".format(
                X.ndim))
            raise ValueError(msg)
            
        X_with_identity = np.c_[node_identities, X]

    else:
        msg = ("[add_node_identities]: attempting to add identities to an "
               "invalid data matrix data type: {}".format(type(X)))
        raise ValueError(msg)
        
    return X_with_identity
        
    
_training_results_fields = (
    "model_fit_c",
    "model_fit_i",
    "y_pred_cc",
    "y_pred_ic",
    "y_pred_ci",
    "y_pred_ii",
    "y_test"
)
_training_results_fields = ' '.join(_training_results_fields)
TrainingResults = collections.namedtuple(
    "TrainingResults",
    _training_results_fields
)

def train_on_incomplete_data(model, incomplete_data):
    """ Perform all combinations of training and testing for the model and
    incomplete data set structure.
    
    In particular, this function fits the model using both the complete and
    incomplete versions of the data. It then makes predictions on both the 
    complete and incomplete versions of the test data.
    
    Parameters
    ----------
    model: an sklearn model
        In particular, the model must support cloning via `sklearn.clone`,
        have a `fit` method and have a `predict` method after fitting.
        
    incomplete_data: an IncompleteData named tuple
        A structure containing both complete and incomplete data. Presumably,
        this was created using `get_incomplete_data_splits`.
    
    Returns (as a named tuple)
    -------
    model_fit_c: fit sklearn model
        The model fit using the complete data
        
    model_fit_i: fit sklearn model
        The model fit using the incomplete data
        
    y_pred_cc: np.array
        The predictions from `model_fit_c` on the complete test dataset
        
    y_pred_ci: np.array
        The predictions from `model_fit_c` on the incomplete test data
        
    y_pred_ic: np.array
        The predictions from `model_fit_i` on the complete test data
        
    y_pred_ii: np.array
        The predictions from `model_fit_i` on the incomplete test data

    y_test: np.array
        The true test values
    """
    model_fit_c = sklearn.clone(model)
    model_fit_c.fit(incomplete_data.X_train_complete, incomplete_data.y_train)
    
    model_fit_i = sklearn.clone(model)
    model_fit_i.fit(incomplete_data.X_train_incomplete, incomplete_data.y_train)
    
    y_pred_cc = model_fit_c.predict(incomplete_data.X_test_complete)
    y_pred_ci = model_fit_c.predict(incomplete_data.X_test_incomplete)
    y_pred_ic = model_fit_i.predict(incomplete_data.X_test_complete)
    y_pred_ii = model_fit_i.predict(incomplete_data.X_test_incomplete)
    
    ret = TrainingResults(
        model_fit_c,
        model_fit_i,
        y_pred_cc,
        y_pred_ic,
        y_pred_ci,
        y_pred_ii,
        incomplete_data.y_test
    )
    
    return ret

###
#   Helpers to create preprocessing missing data
###
def replace_nans_with_flag(X, y=None):
    X = X.copy()

    for col in range(X.shape[1]):
        c = X[:,col]
        max_val = np.nanmax(c)
        missing_flag = max_val + 1
        m_nan = np.isnan(c)
        c[m_nan] = missing_flag
    return X

def replace_nans_with_zero(X, y=None, try_cast=True):
    if try_cast:
        try:
            X = X.astype(float)
        except TypeError as te:
            msg = ("[replace_nans_with_zero] the input data type could not "
                "be cast as a floating type. Please check the input type.")
            raise TypeError(msg)

    else:
        X = X.copy()
    
    m_nan = pd.isnull(X)
    X[m_nan] = 0

    m_finite = np.isfinite(X)
    X[~m_finite] = 0
    return X


def get_nan_preprocessing_pipeline(dataset_manager, fill_categoricals="flag",
        fields_to_ignore=None):
    """ Retrieve a simple pipeline for preprocessing missing data

    The pipeline includes the following steps:

    1. scale the non-missing numerical values
    2. zero-fill the missing numerical values
    3. replace missing categorical values with the mode
    4. one-hot encode the categorical values
    """

    preprocessing = []

    allowed_fill_categoricals = set(['flag', 'mode'])
    if fill_categoricals not in allowed_fill_categoricals:
        msg = ("[missing_data_utils.get_nan_preprocessing_pipeline] allowed "
            "`fill_categoricals` are: {}. found: {}".format(
            allowed_fill_categoricals, fill_categoricals))
        raise ValueError(msg)

    ###
    # First, build up the missing data part of the pipeline
    ###

    # first, scale the numerical values
    nan_scaler = NaNStandardScaler(
        columns=dataset_manager.get_numerical_field_indices(fields_to_ignore)
    )

    # and zero-fill the missing numeric values
    # due to scaling, this is the same as replacing the values
    # by the mean, but by scaling first, we do not falsely add
    # mass around the mean from the missing values
    num_imputer_template = FunctionTransformer(
        func=replace_nans_with_zero,
        validate=False
    )

    num_imputer = MultiColumnImputer(
        imputer_template=num_imputer_template,
        columns=dataset_manager.get_numerical_field_indices(fields_to_ignore)
    )

    preprocessing.append(("nan_scaler", nan_scaler))
    preprocessing.append(("num_imputer", num_imputer))


    # only add the categorical part if there are encoded categoricals
    if hasattr(dataset_manager, "le_"):
        if fill_categoricals == "mode":
            # replace missing categorical values with the mode
            cat_imputer_template = CategoricalImputer(
                missing_values="NaN"
            )
        elif fill_categoricals == "flag":
            cat_imputer_template = FunctionTransformer(
                func=replace_nans_with_flag,
                validate=False
            )

        cat_imputer = MultiColumnImputer(
            imputer_template=cat_imputer_template,
            columns=dataset_manager.get_categorical_field_indices(fields_to_ignore)
        )
        
        # finally, we one-hot encode the categorical variables

        # use the label encoder from the dataset manager to pull out
        # the number of values for each variable
        n_values = np.array([
            len(dataset_manager.le_.le_[f].classes_)
                for f in dataset_manager.get_categorical_field_names(fields_to_ignore)
        ])

        one_hot_encoder = NaNOneHotEncoder(
            categorical_features=dataset_manager.get_categorical_field_indices(fields_to_ignore),
            sparse=False,
            n_values=n_values
            #handle_unknown='ignore'
        )
        
        preprocessing.append(("cat_imputer", cat_imputer))
        preprocessing.append(("one_hot_encoder", one_hot_encoder))

    else:
        msg = ("[missing_data_utils.get_nan_preprocessing_pipeline]: did not "
            "find a label encoder in the dataset manager, so not attempting "
            "to handle categorical variables")
        logger.warning(msg)

    preprocessing = sklearn.pipeline.Pipeline(preprocessing)
    return preprocessing



def distance_with_nans(x, y, metric, normalize=True):
    """ Compute the distance between the two vectors considering only features
    observed in both. The distance is then normalized by the number of features
    in both.
    
    If the vectors share no common features, then the distance is taken to
    be np.inf.
    
    Parameters
    ----------
    x,y: np.arrays
        The two feature vectors
        
    metric: callable, which takes two arguments (x and y)
        The function used to calculate the distance between the common features.
        
        N.B. While the indices of features passed to this function will match,
            they will not, in general, match the indices from the original
            vectors. Thus, care should be taken in "metric" if a specific
            meaning is assigned to particular indices (e.g., "our bag-of-words
            begins at index 3" may not hold unless some prior knowledge is
            available about the structure of the missing values).
            
    normalize: bool
        Whether to normalize the distance by the number of observations. For
        example, it may not make sense to normalize something like cosine
        distance.
            
    Returns
    -------
    normalized_distance: float
        The distance between `x` and `y`, accounting for missing values as
        described above. If `normalize` is `True`, then the distance is
        normalized by the number of observed values.
    """
    # first, find the set of nan's in both
    nan_x = np.isnan(x)
    nan_y = np.isnan(y)

    nan_either = nan_x | nan_y
    remaining = ~nan_either
    
    remaining_x = x[remaining]
    remaining_y = y[remaining]
    
    num_remaining = remaining_x.shape[0]
    
    # it is possible there is no overlap
    if num_remaining == 0:
        # just say they will not be connected
        return np.inf
    
    distance = metric(remaining_x, remaining_y) / num_remaining
    
    return distance