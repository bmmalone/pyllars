"""
Helper functions related to machine learning tasks
"""
import collections
import joblib
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import tqdm
import warnings

import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

from copy import deepcopy

import misc.utils as utils
import misc.validation_utils as validation_utils

###
# Data structures
###

_fold_data_fields = [
    'X_train',
    'y_train',
    'X_test',
    'y_test',
    'X_validation',
    'y_validation',
    'train_indices',
    'test_indices',
    'validation_indices'
]
fold_data = collections.namedtuple('fold_data', ' '.join(_fold_data_fields))

_split_masks_fields = [
    'training',
    'validation',
    'test'
]
split_masks = collections.namedtuple('split_masks', ' '.join(_split_masks_fields))

def get_cv_folds(y,
        num_splits=10,
        use_stratified=True,
        shuffle=True,
        random_state=8675309):
    """ Assign a split to each row based on the values of `y`
    
    Parameters
    ----------
    y : np.array-like of ints
        The target variable for each row in a data frame. This is used to
        determine the stratification.
        
    num_splits : int
        The number of stratified splits to use

    use_stratified : bool
        Whether to use stratified cross-validation. For example, this may be
        set to False if choosing folds for regression.
        
    shuffle : bool
        Whether to shuffle during the split
        
    random_state : int
        The state for the random number generator
        
    Returns
    -------
    splits : np.array of ints
        The split of each row
    """
    if use_stratified:
        cv = sklearn.model_selection.StratifiedKFold(
            n_splits=num_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    else:
        cv = sklearn.model_selection.KFold(
            n_splits=num_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    
    splits = np.zeros(len(y), dtype=int)
    for fold, (train, test) in enumerate(cv.split(y,y)):
        splits[test] = fold
        
    return splits
        

def get_train_val_test_splits(
        df,
        training_splits=None,
        validation_splits=None,
        test_splits=None,
        split_field='split'):
    """ Get the appropriate training, validation, and testing split masks
    
    The `split_field` column in `df` is used to assign each row to a particular
    split. Then, the splits specified in the parameters are assigned as
    indicated.
    
    By default, all splits not in `validation_splits` and `test_splits` are
    assumed to belong to the training set. Thus, unless a particular training
    set is given, the returned masks will cover the entire dataset.
    
    This function does not check whether the different splits overlap. So
    care should be taken, especially if specifying the training splits
    explicitly.
    
    It is not necessary that the `split_field` values are numeric. They
    must be compatible with `isin`, however.
    
    Parameters
    ----------
    df : pd.DataFrame
        A data frame. It must contain a column named `split_field`, but
        it is not otherwise validated.
        
    training_splits : None or set-like
        The splits to use for the training set. By default, anything not
        in the `validation_splits` or `test_splits` will be placed in the
        training set.
        
        If given, this container must be compatible with `isin`.
        
    {validation,test}_splits : None or set-like
        The splits to use for the validation and test sets, respectively.
        
    split_field : string
        The name of the column indicating the split for each row.
        
    Returns
    -------
    split_masks : namedtuple with the following fields:
            * training
            * validation
            * test
    
        Masks for the respective sets. `True` positions indicate the
        rows which belong to the respective sets. All three masks are
        always returned, but a mask may be always `False` if the given
        split does not contain any rows.
    """
        
    if validation_splits is None:
        validation_splits = set()
        
    if test_splits is None:
        test_splits = set()
    
    if training_splits is None:
        training_splits = set(df[split_field].unique())
        training_splits = training_splits - validation_splits - test_splits


    m_train = df[split_field].isin(training_splits)
    m_validation = df[split_field].isin(validation_splits)
    m_test = df[split_field].isin(test_splits)
    
    ret = split_masks(
        m_train,
        m_validation,
        m_test
    )
    
    return ret

def get_fold_data(
        df,
        target_field,
        m_train,
        m_test,
        m_validation=None,
        attribute_fields=None,
        attributes_are_np_arrays=False):
    """ Create an `ml_utils.fold_data` with the given splits
    
    N.B. This function creates copies of the data, so it is not appropriate
    for very large datasets.
    
    Parameters
    ----------
    df : pd.DataFrame
        A data frame
        
    target_field : string
        The name of the column containing the target variable
        
    m_{train,test,validation} : boolean mask
        Mask indicating the training, testing, and validation set rows. If
        `m_validation` is `None` (default), then no validation set will be
        included.
        
    attribute_fields : list of strings, or None
        The names of the columns to use for attributes (that is, `X`). If
        `None` (default), then all columns except the `target_field` will
        be used as attributes
        
    attributes_are_np_arrays : bool
        Whether to stack the values from the individual rows. This should
        be set to `True` when some of the columns in `attribute_fields`
        contain numpy arrays.
        
    Returns
    -------
    fold_data : ml_utils.fold_data tuple
        A fold data with the given splits
    """
    
    # first, grab the attribute columns if not specified
    if attribute_fields is None:
        attribute_fields = df.columns.values.tolist()
        attribute_fields = attribute_fields.remove(target_field)
        
    # TODO: should check if attribute_fields is a sequence
    
    X_train = df.loc[m_train, attribute_fields].values
    X_test = df.loc[m_test, attribute_fields].values
    
    if attributes_are_np_arrays:
        X_train = np.stack(X_train)
        X_test = np.stack(X_test)
    
    y_train = df.loc[m_train, target_field].values
    y_test = df.loc[m_test, target_field].values
    
    train_indices = np.where(m_train)[0]
    test_indices = np.where(m_test)[0]
    
    X_val = None
    y_val = None
    val_indices = None
    
    if m_validation is not None:
        
        # in case we were not given any validation instances, also ignore it
        if np.sum(m_validation) > 0:

            X_val = df.loc[m_validation, attribute_fields].values
            y_val = df.loc[m_validation, target_field].values
            val_indices = np.where(m_validation)[0]

            if attributes_are_np_arrays:
                X_val = np.stack(X_val)
        
    ret = fold_data(
        X_train, y_train,
        X_test, y_test,
        X_val, y_val,
        train_indices, test_indices, val_indices
    )
    
    return ret
    

###
# Training helpers
###
def train_binary_classification_model(
        estimator_template,
        hyperparameters,
        fold_data,
        return_predictions=False):
    """ Train a copy of `estimator_template` with `hyperparameters` for
    a binary classification problem.
    
    This function also calculates the binary classification metrics on
    the test split specified by `fold_data`. Optionally, it returns the
    predictions.
    
    Parameters
    ----------
    estimator_template : sklearn.ClassifierMixin
        An sklearn classifier. It will be cloned, so it should not
        already have been `fit`. This could also be a pipeline
        
    hyperparameters : dictionary
        The hyperparameters to use for training the esimator. This
        should be something that `set_params` for `estimator_template`
        can use. For example, pipelines use the form `<component>__<parameter>`.
        
    fold_data : a `fold_data` tuple
        The training and testing split.
        
        N.B. Currently, this function *does not* use the validation
        split.
        
    return_predictions : bool
        Whether to return the actual predictions.
    
    Returns
    -------
    fit_estimator : sklearn.ClassifierMixin
        A copy of `estimator_template` after fitting it on the training
        data.
        
    metrics : dictionary
        Various binary classification metrics characterizing the performance
        on the testing data. Please see `collect_binary_classification_metrics`
        for more details.
        
    y_proba_pred : np.array if `return_predictions` is `True`
        The predicted probabilities of each class
    """
    
    estimator = sklearn.clone(estimator_template)
    estimator.set_params(**hyperparameters)
    
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=sklearn.exceptions.UndefinedMetricWarning)
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        estimator_fit = estimator.fit(fold_data.X_train, fold_data.y_train)
        y_proba_pred = estimator_fit.predict_proba(fold_data.X_test)
        metrics = collect_binary_classification_metrics(fold_data.y_test, y_proba_pred)

    ret = [estimator_fit, metrics]
    
    if return_predictions:
        ret.append(y_proba_pred)
        
    return ret

###
# Evaluation helpers
###
def collect_regression_metrics(y_true, y_pred):
    """ Collect various classification performance metrics for the predictions

    Parameters
    ----------
    y_true: np.array of real values
        The true value of each instance

    y_pred: np.array of floats
        The prediction for each instance
    
    Returns
    -------
    metrics: dict
        A mapping from the metric name to the respective value
    """
    validation_utils.validate_equal_shape(y_true, y_pred)

    ret = {
        "explained_variance": sklearn.metrics.explained_variance_score(y_true, y_pred),
        "mean_absolute_error": sklearn.metrics.mean_absolute_error(y_true, y_pred),
        "mean_squared_error": sklearn.metrics.mean_squared_error(y_true, y_pred),
        #"mean_squared_log_error": sklearn.metrics.mean_squared_log_error(y_true, y_pred),
        "median_absolute_error": sklearn.metrics.median_absolute_error(y_true, y_pred),
        "r2": sklearn.metrics.r2_score(y_true, y_pred)
    }

    return ret


def collect_multiclass_classification_metrics(y_true, y_score):
    """ Calculate various multi-class classification performance metrics
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though
        they are not required to be probabilities
        
    Returns
    -------
    metrics: dict
        A mapping from the metric name to the respective value
    """

    # make hard predictions
    y_pred = np.argmax(y_score, axis=1)

    # now collect all statistics
    ret = {
         "cohen_kappa":  sklearn.metrics.cohen_kappa_score(y_true, y_pred),
         #"matthews_corrcoef":  sklearn.metrics.matthews_corrcoef(y_true, y_pred),
         "accuracy":  sklearn.metrics.accuracy_score(y_true, y_pred),
         "micro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='micro'),
         "macro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='macro'),
         "hamming_loss":  sklearn.metrics.hamming_loss(y_true, y_pred),
         "micro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='micro'),
         "macro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='macro'),
         "micro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='micro'),
         "macro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='macro'),
         "hand_and_till_m_score": calc_hand_and_till_m_score(y_true, y_score),
         "provost_and_domingos_auc": calc_provost_and_domingos_auc(y_true, y_score)
    }

    return ret


def collect_binary_classification_metrics(y_true, y_probas_pred, threshold=0.5,
        pos_label=1):
    """ Collect various classification performance metrics for the predictions

    N.B. This function assumes the second column in y_probas_pred gives the
    score for the "positive" class.

    Parameters
    ----------
    y_true: np.array of binary-like values
        The true class of each instance

    y_probas_pred: 2-d np.array of floats, shape is (num_instances, 2)
        The score of each prediction for each instance
    
    threshold: float
        The score threshold to choose "positive" predictions

    pos_label: str or int
        The "positive" class for some metrics

    Returns
    -------
    metrics: dict
        A mapping from the metric name to the respective value
    """

    # first, validate the input
    if y_true.shape[0] != y_probas_pred.shape[0]:
        msg = ("[math_utils.collect_binary_classification_metrics]: y_true "
            "and y_probas_pred do not have matching shapes. y_true: {}, "
            "y_probas_pred: {}".format(y_true.shape, y_probas_pred.shape))
        raise ValueError(msg)

    if y_probas_pred.shape[1] != 2:
        msg = ("[math_utils.collect_binary_classification_metrics]: "
            "y_probas_pred does not have scores for exactly two classes: "
            "y_probas_pred.shape: {}".format(y_probas_pred.shape))
        raise ValueError(msg)


    # first, pull out the probability of positive classes
    y_score = y_probas_pred[:,1]

    # and then make a hard prediction
    y_pred = (y_score >= threshold)

    # now collect all statistics
    ret = {
         "cohen_kappa":  sklearn.metrics.cohen_kappa_score(y_true, y_pred),
         "hinge_loss":  sklearn.metrics.hinge_loss(y_true, y_score),
         "matthews_corrcoef":  sklearn.metrics.matthews_corrcoef(y_true, y_pred),
         "accuracy":  sklearn.metrics.accuracy_score(y_true, y_pred),
         "binary_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='binary', pos_label=pos_label),
         "micro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='micro', pos_label=pos_label),
         "macro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='macro', pos_label=pos_label),
         "hamming_loss":  sklearn.metrics.hamming_loss(y_true, y_pred),
         "jaccard_similarity_score":  sklearn.metrics.jaccard_similarity_score(
            y_true, y_pred),
         "log_loss":  sklearn.metrics.log_loss(y_true, y_probas_pred),
         "micro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='micro', pos_label=pos_label),
         "binary_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='binary', pos_label=pos_label),
         "macro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='macro', pos_label=pos_label),
         "micro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='micro', pos_label=pos_label),
         "macro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='macro', pos_label=pos_label),
         "binary_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='binary', pos_label=pos_label),
         "zero_one_loss":  sklearn.metrics.zero_one_loss(y_true, y_pred),
         "micro_average_precision":  sklearn.metrics.average_precision_score(
            y_true, y_score, average='micro'),
         "macro_average_precision":  sklearn.metrics.average_precision_score(
            y_true, y_score, average='macro'),
         "micro_roc_auc_score":  sklearn.metrics.roc_auc_score(y_true, y_score,
            average='micro'),
         "macro_roc_auc_score":  sklearn.metrics.roc_auc_score(y_true, y_score,
            average='macro')
    }

    return ret

def _calc_hand_and_till_a_value(y_true, y_score, i, j):
    """ Calculate the \hat{A} value in Equation (3) of:
    
    Hand, D. & Till, R. A Simple Generalisation of the Area Under the ROC Curve
    for Multiple Class Classification Problems Machine Learning, 2001, 45, 171-186.
    
    Specifically:
        A(i|j) = \frac{ S_i - n_i*(n_i + 1)/2 }{n_i * n_j},

        where n_i, n_j are the count of instances of the respective classes and
        S_i is the (base-1) sum of the ranks of class i
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded with
        integers [0, 1, ... n_classes-1]. The respective columns in y_prob should
        give the probabilities of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though they
        are not required to be probabilities
        
    i, j: integers
        The class indices
        
    Returns
    -------
    a_hat: float
        The \hat{A} value from Equation (3) referenced above. Specifically, this is
        the probability that a randomly drawn member of class j will have a lower
        estimated score for belonging to class i than a randomly drawn member
        of class i.
    """
    # so first pull out all elements of class i or j
    m_j = (y_true == j)
    m_i = (y_true == i)
    m_ij = (m_i | m_j)

    y_true_ij = y_true[m_ij]
    y_score_ij = y_score[m_ij]

    # count them
    n_i = np.sum(m_i)
    n_j = np.sum(m_j)

    # likelihood of class i
    y_score_i_ij = zip(y_true_ij, y_score_ij[:,i])

    # rank the instances
    sorted_c_pi = np.array(sorted(y_score_i_ij, key=lambda a: a[1]))

    # sum the ranks for class i

    # first, find where the class_i's are
    m_ci = sorted_c_pi[:,0] == i

    # ranks are base-1, so add 1
    ci_ranks = np.where(m_ci)[0] + 1
    s_i = np.sum(ci_ranks)

    a_i_given_j = s_i - n_i * (n_i + 1)/2
    a_i_given_j /= (n_i * n_j)

    return a_i_given_j

def calc_hand_and_till_m_score(y_true, y_score):
    """ Calculate the "M" score from Equation (7) of:
    
    Hand, D. & Till, R. A Simple Generalisation of the Area Under the ROC Curve
    for Multiple Class Classification Problems Machine Learning, 2001, 45,
    171-186.
    
    This is typically taken as a good multi-class extension of the AUC score.
    For more details, see:
    
    Fawcett, T. An introduction to ROC analysis Pattern Recognition Letters,
    2006, 27, 861 - 874.

    N.B. This function *can* handle unobserved labels, except for the label
    with the highest index. In particular:

    y_score.shape[1] != np.max(np.unique(y_true)) + 1 causes an error.

    N.B. In case y_score contains any np.nan's, those will be removed before
    calculating the M score.
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though
        they are not required to be probabilities
        
    Returns
    -------
    m: float
        The "multi-class AUC" score referenced above
    """

    #msg = ("[hand_and_till] y_true: {}. y_score: {}".format(y_true, y_score))
    #print(msg)
    
    classes = np.unique(y_true)

    # make sure the classes are integers, or we will have problems indexing
    classes = np.array(classes, dtype=int)
    num_classes = np.max(classes)+1

    # first, validate our input
    if y_true.shape[0] != y_score.shape[0]:
        msg = ("[math_utils.m_score]: y_true and y_score do not have matching "
            "shapes. y_true: {}, y_score: {}".format(y_true.shape,
            y_score.shape))
        raise ValueError(msg)
    
    if y_score.shape[1] != (num_classes):
        msg = ("[math_utils.m_score]: y_score does not have the expected "
            "number of columns based on the maximum observed class in y_true. "
            "y_score.shape: {}. expected number of columns: {}".format(
            y_score.shape, num_classes))
        raise ValueError(msg)

    # clear out the np.nan's
    m_nan = np.any(np.isnan(y_score), axis=1)
    y_score = y_score[~m_nan]
    y_true = y_true[~m_nan]
    
    # the specific equation is:
    #
    # M = \frac{2}{c*(c-1)}*\sum_{i<j} {\hat{A}(i,j)},
    #
    # where \hat{A}(i,j) is \frac{A(i|j) + A(i|j)}{2}
    ij_pairs = itertools.combinations(classes, 2)

    m = 0
    for i,j in ij_pairs:
        a_ij = _calc_hand_and_till_a_value(y_true, y_score, i,j)
        a_ji = _calc_hand_and_till_a_value(y_true, y_score, j, i)

        m += (a_ij + a_ji) / 2
    
    m_1 = num_classes * (num_classes - 1)
    m_1 = 2 / m_1
    m = m_1 * m

    #print("[hand_and_till] m: {}".format(m))
    return m


def calc_provost_and_domingos_auc(y_true, y_score):
    """ Calculate the "M" score from Equation (7) of:
    
    Provost, F. & Domingos, P. Well-Trained PETs: Improving Probability
    Estimation Trees Sterm School of Business, NYU, Sterm School of
    Business, NYU, 2000.
    
    This is typically taken as a good multi-class extension of the AUC score.
    For more details, see:
    
    Fawcett, T. An introduction to ROC analysis Pattern Recognition Letters,
    2006, 27, 861 - 874.

    N.B. This function *can* handle unobserved labels, except for the label
    with the highest index. In particular:

    y_score.shape[1] != np.max(np.unique(y_true)) + 1 causes an error.
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though
        they are not required to be probabilities
        
    Returns
    -------
    m: float
        The "multi-class AUC" score referenced above
    """
    
    classes = np.unique(y_true)

    # make sure the classes are integers, or we will have problems indexing
    classes = np.array(classes, dtype=int)

    num_classes = np.max(classes)+1
    
    # first, validate our input
    if y_true.shape[0] != y_score.shape[0]:
        msg = ("[math_utils.m_score]: y_true and y_score do not have matching "
            "shapes. y_true: {}, y_score: {}".format(y_true.shape,
            y_score.shape))
        raise ValueError(msg)

    if y_score.shape[1] != (num_classes):
        msg = ("[math_utils.m_score]: y_score does not have the expected "
            "number of columns based on the maximum observed class in y_true. "
            "y_score.shape: {}. expected number of columns: {}".format(
            y_score.shape, num_classes))
        raise ValueError(msg)
        
    m = 0
    
    for c in classes:
        m_c = y_true == c
        p_c = np.sum(m_c) / len(y_true)

        y_true_c = (y_true == c)
        y_score_c = y_score[:,c]

        m_nan = np.isnan(y_score_c)
        y_score_c = y_score_c[~m_nan]
        y_true_c = y_true_c[~m_nan]

        auc_c = sklearn.metrics.roc_auc_score(y_true_c, y_score_c)
        a_c = auc_c * p_c
        
        m += a_c
        
    return m

###
# Preprocessing helpers
###
def scale_single_epitope_features(epitope_features, scalers):
    """ Scale the epitope features
    
    Parameters
    ----------
    epitope_features : np.array
        The features for a single epitope
        
    scalers : list of StandardScalers
        The fit scalers for each feature. Presumably, these come
        from `scale_epitope_features`.
        
    Returns
    -------
    scaled_epitope_features : np.array
        The scaled features
    """
    res = np.zeros_like(epitope_features)

    for i in range(epitope_features.shape[1]):
        scaler = scalers[i]

        vals = epitope_features[:,i]
        vals = vals.reshape(1, -1)
        vals = scaler.transform(vals)
        vals = vals.flatten()

        res[:,i] = vals
        
    return res

def scale_features(X):
    """ Scale each feature independently
    
    Parameters
    ----------
    X : 3d np.array
    
    Returns
    -------
    X_scaled : the scaled data
        Either a copy or updated version of `X`
    
    scalers : list of sklearn.StandardScalers
        The scalers for all features
    """
    X_scaled = np.zeros_like(X)
    
    scalers = []
    for i in range(X.shape[2]):

        # fit the scaler
        s = X[:,:,i].shape
        vals = X[:,:,i].reshape(-1, 1)
        scaler = sklearn.preprocessing.StandardScaler().fit(vals)

        # scale the training data
        vals = scaler.transform(vals).reshape(s)    
        X_scaled[:,:,i] = vals
        
        scalers.append(scaler)
        
    return X_scaled, scalers

def scale_epitope_features(fold_data, copy=True):
    """ Scale each epitope feature independently
    
    This function uses only the training set for estimating
    the mean and variance for scaling.
    
    Parameters
    ----------
    fold_data : fold_data named tuple
    
    copy : bool
        Whether to make a copy
    
    Returns
    -------
    scaled_fold_data : the scaled fold data
        Either a copy or updated version of `fold_data`
    
    scalers : list of sklearn.StandardScalers
        The scalers for all features
    """
    
    if copy:
        fold_data = deepcopy(fold_data)
    
    scalers = []
    for i in range(fold_data.X_train.shape[2]):

        # fit the scaler
        s = fold_data.X_train[:,:,i].shape
        vals = fold_data.X_train[:,:,i].reshape(-1, 1)
        scaler = sklearn.preprocessing.StandardScaler().fit(vals)

        # scale the training data
        vals = scaler.transform(vals).reshape(s)    
        fold_data.X_train[:,:,i] = vals

        # scale the validation data (with the same scaler)
        if fold_data.X_validation is not None:
            s = fold_data.X_validation[:,:,i].shape
            vals = fold_data.X_validation[:,:,i].reshape(-1, 1)
            vals = scaler.transform(vals).reshape(s)
            fold_data.X_validation[:,:,i] = vals

        # scale the test data (with the same scaler)
        if fold_data.X_test is not None:
            s = fold_data.X_test[:,:,i].shape
            vals = fold_data.X_test[:,:,i].reshape(-1, 1)
            vals = scaler.transform(vals).reshape(s)
            fold_data.X_test[:,:,i] = vals

        scalers.append(scaler)
        
    return fold_data, scalers

def make_two_class_set(fd):
    """ Convert boolean output into two-tuple output
    
    This is helpful for certain neural network structures
    
    Parameters
    ----------
    fd : fold_data named tuple
    
    Returns
    -------
    updated_fold_data : the scaled fold data
        Either a copy or updated version of `fold_data`
    """

    y_train = np.ones((fd.y_train.shape[0], 2))
    y_train[:,1] = fd.y_train
    y_train[:,0] = 1 - fd.y_train

    y_val = None
    if fd.y_validation is not None:
        y_val = np.ones((fd.y_validation.shape[0], 2))
        y_val[:,1] = fd.y_validation
        y_val[:,0] = 1 - fd.y_validation

    y_test = None
    if fd.y_test is not None:
        y_test = np.ones((fd.y_test.shape[0], 2))
        y_test[:,1] = fd.y_test
        y_test[:,0] = 1 - fd.y_test
    
    ret = fold_data(
        fd.X_train, y_train,
        fd.X_test, y_test,
        fd.X_validation, y_val,
        fd.train_indices, fd.test_indices, fd.validation_indices
    )
    
    return ret


def collect_regression_metrics(y_true, y_pred):
    """ Collect various classification performance metrics for the predictions

    Parameters
    ----------
    y_true: np.array of real values
        The true value of each instance

    y_pred: np.array of floats
        The prediction for each instance
    
    Returns
    -------
    metrics: dict
        A mapping from the metric name to the respective value
    """
    validation_utils.validate_equal_shape(y_true, y_pred)

    ret = {
        "explained_variance": sklearn.metrics.explained_variance_score(y_true, y_pred),
        "mean_absolute_error": sklearn.metrics.mean_absolute_error(y_true, y_pred),
        "mean_squared_error": sklearn.metrics.mean_squared_error(y_true, y_pred),
        #"mean_squared_log_error": sklearn.metrics.mean_squared_log_error(y_true, y_pred),
        "median_absolute_error": sklearn.metrics.median_absolute_error(y_true, y_pred),
        "r2": sklearn.metrics.r2_score(y_true, y_pred)
    }

    return ret


def collect_multiclass_classification_metrics(y_true, y_score):
    """ Calculate various multi-class classification performance metrics
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though
        they are not required to be probabilities
        
    Returns
    -------
    metrics: dict
        A mapping from the metric name to the respective value
    """

    # make hard predictions
    y_pred = np.argmax(y_score, axis=1)

    # now collect all statistics
    ret = {
         "cohen_kappa":  sklearn.metrics.cohen_kappa_score(y_true, y_pred),
         #"matthews_corrcoef":  sklearn.metrics.matthews_corrcoef(y_true, y_pred),
         "accuracy":  sklearn.metrics.accuracy_score(y_true, y_pred),
         "micro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='micro'),
         "macro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='macro'),
         "hamming_loss":  sklearn.metrics.hamming_loss(y_true, y_pred),
         "micro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='micro'),
         "macro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='macro'),
         "micro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='micro'),
         "macro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='macro'),
         "hand_and_till_m_score": calc_hand_and_till_m_score(y_true, y_score),
         "provost_and_domingos_auc": calc_provost_and_domingos_auc(y_true, y_score)
    }

    return ret


def collect_binary_classification_metrics(y_true, y_probas_pred, threshold=0.5,
        pos_label=1):
    """ Collect various classification performance metrics for the predictions

    N.B. This function assumes the second column in y_probas_pred gives the
    score for the "positive" class.

    Parameters
    ----------
    y_true: np.array of binary-like values
        The true class of each instance

    y_probas_pred: 2-d np.array of floats, shape is (num_instances, 2)
        The score of each prediction for each instance
    
    threshold: float
        The score threshold to choose "positive" predictions

    pos_label: str or int
        The "positive" class for some metrics

    Returns
    -------
    metrics: dict
        A mapping from the metric name to the respective value
    """

    # first, validate the input
    if y_true.shape[0] != y_probas_pred.shape[0]:
        msg = ("[math_utils.collect_binary_classification_metrics]: y_true "
            "and y_probas_pred do not have matching shapes. y_true: {}, "
            "y_probas_pred: {}".format(y_true.shape, y_probas_pred.shape))
        raise ValueError(msg)

    if y_probas_pred.shape[1] != 2:
        msg = ("[math_utils.collect_binary_classification_metrics]: "
            "y_probas_pred does not have scores for exactly two classes: "
            "y_probas_pred.shape: {}".format(y_probas_pred.shape))
        raise ValueError(msg)


    # first, pull out the probability of positive classes
    y_score = y_probas_pred[:,1]

    # and then make a hard prediction
    y_pred = (y_score >= threshold)

    # now collect all statistics
    ret = {
         "cohen_kappa":  sklearn.metrics.cohen_kappa_score(y_true, y_pred),
         "hinge_loss":  sklearn.metrics.hinge_loss(y_true, y_score),
         "matthews_corrcoef":  sklearn.metrics.matthews_corrcoef(y_true, y_pred),
         "accuracy":  sklearn.metrics.accuracy_score(y_true, y_pred),
         "binary_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='binary', pos_label=pos_label),
         "micro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='micro', pos_label=pos_label),
         "macro_f1_score":  sklearn.metrics.f1_score(y_true, y_pred,
            average='macro', pos_label=pos_label),
         "hamming_loss":  sklearn.metrics.hamming_loss(y_true, y_pred),
         "jaccard_similarity_score":  sklearn.metrics.jaccard_similarity_score(
            y_true, y_pred),
         "log_loss":  sklearn.metrics.log_loss(y_true, y_probas_pred),
         "micro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='micro', pos_label=pos_label),
         "binary_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='binary', pos_label=pos_label),
         "macro_precision":  sklearn.metrics.precision_score(y_true, y_pred,
            average='macro', pos_label=pos_label),
         "micro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='micro', pos_label=pos_label),
         "macro_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='macro', pos_label=pos_label),
         "binary_recall":  sklearn.metrics.recall_score(y_true, y_pred,
            average='binary', pos_label=pos_label),
         "zero_one_loss":  sklearn.metrics.zero_one_loss(y_true, y_pred),
         "micro_average_precision":  sklearn.metrics.average_precision_score(
            y_true, y_score, average='micro'),
         "macro_average_precision":  sklearn.metrics.average_precision_score(
            y_true, y_score, average='macro'),
         "micro_roc_auc_score":  sklearn.metrics.roc_auc_score(y_true, y_score,
            average='micro'),
         "macro_roc_auc_score":  sklearn.metrics.roc_auc_score(y_true, y_score,
            average='macro')
    }

    return ret



def _calc_hand_and_till_a_value(y_true, y_score, i, j):
    """ Calculate the \hat{A} value in Equation (3) of:
    
    Hand, D. & Till, R. A Simple Generalisation of the Area Under the ROC Curve
    for Multiple Class Classification Problems Machine Learning, 2001, 45, 171-186.
    
    Specifically:
        A(i|j) = \frac{ S_i - n_i*(n_i + 1)/2 }{n_i * n_j},

        where n_i, n_j are the count of instances of the respective classes and
        S_i is the (base-1) sum of the ranks of class i
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded with
        integers [0, 1, ... n_classes-1]. The respective columns in y_prob should
        give the probabilities of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though they
        are not required to be probabilities
        
    i, j: integers
        The class indices
        
    Returns
    -------
    a_hat: float
        The \hat{A} value from Equation (3) referenced above. Specifically, this is
        the probability that a randomly drawn member of class j will have a lower
        estimated score for belonging to class i than a randomly drawn member
        of class i.
    """
    # so first pull out all elements of class i or j
    m_j = (y_true == j)
    m_i = (y_true == i)
    m_ij = (m_i | m_j)

    y_true_ij = y_true[m_ij]
    y_score_ij = y_score[m_ij]

    # count them
    n_i = np.sum(m_i)
    n_j = np.sum(m_j)

    # likelihood of class i
    y_score_i_ij = zip(y_true_ij, y_score_ij[:,i])

    # rank the instances
    sorted_c_pi = np.array(sorted(y_score_i_ij, key=lambda a: a[1]))

    # sum the ranks for class i

    # first, find where the class_i's are
    m_ci = sorted_c_pi[:,0] == i

    # ranks are base-1, so add 1
    ci_ranks = np.where(m_ci)[0] + 1
    s_i = np.sum(ci_ranks)

    a_i_given_j = s_i - n_i * (n_i + 1)/2
    a_i_given_j /= (n_i * n_j)

    return a_i_given_j

def calc_hand_and_till_m_score(y_true, y_score):
    """ Calculate the "M" score from Equation (7) of:
    
    Hand, D. & Till, R. A Simple Generalisation of the Area Under the ROC Curve
    for Multiple Class Classification Problems Machine Learning, 2001, 45,
    171-186.
    
    This is typically taken as a good multi-class extension of the AUC score.
    For more details, see:
    
    Fawcett, T. An introduction to ROC analysis Pattern Recognition Letters,
    2006, 27, 861 - 874.

    N.B. This function *can* handle unobserved labels, except for the label
    with the highest index. In particular:

    y_score.shape[1] != np.max(np.unique(y_true)) + 1 causes an error.

    N.B. In case y_score contains any np.nan's, those will be removed before
    calculating the M score.
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though
        they are not required to be probabilities
        
    Returns
    -------
    m: float
        The "multi-class AUC" score referenced above
    """

    #msg = ("[hand_and_till] y_true: {}. y_score: {}".format(y_true, y_score))
    #print(msg)
    
    classes = np.unique(y_true)

    # make sure the classes are integers, or we will have problems indexing
    classes = np.array(classes, dtype=int)
    num_classes = np.max(classes)+1

    # first, validate our input
    if y_true.shape[0] != y_score.shape[0]:
        msg = ("[math_utils.m_score]: y_true and y_score do not have matching "
            "shapes. y_true: {}, y_score: {}".format(y_true.shape,
            y_score.shape))
        raise ValueError(msg)
    
    if y_score.shape[1] != (num_classes):
        msg = ("[math_utils.m_score]: y_score does not have the expected "
            "number of columns based on the maximum observed class in y_true. "
            "y_score.shape: {}. expected number of columns: {}".format(
            y_score.shape, num_classes))
        raise ValueError(msg)

    # clear out the np.nan's
    m_nan = np.any(np.isnan(y_score), axis=1)
    y_score = y_score[~m_nan]
    y_true = y_true[~m_nan]
    
    # the specific equation is:
    #
    # M = \frac{2}{c*(c-1)}*\sum_{i<j} {\hat{A}(i,j)},
    #
    # where \hat{A}(i,j) is \frac{A(i|j) + A(i|j)}{2}
    ij_pairs = itertools.combinations(classes, 2)

    m = 0
    for i,j in ij_pairs:
        a_ij = _calc_hand_and_till_a_value(y_true, y_score, i,j)
        a_ji = _calc_hand_and_till_a_value(y_true, y_score, j, i)

        m += (a_ij + a_ji) / 2
    
    m_1 = num_classes * (num_classes - 1)
    m_1 = 2 / m_1
    m = m_1 * m

    #print("[hand_and_till] m: {}".format(m))
    return m


def calc_provost_and_domingos_auc(y_true, y_score):
    """ Calculate the "M" score from Equation (7) of:
    
    Provost, F. & Domingos, P. Well-Trained PETs: Improving Probability
    Estimation Trees Sterm School of Business, NYU, Sterm School of
    Business, NYU, 2000.
    
    This is typically taken as a good multi-class extension of the AUC score.
    For more details, see:
    
    Fawcett, T. An introduction to ROC analysis Pattern Recognition Letters,
    2006, 27, 861 - 874.

    N.B. This function *can* handle unobserved labels, except for the label
    with the highest index. In particular:

    y_score.shape[1] != np.max(np.unique(y_true)) + 1 causes an error.
    
    Parameters
    ----------
    y_true: np.array with shape [n_samples]
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
    y_score: np.array with shape [n_samples, n_classes]
        The score predictions for each class, e.g., from pred_proba, though
        they are not required to be probabilities
        
    Returns
    -------
    m: float
        The "multi-class AUC" score referenced above
    """
    
    classes = np.unique(y_true)

    # make sure the classes are integers, or we will have problems indexing
    classes = np.array(classes, dtype=int)

    num_classes = np.max(classes)+1
    
    # first, validate our input
    if y_true.shape[0] != y_score.shape[0]:
        msg = ("[math_utils.m_score]: y_true and y_score do not have matching "
            "shapes. y_true: {}, y_score: {}".format(y_true.shape,
            y_score.shape))
        raise ValueError(msg)

    if y_score.shape[1] != (num_classes):
        msg = ("[math_utils.m_score]: y_score does not have the expected "
            "number of columns based on the maximum observed class in y_true. "
            "y_score.shape: {}. expected number of columns: {}".format(
            y_score.shape, num_classes))
        raise ValueError(msg)
        
    m = 0
    
    for c in classes:
        m_c = y_true == c
        p_c = np.sum(m_c) / len(y_true)

        y_true_c = (y_true == c)
        y_score_c = y_score[:,c]

        m_nan = np.isnan(y_score_c)
        y_score_c = y_score_c[~m_nan]
        y_true_c = y_true_c[~m_nan]

        auc_c = sklearn.metrics.roc_auc_score(y_true_c, y_score_c)
        a_c = auc_c * p_c
        
        m += a_c
        
    return m