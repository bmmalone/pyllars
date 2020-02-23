"""
This module contains utilities for common machine learning tasks.

In particular, this module focuses on tasks "surrounding" machine learning,
such as cross-fold splitting, performance evaluation, etc. It does not include
helpers for use directly in :py:class:`sklearn.pipeline.Pipeline`.
"""
import logging
logger = logging.getLogger(__name__)

import collections
import itertools
import joblib
import json
import networkx as nx
import numpy as np
import pandas as pd
import sklearn
import tqdm
import warnings

import sklearn
import sklearn.metrics
import sklearn.model_selection
import sklearn.preprocessing

from copy import deepcopy

import pyllars.collection_utils as collection_utils
import pyllars.utils as utils
import pyllars.validation_utils as validation_utils

from typing import Any, Callable, Container, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

###
# Data structures
###


class fold_data(NamedTuple):
    """
    A named tuple for holding train, validation, and test datasets suitable for use
    in `sklearn`.
    
    This class can be more convenient than :class:`pyllars.ml_utils.split_masks` for
    modest-sized datasets.

    Attributes
    ------------
    X_{train,test,validation} : numpy.ndarray
        The `X` data (features) for the respective dataset splits
    y_{train,test,validation} : numpy.ndarray
        The `y` data (target) for the respective dataset splits
    {train_test,validation}_indices : numpy.ndarray
        The row indices from the original dataset of the respective dataset splits
    """
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    X_validation : np.ndarray
    y_validation : np.ndarray
    train_indices : np.ndarray
    test_indices : np.ndarray
    validation_indices : np.ndarray

class split_masks(NamedTuple):
    """
    A named tuple for holding boolean masks for the train, validation, and test splits
    of a complete dataset.
    
    These masks can be used to index :py:class:`numpy.ndarray` or :py:class:`pandas.DataFrame`
    objects to extract the relevant dataset split for `sklearn`. This class can  be more
    appropriate than :class:`pyllars.ml_utils.fold_data` for large objects since it avoids
    any copies of the data.
    
    Attributes
    -----------
    training,test,validation : numpy.ndarray
        Boolean masks for the respective dataset splits
    """
    training : np.ndarray
    validation : np.ndarray
    test : np.ndarray
    
class estimators_predictions_metrics(NamedTuple):
    """
    A named tuple for holding fit estimators, predictions on the
    respective datasets, and results.
    
    Attributes
    ------------
    estimator_{val,test} : sklearn.base.BaseEstimators
        Estimators fit on the respective datasets.
    predictions_{val,test} : numpy.ndarray
        Predictions of the respective models.
    metrics_{val,test} : typing.Dict
        Metrics for the respective datasets.
    fold_{train,val,test} : typing.Any
        The identifiers of the respective folds. 
    hyperparameters{_str} : typing.Optional[typing.Dict]
        The hyperparameters (in a string format) for
        training the models.
    """
    estimator_val : sklearn.base.BaseEstimator
    estimator_test : sklearn.base.BaseEstimator
    predictions_val : np.ndarray
    predictions_test : np.ndarray
    true_val : np.ndarray
    true_test : np.ndarray
    metrics_val : Dict
    metrics_test : Dict
    fold_train : Any
    fold_val : Any
    fold_test : Any
    hyperparameters : Optional[Dict]
    hyperparameters_str : str
    
###
# Cross-validation helpers
###


def get_cv_folds(y:np.ndarray,
        num_splits:int=10,
        use_stratified:bool=True,
        shuffle:bool=True,
        random_state:int=8675309) -> np.ndarray:
    """ Assign a split to each row based on the values of `y`
    
    Parameters
    ----------
    y : numpy.ndarray
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
    splits : numpy.ndarray
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
        df:pd.DataFrame,
        training_splits:Optional[Set]=None,
        validation_splits:Optional[Set]=None,
        test_splits:Optional[Set]=None,
        split_field:str='split') -> split_masks:
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
    df : pandas.DataFrame
        A data frame. It must contain a column named `split_field`, but
        it is not otherwise validated.
        
    training_splits : typing.Optional[typing.Set]
        The splits to use for the training set. By default, anything not
        in the `validation_splits` or `test_splits` will be placed in the
        training set.
        
        If given, this container must be compatible with `isin`. Otherwise,
        it will be wrapped in a set.
        
    {validation,test}_splits :  typing.Optional[typing.Set]
        The splits to use for the validation and test sets, respectively.
        
        If given, this container must be compatible with `isin`. Otherwise,
        it will be wrapped in a set.
        
    split_field : str
        The name of the column indicating the split for each row.
        
    Returns
    -------
    split_masks : pyllars.ml_utils.split_masks
    
        Masks for the respective sets. `True` positions indicate the
        rows which belong to the respective sets. All three masks are
        always returned, but a mask may be always `False` if the given
        split does not contain any rows.
    """ 
    validation_splits = collection_utils.wrap_in_set(validation_splits)
    test_splits = collection_utils.wrap_in_set(test_splits)
    
    if training_splits is None:
        training_splits = set(df[split_field].unique())
        training_splits = training_splits - validation_splits - test_splits
    else:
        training_splits = collection_utils.wrap_in_set(training_splits)


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
        df:pd.DataFrame,
        target_field:str,
        m_train:np.ndarray,
        m_test:np.ndarray,
        m_validation:Optional[np.ndarray]=None,
        attribute_fields:Optional[Iterable[str]]=None,
        fields_to_ignore:Optional[Iterable[str]]=None,
        attributes_are_np_arrays:bool=False) -> fold_data:
    """ Prepare a data frame for `sklearn` according to the given splits
    
    **N.B.** This function creates copies of the data, so it is not appropriate
    for very large datasets.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A data frame
        
    target_field : str
        The name of the column containing the target variable
        
    m_{train,test,validation} : np.ndarray
        Boolean masks indicating the training, testing, and validation set rows.
        If `m_validation` is `None` (default), then no validation set will be
        included.
        
    attribute_fields : typing.Optional[typing.Iterable[str]]
        The names of the columns to use for attributes (that is, `X`). If
        `None` (default), then all columns except the `target_field` will
        be used as attributes.
        
    fields_to_ignore : typing.Optional[typing.Container[str]]
        The names of the columns to ignore.
        
    attributes_are_np_arrays : bool
        Whether to stack the values from the individual rows. This should
        be set to `True` when some of the columns in `attribute_fields`
        contain numpy arrays.
        
    Returns
    -------
    fold_data : pyllars.ml_utils.fold_data
        A named tuple with the given splits
    """
    caller = 'ml_utils.get_fold_data' # for validation
    
    # first, grab the attribute columns if not specified
        
    if attribute_fields is None:
        attribute_fields = df.columns.values.tolist()
        attribute_fields.remove(target_field)
        
    attribute_fields = deepcopy(attribute_fields)
    
    if fields_to_ignore is not None:
        # make sure to wrap strings, etc., so they behave as expected
        fields_to_ignore = collection_utils.wrap_in_set(fields_to_ignore)
        attribute_fields = collection_utils.list_remove_list(
            attribute_fields, fields_to_ignore)
        
    validation_utils.validate_is_sequence(
        attribute_fields,
        name='attribute_fields',
        caller=caller
    )
    
    if len(attribute_fields) == 1:
        msg = ("[{}]: found a single attribute field. Treating as a "
            "single column".format(caller))
        logger.warning(msg)
        attribute_fields = attribute_fields[0]
    
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

def _train_and_evaluate(
        estimator,
        X_train,
        y_train,
        X_test,
        y_test,
        target_transform,
        target_inverse_transform,
        collect_metrics,
        collect_metrics_kwargs,
        use_predict_proba):
    """ Train and evaluate `estimator` on the given datasets
    
    This function is a helper for `evaluate_hyperparameters`. It is
    not intended for external use.
    """
    # transform the target, if necessary
    if target_transform is not None:
        y_train = target_transform(y_train)
        
    # train the estimator
    estimator_fit = estimator.fit(X_train, y_train)
    
    # make predictions
    if use_predict_proba:
        y_pred = estimator_fit.predict_proba(X_test)
    else:        
        y_pred = estimator_fit.predict(X_test)
    
    # transform back, if needed
    if target_inverse_transform is not None:
        y_pred = target_inverse_transform(y_pred)
        
    # evaluate
    metrics = collect_metrics(
        y_test,
        y_pred,
        **collect_metrics_kwargs
    )
    
    return estimator_fit, y_pred, metrics
    
def evaluate_hyperparameters(
        estimator_template:sklearn.base.BaseEstimator,
        hyperparameters:Dict,
        validation_folds:Any,
        test_folds:Any,
        data:pd.DataFrame,
        collect_metrics:Callable,
        use_predict_proba:bool=False,
        train_folds:Optional[Any]=None,
        split_field:str='fold',
        target_field:str='target',
        target_transform:Optional[Callable]=None,
        target_inverse_transform:Optional[Callable]=None,
        collect_metrics_kwargs:Optional[Dict]=None,
        attribute_fields:Optional[Iterable[str]]=None,
        fields_to_ignore:Optional[Container[str]]=None,
        attributes_are_np_arrays:bool=False) -> estimators_predictions_metrics:
    """ Evaluate `hyperparameters` for `fold`
    
    **N.B.** This function is not particularly efficient with
    creating copies of data.
    
    This function performs the following steps:
    
    0.  Create `estimator_val` and `estimator_test` based on
        `estimator_template` and `hyperparameters`
    1.  Split `data` into `train`, `val`, `test` based on `validation_fold`
        and `test_fold`
    2.  Transform `target_field` using the `target_transform` function
    3.  Train `estimator_val` using `train`
    4.  Evaluate the trained `estimator_val` on `val` using `collect_metrics`
    5.  Train `estimator_test` using both `train` and `val`
    6.  Evaluate the trained `estimator_test` on `test` using `collect_metrics`
    
    Parameters
    ---------
    estimator_template : sklearn.base.BaseEstimator
        The template for creating the `estimator`.
        
    hyperparameters : typing.Dict
        The hyperparameters for the model. These should be compatible
        with `estimator_template.set_params`.
        
    validation_folds : typing.Any
        The fold(s) to use for validation. The validation fold will be selected
        based on `isin`. If `validation_fold` is not a container, it will be
        cast as one.
        
    test_folds : typing.Any
        The fold(s) to use for testing. The test fold will be selected
        based on `isin`. If `test_fold` is not a container, it will be
        cast as one.
        
    data : pandas.DataFrame
        The data.
        
    collect_metrics : typing.Callable
        The function for evaluating the model performance. It should have
        at least two arguments, `y_true` and `y_pred`, in that order. This
        function will eventually return whatever this function returns.
        
    use_predict_proba : bool
        Whether to use `predict` (when `False`, the default) or `predict_proba`
        on the trained model.
        
    train_folds : typing.Optional[typing.Any]
        The fold(s) to use for training. If not given, the training fold
        will be taken as all rows in `data` which are not part of the
        validation or testing set.
        
    split_field : str
        The name of the column with the fold identifiers
        
    target_field : str
        The name of the column with the target value
        
    target_transform : typing.Optional[typing.Callable]
        A function for transforming the target before training models.
        Example: :py:func:`numpy.log1p`
    
    target_inverse_transform : typing.Optional[typing.Callable]
        A function for transforming model predictions back to the original
        domain. This should be a mathematical inverse of `target_transform`.
        Example: :py:func:`numpy.expm1` is the inverse of :py:func:`numpy.log1p`.
        
    collect_metrics_kwargs : typing.Optional[typing.Dict]
        Additional keyword arguments for `collect_metrics`.
        
    attribute_fields : typing.Optional[typing.Iterable[str]]
        The names of the columns to use for attributes (that is, `X`). If
        `None` (default), then all columns except the `target_field` will
        be used as attributes.
        
    fields_to_ignore : typing.Optional[typing.Container[str]]
        The names of the columns to ignore.
        
    attributes_are_np_arrays : bool
        Whether to stack the values from the individual rows. This should
        be set to `True` when some of the columns in `attribute_fields`
        contain numpy arrays.
        
    Returns
    -------
    estimators_predictions_metrics : typing.NamedTuple
        The fit estimators, predictions on the respective datasets,
        and results from `collect_metrics`.
    """
    ###
    # Based on the template of our estimator pipeline template
    # and hyperparameters, create a concrete estimator with the
    # specified hyperparameters.
    ###
    estimator_val = sklearn.clone(estimator_template)
    estimator_val = estimator_val.set_params(**hyperparameters)
    
    estimator_test = sklearn.clone(estimator_template)
    estimator_test = estimator_test.set_params(**hyperparameters)
        
    ###
    # Split `data` into `train`, `val`, `test` based on
    # `validation_fold` and `test_fold`.
    ###
    split_masks = get_train_val_test_splits(
        df=data,
        training_splits=train_folds,
        validation_splits=validation_folds,
        test_splits=test_folds,
        split_field=split_field
    )
    
    ###
    # Create the data matrices necessary for the various
    #    sklearn operations we will perform later.
    ###
    if fields_to_ignore is None:
        fields_to_ignore = list()
        
    fields_to_ignore = fields_to_ignore + [split_field]

    val_fold_data = get_fold_data(
        df=data,
        target_field=target_field,
        m_train=split_masks.training,
        m_test=split_masks.test,
        m_validation=split_masks.validation,
        attribute_fields=attribute_fields,
        fields_to_ignore=fields_to_ignore,
        attributes_are_np_arrays=attributes_are_np_arrays
    )
    
    if collect_metrics_kwargs is None:
        collect_metrics_kwargs= dict()
    
    # get the validation performance
    estimator_val_fit, y_val, metrics_val = _train_and_evaluate(
        estimator_val,
        X_train=val_fold_data.X_train,
        y_train=val_fold_data.y_train,
        X_test=val_fold_data.X_validation,
        y_test=val_fold_data.y_validation,
        target_transform=target_transform,
        target_inverse_transform=target_inverse_transform,
        collect_metrics=collect_metrics,
        collect_metrics_kwargs=collect_metrics_kwargs,
        use_predict_proba=use_predict_proba
    )
    
    # for predictions on the test set, we will train on
    # both the training and validation sets
    X_train = np.concatenate([val_fold_data.X_train, val_fold_data.X_validation])
    y_train = np.concatenate([val_fold_data.y_train, val_fold_data.y_validation])
    
    # get the testing performance
    estimator_test_fit, y_test, metrics_test = _train_and_evaluate(
        estimator_test,
        X_train=X_train,
        y_train=y_train,
        X_test=val_fold_data.X_test,
        y_test=val_fold_data.y_test,
        target_transform=target_transform,
        target_inverse_transform=target_inverse_transform,
        collect_metrics=collect_metrics,
        collect_metrics_kwargs=collect_metrics_kwargs,
        use_predict_proba=use_predict_proba
    )
    
    hyperparameters_str = json.dumps(hyperparameters)
    
    ret = estimators_predictions_metrics(
        estimator_val=estimator_val_fit,
        estimator_test=estimator_test_fit,
        predictions_val=y_val,
        predictions_test=y_test,
        true_val=val_fold_data.y_train,
        true_test=val_fold_data.y_test,
        metrics_val=metrics_val,
        metrics_test=metrics_test,
        fold_train=train_folds,
        fold_val=validation_folds,
        fold_test=test_folds,
        hyperparameters=hyperparameters,
        hyperparameters_str=hyperparameters_str,
    )
    
    return ret

###
# Evaluation helpers
###
def collect_regression_metrics(
        y_true : np.ndarray,
        y_pred : np.ndarray,
        prefix:str = "") -> Dict:
    """ Collect various regression performance metrics for the predictions

    Parameters
    ----------
    y_true : numpy.ndarray
        The true value of each instance

    y_pred : numpy.ndarray
        The prediction for each instance
        
    prefix : str
        An optional prefix for the keys in the `metrics` dictionary
    
    Returns
    -------
    metrics: typing.Dict
        A mapping from the metric name to the respective value. Currently,
        the following metrics are included:
        
        * :py:func:`sklearn.metrics.explained_variance_score`
        * :py:func:`sklearn.metrics.mean_absolute_error`
        * :py:func:`sklearn.metrics.mean_squared_error`
        * :py:func:`sklearn.metrics.median_absolute_error`
        * :py:func:`sklearn.metrics.r2_score`
    """
    validation_utils.validate_equal_shape(y_true, y_pred)

    ret = {
        "{}explained_variance".format(prefix): sklearn.metrics.explained_variance_score(y_true, y_pred),
        "{}mean_absolute_error".format(prefix): sklearn.metrics.mean_absolute_error(y_true, y_pred),
        "{}mean_squared_error".format(prefix): sklearn.metrics.mean_squared_error(y_true, y_pred),
        #"{}mean_squared_log_error".format(prefix): sklearn.metrics.mean_squared_log_error(y_true, y_pred),
        "{}median_absolute_error".format(prefix): sklearn.metrics.median_absolute_error(y_true, y_pred),
        "{}r2".format(prefix): sklearn.metrics.r2_score(y_true, y_pred)
    }

    return ret


def collect_multiclass_classification_metrics(
        y_true : np.ndarray,
        y_score : np.ndarray,
        prefix:str = "") -> Dict:
    """ Calculate various multi-class classification performance metrics
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The true label of each instance. The labels
        are assumed to be encoded with integers [0, 1, ... n_classes-1]. The respective
        columns in `y_score` should give the scores of the matching label.
        
        This should have shape (n_samples,).
        
    y_score : numpy.ndarray
        The score predictions for each class, e.g., from` pred_proba`, though
        they are not required to be probabilities.
        
        This should have shape (n_samples, n_classes).
        
    prefix : str
        An optional prefix for the keys in the `metrics` dictionary
        
    Returns
    -------
    metrics : typing.Dict
        A mapping from the metric name to the respective value. Currently,
        the following metrics are included:
        
        * :py:func:`sklearn.metrics.cohen_kappa_score`
        * :py:func:`sklearn.metrics.accuracy_score`
        * :py:func:`sklearn.metrics.f1_score` (micro)
        * :py:func:`sklearn.metrics.f1_score` (macro)
        * :py:func:`sklearn.metrics.hamming_loss`
        * :py:func:`sklearn.metrics.precision_score` (micro)
        * :py:func:`sklearn.metrics.precision_score` (macro)
        * :py:func:`sklearn.metrics.recall_score` (micro)
        * :py:func:`sklearn.metrics.recall_score` (macro)
        * :py:func:`pyllars.ml_utils.calc_hand_and_till_m_score`
        * :py:func:`pyllars.ml_utils.calc_provost_and_domingos_auc`
    """

    # make hard predictions
    y_pred = np.argmax(y_score, axis=1)

    # now collect all statistics
    ret = {
         "{}cohen_kappa".format(prefix):  sklearn.metrics.cohen_kappa_score(y_true, y_pred),
         #"{}matthews_corrcoef".format(prefix):  sklearn.metrics.matthews_corrcoef(y_true, y_pred),
         "{}accuracy".format(prefix):  sklearn.metrics.accuracy_score(y_true, y_pred),
         "{}micro_f1_score".format(prefix):  sklearn.metrics.f1_score(y_true, y_pred,
            average='micro'),
         "{}macro_f1_score".format(prefix):  sklearn.metrics.f1_score(y_true, y_pred,
            average='macro'),
         "{}hamming_loss".format(prefix):  sklearn.metrics.hamming_loss(y_true, y_pred),
         "{}micro_precision".format(prefix):  sklearn.metrics.precision_score(y_true, y_pred,
            average='micro'),
         "{}macro_precision".format(prefix):  sklearn.metrics.precision_score(y_true, y_pred,
            average='macro'),
         "{}micro_recall".format(prefix):  sklearn.metrics.recall_score(y_true, y_pred,
            average='micro'),
         "{}macro_recall".format(prefix):  sklearn.metrics.recall_score(y_true, y_pred,
            average='macro'),
         "{}hand_and_till_m_score".format(prefix): calc_hand_and_till_m_score(y_true, y_score),
         "{}provost_and_domingos_auc".format(prefix): calc_provost_and_domingos_auc(y_true, y_score)
    }

    return ret


def collect_binary_classification_metrics(
        y_true:np.ndarray,
        y_probas_pred:np.ndarray,
        threshold:float=0.5,
        pos_label=1,
        k:int=10,
        include_roc_curve:bool=True,
        include_pr_curve:bool=True,
        prefix:str = "") -> Dict:
    """ Collect various binary classification performance metrics for the predictions

    Parameters
    ----------
    y_true: numpy.ndarray
        The true class of each instance.
        
        This should have shape (n_samples,).

    y_probas_pred: numpy.ndarray
        The score of each prediction for each instance.
        
        This should have shape (n_samples, n_classes).
    
    threshold: float
        The score threshold to choose "positive" predictions

    pos_label: str or int
        The "positive" class for some metrics
        
    k : int
        The value of `k` to use for `precision_at_k`
        
    include_roc_curve : bool
        Whether to include the fpr and trp points necessary to draw
        a roc curve
        
    include_pr_curve : bool
        Whether to include details on the precision-recall curve
        
    prefix : str
        An optional prefix for the keys in the `metrics` dictionary

    Returns
    -------
    metrics: dict
        A mapping from the metric name to the respective value. Currently,
        the following metrics are included:
        
        * :py:func:`sklearn.metrics.cohen_kappa_score`
        * :py:func:`sklearn.metrics.hinge_loss`
        * :py:func:`sklearn.metrics.matthews_corrcoef`
        * :py:func:`sklearn.metrics.accuracy_score`
        * :py:func:`sklearn.metrics.f1_score` (binary)
        * :py:func:`sklearn.metrics.f1_score` (macro)
        * :py:func:`sklearn.metrics.f1_score` (micro)
        * :py:func:`sklearn.metrics.hamming_loss`
        * :py:func:`sklearn.metrics.jaccard_score`
        * :py:func:`sklearn.metrics.log_loss`
        * :py:func:`sklearn.metrics.precision_score` (binary)
        * :py:func:`sklearn.metrics.precision_score` (macro)
        * :py:func:`sklearn.metrics.precision_score` (micro)
        * :py:func:`sklearn.metrics.recall_score` (binary)
        * :py:func:`sklearn.metrics.recall_score` (macro)
        * :py:func:`sklearn.metrics.recall_score` (micro)
        * :py:func:`sklearn.metrics.zero_one_loss`
        * :py:func:`sklearn.metrics.average_precision_score` (macro)
        * :py:func:`sklearn.metrics.average_precision_score` (micro)
        * :py:func:`sklearn.metrics.roc_auc_score` (macro)
        * :py:func:`sklearn.metrics.roc_auc_score` (micro)
        * :py:func:`pyllars.ml_utils.precision_at_k`
        * `auprc`: area under the PR curve
        * `minpse`: See [Harutyunyan et al., 2019] for details
        * `roc_` {`fpr`, `tpr`, `thresholds`}: :py:func:`sklearn.metrics.roc_curve`
        * `pr_` {`precisions`, `recalls`, `thresholds`}: :py:func:`sklearn.metrics.precision_recall_curve`
    """

    # first, validate the input
    if y_true.shape[0] != y_probas_pred.shape[0]:
        msg = ("[ml_utils.collect_binary_classification_metrics]: y_true "
            "and y_probas_pred do not have matching shapes. y_true: {}, "
            "y_probas_pred: {}".format(y_true.shape, y_probas_pred.shape))
        raise ValueError(msg)

    if y_probas_pred.shape[1] != 2:
        msg = ("[ml_utils.collect_binary_classification_metrics]: "
            "y_probas_pred does not have scores for exactly two classes: "
            "y_probas_pred.shape: {}".format(y_probas_pred.shape))
        raise ValueError(msg)


    # first, pull out the probability of positive classes
    y_score = y_probas_pred[:,pos_label]

    # and then make a hard prediction
    y_pred = (y_score >= threshold)
    

    precisions, recalls, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    auprc = sklearn.metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

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
         "jaccard_score":  sklearn.metrics.jaccard_score(
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
            average='macro'),
         "precision_at_k": precision_at_k(y_true, y_score, k, pos_label),
         "auprc": auprc,
         "minpse": minpse
    }
    
    if include_roc_curve:
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
        ret['roc_fpr'] = fpr
        ret['roc_tpr'] = tpr
        ret['roc_thresholds'] = thresholds
        
    if include_pr_curve:        
        ret['pr_precisions'] = precisions
        ret['pr_recalls'] = recalls
        ret['pr_thresholds'] = thresholds
        
    # add the prefix, if given
    if len(prefix) > 0:
        ret = {
            "{}{}".format(prefix, key): value
                for key, value in ret.items()
        }
        
    return ret

def precision_at_k(y_true, y_score, k=10, pos_label=1):
    """Precision at rank k
    
    This code was adapted from this gist: https://gist.github.com/mblondel/7337391
    
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
        
    y_score : array-like, shape = [n_samples]
        Predicted scores.
        
    k : int
        Rank.
        
    pos_label : int
        The label for "positive" instances
        
    Returns
    -------
    precision @k : float
    """
    
    # how many positives in total?
    n_pos = np.sum(y_true == pos_label)

    # pull out the top-k according to score
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    
    # how many positives were retrieved?
    n_relevant = np.sum(y_true == pos_label)

    # Divide by min(n_pos, k) such that the best achievable score is always 1.0.
    pak = float(n_relevant) / min(n_pos, k)
    
    return pak

def _calc_hand_and_till_a_value(y_true:np.ndarray, y_score:np.ndarray, i:int, j:int) -> float:
    """ Calculate the :math:`\hat{A}` value in Equation (3) of [1]_. Specifically;
    
    .. math::
        \\hat{A}(i|j) = \\frac{ S_i - n_i*(n_i + 1)/2 }{n_i * n_j},

    where :math:`n_i`, :math:`n_j` are the count of instances of the respective
    classes and :math:`S_i` is the (base-1) sum of the ranks of class :math:`i`.
    
    Parameters
    ----------
    y_true : numpy.ndarray 
        The true label of each instance. The labels are assumed to be encoded with
        integers [0, 1, ... n_classes-1]. The respective columns in `y_score` should
        give the probabilities of the matching label.
        
        This should have shape (n_samples,).
        
    y_score : numpy.ndarray
        The score predictions for each class, e.g., from `pred_proba`, though they
        are not required to be probabilities.
        
        This should have shape (n_samples, n_classes).
        
    {i,j} : int
        The class indices
        
    Returns
    -------
    a_hat : float
        The :math:`\hat{A}` value from Equation (3) referenced above. Specifically,
        this is the probability that a randomly drawn member of class :math:`j` will have
        a lower estimated score for belonging to class :math:`i` than a randomly drawn member
        of class :math:`i`.
            
    References
    ----------
    .. [1] Hand, D. & Till, R. A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning, 2001, 45, 171-186. `Springer link <https://link.springer.com/article/10.1023/A:1010920819831>`_.
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

def calc_hand_and_till_m_score(y_true:np.ndarray, y_score:np.ndarray) -> float:
    """ Calculate the (multi-class AUC) :math:`M` score from Equation (7) of Hand and Till (2001).
    
    This is typically taken as a good multi-class extension of the AUC score. Please see [2]_
    for more details about this score in particular and [3]_ for multi-class AUC in general.

    **N.B.** In case y_score contains any `np.nan` values, those will be removed before
    calculating the :math:`M` score.

    **N.B.** This function *can* handle unobserved labels, except for the label
    with the highest index. In particular, ``y_score.shape[1] != np.max(np.unique(y_true)) + 1``
    causes an error.
    
    Parameters
    ----------
    y_true: numpy.ndarray
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
        This should have shape (n_samples,).
        
    y_score: numpy.ndarray
        The score predictions for each class, e.g., from `pred_proba`, though
        they are not required to be probabilities.
        
        This should have shape (n_samples, n_classes).
        
    Returns
    -------
    m : float
        The "multi-class AUC" score referenced above
        
    See Also
    --------
    _calc_hand_and_till_a_value : for calculating the :math:`\\hat{A}` value
            
    References
    ----------
    .. [2] Hand, D. & Till, R. A Simple Generalisation of the Area Under the ROC Curve for Multiple Class Classification Problems. Machine Learning, 2001, 45, 171-186. `Springer link <https://link.springer.com/article/10.1023/A:1010920819831>`_.
    .. [3] Fawcett, T. An introduction to ROC analysis. Pattern Recognition Letters,  2006, 27, 861 - 874. `Elsevier link <https://www.sciencedirect.com/science/article/abs/pii/S016786550500303X>`_.
    """
    classes = np.unique(y_true)

    # make sure the classes are integers, or we will have problems indexing
    classes = np.array(classes, dtype=int)
    num_classes = np.max(classes)+1

    # first, validate our input
    if y_true.shape[0] != y_score.shape[0]:
        msg = ("[ml_utils.m_score]: y_true and y_score do not have matching "
            "shapes. y_true: {}, y_score: {}".format(y_true.shape,
            y_score.shape))
        raise ValueError(msg)
    
    if y_score.shape[1] != (num_classes):
        msg = ("[ml_utils.m_score]: y_score does not have the expected "
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


def calc_provost_and_domingos_auc(y_true:np.ndarray, y_score:np.ndarray) -> float:
    """ Calculate the (multi-class AUC) :math:`M` score from Equation (7) of Provost and Domingos (2000).
    
    This is typically taken as a good multi-class extension of the AUC score. Please see [4]_
    for more details about this score in particular and [5]_ for multi-class AUC in general.

    N.B. This function *can* handle unobserved labels, except for the label
    with the highest index. In particular, ``y_score.shape[1] != np.max(np.unique(y_true)) + 1``
    causes an error.
    
    Parameters
    ----------
    y_true : numpy.ndarray
        The true label of each instance. The labels are assumed to be encoded 
        with integers [0, 1, ... n_classes-1]. The respective columns in
        y_score should give the scores of the matching label.
        
        This should have shape (n_samples,).
        
    y_score : numpy.ndarray
        The score predictions for each class, e.g., from `pred_proba`, though
        they are not required to be probabilities.
        
        This should have shape (n_samples, n_classes).
        
    Returns
    -------
    m : float
        The "multi-class AUC" score referenced above
            
    References
    ----------
    .. [4] Provost, F. & Domingos, P. Well-Trained PETs: Improving Probability Estimation Trees. Sterm School of Business, NYU, Sterm School of Business, NYU, 2000. `Citeseer link <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.33.309>`_.
    .. [5] Fawcett, T. An introduction to ROC analysis. Pattern Recognition Letters,  2006, 27, 861 - 874. `Elsevier link <https://www.sciencedirect.com/science/article/abs/pii/S016786550500303X>`_.
    """
    classes = np.unique(y_true)

    # make sure the classes are integers, or we will have problems indexing
    classes = np.array(classes, dtype=int)

    num_classes = np.max(classes)+1
    
    # first, validate our input
    if y_true.shape[0] != y_score.shape[0]:
        msg = ("[ml_utils.m_score]: y_true and y_score do not have matching "
            "shapes. y_true: {}, y_score: {}".format(y_true.shape,
            y_score.shape))
        raise ValueError(msg)

    if y_score.shape[1] != (num_classes):
        msg = ("[ml_utils.m_score]: y_score does not have the expected "
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