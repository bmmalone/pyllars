"""
This module contains very high-level helpers for selecting hyperparameters
for machine learning models using a train-validation-test strategy. Typical
usage looks as follows:

```
# create the hyperparameter grid
hp_grid = sklearn.model_selection.ParameterGrid({
    ...
})

# create an iterator over the hyperparameter grid and folds
hp_fold_it = hp_utils.get_hp_fold_iterator(hp_grid, num_folds)

# distribute training to the dask cluster
f_res = dask_utils.apply_iter(
    hp_fold_it,
    dask_client,
    hp_utils.evaluate_hyperparameters_helper,
    args=args,
    ...,
    return_futures=True
)

# collect the results from dask
all_res = dask_utils.collect_results(f_res)

# parse the results
df_results = hp_utils.get_hp_results(all_res)

# select the best hyperparameters using the validation set results
evaluation_metric = 'micro_roc_auc_score'
best_val_hps = hp_utils.get_best_hyperparameters(
    df_results,
    evaluation_metric=evaluation_metric,
    selection_function=np.argmax # **this depends on the evaluation metric**
)

# pull out the results on those folds
m_val_best = (df_results['hyperparameters_str'] == val_best)
```

"""
import logging
logger = logging.getLogger(__name__)

import itertools
import json
import numpy as np
import pandas as pd
import tqdm

import sklearn.model_selection

import pyllars.ml_utils as ml_utils
import pyllars.pandas_utils as pd_utils

_ESTIMATOR_PREFIX = "estimator__"

def get_log_reg_hp_grid(estimator_prefix=_ESTIMATOR_PREFIX):

    log_reg_hp_grid = list(sklearn.model_selection.ParameterGrid({
        '{}class_weight'.format(estimator_prefix): ['balanced', None],
        '{}penalty'.format(estimator_prefix): ['l1', 'l2'],
        '{}C'.format(estimator_prefix): np.concatenate([10.0**-np.arange(0,6)]),#, 10.0**np.arange(1,4)]),
        '{}random_state'.format(estimator_prefix): [8675309],
        '{}max_iter'.format(estimator_prefix): [1000],
        '{}solver'.format(estimator_prefix): ['liblinear']
    }))

    for p in log_reg_hp_grid:
        if p['{}penalty'.format(estimator_prefix)] == 'l2':
            p['{}solver'.format(estimator_prefix)] = 'lbfgs'

    return log_reg_hp_grid

def get_xgb_classifier_hp_grid(estimator_prefix=_ESTIMATOR_PREFIX):

    xgb_hp_grid = list(sklearn.model_selection.ParameterGrid({
        '{}scale_features'.format(estimator_prefix): [True],
        '{}num_boost_round'.format(estimator_prefix): [2, 5, 10],
        '{}max_depth'.format(estimator_prefix): [2, 4, 6],
        '{}min_child_weight'.format(estimator_prefix): [1, 2, 4],
        '{}gamma'.format(estimator_prefix): [0, 0.1],
        '{}subsample'.format(estimator_prefix): [0.25, 0.5, 1.0],
        '{}scale_pos_weight'.format(estimator_prefix): [1.0],
    }))

    return xgb_hp_grid



def get_svc_hyperparameter_grid(estimator_prefix=_ESTIMATOR_PREFIX):
                
    svc_hp_grid = list(sklearn.model_selection.ParameterGrid({
        #'{}C'.format(estimator_prefix):  np.concatenate([10.**-np.arange(0,7), 10.**np.arange(1,8)]),
        '{}C'.format(estimator_prefix):  np.concatenate([10.**-np.arange(0,2), 10.**np.arange(1,3)]),
        '{}gamma'.format(estimator_prefix): [1,0.1,0.01,0.001, 'scale', 'auto'],
        '{}kernel'.format(estimator_prefix): ['rbf', 'poly', 'sigmoid']
    }))

    return svc_hp_grid




def get_hp_fold_iterator(hp_grid, num_folds, use_tqdm=True):
    """ Create an iterator over all combinations of hyperparameters and folds
    """
    hp_grid = list(hp_grid)
    val_folds = np.array(list(range(num_folds)))
    test_folds = (val_folds +1) % num_folds

    hp_fold_it = itertools.product(hp_grid, val_folds, test_folds)
    hp_fold_it = list(hp_fold_it)

    if use_tqdm:
        hp_fold_it = tqdm.tqdm(hp_fold_it)
    
    return hp_fold_it

def evaluate_hyperparameters_helper(hv, *args, **kwargs):
    
    # these come from our iterator
    hyperparameters = hv[0]
    validation_folds = hv[1]
    test_folds = hv[2]
    
    res = ml_utils.evaluate_hyperparameters(
        hyperparameters=hyperparameters,
        validation_folds=validation_folds,
        test_folds=test_folds,
        *args,
        **kwargs
    )
    
    return res

def _get_res(res):
    ret_val = {
        'validation_{}'.format(k): v
            for k,v in res.metrics_val.items()
    }
    
    ret_test = {
        'test_{}'.format(k): v
            for k,v in res.metrics_test.items()
    }
    
    ret = ret_val
    ret.update(ret_test)
    
    hp_string = json.dumps(res.hyperparameters)
    ret['hyperparameters_str'] = hp_string
    
    ret['hyperparameters'] = res.hyperparameters
    ret['validation_fold'] = res.fold_val
    ret['test_fold'] = res.fold_test
    return ret

def get_hp_results(all_res):
    """ Create the results data frame
    """
    results = [
        _get_res(res) for res in all_res
    ]

    df_results = pd.DataFrame(results)
    
    return df_results

def get_best_hyperparameters(
        results,
        evaluation_metric,
        ex_type='max',
        group_fields='fold_val'):
    """ Based on the performance on the validation, select the best hyperparameters
    """
    if not isinstance(results, pd.DataFrame):
        results = pd.DataFrame(results)

    validation_evaluation_metric = "val_{}".format(evaluation_metric)

    df_best_hps = pd_utils.get_group_extreme(
        df=results,
        ex_field=validation_evaluation_metric,
        ex_type=ex_type,
        group_fields=group_fields
    )

    return df_best_hps
