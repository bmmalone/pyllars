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
import pandas as pd

import pyllars.ml_utils as ml_utils
import pyllars.pandas_utils as pd_utils

def get_hp_fold_iterator(hp_grid, num_folds):
    """ Create an iterator over all combinations of hyperparameters and folds
    """
    hp_grid = list(hp_grid)
    folds = list(range(num_folds))

    hp_fold_it = itertools.product(hp_grid, folds)
    hp_fold_it = list(hp_fold_it)
    
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

def get_best_hyperparameters(df_results, evaluation_metric, selection_function):
    """ Based on the performance on the validation, select the best hyperparameters
    """
    hp_groups = df_results.groupby('hyperparameters_str')

    validation_evaluation_metric = "validation_{}".format(evaluation_metric)
    test_evaluation_metric = "test_{}".format(evaluation_metric)

    # find the mean of each set of hp's across all folds
    val_performance = hp_groups[validation_evaluation_metric].mean()

    # now, select the best
    val_best = selection_function(val_performance) 
    
    return val_best