Machine learning utilities
*****************************

.. automodule:: pyllars.ml_utils
    :noindex:
    
.. currentmodule::  pyllars.ml_utils

Creating and managing cross-validation
-----------------------------------------

.. autosummary::
    get_cv_folds
    get_train_val_test_splits
    get_fold_data
    
Evaluating results
-------------------

.. autosummary::
    collect_binary_classification_metrics
    collect_multiclass_classification_metrics
    collect_regression_metrics
    calc_hand_and_till_m_score
    calc_provost_and_domingos_auc
    

Data structures
----------------

.. autosummary::
    fold_data
    split_masks
    

Definitions
-------------
.. automodule:: pyllars.ml_utils
    :members:
    :private-members:
    :exclude-members: fold_data, split_masks
    
.. autoclass:: fold_data
.. autoclass:: split_masks