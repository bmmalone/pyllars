"""
This module contains helpers for the XGBoost python wrapper: https://xgboost.readthedocs.io/en/latest/python/index.html

The largest part of the module are helper classes which make
using a validation set to select the number of trees transparent.
"""
import logging
logger = logging.getLogger(__name__)

import joblib
import numpy as np
import pathlib
import sklearn.exceptions
import sklearn.metrics
import sklearn.preprocessing
import xgboost as xgb

import pyllars.validation_utils as validation_utils

from typing import Optional

    
def xgbooster_predict_proba(
        booster:xgb.Booster,
        d_x:xgb.DMatrix) -> np.ndarray:
    """ Simulate the `predict_proba` interface from sklearn
    
    This function will only work as expected if `booster` has been
    training using the `binary:logistic` loss.
    
    Parameters
    ----------
    booster : xgboost.Booster
        The trained booster
        
    d_x : xgboost.DMatrix
        The dataset
        
    Returns
    -------
    y_proba_pred : numpy.ndarray
        The probabilistic predictions. The shape of the array
        is (n_row, 2).
    """
    y_score = booster.predict(d_x)
    y_false = 1-y_score
    size = (d_x.num_row(), 2)

    y_probas_pred = np.zeros(size)
    y_probas_pred[:,0] = y_false
    y_probas_pred[:,1] = y_score
    
    return y_probas_pred


class XGBClassifierWrapper(object):
    """ This class wraps xgboost to facilitate transparent
    use of a validation set to select the number of trees.
    
    It also optionally scales the input features. (In principle,
    it is not necessary to scale input features for trees. Still,
    in practice, it annecdotally helps, and the theory also suggests
    that is should not hurt.)
    
    **N.B.** Currently, this class is hard-coded to use (binary) AUC
    as the metric for selecting the best model on the validation set.
    
    Attributes
    ----------
    num_boost_round : int
        The number of boosting rounds
        
    scale_features : bool
        Whether to fit a StandardScaler on the training data
        and use it to transform the validation and test data.
        
    validation_period : int
        The number of training iterations (that is, the number of new
        trees) between checking the validation set.
        
    name : str
        A name for use in logging statements.
        
    booster_ : xgboost.Booster
        The trained model.
        
    best_booster_ : xgboost.Booster
        The best booster found according to performance on the
        validation set.
        
    scaler_ : sklearn.preprocessing.StandardScaler
        The scaler fit on the training data set.
        
    **kwargs : key=value pairs
        Additional keyword arguments are passed through to the
        xgboost.train constructor.
    """
    def __init__(
            self,
            num_boost_round:int=10,
            scale_features:bool=False,
            validation_period:int=1,
            name:str="XGBClassiferWrapper",
            **kwargs):
        
        self.num_boost_round = num_boost_round
        self.scale_features = scale_features
        self.validation_period = validation_period
        self.name = name
        self.kwargs = kwargs
        
    def log(self, msg, level=logging.INFO):
        msg = "[{}]: {}".format(self.name, msg)
        logger.log(level, msg)

    def _validate(self, xgboost_callback_env):

        iteration = xgboost_callback_env.iteration
        booster = xgboost_callback_env.model

        y_score = booster.predict(self._dval)

        # TODO: allow other validation metrics
        validation_roc = sklearn.metrics.roc_auc_score(
            y_true=self._dval.get_label(),
            y_score=y_score
        )

        msg = "{}\tValidation AUC: {:.6f}".format(iteration, validation_roc)
        self.log(msg, logging.DEBUG)

        if validation_roc > self._best_validation_roc:
            self._best_validation_roc = validation_roc
            self.best_booster_ = booster.copy()

            msg = "*** New Best ***"
            self.log(msg, logging.DEBUG)
            
    def _callback(self, xgboost_callback_env):

        iteration = xgboost_callback_env.iteration
        if iteration % self.validation_period == 0:
            self._validate(xgboost_callback_env)
        
    def fit(self,
            X_t:np.ndarray,
            y_t:np.ndarray,
            X_v:Optional[np.ndarray]=None,
            y_v:Optional[np.ndarray]=None):
        """ Fit a model
        
        Parameters
        ----------
        {X,y}_t : numpy.ndarray
            The training data. **N.B.** 
            
        {X,y}_v : typing.Optional[numpy.ndarray]
            The validation data
            
        Returns
        -------
        self
        """
        
        if self.scale_features:
            msg = "scaling the training data"
            self.log(msg)
            
            self.scaler_ = sklearn.preprocessing.StandardScaler()
            X_t = self.scaler_.fit_transform(X_t)
            
            if X_v is not None:
                msg = "scaling the validation data"
                X_v = self.scaler_.transform(X_v)
        else:
            self.scaler_ = None
        
        self._dtrain = xgb.DMatrix(X_t, label=y_t)
        
        callbacks = None # we will not use any callbacks by default
        if X_v is not None:            
            self._dval = xgb.DMatrix(X_v, label=y_v)
            
            # we *will* use a callback if we want to use the
            # validation set
            callbacks = [self._callback]
            
        # we can set these either way. they will just not be used
        # if there is no validation set.
        self._best_validation_roc = -np.inf
        self.best_booster_ = None
            
        msg = "training the model"
        self.log(msg)
        
        self.booster_ = xgb.train(
            self.kwargs,
            self._dtrain,
            self.num_boost_round,
            callbacks=callbacks
        )
        
        # if we did not use a validation set, then just use the
        # final learned model as the best model
        if self.best_booster_ is None:
            self.best_booster_ = self.booster_
            
        return self
            
    def predict_proba(self, X:np.ndarray) -> np.ndarray:
        """ Predict the likelihood of each class.
        
        This function will only work as expected if training
        used the `binary:logistic` loss.
        
        Parameters
        ----------
        X : numpy.ndarray
            The input data
            
        Returns
        -------
        y_proba_pred : numpy.ndarray
            The probabilistic predictions
        """
        validation_utils.check_is_fitted(self, 'best_booster_', self.name)
        
        if self.scaler_ is not None:
            msg = "transforming the input data"
            self.log(msg, logging.DEBUG)
            X = self.scaler_.transform(X)

        d_x = xgb.DMatrix(X)
        y_proba_pred = xgbooster_predict_proba(self.best_booster_, d_x)
        return y_proba_pred
    
    def get_params(self, deep=False):
        """ Get the hyperparameters and other meta data about this model
        """
        params = {
            'num_boost_round': self.num_boost_round,
            'scale_features': self.scale_features,
            'validation_period': self.validation_period,
            'name': self.name
        }
        
        params.update(self.kwargs)
        
        # we do not do anything with `deep`
        
        return params
    
    def set_params(self, **params):
        """ Set the hyperparameters of the model
        """
        # very similar to the sklearn implementation
        valid_params = self.get_params(deep=True)
        
        for k,v in params.items():
            if k not in valid_params:
                msg = "Invalid parameter: {}".format(k)
                raise ValueError(msg)
                
            # then this is a valid hyperparameter
            setattr(self, key, value)
            
        return self
    
    def save_model(self, out):
        """ Save the scaler (if present) and best model to disk.
        
        This *does not* save the training or validation datasets.
        """
        out = pathlib.Path(out)
        out.mkdir(parents=True, exist_ok=True)

        scaler_out = out / "scaler.jpkl"
        joblib.dump(self.scaler_, str(scaler_out))

        booster_out = out / "booster.jpkl"
        joblib.dump(self.best_booster_, str(booster_out))
        
        params_out = out / "params.jpkl"
        joblib.dump(self.get_params(deep=True), str(params_out))
        
    @classmethod
    def load_model(klass, f):
        """ Load the scaler, model, and hyperparameters from disk
        """
        
        in_f = pathlib.Path(f)
        
        params_in = in_f / "params.jpkl"
        params = joblib.load(str(params_in))
        model = klass(**params)
        
        scaler_in = in_f / "scaler.jpkl"
        scaler = joblib.load(str(scaler_in))
        model.scaler_ = scaler

        booster_in = in_f / "booster.jpkl"
        booster = joblib.load(str(booster_in))
        model.best_booster_ = booster
        
        return model