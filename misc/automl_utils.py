###
#   These are helpers related to auto-sklearn, AutoFolio, ASlib, etc.
###

import logging
logger = logging.getLogger(__name__)

# system imports
import collections
import os

# pydata imports
import joblib
import numpy as np
import sklearn.preprocessing

# widely-used package imports
import networkx as nx
import statsmodels.stats.weightstats

# less-used imports
from aslib_scenario.aslib_scenario import ASlibScenario
import misc.utils as utils

imputer_strategies = [
    'mean', 
    'median', 
    'most_frequent',
    'zero_fill'
]

def check_imputer_strategy(imputer_strategy, raise_error=True, error_prefix=""):
    """ Ensure that the imputer strategy is a valid selection. 

    Parameters
    ----------
    imputer_strategy: str
        The name of the strategy to check

    raise_error: bool
        Whether to raise a ValueError (or print a warning message) if the
        strategy is not in the valid list.

    error_prefix: str
        A string to prepend to the error/warning message

    Returns
    -------
    is_valid: bool
        True if the imputer strategy is in the allowed list (and an error is
        not raised)    
    """
    if imputer_strategy not in imputer_strategies:
        imputer_strategies_str = ','.join(imputer_strategies)
        msg = ("{}The imputer strategy is not allowed. given: {}. allowed: {}".
            format(error_prefix, imputer_strategy, imputer_strategies_str))

        if raise_error:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    return True

def get_imputer(imputer_strategy, verify=False):
    """ Get the imputer for use in an sklearn pipeline

    Parameters
    ----------
    imputer_strategy: str
        The name of the strategy to use. This must be one of the imputer
        strategies. check_imputer_strategy can be used to verify the string.

    verify: bool
        If true, check_imputer_strategy will be called before creating the
        imputer

    Returns
    -------
    imputer: sklearn.transformer
        A transformer with the specified strategy
    """

    if verify:
        check_imputer_strategy(imputer_strategy)

    imputer = None
    if imputer_strategy == 'zero_fill':
        imputer = sklearn.preprocessing.FunctionTransformer(
            np.nan_to_num,
            validate=False
        )

    else:
        imputer = sklearn.preprocessing.Imputer(strategy=imputer_strategy)

    return imputer

import autosklearn.pipeline.components.classification as c
import autosklearn.pipeline.components.feature_preprocessing as fp
import autosklearn.pipeline.components.regression as r

all_classifiers = c._classifiers.keys()
all_preprocessors = fp._preprocessors.keys()
all_regressors = r._regressors.keys()

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier

ASL_REGRESSION_PIPELINE_TYPES = {
    utils.get_type("autosklearn.pipeline.regression.SimpleRegressionPipeline")
}

ASL_CLASSIFICATION_PIPELINE_TYPES = {
    utils.get_type("autosklearn.pipeline.classification.SimpleClassificationPipeline")
}

ASL_PIPELINE_TYPES = ASL_REGRESSION_PIPELINE_TYPES | ASL_CLASSIFICATION_PIPELINE_TYPES

DUMMY_CLASSIFIER_TYPES = {
    utils.get_type("autosklearn.evaluation.abstract_evaluator.MyDummyClassifier"),
    utils.get_type("autosklearn.evaluation.abstract_evaluator.DummyClassifier"),
}

DUMMY_REGRESSOR_TYPES = {
    utils.get_type("autosklearn.evaluation.abstract_evaluator.MyDummyRegressor"),   
    utils.get_type("autosklearn.evaluation.abstract_evaluator.DummyRegressor"),  
}

DUMMY_PIPELINE_TYPES= DUMMY_CLASSIFIER_TYPES | DUMMY_REGRESSOR_TYPES

CLASSIFIER_CHOICE_TYPES = {
    utils.get_type("autosklearn.pipeline.components.classification.ClassifierChoice"),
}
REGRESSOR_CHOICE_TYPES = {
    utils.get_type("autosklearn.pipeline.components.regression.RegressorChoice"),
}

ESTIMATOR_CHOICE_TYPES = REGRESSOR_CHOICE_TYPES | CLASSIFIER_CHOICE_TYPES

ESTIMATOR_NAMED_STEPS = {
    'regressor',
    'classifier'
}

regression_pipeline_steps = [
    'one_hot_encoding', 
    'imputation', 
    'rescaling', 
    'preprocessor', 
    'regressor'
]

classificaation_pipeline_steps = [
    'one_hot_encoding',
    'imputation',
    'rescaling',
    'balancing',
    'preprocessor',
    'classifier'
]

pipeline_step_names_map = {
    # preprocessors
    "<class 'sklearn.feature_selection.from_model.SelectFromModel'>": "Model-based",
    "<class 'sklearn.feature_selection.univariate_selection.SelectPercentile'>": "Percentile-based",
    "<class 'sklearn.cluster.hierarchical.FeatureAgglomeration'>": "Feature agglomeration",
    "<class 'int'>": "None",
    "<class 'sklearn.preprocessing.data.PolynomialFeatures'>": "Polynomial expansion",
    "<class 'sklearn.decomposition.pca.PCA'>": "PCA",
    "<class 'sklearn.ensemble.forest.RandomTreesEmbedding'>": "Random tree embedding",
    "<class 'sklearn.decomposition.fastica_.FastICA'>": "ICA",
    "<class 'sklearn.kernel_approximation.RBFSampler'>": "RBF Sampler",
    "<class 'sklearn.decomposition.kernel_pca.KernelPCA'>": "Kernel PCA",
    "<class 'sklearn.kernel_approximation.Nystroem'>": "Nystroem approximation",
    
    # regression models
    "<class 'sklearn.linear_model.stochastic_gradient.SGDRegressor'>": "Linear model trained with stochastic gradient descent",
    "<class 'sklearn.ensemble.weight_boosting.AdaBoostRegressor'>": "AdaBoost",
    "<class 'sklearn.tree.tree.DecisionTreeRegressor'>": "Regression trees",
    "<class 'sklearn.ensemble.forest.ExtraTreesRegressor'>": "Extremely randomized trees",
    "<class 'sklearn.svm.classes.SVR'>": "episilon-support vector regression",
    "<class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>": "Gradient-boosted regression trees",
    "<class 'xgboost.sklearn.XGBRegressor'>": "Extreme gradient-boosted regression trees",
    "<class 'sklearn.gaussian_process.gaussian_process.GaussianProcess'>": "Gaussian process regression",
    "<class 'sklearn.ensemble.forest.RandomForestRegressor'>": "Random forest regression",
    "<class 'sklearn.linear_model.ridge.Ridge'>": "Ridge regression",
    "<class 'sklearn.linear_model.bayes.ARDRegression'>": "Bayesian ridge regression",
    "<class 'sklearn.svm.classes.LinearSVR'>": "Linear support vector regression",

    # classification models
    "<class 'autosklearn.evaluation.abstract_evaluator.MyDummyClassifier'>": "Dummy classifier",
    "<class 'sklearn.ensemble.forest.RandomForestClassifier'>": "Random forest classification",
    "<class 'sklearn.svm.classes.LinearSVC'>": "Support vector classification (liblinear)",
    "<class 'sklearn.svm.classes.SVC'>": "Support vector classification (libsvm)",
    "<class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>": "Linear discriminant analysis",
    "<class 'sklearn.ensemble.gradient_boosting.GradientBoostingClassifier'>": "Gradient-boosted regression trees for classification",
    "<class 'sklearn.linear_model.stochastic_gradient.SGDClassifier'>": "Linear model trained with stochastic gradient descent",
    "<class 'sklearn.ensemble.weight_boosting.AdaBoostClassifier'>": "AdaBoost (SAMME), default base: DecisionTree"
    
}

PRETTY_NAME_TYPE_MAP = {
    utils.get_type(k.split("'")[1]): v for k,v in pipeline_step_names_map.items()
}

# the names of the actual "element" (estimator, etc.) for steps in
# autosklearn pipelines
pipeline_step_element_names = {
    "one_hot_encoding": None,
    "imputation": None,
    "rescaling": 'preprocessor',
    "preprocessor": 'preprocessor',
    "regressor": 'estimator',
    "classifier": 'estimator',
    "balancing": None
}

# pipeline types which do not do anything and can be ignored
passthrough_types = {
    utils.get_type("autosklearn.pipeline.components.data_preprocessing.rescaling.none.NoRescalingComponent"),
    int
}

def get_simple_pipeline(asl_pipeline, as_array=False):
    """ Extract the "meat" elements from an auto-sklearn pipeline to create
    a "normal" sklearn pipeline.

    N.B.
    
    * The new pipeline is based on clones.

    * In case the pipeline is actually one of the DUMMY_PIPELINE_TYPES, it will
      be returned without a pipeline wrapper.

    * If asl_pipeline is not one of ASL_PIPELINE_TYPES, then a simple clone
      is created and returned.

    Parameters
    ----------
    asl_pipeline: autosklearn.Pipeline
        The pipeline learned by auto-sklearn

    as_array: bool
        Whether to return the selected steps as a list of 2-tuples of an
        sklearn.pipeline.Pipeline

    Returns
    -------
    sklearn_pipeline: sklearn.pipeline.Pipeline
        A pipeline containing clones of the elements from the auto-sklearn
        pipeline, but in a "normal" sklearn pipeline
    """
    if type(asl_pipeline) not in ASL_PIPELINE_TYPES:
        msg = "[automl_utils]: asl_pipeline is not an auto-sklearn pipeline"
        logger.debug(msg)

        pipeline = sklearn.base.clone(asl_pipeline)
        return pipeline

    simple_pipeline = []

    # if it is one of the dummy classifiers, then just wrap it and return it
    if type(asl_pipeline) in DUMMY_PIPELINE_TYPES:
        return asl_pipeline

        dummy_estimator_name = 'regressor'
        if type(asl_pipeline) in DUMMY_CLASSIFIER_TYPES:
            dummy_estimator_name = 'classifier'

        simple_pipeline = [(dummy_estimator_name, asl_pipeline)] # list of 2-tuples

        if not as_array:
            simple_pipeline = sklearn.pipeline.Pipeline(simple_pipeline)
        return simple_pipeline

    for step, element in asl_pipeline.steps:
        attr = pipeline_step_element_names[step]

        if (attr is not None): # and (type(element) not in base_types):

            choice = element.choice
            
            if type(choice) in passthrough_types:
                continue
            
            choice = choice.__getattribute__(attr)

            if type(choice) in passthrough_types:
                continue

            choice = sklearn.base.clone(choice)

            # check if this has a "prefit" attribute
            if hasattr(choice, 'prefit'):
                # if so and it is true
                choice.prefit = False                    
        else:
            choice = element


        simple_pipeline.append([step, choice])
        
    if not as_array:
        simple_pipeline = sklearn.pipeline.Pipeline(simple_pipeline)
    return simple_pipeline

def retrain_asl_wrapper(asl_wrapper, X_train, y_train):
    """ Retrain the ensemble in the asl_wrapper using the new training data

    The relative weights of the members of the ensemble will remaining
    unchanged. If some member cannot be retrained for some reason, it is
    discarded (and a warning message is logged).

    Parameters
    ----------
    asl_wrapper: AutoSklearnWrapper 
        The (fit) wrapper

    {X,y}_train: data matrices
        Data matrics of the same type used to originally train the wrapper

    Returns
    -------
    weights: np.array of floats
        The weights of members of the ensemble

    pipelines: np.array of sklearn pipelines
        The updated pipelines which could be retrained
    """

    new_weights = []
    new_ensemble = []

    it = zip(asl_wrapper.ensemble_[0], asl_wrapper.ensemble_[1])
    for (weight, asl_pipeline) in it:
        p = get_simple_pipeline(asl_pipeline)

        try:
            p.fit(X_train, y_train)
            new_weights.append(weight)
            new_ensemble.append(p)
        except Exception as ex:
            msg = "Failed to fit a pipeline. Error: {}".format(str(ex))
            logger.warning(msg)

    new_weights = np.array(new_weights) / np.sum(new_weights)
    new_ensemble = [new_weights, new_ensemble]
    
    return new_ensemble

def _validate_fit_asl_wrapper(asl_wrapper):
    """ Check that the AutoSklearnWrapper contains a valid ensemble
    """
    if len(asl_wrapper.ensemble_) != 2:
        msg = ("[asl_wrapper]: the ensemble_ must be a list-like with two "
            "elements. Presumably, it is either read from a pickle file using "
            "read_asl_wrapper or extracted from a fit autosklearn object.")
        raise ValueError(msg)

    if len(asl_wrapper.ensemble_[0]) != len(asl_wrapper.ensemble_[1]):
        msg = ("[asl_wrapper]: the ensemble_[0] and ensemble_[1] list-likes "
            "must be the same length.")
        raise ValueError(msg)

    if asl_wrapper.estimator_named_step not in ESTIMATOR_NAMED_STEPS:
        s = sorted([e for e in ESTIMATOR_NAMED_STEPS])
        msg = ("[asl_wrapper]: the estimator named step must be one of: {}".
            format(s))
        raise ValueError(msg)

    if asl_wrapper.autosklearn_optimizer is not None:
        msg = ("[asl_wrapper]: the fit wrapper should not include the "
            "autosklearn optimizer")
        raise ValueError(msg)


def _get_asl_estimator(asl_pipeline, pipeline_step='regressor'):
    """ Extract the concrete estimator (RandomForest, etc.) from the asl
    pipeline.

    In case the pipeline is one of the DUMMY_PIPELINE_TYPES, then the asl
    pipeline is actually the estimator, and it is returned as-is.
    """
    asl_pipeline_type = type(asl_pipeline)
    if asl_pipeline_type in DUMMY_PIPELINE_TYPES:
        return asl_pipeline

    asl_model = asl_pipeline.named_steps[pipeline_step]
    
    # this is from the old development branch
    #aml_model_estimator = aml_model_model.estimator

    # for the 0.1.3 branch, grab the "choice" estimator

    # this may be either a "choice" type or an actual model
    if type(asl_model) in ESTIMATOR_CHOICE_TYPES:
        asl_model = asl_model.choice.estimator

    return asl_model

def _get_asl_pipeline(aml_model):
    """ Extract the pipeline object from an autosklearn_optimizer model.
    """
    # this is from the old development branch
    # the model *contained* a pipeline
    #aml_model_pipeline = aml_model.pipeline_

    # this is the updated 0.1.3 branch

    # that is, the model simply *is* a pipeline now
    asl_pipeline = aml_model
    return asl_pipeline

def _extract_autosklearn_ensemble(autosklearn_optimizer,
        estimtor_named_step='regressor'):
    """ Extract the nonzero weights, associated pipelines and estimators from
    a fit autosklearn optimizer (i.e., an AutoSklearnRegressor or an
    AutoSklearnClassifier).
    """
    import numpy as np

    aml = autosklearn_optimizer._automl._automl
    models = aml.models_

    e = aml.ensemble_
    weights = e.weights_
    model_identifiers = np.array(e.identifiers_)

    nonzero_weight_indices = np.nonzero(weights)[0]
    nonzero_weights = weights[nonzero_weight_indices]
    nonzero_model_identifiers = model_identifiers[nonzero_weight_indices]
    
    asl_models = [
        models[tuple(m)] for m in nonzero_model_identifiers
    ]

    asl_pipelines = [
        _get_asl_pipeline(m) for m in asl_models
    ]
    
    asl_estimators = [
        _get_asl_estimator(p, estimtor_named_step) for p in asl_pipelines
    ]

    return (nonzero_weights, asl_pipelines, asl_estimators)

def filter_model_types(aml, model_types):
    """ Remove all models of the specified type from the ensemble.

    Parameters
    ----------
    aml : tuple of (weights, pipelines)
        A tuple representing a trained ensemble, read in with read_automl
        (or similarly constructed)

    model_types : set-like of types
        A set containing the types to remove. N.B. This *should not* be a set
        of strings; it should include the actual type objects.

    Returns
    -------
    filtered_aml : (weights, pipelines) tuple
        The weights and pipelines from the original aml, with the types
        specified by model_types removed.

        The weights are renormalized.
    """

    (weights, pipelines) = (np.array(aml[0]), np.array(aml[1]))
    estimator_types = [
        type(get_aml_estimator(p)) for p in pipelines
    ]
    
    to_filter = [
            i for i, et in enumerate(estimator_types) if not et in model_types
    ]
    
    weights = weights[to_filter]
    weights = weights / np.sum(weights)
    
    return (weights, pipelines[to_filter])

class AutoSklearnWrapper(object):
    """ A wrapper for an autosklearn optimizer to easily integrate it within a
    larger sklearn.Pipeline. The purpose of this class is largely to minimize
    the amount of time the autosklearn.AutoSklearnXXX objects are present. They
    include many functions related to the Bayesian optimization which are not
    relevant after the parameters of the ensemble have been fit. Thus, outside
    of the fit method, there is no need to keep it around.
    """

    def __init__(self,
            ensemble=None,
            autosklearn_optimizer=None,
            estimator_named_step=None,
            args=None,
            le=None,
            metric=None,
            **kwargs):

        msg = ("[asl_wrapper]: initializing a wrapper. ensemble: {}. "
            "autosklearn: {}".format(ensemble, autosklearn_optimizer))
        logger.debug(msg)

        self.args = args
        self.kwargs = kwargs
        self.ensemble_ = ensemble
        self.autosklearn_optimizer = autosklearn_optimizer
        self.estimator_named_step = estimator_named_step
        self.metric = metric
        self.le_ = le

    def create_classification_optimizer(self, args, **kwargs):        
        """ Create an AutoSklearnClassifier and use it as the autosklearn
        optimizer.

        The parameters can either be passed via an argparse.Namespace or using
        keyword arguments. The keyword arguments take precedence over args. The
        following keywords are used:

            * total_training_time
            * iteration_time_limit
            * ensemble_size
            * ensemble_nbest
            * seed
            * estimators
            * tmp

        Parameters
        ----------
        args: Namespace
            An argparse.Namepsace-like object which  presumably comes from
            parsing the add_automl_options arguments.

        kwargs: key=value pairs
            Additional options for creating the autosklearn classifier
        Returns
        -------
        self
        """

        args_dict = args.__dict__
        args_dict.update(kwargs)

        asl_classification_optimizer = AutoSklearnClassifier(
            time_left_for_this_task=args_dict.get('total_training_time', None),
            per_run_time_limit=args_dict.get('iteration_time_limit', None),
            ensemble_size=args_dict.get('ensemble_size', None),
            ensemble_nbest=args_dict.get('ensemble_nbest', None),
            seed=args_dict.get('seed', None),
            include_estimators=args_dict.get('estimators', None),
            tmp_folder=args_dict.get('tmp', None)
        )

        self.autosklearn_optimizer = asl_classification_optimizer
        self.estimator_named_step = "classifier"
        return self


    def create_regression_optimizer(self, args, **kwargs):        
        """ Create an AutoSklearnRegressor and use it as the autosklearn
        optimizer.

        The parameters can either be passed via an argparse.Namespace or using
        keyword arguments. The keyword arguments take precedence over args. The
        following keywords are used:

            * total_training_time
            * iteration_time_limit
            * ensemble_size
            * ensemble_nbest
            * seed
            * estimators
            * tmp

        Parameters
        ----------
        args: Namespace
            An argparse.Namepsace-like object which  presumably comes from
            parsing the add_automl_options arguments.

        kwargs: key=value pairs
            Additional options for creating the autosklearn regressor

        Returns
        -------
        self
        """

        args_dict = args.__dict__
        args_dict.update(kwargs)

        asl_regression_optimizer = AutoSklearnRegressor(
            time_left_for_this_task=args_dict.get('total_training_time', None),
            per_run_time_limit=args_dict.get('iteration_time_limit', None),
            ensemble_size=args_dict.get('ensemble_size', None),
            ensemble_nbest=args_dict.get('ensemble_nbest', None),
            seed=args_dict.get('seed', None),
            include_estimators=args_dict.get('estimators', None),
            tmp_folder=args_dict.get('tmp', None)
        )

        self.autosklearn_optimizer = asl_regression_optimizer
        self.estimator_named_step = "regressor"
        return self

    def fit(self, X_train, y, metric=None, encode_y=True):
        """ Optimize the ensemble parameters with autosklearn """

        # overwrite our metric, if one was given
        if metric is not None:
            self.metric = metric

        # check if we have either args or a learner
        if self.args is not None:
            if self.autosklearn_optimizer is not None:
                msg = ("[asl_wrapper]: have both args and an autosklearn "
                    "optimizer. Please set only one or the other.")
                raise ValueError(msg)

            if self.estimator_named_step == 'regressor':
                msg = ("[asl_wrapper]: creating an autosklearn regression "
                    "optimizer")
                logger.debug(msg)

                self.create_regression_optimizer(self.args, **self.kwargs)
            else:
                msg = ("[asl_wrapper]: creating an autosklearn classification "
                    "optimizer")
                logger.debug(msg)

                self.create_classification_optimizer(self.args, **self.kwargs)

                # sometimes, it seems we need to encode labels to keep them
                # around.
                if encode_y:
                    # In some cases, members of the ensemble drop some of the
                    # labels. Make sure we can reconstruct those
                    self.le = sklearn.preprocessing.LabelEncoder()
                    self.le_ = self.le.fit(y)
                    y = self.le_.transform(y)
                else:
                    # we still need to keep around the number of classes
                    # we assume y is a data frame which gives this value.
                    self.le = sklearn.preprocessing.LabelEncoder()
                    self.le_ = self.le
                    self.le_.classes = np.unique(y.columns)

        elif self.autosklearn_optimizer is None:
            msg = ("[asl_wrapper]: have neither args nor an autosklearn "
                "optimizer. Please set one or the other (but not both).")
            raise ValueError(msg)

        msg = ("[asl_wrapper]: fitting a wrapper with metric: {}".format(
            self.metric))
        logger.debug(msg)

        self.autosklearn_optimizer.fit(X_train, y, metric=self.metric)
        
        vals = _extract_autosklearn_ensemble(
            self.autosklearn_optimizer,
            self.estimator_named_step
        )

        (weights, pipelines, estimators) = vals
        self.ensemble_ = (weights, pipelines)
        
        # since we have the ensemble, we can get rid of the Bayesian optimizer
        self.autosklearn_optimizer = None

        # also, print the unique classes:
        #for e in estimators:
        #    print("estimator:", e, "unique class labels:", e.classes_)

        return self


    def _predict_regression(self, X_test):
        """ Predict the values using the fitted ensemble
        """
        _validate_fit_asl_wrapper(self)
        (weights, pipelines) = self.ensemble_

        y_pred = np.array([w*p.predict(X_test) 
                                for w,p in zip(weights, pipelines)])

        y_pred = y_pred.sum(axis=0)
        return y_pred

    def _predict_classification(self, X_test):
        """ Predict the class using the fitted ensemble and weighted majority
        voting
        """

        # first, get the weighted predictions from each member of the ensemble
        y_pred = self.predict_proba(X_test)

        # now take the majority vote
        y_pred = y_pred.argmax(axis=1)

        return y_pred        

    def predict(self, X_test):
        """ Use the fit ensemble to predict on the given test set.
        """
        _validate_fit_asl_wrapper(self)
        
        predict_f = self._predict_classification
        if self.estimator_named_step == "regressor":
            predict_f = self._predict_regression        
        
        return predict_f(X_test)

    def predict_proba(self, X_test):
        """ Use the automl ensemble to estimate class probabilities
        """
        if self.estimator_named_step == "regressor":
            msg = ("[asl_wrapper]: cannot use predict_proba for regression "
                "problems")
            raise ValueError(msg)

        _validate_fit_asl_wrapper(self)
        (weights, pipelines) = self.ensemble_

        res_shape = (X_test.shape[0], len(self.le_.classes_))
        res = np.zeros(shape=res_shape)

        for w,p in zip(weights, pipelines):

            # get the weighted class probabilities
            y_pred = w*p.predict_proba(X_test)
            
            # and make sure we are looking in the correct columns
            e = _get_asl_estimator(p, pipeline_step='classifier')
            p_classes = e.classes_
            res[:,p_classes] += y_pred

        return res

    def predict_dist(self, X_test, weighted=True):
        """ Use the ensemble to predict a normal distribution for each instance

        This method uses statsmodels.stats.weightstats.DescrStatsW to handle
        weights.

        Parameters
        ----------
        X_test: np.array
            The observations

        weighted: bool
            Whether to weight the predictions of the ensemble members

        Returns
        -------
        y_pred_mean: np.array
            The (weighted) predicted mean estimate

        y_pred_std: np.array
            The (weighted) standard deviation of the estimates
        """

        if self.estimator_named_step != "regressor":
            msg = ("[asl_wrapper]: predict_dist can only be used for "
                "regression problems")
            raise ValueError(msg)

        _validate_fit_asl_wrapper(self)
        (weights, pipelines) = self.ensemble_

        # make the predictions
        y_pred = np.array([p.predict(X_test) for p in pipelines])

        # extract summary statistics

        # the weighting does not work properly with single-member "ensembles"
        if weighted and len(weights) > 1:
            s = statsmodels.stats.weightstats.DescrStatsW(y_pred, weights=weights)
        else:
            s = statsmodels.stats.weightstats.DescrStatsW(y_pred)

        return (s.mean, s.std)

    def get_estimators(self):
        """ Extract the concrete estimators from the ensemble (RandomForest,
        etc.) as a list

        N.B. Predictions use the entire pipelines in the ensemble
        (preprocessors, etc.). This is primarily a convenience method for
        inspecting the learned estimators.
        """
        _validate_fit_asl_wrapper(self)
        pipelines = self.ensemble_[1]
        estimators = [
            _get_asl_estimator(p, self.estimator_named_step) for p in pipelines 
        ]
        return estimators

    def get_ensemble_model_summary(self):
        """ Create a mapping from pretty model names to their total weight in
        the ensemble
        
        Parameters
        ----------
        asl_wrapper: a fit AutoSklearnWrapper
        
        Returns
        -------
        ensemble_summary: dict of string -> float
            A mapping from model names to weights
        """
        weights = self.ensemble_[0]    
        estimators = self.get_estimators()
        
        summary = collections.defaultdict(float)
        
        for w, e in zip(weights, estimators):
            type_str = PRETTY_NAME_TYPE_MAP[type(e)]
            summary[type_str] += w
            
        return summary

    def get_params(self, deep=True):
        params = {
            'autosklearn_optimizer': self.autosklearn_optimizer,
            'ensemble_': self.ensemble_,
            'estimator_named_step': self.estimator_named_step,
            'args': self.args,
            'kwargs': self.kwargs,
            "le_": self.le_,
            'metric': self.metric
        }
        return params

    def set_params(self, **parameters):
        if 'args' in parameters:
            self.args = parameters.pop('args')

        if 'kwargs' in parameters:
            self.kwargs = parameters.pop('kwargs')

        if 'autosklearn_optimizer' in parameters:
            self.autosklearn_optimizer = parameters.pop('autosklearn_optimizer')

        if 'ensemble_' in parameters:
            self.ensemble_ = parameters.pop('ensemble_')

        if 'estimator_named_step' in parameters:
            self.estimator_named_step = parameters.pop('estimator_named_step')

        if 'le_' in parameters:
            self.le_ = parameters.pop('le_')

        if 'metric' in parameters:
            self.metric = parameters.pop('metric')

        return self


    def __getstate__(self):
        """ Returns everything to be pickled """
        state = {}
        state['args'] = self.args
        state['kwargs'] = self.kwargs
        state['autosklearn_optimizer'] = self.autosklearn_optimizer
        state['ensemble_'] = self.ensemble_
        state['estimator_named_step'] = self.estimator_named_step
        state['le_'] = self.le_
        state['metric'] = self.metric

        return state

    def __setstate__(self, state):
        """ Re-creates the object after pickling """
        self.args = state['args']
        self.kwargs = state['kwargs']
        self.autosklearn_optimizer = state['autosklearn_optimizer']
        self.estimator_named_step = state['estimator_named_step']
        self.ensemble_ = state['ensemble_']
        self.le_ = state['le_']
        self.metric = state['metric']

    def write(self, out_file):
        """ Validate that the wrapper has been fit and then write it to disk
        """
        _validate_fit_asl_wrapper(self)
        joblib.dump(self, out_file)

    @classmethod
    def read(cls, in_file):
        """ Read back in an asl_wrapper which was written to disk """
        asl_wrapper = joblib.load(in_file)
        return asl_wrapper

def add_automl_options(parser,
    default_out = ".",
    default_tmp = None,
    default_estimators = None,
    default_seed = 8675309,
    default_total_training_time = 3600,
    default_iteration_time_limit = 360,
    default_ensemble_size = 50,
    default_ensemble_nbest = 50):

    """ This function adds standard automl command line options to the parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        the argparse parser

    all others
        the default options to use in the parser

    Returns
    -------
    None, but the automl options are added to the parser
    """
    automl_options = parser.add_argument_group("auto-sklearn options")

    automl_options.add_argument('--out', help="The output folder", 
        default=default_out)
    automl_options.add_argument('--tmp', help="If specified, this will be used "
        "as the temp directory rather than the default", default=default_tmp)

    automl_options.add_argument('--estimators', help="The names of the estimators "
        "to use for learning. If not specified, all available estimators will "
        "be used.", nargs='*', default=default_estimators)

    automl_options.add_argument('--seed', help="The random seed", type=int,
        default=default_seed)

    automl_options.add_argument('--total-training-time', help="The total training "
        "time for auto-sklearn.\n\nN.B. This appears to be more of a "
        "\"suggestion\".", type=int, default=default_total_training_time)

    automl_options.add_argument('--iteration-time-limit', help="The maximum "
        "training time for a single model during the search.\n\nN.B. This also "
        "appears to be more of a \"suggestion\".", type=int,
        default=default_iteration_time_limit)

    automl_options.add_argument('--ensemble-size', help="The number of models to keep "
        "in the learned ensemble.", type=int, default=default_ensemble_size)
    automl_options.add_argument('--ensemble-nbest', help="The number of models to use "
        "for prediction.", type=int, default=default_ensemble_nbest)

def add_automl_values_to_args(args,
        out = ".",
        tmp = None,
        estimators = None,
        seed = 8675309,
        total_training_time = 3600,
        iteration_time_limit = 360,
        ensemble_size = 50,
        ensemble_nbest = 50):
        
    """ Add the automl options to the given argparse namespace

    This function is mostly intended as a helper for use in ipython notebooks.
    """
    args.out = out
    args.tmp = tmp
    args.estimators = estimators
    args.seed = seed
    args.total_training_time = total_training_time
    args.iteration_time_limit = iteration_time_limit
    args.ensemble_size = ensemble_size
    args.ensemble_nbest = ensemble_nbest


def get_automl_options_string(args):
    """ This function creates a string suitable for passing to another script
        of the automl command line options.

    The expected use case for this function is that a "driver" script is given
    the automl command line options (added to its parser with add_automl_options),
    and it calls multiple other scripts which also accept the automl options.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace containing all of the options from add_automl_options

    Returns
    -------
    string:
        A string containing all of the automl options
    """

    args_dict = vars(args)

    # first, pull out the text arguments
    automl_options = ['out', 'tmp', 'seed', 'total_training_time', 
        'iteration_time_limit', 'ensemble_size', 'ensemble_nbest']

    # create a new dictionary mapping from the flag to the value
    automl_flags_and_vals = {'--{}'.format(o.replace('_', '-')) : args_dict[o] 
        for o in automl_options if args_dict[o] is not None}

    estimators = ""
    if args_dict['estimators'] is not None:
        estimators = " ".join(mt for mt in args_dict['estimators'])
        estimators = "--estimators {}".format(estimators)

    s = ' '.join("{} {}".format(k,v) for k,v in automl_flags_and_vals.items())
    s = "{} {}".format(estimators, s)

    return s

###
#   Utilities to help with OpenBLAS
###
def add_blas_options(parser, default_num_blas_cpus=1):
    """ Add options to the parser to control the number of BLAS threads
    """
    blas_options = parser.add_argument_group("blas options")

    blas_options.add_argument('--num-blas-threads', help="The number of threads to "
        "use for parallelizing BLAS. The total number of CPUs will be "
        "\"num_cpus * num_blas_cpus\". Currently, this flag only affects "
        "OpenBLAS and MKL.", type=int, default=default_num_blas_cpus)

    blas_options.add_argument('--do-not-update-env', help="By default, num-blas-threads "
        "requires that relevant environment variables are updated. Likewise, "
        "if num-cpus is greater than one, it is necessary to turn off python "
        "assertions due to an issue with multiprocessing. If this flag is "
        "present, then the script assumes those updates are already handled. "
        "Otherwise, the relevant environment variables are set, and a new "
        "processes is spawned with this flag and otherwise the same "
        "arguments. This flag is not inended for external users.",
        action='store_true')

def get_blas_options_string(args):
    """  Create a string suitable for passing to another script of the BLAS
    command line options
    """
    s = "--num-blas-threads {}".format(args.num_blas_threads)
    return s


def spawn_for_blas(args):
    """ Based on the BLAS command line arguments, update the environment and
    spawn a new version of the process

    Parameters
    ----------
    args: argparse.Namespace
        A namespace containing num_cpus, num_blas_threads and do_not_update_env

    Returns
    -------
    spawned: bool
        A flag indicating whether a new process was spawned. Presumably, if it
        was, the calling context should quit
    """
    import os
    import sys
    import shlex

    import misc.shell_utils as shell_utils

    spawned = False
    if not args.do_not_update_env:

        ###
        #
        # There is a lot going on with settings these environment variables.
        # please see the following references:
        #
        #   Turning off assertions so we can parallelize sklearn across
        #   multiple CPUs for different solvers/folds
        #       https://github.com/celery/celery/issues/1709
        #
        #   Controlling OpenBLAS threads
        #       https://github.com/automl/auto-sklearn/issues/166
        #
        #   Other environment variables controlling thread usage
        #       http://stackoverflow.com/questions/30791550
        #
        ###
        
        # we only need to turn off the assertions if we parallelize across cpus
        if args.num_cpus > 1:
            os.environ['PYTHONOPTIMIZE'] = "1"

        # openblas
        os.environ['OPENBLAS_NUM_THREADS'] = str(args.num_blas_threads)
        
        # mkl blas
        os.environ['MKL_NUM_THREADS'] = str(args.num_blas_threads)

        # other stuff from the SO post
        os.environ['OMP_NUM_THREADS'] = str(args.num_blas_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(args.num_blas_threads)

        cmd = ' '.join(shlex.quote(a) for a in sys.argv)
        cmd += " --do-not-update-env"
        shell_utils.check_call(cmd)
        spawned = True

    return spawned


###
#   Utilities to help with the autofolio package:
#
#       https://github.com/mlindauer/AutoFolio
###

def load_autofolio(fn:str):
    """ Read a pickled autofolio model.

    Parameters
    ----------
    fn: string
        The path to the file

    Returns
    -------
    A namedtuple with the following fields:
        - scenario: ASlibScenario
            The aslib scenario information used to learn the model

        - preprocessing: list of autofolio.feature_preprocessing objects
            All of the preprocessing objects

        - pre_solver: autofolio.pre_solving.aspeed_schedule.Aspeed
            Presolving schedule

        - selector: autofolio.selector
            The trained pairwise selection model

        - config: ConfigSpace.configuration_space.Configuration
            The dict-like configuration information
    """
    import collections
    import pickle

    af = collections.namedtuple("af",
        "scenario,preprocessing,pre_solver,selector,config"
    )

    with open(fn, "br") as fp:
        autofolio_model = pickle.load(fp)

    autofolio_model = af(
        scenario = autofolio_model[0],
        preprocessing = autofolio_model[1],
        pre_solver = autofolio_model[2],
        selector = autofolio_model[3],
        config = autofolio_model[4]
    )

    return autofolio_model

###
#   Utilities to help with the ASlib package:
#
#       https://github.com/mlindauer/ASlibScenario
###

def _get_feature_step_dependencies(scenario, feature_step):
    fg = scenario.feature_group_dict[feature_step]
    return set(fg.get('requires', []))
    
def check_all_dependencies(scenario, feature_steps):
    """ Ensure all dependencies all included for all feature sets
    
    Parameters
    ----------
    scenario: ASlibScenario
        The ASlib scenario
        
    feature_steps: list-like of strings
        A list of feature sets
        
    Returns
    -------
    dependencies_met: bool
        True if all dependencies for all feature sets are included,
        False otherwise
    """
    feature_steps = set(feature_steps)
    for feature_step in feature_steps:
        dependencies = _get_feature_step_dependencies(scenario, feature_step)
        if not dependencies.issubset(feature_steps):
            return False
    return True

def extract_feature_step_dependency_graph(scenario):
    """ Create a graph encoding dependencies among the feature steps

    Specifically, an edge in the graph from A to B implies B requires A. Root
    nodes have no dependencies.

    Parameters
    ----------
    scenario: ASlibScenario
        The ASlib scenario

    Returns
    -------
    dependency_graph: networkx.DiGraph
        A directed graph in which each node corresponds to a feature step and
        edges encode the dependencies among the steps
    """
    # build the feature_group graph
    dependency_graph = nx.DiGraph()
    dependency_graph.add_nodes_from(scenario.feature_steps)

    # an edge from A -> B implies B requires A
    for feature_step in scenario.feature_steps:
        dependencies = _get_feature_step_dependencies(scenario, feature_step)
        
        for dependency in dependencies:
            dependency_graph.add_edge(dependency, feature_step)

    return dependency_graph

def extract_feature_names(scenario, feature_steps):
    """ Extract the names of features for the specified feature steps

    Parameters
    ----------
    scenario: ASlibScenario
        The ASlib scenario

    feature_steps: list-like of strings
        The names of the feature steps

    Returns
    -------
    feature_names: list of strings
        The names of all features in the given steps
    """
    feature_names = [
        scenario.feature_group_dict[f]['provides']
            for f in feature_steps
    ]
    feature_names = utils.flatten_lists(feature_names)
    return feature_names

def load_scenario(scenario_path, return_name=True):
    """ Load the ASlibScenario in the given path

    This is a convenience function that can be more easily used in list
    comprehensions, etc.

    Parameters
    ----------
    scenario_path: path-like
        The location of the scenario directory

    return_name: bool
        Whether to return the name of the scenario
    Returns
    -------
    scenario_name: string
        The name of the scenario (namely, `scenario.scenario`). This is only
        present if return_name is True

    scenario: ASlibScenario
        The actual scenario
    """
    scenario = ASlibScenario()
    scenario.read_scenario(scenario_path)

    if return_name:
        return scenario.scenario, scenario
    else:
        return scenario


def load_all_scenarios(scenarios_dir):
    """ Load all scenarios in scenarios_dir into a dictionary

    In particular, this function assumes all subdirectories within
    scenarios_dir are ASlibScenarios

    Parameters
    ----------
    scenarios_dir: path-like
        The location of the scenarios

    Returns
    -------
    scenarios: dictionary of string -> ASlibScenario
        A dictionary where the key is the name of the scenario and the value
        is the corresponding ASlibScenario
    """
        
    # first, just grab everything in the directory
    scenarios = [
        os.path.join(scenarios_dir, o) for o in os.listdir(scenarios_dir)
    ]

    # only keep the subdirectories
    scenarios = sorted([
        t for t in scenarios if os.path.isdir(t)
    ])

    # load the scenarios
    scenarios = [
        load_scenario(s) for s in scenarios
    ]
    
    # the list was already (key,value) pairs, so create the dictionary
    scenarios = dict(scenarios)

    return scenarios

def create_cv_splits(scenario):
    """ Create cross-validation splits for the scenario and save to file

    In particular, this is useful if a scenario does not already include cv
    splits, and cv splits will be used in multiple locations.

    Parameters
    ----------
    scenario: ASlibScenario
        The scenario

    Returns
    -------
    None, but new cv splits will be assigned to the instances and the file
    scenario.dir_.cv.arff will be (over)written.
    """
    import arff

    # first, make sure the cv splits exist
    if scenario.cv_data is None:
        scenario.create_cv_splits()

    # format the cv splits for the arff file
    scenario.cv_data.index.name = 'instance_id'
    cv_data = scenario.cv_data.reset_index()
    cv_data['repetition'] = 1
    cv_data['fold'] = cv_data['fold'].astype(int)

    # and put the fields in the correct order
    fields = ['instance_id', 'repetition', 'fold']
    cv_data_np = cv_data[fields].values

    # create the yaml-like dict for writing
    cv_dataset = {
        'description': "CV_{}".format(scenario.scenario),
        'relation': "CV_{}".format(scenario.scenario),
        'attributes': [
            ('instance_id', 'STRING'),
            ('repetition', 'NUMERIC'),
            ('fold', 'NUMERIC')
        ],
        'data': cv_data_np
    }

    # and write the file
    cv_loc = os.path.join(scenario.dir_, "cv.arff")
    with open(cv_loc, 'w') as f:
        arff.dump(cv_dataset, f)
