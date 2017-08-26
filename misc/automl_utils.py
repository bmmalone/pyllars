###
#   These are helpers related to auto-sklearn and AutoFolio
###

import logging
logger = logging.getLogger(__name__)

import misc.utils as utils

import joblib
import numpy as np
import sklearn.preprocessing

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

DUMMY_PIPELINE_TYPES= {
    utils.get_type("autosklearn.evaluation.abstract_evaluator.MyDummyClassifier"),
    utils.get_type("autosklearn.evaluation.abstract_evaluator.MyDummyRegressor"),   
    utils.get_type("autosklearn.evaluation.abstract_evaluator.DummyClassifier"),
    utils.get_type("autosklearn.evaluation.abstract_evaluator.DummyRegressor"),   
}

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
    "<class 'sklearn.svm.classes.LinearSVR'>": "linear support vector regression"
}


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
    asl_estimator = asl_model.choice.estimator

    return asl_estimator

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
            **kwargs):

        msg = ("[asl_wrapper]: initializing a wrapper. ensemble: {}. "
            "autosklearn: {}".format(ensemble, autosklearn_optimizer))
        logger.debug(msg)

        self.args = args
        self.kwargs = kwargs
        self.ensemble_ = ensemble
        self.autosklearn_optimizer = autosklearn_optimizer
        self.estimator_named_step = estimator_named_step

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

    def fit(self, X_train, y, metric=None):
        """ Optimize the ensemble parameters with autosklearn """

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

        elif self.autosklearn_optimizer is None:
            msg = ("[asl_wrapper]: have neither args nor an autosklearn "
                "optimizer. Please set one or the other (but not both).")
            raise ValueError(msg)

        msg = "[asl_wrapper]: fitting a wrapper"
        logger.debug(msg)

        self.autosklearn_optimizer.fit(X_train, y)
        
        vals = _extract_autosklearn_ensemble(
            self.autosklearn_optimizer,
            self.estimator_named_step
        )

        (weights, pipelines, estimators) = vals
        self.ensemble_ = (weights, pipelines)
        
        # since we have the ensemble, we can get rid of the Bayesian optimizer
        self.autosklearn_optimizer = None

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
        
        return predict_f(X_test, self)

    def predict_proba(self, X_test):
        """ Use the automl ensemble to estimate class probabilities
        """
        if self.estimator_named_step == "regressor":
            msg = ("[asl_wrapper]: cannot use predict_proba for regression "
                "problems")
            raise ValueError(msg)

        _validate_fit_asl_wrapper(self)
        (weights, pipelines) = self.ensemble_

        y_pred = np.array([w*p.predict_proba(X_test) 
                                for w,p in zip(weights, pipelines)])

        y_pred = y_pred.sum(axis=0)
        return y_pred

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

    def get_params(self, deep=True):
        msg = "[asl_wrapper]: get_params"
        logger.debug(msg)

        params = {
            'autosklearn_optimizer': self.autosklearn_optimizer,
            'ensemble_': self.ensemble_,
            'estimator_named_step': self.estimator_named_step,
            'args': self.args,
            'kwargs': self.kwargs
        }
        return params

    def set_params(self, **parameters):
        msg = "[asl_wrapper]: set_params"
        logger.debug(msg)

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

        return self


    def __getstate__(self):
        """ Returns everything to be pickled """
        import sys
        caller_name = sys._getframe(1).f_code.co_name

        msg = "[asl_wrapper]: calling __getstate__. caller: {}".format(caller_name)
        logger.debug(msg)
        
        state = {}
        state['args'] = self.args
        state['kwargs'] = self.kwargs
        state['autosklearn_optimizer'] = self.autosklearn_optimizer
        state['ensemble_'] = self.ensemble_
        state['estimator_named_step'] = self.estimator_named_step

        return state

    def __setstate__(self, state):
        """ Re-creates the object after pickling """
        msg = "[asl_wrapper]: __setstate__"
        logger.debug(msg)

        self.args = state['args']
        self.kwargs = state['kwargs']
        self.autosklearn_optimizer = state['autosklearn_optimizer']
        self.estimator_named_step = state['estimator_named_step']
        self.ensemble_ = state['ensemble_']

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

