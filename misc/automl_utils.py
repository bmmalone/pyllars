###
#   These are helpers related to auto-sklearn and AutoFolio
###

import logging
logger = logging.getLogger(__name__)


imputer_strategies = [
    'mean', 
    'median', 
    'most_frequent'
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


import autosklearn.pipeline.components.classification as c
import autosklearn.pipeline.components.feature_preprocessing as fp
import autosklearn.pipeline.components.regression as r

all_classifiers = c._classifiers.keys()
all_preprocessors = fp._preprocessors.keys()
all_regressors = r._regressors.keys()

from autosklearn.regression import AutoSklearnRegressor
from autosklearn.classification import AutoSklearnClassifier

pipeline_steps = [
    'one_hot_encoding', 
    'imputation', 
    'rescaling', 
    'preprocessor', 
    'regressor'
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


def _validate_aml(aml):
    """ Check that aml looks like a valid ensemble
    """

    if len(aml) < 2:
        msg = ("aml must be a list-like with at least two elements. Presumably, "
            "it is either read from a pickle file using read_automl or created "
            "by calling extract_automl_results on a fit AutoSklearnXXX model")
        raise ValueError(msg)


def get_aml_estimator(aml_model_pipeline, pipeline_step='regressor'):
    """ This function extracts the estimator (stored at the specified step in
        the pipeline.
    """
    
    print(aml_model_pipeline)
    aml_model_model = aml_model_pipeline.named_steps[pipeline_step]
    
    # this is from the old development branch
    #aml_model_estimator = aml_model_model.estimator

    # for the 0.1.3 branch, grab the "choice" estimator
    aml_model_estimator = aml_model_model.choice.estimator

    return aml_model_estimator

def get_aml_pipeline(aml_model):
    """ This function extracts the pipeline object from an aml_model.
    """
    # this is from the old development branch
    # the model *contained* a pipeline
    #aml_model_pipeline = aml_model.pipeline_

    # this is the updated 0.1.3 branch

    # that is, the model simply *is* a pipeline now
    aml_model_pipeline = aml_model
    return aml_model_pipeline

def extract_automl_results(automl, estimtor_named_step='regressor'):
    """ This returns the nonzero weights, associated pipelines and estimators
        from a fitted "automl" object (i.e., an AutoSklearnRegressor or an
        AutoSklearnClassifier).

        Imports:
            numpy
    """
    import numpy as np

    aml = automl._automl._automl
    models = aml.models_

    e = aml.ensemble_
    weights = e.weights_
    model_identifiers = np.array(e.identifiers_)

    nonzero_weight_indices = np.nonzero(weights)[0]
    nonzero_weights = weights[nonzero_weight_indices]
    nonzero_model_identifiers = model_identifiers[nonzero_weight_indices]
    
    aml_models = [
        models[tuple(m)] for m in nonzero_model_identifiers
    ]

    for m in aml_models:
        print("model: ", m)
    
    aml_pipelines = [
        get_aml_pipeline(m) for m in aml_models
    ]
    
    aml_estimators = [
        get_aml_estimator(m, estimtor_named_step) for m in aml_pipelines
    ]

    return (nonzero_weights, aml_pipelines, aml_estimators)

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
    import numpy as np

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

def write_automl(automl, out_file, estimator_named_step='regressor'):
    """ This function extracts the necessary bits to reconstruct a fitted
        "automl" object. It then writes that to a pickle file.

        Imports:
            pickle
    """
    import pickle

    (weights, pipelines, estimators) = extract_automl_results(automl)
    
    with open(out_file, 'wb') as out:
        pickle.dump((weights, pipelines), out)

def read_automl(in_file):
    """ This function reads back in the weights, pipelines, and estimators
        written to disk with write_automl.

        Imports:
            pickle
    """
    import pickle

    automl = pickle.load(open(in_file, 'rb'))
    automl = AutoML(automl)
    return automl

def automl_predict_regression(X_test, aml):
    """ Predict the values using the fitted automl object
    """
    import numpy as np

    _validate_aml(aml)
    (weights, pipelines) = (aml[0], aml[1])

    y_pred = np.array([w*p.predict(X_test) 
                            for w,p in zip(weights, pipelines)])

    y_pred = y_pred.sum(axis=0)
    return y_pred

def automl_predict_classification(X_test, aml):
    """ Predict the class using the fitted automl object and weighted majority
    voting
    """

    # first, get the weighted predictions from each member of the ensemble
    y_pred = automl_predict_proba(X_test, aml)

    # now take the majority vote
    y_pred = y_pred.argmax(axis=1)

    return y_pred


def automl_predict_proba(X_test, aml):
    """ Predict the class probabilities using the fitted automl object
    """
    import numpy as np

    _validate_aml(aml)
    (weights, pipelines) = (aml[0], aml[1])

    y_pred = np.array([w*p.predict_proba(X_test) 
                            for w,p in zip(weights, pipelines)])

    y_pred = y_pred.sum(axis=0)
    return y_pred




class AutoSklearnWrapper(object):
    """ A wrapper for an autosklearn wrapper to easily integrate it within a
    larger sklearn.Pipeline. The purpose of this class is largely to buffer
    usage while auto-sklearn is still under development.
    """

    def __init__(self, aml=None, autosklearn_model=None):
        self.aml = aml
        self.autosklearn_model = autosklearn_model
        self.estimator_named_step = None

    def create_classifier(self, args, **kwargs):        
        """ Create an AutoSklearnClassifier and use it as the autosklearn_model.

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

        classifier = AutoSklearnClassifier(
            time_left_for_this_task=args_dict.get('total_training_time', None),
            per_run_time_limit=args_dict.get('iteration_time_limit', None),
            ensemble_size=args_dict.get('ensemble_size', None),
            ensemble_nbest=args_dict.get('ensemble_nbest', None),
            seed=args_dict.get('seed', None),
            include_estimators=args_dict.get('estimators', None),
            tmp_folder=args_dict.get('tmp', None)
        )

        self.autosklearn_model = classifier
        self.estimator_named_step = "classifier"
        return self


    def create_regressor(self, args, **kwargs):        
        """ Create an AutoSklearnRegressor and use it as the autosklearn_model.

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

        regressor = AutoSklearnRegressor(
            time_left_for_this_task=args_dict.get('total_training_time', None),
            per_run_time_limit=args_dict.get('iteration_time_limit', None),
            ensemble_size=args_dict.get('ensemble_size', None),
            ensemble_nbest=args_dict.get('ensemble_nbest', None),
            seed=args_dict.get('seed', None),
            include_estimators=args_dict.get('estimators', None),
            tmp_folder=args_dict.get('tmp', None)
        )

        self.autosklearn_model = regressor
        self.estimator_named_step = "regressor"
        return self

    def predict(self, X_test):
        """ Use the automl ensemble to predict on the given test set.
        """
        
        predict_f = automl_predict_classification
        if self.estimator_named_step == "regressor":
            predict_f = automl_predict_regression        

        if self.aml is not None:
            return predict_f(X_test, self.aml)
        elif self.autosklearn_model is not None:
            vals = extract_automl_results(self.autosklearn_model)
            (weights, pipelines, estimators) = vals
            self.aml = (weights, pipelines)
            return predict_f(X_test, self.aml)
        else:
            msg = ("[AutoML]: cannot predict without setting aml or "
                "autosklearn_model")
            raise ValueError(msg)

    def predict_proba(self, X_test):
        """ Use the automl ensemble to estimate class probabilities
        """

        if self.estimator_named_step == "regressor":
            msg = ("[AutoML]: cannot use predict_proba for regression "
                "problems")
            raise ValueError(msg)

        if self.aml is not None:
            return automl_predict_proba(X_test, self.aml)
        elif self.autosklearn_model is not None:
            vals = extract_automl_results(self.autosklearn_model)
            (weights, pipelines, estimators) = vals
            self.aml = (weights, pipelines)
            return automl_predict_proba(X_test, self.aml)
        else:
            msg = ("[AutoML]: cannot predict without setting aml or "
                "autosklearn_model")
            raise ValueError(msg)



    def fit(self, X_train, y):
        """ Fit the autosklearn model. """
        if self.autosklearn_model is None:
            msg = "[AutoML]: cannot fit with having an autosklearn_model"
            raise ValueError(msg)

        self.autosklearn_model.fit(X_train, y)
        vals = extract_automl_results(self.autosklearn_model,
            self.estimator_named_step)
        (weights, pipelines, estimators) = vals
        self.aml = (weights, pipelines)

        return self

    def __getstate__(self):
        """ This returns everything to be pickled. """
        state = {}
        if self.autosklearn_model is not None:
            vals = extract_automl_results(self.autosklearn_model,
                self.estimator_named_step)
            (weights, pipelines, estimators) = vals
            state['weights'] = weights
            state['pipelines'] = pipelines
            state['estimator_named_step'] = self.estimator_named_step
        elif self.aml is not None:
            state['weights'] = self.aml[0]
            state['pipelines'] = self.aml[1]
            state['estimator_named_step'] = self.estimator_named_step
        else:
            msg = ("[AutoML]: cannot pickle without setting aml or "
                "autosklearn_model")
            raise ValueError(msg)
        return state

    def __setstate__(self, state):
        """ This re-creates the object after pickling. """
        self.aml = (state['weights'], state['pipelines'])
        self.estimator_named_step = state['estimator_named_step']
        self.autosklearn_model = None

        

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

    automl_options.add_argument('-o', '--out', help="The output folder", 
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
    parser.add_argument('--num-blas-threads', help="The number of threads to "
        "use for parallelizing BLAS. The total number of CPUs will be "
        "\"num_cpus * num_blas_cpus\". Currently, this flag only affects "
        "OpenBLAS and MKL.", type=int, default=default_num_blas_cpus)

    parser.add_argument('--do-not-update-env', help="By default, num-blas-threads "
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

