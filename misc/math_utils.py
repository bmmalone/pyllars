"""
This module contains utilities which simplify working with the numpy stack:
    * numpy
    * pandas (but see the note below about pandas_utils)
    * scipy
    * sklearn

It also contains other math helpers.

This module differs from pandas_utils because this module treats pandas
data frames as data matrices (in a statistical/machine learning sense),
while that module considers data frames more like database tables which
hold various types of records.
"""

import collections
import itertools
from enum import Enum

import more_itertools
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
import sklearn.model_selection

import logging
logger = logging.getLogger(__name__)

def calculate_symmetric_kl_divergence(p, q, calculate_kl_divergence):
    """ This function calculates the symmetric KL-divergence between 
    distributions p and q. In particular, it defines the symmetric
    KL-divergence to be:
    
    .. math::
        D_{sym}(p||q) := \frac{D(p||q) + D(p||p)}{2}

    Args:
        p, q: a representation of a distribution that can be used by the 
            function ``calculate_kl_divergence"
        calculate_kl_divergence: a function the calculates the KL-divergence 
            between :math:`p` and :math:`q`

    Returns:
        float: the symmetric KL-divergence between :math:`p` and :math:`q`

    """
    kl_1 = calculate_kl_divergence(p, q)
    kl_2 = calculate_kl_divergence(q, p)
    symmetric_kl = (kl_1 + kl_2) / 2
    return symmetric_kl

def symmetric_entropy(p, q):
    """ Calculate the symmetric scipy.stats.entropy. """
    return calculate_symmetric_kl_divergence(p, q, scipy.stats.entropy)

def symmetric_gaussian_kl(p, q):
    """ Calculate the symmetric gaussian KL divergence. """
    return calculate_symmetric_kl_divergence(
        p, q, 
        calculate_univariate_gaussian_kl
    )

def random_pick(probs):
    """ Select an item according to the specified categorical distribution
    """
    '''
    >>> probs = [.3, .7]
    >>> random_pick(probs)
    '''
    cutoffs = np.cumsum(probs)
    idx = cutoffs.searchsorted(np.random.uniform(0, cutoffs[-1]))
    return idx

def mask_random_values(X, likelihood=0.1, return_mask=False, random_state=None):
    """ Replace values of X with np.nan, presumably to simulate missing data.

    Specifically, this matrix simulates a missing completely at random (MCAR)
    mechanism. Thus, the likelihood that a particular cell of X is replaced
    with np.nan does not depend on its own value or those of any other any
    other cells.

    Also, this function does not treat np.nan values already present in X in
    any special way.

    Parameters
    ----------
    X: np.array-like
        The data. It must have `shape` and `copy` methods, as well as support
        Boolean mask indexing. Thus, some forms of scipy.sparse_matrix may
        also work. The function also handles pandas data frames (by directly
        updating the `values`).

    likelihood: float between 0 and 1
        The likelihood that each value in X is replaced with np.nan.

    return_mask: bool
        Whether to include the Boolean indicator matrix of values selected
        to remove.

    random_state: int
        An attempt to make things reproducible

    Returns
    -------
    X_incomplete: np.array
        An array with the same shape as X, but with some values replaced with
        np.nan.

    missing_mask: np.array (if return_mask is True)
        A Boolean array with the same shape as where True values in the mask
        indicating that the respective cell in X was replaced.
    """
    np.random.seed(random_state)
    missing_mask = np.random.rand(*X.shape) < likelihood

    # missing entries indicated with NaN
    if isinstance(X, pd.DataFrame):
        ###
        # see this SO thread for using `where` instead of masking
        #   https://stackoverflow.com/questions/30519140
        ###

        #X_incomplete.values[missing_mask] = np.nan
        X_incomplete = X.where(~missing_mask, other=np.nan)
    else:
        X_incomplete = X.copy()
        X_incomplete[missing_mask] = np.nan

    ret = X_incomplete
    if return_mask:
        ret = (X_incomplete, missing_mask)

    return ret

def permute_matrix(m, is_flat=False, shape=None):
    """ Randomly permute the entries of the matrix. The matrix is first 
    flattened.

    Parameters
    ----------
    m: np.array
        The matrix (tensor, etc.)

    is_flat: bool
        Whether the matrix values have already been flattened. If they have
        been, then the desired shape must be passed.

    shape: tuple of ints
        The shape of the output matrix, if m is already flattened

    Returns
    -------
    permuted_m: np.array
        A copy of m (with the same shape as m) with the values randomly 
        permuted. 
    """

    if shape is None:
        shape = m.shape

    if not is_flat:
        m = np.reshape(m, -1)

    # shuffle the actual values
    permuted_m = np.random.permutation(m)

    # and put them back in the correct shape
    permuted_m = np.reshape(permuted_m, shape)

    return permuted_m

def calculate_univariate_gaussian_kl(mean_p_var_p, mean_q_var_q):
    """ This function calculates the (asymmetric) KL-divergence between
        the univariate Gaussian distributions p and q.

        N.B. This function uses the variance!

        N.B. The parameters for each distribution are passed as pairs 
        for easy use with calculate_symmetric_kl_divergence.

        See, for example, (Penny, 2001) for the formula.

        Args:
            mean_p, var_p, mean_q, var_q (reals) : the parameters of the
                distributions.

        Returns:
            float : the KL divergence.
    """
    import numpy as np

    (mean_p, var_p) = mean_p_var_p
    (mean_q, var_q) = mean_q_var_q

    t_1 = 0.5 * (np.log(var_p) - np.log(var_q))
    t_2 = np.log(mean_q*mean_q + mean_p*mean_p + var_q - 2*mean_q*mean_p)
    t_3 = np.log(2*var_p)
    kl = t_1 - 0.5 + np.exp(t_2 - t_3)
    return kl

def has_nans(X):
    """ Check if `X` has any np.nan values

    Parameters
    ----------
    X: np.array
        The array

    Returns
    -------
    has_nans: bool
        True if any np.nan's are in `X`, False otherwise
    """
    # please see: https://stackoverflow.com/questions/6736590 for details
    return np.isnan(np.sum(X))


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

def remove_negatives(x): 
    """ Remove all negative and NaN values from x

    Parameters
    ----------
    x: np.array
        An array

    Returns
    -------
    non_negative_x: np.array
        A copy of x which does not contain any negative (or NaN) values. The
        shape of non_negative_x depends on the number of negative/NaN values
        in x.
    """
    x = x[x >= 0]
    return x



def is_monotonic(x, increasing=True):
    """ This function checks whether a given list is monotonically increasing
        (or decreasing). By definition, "monotonically increasing" means the
        same as "strictly non-decreasing".

        Adapted from: http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity

        Args:
            x (sequence) : a 1-d list of numbers

            increasing (bool) : whether to check for increasing monotonicity
    """
    import numpy as np

    dx = np.diff(x)

    if increasing:
        return np.all(dx >= 0)
    else:
        return np.all(dx <= 0)

def check_range(val, min_, max_, min_inclusive=True, max_inclusive=True, 
        variable_name=None, raise_on_invalid=True, logger=logger):

    """ This function checks whether the given value falls within the
        specified range. If not, either an exception is raised or a
        warning is logged.

        Args:
            val (number): the value to check

            min_, max_ (numbers): the acceptable range

            min_inclusive, max_inclusive (bools): whether the end
                points are included in the acceptable range

            variable_name (string): for the exception/warning, the
                name to use in the message

            raise_on_invalid (bool): whether to raise an exception (True)
                or issue a warning (False) when the value is invalid

            logger (logging.Logger): the logger to use in case a
                warning is issued

        Returns:
            None

        Raises:
            ValueError, if val is not within the valid range and
                raise_on_invalid is True

        Imports:
            operator
    """
    import operator

    # first, check min
    min_op = operator.le
    if min_inclusive:
        min_op = operator.lt

    if min_op(val, min_):
        msg = ("Variable: {}. The given value ({}) was less than the "
            "acceptable minimum ({})".format(variable_name, val, min_))

        if raise_on_invalid:
            raise ValueError(msg)
        else:
            logger.warning(msg)

    # now max
    max_op = operator.ge
    if max_inclusive:
        max_op = operator.gt

    if max_op(val, max_):
        msg = ("Variable: {}. The given value ({}) was greater than the "
            "acceptable maximum ({})".format(variable_name, val, max_))

        if raise_on_invalid:
            raise ValueError(msg)
        else:
            logger.warning(msg)


def write_sparse_matrix(target, a, compress=True, **kwargs):
    """ This function is a drop-in replacement for scipy.io.mmwrite. The only
        difference is that is compresses the output by default. It _does not_
        alter the file extension, which should likely end in "mtx.gz" except
        in special circumstances.

        Args:
            compress (bool): whether to compress the output.

            All other arguments are exactly the same as for scipy.io.mmwrite.
            Please see the scipy documentation for more details.
        
        Returns:
            None

        Imports:
            gzip
            scipy.io
    """
    import gzip
    import scipy.io

    if compress:
        with gzip.open(target, 'wb') as target_gz:
            scipy.io.mmwrite(target_gz, a, **kwargs)
    else:
        scipy.io.mmwrite(target_gz, a, **kwargs)

def row_op(m, op):
    """Apply op to each row in the matrix."""
    return op(m, axis=1)

def col_op(m, op):
    """Apply op to each column in the matrix."""
    return op(m, axis=0)

def row_sum(m):
    """Calculate the sum of each row in the matrix."""
    return  np.sum(m, axis=1)

def col_sum(m):
    """Calculate the sum of each column in the matrix."""
    return  np.sum(m, axis=0)

def row_sum_mean(m, var=False):
    """ Calculate the mean of the sum of each row in the matrix. Optionally,
    the variances of the row sums can also be calculated.

    Parameters
    ----------
    m: 2-dimensional np.array
        The matrix

    var: bool
        Whether to calculate the variances

    Returns
    -------
    mean: float
        The mean of the row sums in the matrix

    variance: float
        If var is True, then the variance of the row sums
    """
    row_sums = row_sum(m)
    mean = np.mean(row_sums)

    if var:
        var = np.var(row_sums)
        return mean, var

    return mean

def col_sum_mean(m, var=False):
    """ Calculate the mean of the sum of each column in the matrix. Optionally,
    the variances of the column sums can also be calculated.

    Parameters
    ----------
    m: 2-dimensional np.array
        The matrix

    var: bool
        Whether to calculate the variances

    Returns
    -------
    mean: float
        The mean of the column sums in the matrix

    variance: float
        If var is True, then the variance of the column sums
    """
    col_sums = col_sum(m)
    mean = np.mean(col_sums)

    if var:
        var = np.var(col_sums)
        return mean, var

    return mean

def normalize_rows(matrix):
    """ Normalize the rows of the given (dense) matrix

    Parameters
    ----------
    matrix: 2-dimensional np.array
        The matrix

    Returns
    -------
    normalized_matrix: 2-dimensional np.array
        The matrix normalized s.t. all row sums are 1
    """
    coef = matrix.sum(axis=1)
    coef = coef[:,np.newaxis]
    matrix = np.divide(matrix, coef)
    return matrix

def normalize_columns(m):
    """ Normalize the columns of the given (dense) matrix

    Parameters
    ----------
    m: 2-dimensional np.array
        The matrix

    Returns
    -------
    normalized_matrix: np.array with the same shape as m
        The matrix normalized s.t. all column sums are 1
    """
    col_sums = np.sum(m, axis=0)
    m = np.divide(m, col_sums)
    return m

def to_dense(sparse_matrix):
    """ Convert the scipy.sparse matrix to a dense np.array

    Parameters
    ----------
    sparse_matrix: scipy.sparse
        The sparse scipy matrix

    Returns
    -------
    dense_matrix: 2-dimensional np.array
        The dense np.array
    """
    dense = np.array(sparse_matrix.todense())
    return dense

def matrix_multiply(m1, m2, m3):
    """ This helper function multiplies the three matrices. It minimizes the
        size of the intermediate matrix created by the first multiplication.

        Args:
            m1, m2, m3 (2-dimensional np.arrays): the matrices to multiply

        Returns:
            np.array, with shape m1.shape[0] x m3.shape[1]

        Imports:
            numpy

        Raises:
            ValueError: if the dimensions do not match
    """

    # check the dimensions
    if m1.shape[1] != m2.shape[0]:
        msg = ("The inner dimension between the first and second matrices do "
            "not match. {} and {}".format(m1.shape, m2.shape))
        raise ValueError(msg)

    if m2.shape[1] != m3.shape[0]:
        msg = ("The inner dimensions between the second and third matrices do "
            "not match. {} and {}".format(m2.shape, m3.shape))
        raise ValueError(msg)

    # now, check which order to perform the multiplications
    m1_first_size = m1.shape[0] * m2.shape[1]
    m3_first_size = m2.shape[0] * m3.shape[1]

    if m1_first_size < m3_first_size:
        res = np.dot(m1, m2)
        res = np.dot(res, m3)
    else:
        res = np.dot(m2, m3)
        res = np.dot(m1, res)

    return res

def fit_bayesian_gaussian_mixture(X, 
                                  n_components=100, 
                                  seed=8675309,
                                  **kwargs):

    """ Fit a sklearn.mixture.BayesianGaussianMixture with the parameters.

    This function is mostly used to give slightly more reasonable defaults for
    a few of the parameters for DP-GMMs.

    Parameters
    ----------
    X : np.array
        The data matrix, where rows are instances and columns are features

    n_components : int
        The maximum number of components for the mixture (stick-breaking)

    seed : int or np.RandomState
        A seed for initializing the random number generator

    **kwargs : {key}={value}
        Other parameters to pass to the BGM constructor. Please see the sklearn
        documentation for acceptable options.

    Returns
    -------
    m : sklearn.mixture.BayesianGaussianMixture
        The fit model
    """

    import sklearn.mixture
    
    m = sklearn.mixture.BayesianGaussianMixture(
        n_components=n_components, 
        random_state=seed,
        **kwargs)
    
    m.fit(X)
    
    return m

class polynomial_order(Enum):
    linear = 1
    quadratic = 2

def fit_with_least_squares(x, y, w=None, order=polynomial_order.linear):
    """ Fit a polynomial relationship between x and y. Optionally, the values
    can be weighted.

    Parameters
    ----------
    x, y : arrays of floats
        The input arrays

    w : array of floats
        A weighting of each of the (x,y) pairs for regression

    order : from polynomial_order enum
        The order of the fit

    Returns
    -------
    intercept, slope, power : floats
        The coefficients of the fit. power is 0 if the order is linear.

    r_sqr : float
        The coefficient of determination
    """

    #Calculate trendline
    coeffs = np.polyfit(x, y, order.value, w=w)

    intercept = coeffs[-1]
    slope = coeffs[-2]
    power = coeffs[0] if order != polynomial_order.linear else 0 

    #Calculate R Squared
    p = np.poly1d(coeffs)

    ybar = np.sum(y) / len(y)
    ssreg = np.sum((p(x) - ybar) ** 2)
    sstot = np.sum((y - ybar) ** 2)
    r_sqr = ssreg / sstot
    

    return intercept, slope, power, r_sqr

def bayesian_proportion_test(x, n, prior=(0.5,0.5), prior2=None, 
        num_samples=1000, seed=None):
    """ Perform a Bayesian test to identify significantly different proportions.
    
    This test is based on a beta-binomial conjugate model. It uses simulations to
    estimate the posterior of the difference between the proportions, as well as
    the likelihood that \pi_1 > \pi_2 (where \pi_i is the likelihood of success in 
    sample i).
    
    Parameters
    ----------
    x: two-element array-like of integers
        The number of successes in each sample
        
    n: two-element array-like of integers
        The number of trials in each sample
        
    prior: two-element array-like of floats
        The parameters of the beta distribution used as the prior in the conjugate
        model for the first sample.
        
    prior2: two-element array-like of floats, or None
        The parameters of the beta distribution used as the prior in the conjugate
        model for the second sample. If this is not specified, then prior is used.
        
    num_samples: int
        The number of simulations
        
    seed: int
        The seed for the random number generator
        
    Returns
    -------
    (difference_mean, difference_var) : a 2-tuple of floats
        The posterior mean and variance of the difference in the likelihood of success 
        in the two samples. A negative mean indicates that the likelihood in sample 2 
        is higher.
        
    p_pi_1_greater : float
        The probability that pi_1 > pi_2
    """
    import numpy as np
    import scipy.stats
    
    # copy over the prior if not specified for sample 2
    if prior2 is None:
        prior2 = prior
        
    # check the bounds
    if len(x) != 2:
        msg = "[bayesian_proportion_test]: please ensure x has exactly two elements"
        raise ValueError(msg)
    if len(n) != 2:
        msg = "[bayesian_proportion_test]: please ensure n has exactly two elements"
        raise ValueError(msg)
    if len(prior) != 2:
        msg = "[bayesian_proportion_test]: please ensure prior has exactly two elements"
        raise ValueError(msg)
    if len(prior2) != 2:
        msg = "[bayesian_proportion_test]: please ensure prior2 has exactly two elements"
        raise ValueError(msg)
        
    
    # set the seed
    if seed is not None:
        np.random.seed(seed)
    
    # perform the test
    a = prior[0]+x[0]
    b = prior[0]+n[0]-x[0]
    s1_posterior_samples = scipy.stats.beta.rvs(a, b, size=num_samples)

    a = prior[1]+x[1]
    b = prior[1]+n[1]-x[1]
    s2_posterior_samples = scipy.stats.beta.rvs(a, b, size=num_samples)
    
    diff_posterior_samples = s1_posterior_samples - s2_posterior_samples
    diff_posterior_mean = np.mean(diff_posterior_samples)
    diff_posterior_var = np.var(diff_posterior_samples)
    
    p_pi_1_greater = sum(s1_posterior_samples > s2_posterior_samples) / num_samples
    
    return (diff_posterior_mean, diff_posterior_var), p_pi_1_greater

def normal_inverse_chi_square(m, k, r, s, size=1):
    """ Sample from a normal-inverse-chi-square distribution with parameters
    m, k, r, s.

    This distribution is of interest because it is a conjugate prior for
    Gaussian observations.

    Sampling is described in: https://www2.stat.duke.edu/courses/Fall10/sta114/notes15.pdf
    
    Parameters
    ----------
    m, k: float
        m is the mean of the sampled mean; k relates to the variance of the
        sampled mean.

    r, s: float
        r is the degrees of freedom in the chi-square distribution from which
        the variance is samples; s is something like a scaling factor.

    size: int or tuple of ints, or None
        Output shape. This shares the semantics as other numpy sampling functions.
    """
    
    x = np.random.chisquare(r, size=size)
    v = (r*s)/x
    w = np.random.normal(m, v/k)
        
    return w, v

def bayesian_means_test(x1, x2, jeffreys_prior=True, prior1=None, prior2=None, 
        num_samples=1000, seed=None):
    """ Perform a Bayesian test to identify significantly different means.
    
    The test is based on a Gaussian conjugate model. (The normal-inverse-chi-square
    distribution is the prior.) It uses simulation to estimate the posterior of the 
    difference between the means of the populations, under the (probably dubious) 
    assumption that the observations are Gaussian distributed. It also estimates the
    likelihood that \mu_1 > \mu_2, where \mu_i is the mean of each sample.
    
    Parameters
    ----------
    x{1,2}: array-like of floats
        The observations of each sample
        
    jeffreys_prior: bool
        Whether to use the Jeffreys prior. For more details, see:
        
            Murphy, K. Conjugate Bayesian analysis of the Gaussian distribution. 
            Technical report, 2007.
            
        Briefly, the Jeffreys prior is: (sample mean, n, n-1, sample variance)
            
    prior{1,2}: four-element tuple, or None
        If the Jeffreys prior is not used, then these parameters are used as the
        priors for the normal-inverse-chi-square. If only prior1 is given, then
        those values are also used for prior2, where prior_i is taken as the prior
        for x_i.
        
    num_samples: int
        The number of simulations
        
    seed: int
        The seed for the random number generator
        
    Returns
    -------
    (difference_mean, difference_var) : a 2-tuple of floats
        The posterior mean and variance of the difference in the mean of the two
        samples. A negative difference_mean indicates that the mean of x2 is 
        higher.
        
    p_m1_greater : float
        The probability that mu_1 > mu_2

    """
    
    if jeffreys_prior:
        prior1 = (np.mean(x1), len(x1), int(len(x1)-1), np.var(x1))
        prior2 = (np.mean(x2), len(x2), int(len(x2)-1), np.var(x2))
        
    elif prior2 is None:
        prior2 = prior1
        
    if prior1 is None:
        msg = ("[bayesian_means_test]: either the Jeffreys prior must be "
               "used, or the parameters of the normal-inverse-chi-square "
               "distributions must be given.")
        raise ValueError(msg)
        
    if len(prior1) != 4:
        msg = ("[bayesian_means_test]: please ensure prior1 has exactly four "
            "elements")
        raise ValueError(msg)
    if len(prior2) != 4:
        msg = ("[bayesian_means_test]: please ensure prior2 has exactly four "
            "elements")
        raise ValueError(msg)
        
    if seed is not None:
        np.random.seed(seed)
        
    # now, sample from the posterior distributions
    s1 = normal_inverse_chi_square(*prior1, size=num_samples)
    s2 = normal_inverse_chi_square(*prior2, size=num_samples)

    w1_posterior_samples, v1_posterior_samples = s1
    w2_posterior_samples, v2_posterior_samples = s2
    
    diff_posterior_samples = w1_posterior_samples - w2_posterior_samples
    diff_posterior_mean = np.mean(diff_posterior_samples)
    diff_posterior_var = np.var(diff_posterior_samples)
    
    m_m1_greater = w1_posterior_samples > w2_posterior_samples
    p_m1_greater = np.sum(m_m1_greater) / num_samples

    return (diff_posterior_mean, diff_posterior_var), p_m1_greater

def l1_distance(p, q):
    """ Calculate the l1 distance between the two vectors.

    Parameters
    ----------
    p, q: np.arrays of the same shape
        The vectors (or matrices) for which the distance will be calculated

    Returns
    -------
    l1_distance: float
        The l1 distance (sum of absolute differences) between p and q

    Raises
    ------
    ValueError: if p and q do not have the same shape
    """
    if p.shape != q.shape:
        msg = "[math_utils.l1_distance]: p and q must have the same shape"
        raise ValueError(msg)

    diff = np.abs(p - q)
    return np.sum(diff)


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
    

fold_tuple_fields = [
    'X_train',
    'y_train',
    'X_test',
    'y_test'
]
fold_tuple = collections.namedtuple('fold', ' '.join(fold_tuple_fields))

def get_kth_fold(X, y, fold, num_folds=10, random_seed=8675309):
    """ Select the kth cross-validation fold using stratified CV
    
    In partcular, this function uses `sklearn.model_selection.StratifiedKFold`
    to split the data. It then selects the training and testing splits
    from the k^th fold.

    N.B. If `y` is None, the simple `KFold` is used instead.
    
    Parameters
    ----------
    X, y: sklearn-formated data matrices
    
    fold: int
        The cv fold
        
    num_folds: int
        The total number of folds
        
    random_seed: int or random state
        The value used a the random seed for the k-fold split
    """

    check_range(fold, 0, num_folds, max_inclusive=False, variable_name='fold')

    if y is None:
        cv = sklearn.model_selection.KFold(
            n_splits=num_folds, random_state=random_seed
        )
    else:        
        cv = sklearn.model_selection.StratifiedKFold(
            n_splits=num_folds, random_state=random_seed
        ) 
    
    splits = cv.split(X, y)
    train, test = more_itertools.nth(splits, fold)

    X_train = X[train]
    X_test = X[test]

    if y is None:
        y_train = None
        y_test = None
    else:
        y_train = y[train]
        y_test = y[test]
        
    ret = fold_tuple(X_train, y_train, X_test, y_test)
    return ret
    
