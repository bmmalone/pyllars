"""
This module contains helpers for various statistical calculations.
"""
import logging
logger = logging.getLogger(__name__)

from enum import Enum
import numpy as np
import pandas as pd
import scipy.stats
import typing

from typing import Any, Callable, Iterable, Optional, Tuple

###
# KL-divergence helpers
###
def calculate_symmetric_kl_divergence(
        p:Any,
        q:Any,
        calculate_kl_divergence:Callable) -> float:
    """ Calculates the symmetric KL-divergence between distributions p and q
    
    In particular, this function defines the symmetric KL-divergence to be\:
    
    .. math:: D_{sym}(p||q) \:= \\frac{D(p||q) + D(q||p)}{2}
        
    Parameters
    ----------
    {p,q} : typing.Any
        A representation of a distribution that can be used by the 
        function `calculate_kl_divergence`
            
    calculate_kl_divergence : typing.Callable
        A function the calculates the KL-divergence between :math:`p`
        and :math:`q`

    Returns
    -------
    symmetric_kl : float
        The symmetric KL-divergence between :math:`p` and :math:`q`
    """
    kl_1 = calculate_kl_divergence(p, q)
    kl_2 = calculate_kl_divergence(q, p)
    symmetric_kl = (kl_1 + kl_2) / 2
    return symmetric_kl

def calculate_univariate_gaussian_kl(
        mean_p_var_p:Tuple[float,float],
        mean_q_var_q:Tuple[float,float]) -> float:
    """ Calculate the (asymmetric) KL-divergence between the univariate
    Gaussian distributions :math:`p` and :math:`q`
    
    That is, this calculates KL(p||q).

    **N.B.** This function uses the variance!

    **N.B.** The parameters for each distribution are passed as pairs 
    for easy use with `calculate_symmetric_kl_divergence`.

    See, for example, [1]_ for the formula.

    Parameters
    ----------
    {mean_p_var_p,mean_q_var_q} : Typing.Tuple[float,float]
        The parameters of the distributions.

    Returns
    -------
    kl_divergence : float
        The KL divergence between the two distributions.
            
    References
    ----------
    .. [1] Penny, W. "KL-Divergences of Normal, Gamma, Dirichlet and Wishart densities." Wellcome Department of Cognitive Neurology, University College London, 2001.
    """

    (mean_p, var_p) = mean_p_var_p
    (mean_q, var_q) = mean_q_var_q

    t_1 = 0.5 * (np.log(var_q) - np.log(var_p))
    t_2 = np.log(mean_q*mean_q + mean_p*mean_p + var_p - 2*mean_q*mean_p)
    t_3 = np.log(2*var_q)
    kl = t_1 - 0.5 + np.exp(t_2 - t_3)
    return kl

def symmetric_entropy(p, q) -> float:
    """ Calculate the symmetric :func:`scipy.stats.entropy`. """
    return calculate_symmetric_kl_divergence(p, q, scipy.stats.entropy)

def symmetric_gaussian_kl(p, q) -> float:
    """ Calculate the symmetric :func:`pyllars.stats_utils.calculate_univariate_gaussian_kl`. """
    return calculate_symmetric_kl_divergence(
        p, q, 
        calculate_univariate_gaussian_kl
    )

###
# Frequentist statistics
###
def get_population_statistics(
        subpopulation_sizes:np.ndarray,
        subpopulation_means:np.ndarray,
        subpopulation_variances:np.ndarray) -> Tuple[float,float,float,float]:
    """ Calculate the population size, mean and variance based on subpopulation statistics
    
    This code is based on "Chap"'s answer here: https://stats.stackexchange.com/questions/30495
        
    This calculation seems to underestimate the variance relative
    to :func:`numpy.var` on the entire dataset (determined by simulation). This may
    somehow relate to "biased" vs. "unbiased" variance estimates (basically,
    whether to subtract 1 from the population size). Still, naive
    approaches to correct for that do not produce variance estimates which
    exactly match those from :func:`numpy.var`.
    
    Parameters
    ----------
    subpopulation_{sizes,means,variances} : numpy.ndarray
        The subpopulation sizes, means, and variances, respectively. These should
        all be the same size.
        
    Returns
    -------
    population_{size,mean,variance,std} : float
        The respective statistics about the entire population
    """
    
    n = np.sum(subpopulation_sizes)
    _subpopulation_total_sum = np.multiply(subpopulation_sizes, subpopulation_means)
    _population_total_sum = np.sum(_subpopulation_total_sum)

    population_mean = 1 / n * _population_total_sum
    
    _subpopulation_variance = np.multiply(subpopulation_sizes - 1, subpopulation_variances)
    _subpopulation_variance_sum = np.sum(_subpopulation_variance)
    
    _subpopulation_mean_difference = (subpopulation_means - population_mean)**2
    _subpopulation_means_product = np.multiply(subpopulation_sizes, _subpopulation_mean_difference)
    _subpopulation_means_sum = np.sum(_subpopulation_means_product)
    
    population_variance = (_subpopulation_means_sum + _subpopulation_variance_sum) / (n-1)
    
    population_std = np.sqrt(population_variance)
    
    return n, population_mean, population_variance, population_std

###
# Bayesian hypothesis testing
###

def bayesian_proportion_test(
        x:Tuple[int,int],
        n:Tuple[int,int],
        prior:Tuple[float,float]=(0.5,0.5),
        prior2:Optional[Tuple[float,float]]=None, 
        num_samples:int=1000,
        seed:int=8675309) -> Tuple[float,float,float]:
    """ Perform a Bayesian test to identify significantly different proportions.
    
    This test is based on a beta-binomial conjugate model. It uses Monte Carlo
    simulations to estimate the posterior of the difference between the
    proportions, as well as the likelihood that :math:`\pi_1 > \pi_2` (where
    :math:`\pi_i` is the likelihood of success in sample :math:`i`).
    
    Parameters
    ----------
    x : typing.Tuple[int,int]
        The number of successes in each sample
        
    n : typing.Tuple[int,int]
        The number of trials in each sample
        
    prior : typing.Tuple[float,float]
        The parameters of the beta distribution used as the prior in the conjugate
        model for the first sample.
        
    prior2 : typing.Optional[typing.Tuple[float,float]]
        The parameters of the beta distribution used as the prior in the conjugate
        model for the second sample. If this is not specified, then `prior` is used.
        
    num_samples : int
        The number of simulations
        
    seed : int
        The seed for the random number generator
        
    Returns
    -------
    difference_{mean,var} : float
        The posterior mean and variance of the difference in the likelihood of success 
        in the two samples. A negative mean indicates that the likelihood in sample 2 
        is higher.
        
    p_pi_1_greater : float
        The probability that :math:`\pi_1 > \pi_2`
    """
    
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
    
    return diff_posterior_mean, diff_posterior_var, p_pi_1_greater

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

def bayesian_means_test(
        x1:np.ndarray,
        x2:np.ndarray,
        use_jeffreys_prior:bool=True,
        prior1:Optional[Tuple[float,float,float,float]]=None,
        prior2:Optional[Tuple[float,float,float,float]]=None,
        num_samples:int=1000,
        seed:int=8675309) -> Tuple[float,float,float]:
    """ Perform a Bayesian test to identify significantly different means.
    
    The test is based on a Gaussian conjugate model. (The normal-inverse-chi-square
    distribution is the prior.) It uses Monte Carlo simulation to estimate the
    posterior of the difference between the means of the populations, under the
    (probably dubious) assumption that the observations are Gaussian distributed.
    It also estimates the likelihood that :math:`\mu_1 > \mu_2`, where :math`\mu_i`
    is the mean of each sample.
    
    Parameters
    ----------
    x{1,2} : numpy.ndarray
        The observations of each sample
        
    use_jeffreys_prior : bool
        Whether to use the Jeffreys prior. For more details, see:
        
            Murphy, K. Conjugate Bayesian analysis of the Gaussian distribution. 
            Technical report, 2007.
            
        Briefly, the Jeffreys prior is: :math:`(\\text{sample mean}, n, n-1,
        \\text{sample variance})`, according to a
        :func:`pyllars.stats_utils.normal_inverse_chi_square` distribution.
            
    prior{1,2} : typing.Optional[typing.Tuple[float,float,float,float]]
        If the Jeffreys prior is not used, then these parameters are used as the
        priors for the normal-inverse-chi-square. If only prior1 is given, then
        those values are also used for prior2, where prior_i is taken as the prior
        for x_i.
        
    num_samples : int
        The number of simulations
        
    seed : int
        The seed for the random number generator
        
    Returns
    -------
    difference_{mean,var} : float
        The posterior mean and variance of the difference in the mean of the two
        samples. A negative difference_mean indicates that the mean of x2 is 
        higher.
        
    p_m1_greater : float
        The probability that :math:`\mu_1 > \mu_2`

    """
    
    if use_jeffreys_prior:
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

###
# Model parameter estimation
###

def get_categorical_mle_estimates(
        observations:Iterable[int],
        cardinality:Optional[int]=None,
        use_laplacian_smoothing:bool=False,
        base_1:bool=False) -> np.ndarray:
    """ Calculate the MLE estimates for the categorical observations
    
    Parameters
    ----------
    observations : typing.Iterable[int]
        The observed values. These are taken to already be "label encoded",
        so they should be integers in [0,cardinality).
        
    cardinality : typing.Optional[int]
        The cardinality of the categorical variable. If `None`, then this
        is taken as the number of unique values in `observations`.
        
    use_laplacian_smoothing : bool
        Whether to use Laplacian ("add one") smoothing for the estimates.
        This can also be interpreted as a symmetric Dirichlet prior with 
        a concentration parameter of 1.

    base_1 : bool
        Whether the observations are base 1. If so, then the range is taken
        as [1, cardinality].
        
    Returns
    -------
    mle_estimates : numpy.ndarray
        The estimates. The size of the array is `cardinality`.
    """
    if base_1:
        observations = np.array(observations) - 1

    indices, counts = np.unique(observations, return_counts=True)
    
    if cardinality is None:
        cardinality = np.max(indices) + 1
        
    mle_estimates = np.zeros(shape=cardinality)
    
    for i,c in zip(indices, counts):
        mle_estimates[i] += c
        
    if use_laplacian_smoothing:
        mle_estimates += 1
        
    mle_estimates = mle_estimates / np.sum(mle_estimates)
    return mle_estimates


class polynomial_order(Enum):
    linear = 1
    quadratic = 2

def fit_with_least_squares(
        x:np.ndarray,
        y:np.ndarray,
        w:Optional[np.ndarray]=None,
        order=polynomial_order.linear) -> Tuple[float,float,float,float]:
    """ Fit a polynomial relationship between `x` and `y`.
    
    Optionally, the values can be weighted.

    Parameters
    ----------
    {x,y} : numpy.ndarray
        The input arrays

    w : numpy.ndarray
        A weighting of each of the (x,y) pairs for regression

    order : polynomial_order
        The order of the fit

    Returns
    -------
    {intercept,slope,power} : float
        The coefficients of the fit. power is 0 if the order is linear.

    r_sqr : float
        The coefficient of determination
    """

    # Calculate trendline
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