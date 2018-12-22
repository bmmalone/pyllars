""" Helpers for statistical calculations
"""
import logging
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

def get_population_statistics(
        subpopulation_sizes,
        subpopulation_means,
        subpopulation_variances):
    """ Calculate the population size, mean and variance based on subpopulation statistics
    
    This code is based on "Chap"'s answer here:
        https://stats.stackexchange.com/questions/30495
        
    This calculation seems to underestimate the variance relative
    to `np.var` on the entire dataset (determined by simulation). This may
    somehow relate to "biased" vs. "unbiased" variance estimates (basically,
    whether to subtract 1 from the population size). Still, naive
    approaches to correct for that do not produce variance estimates which
    exactly match those from `np.var`.
    
    Parameters
    ----------
    subpopulation_{sizes,means,variances} : np.arrays of numbers
        The subpopulation sizes, means, and variances, respectively. These should
        all be the same size.
        
    Returns
    -------
    population_{size,mean,variance,std} : floats
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