Statistics utilities
*****************************

.. automodule:: pyllars.stats_utils
    :noindex:
    
.. currentmodule::  pyllars.stats_utils

Analytic KL-divergence calculations
--------------------------------------

.. autosummary::
    calculate_univariate_gaussian_kl
    calculate_symmetric_kl_divergence
    symmetric_entropy
    symmetric_gaussian_kl
    
Sufficient statistics and parameter estimation
-------------------------------------------------

.. autosummary::
    get_population_statistics
    get_categorical_mle_estimates
    fit_with_least_squares
    
Bayesian hypothesis testing
----------------------------

.. autosummary::
    bayesian_proportion_test
    bayesian_means_test

Distribution sampling
---------------------

.. autosummary::
    normal_inverse_chi_square
    sample_dirichlet_multinomial
    sample_beta_binomial
    sample_gamma_poisson
    
Definitions
-------------
.. automodule:: pyllars.stats_utils
    :members:
    :private-members:
    :exclude-members: polynomial_order
    
.. autoclass:: polynomial_order
    :members: