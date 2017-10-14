# pymisc-utils
This project contains many python3 utilities that I find useful.

**Installation**

This package is written in python3. `pip` can be used to install it:

```
git@github.com:bmmalone/pymisc-utils.git
cd pymisc-utils
pip3 install .
```

(The "period" at the end is required.)

If possible, I recommend installing inside a virtual environment. See 
[here](http://www.simononsoftware.com/virtualenv-tutorial-part-2/>), for example.

## Overview

This package contains helpers for a wide variety of python applications.

#### PyData stack utilities

* `automl_utils`. Utilities for AutoML. This mostly contains wrappers for
    working with [`auto-sklearn`](https://github.com/automl/auto-sklearn)
    and related projects, including the algorithm selection framework provided
    by the [`ASlib`](https://github.com/mlindauer/ASlibScenario) project.
    
    **N.B.** This module has been deprecated. Please use the new version in
    the [`automl-utils` repo](https://github.com/bmmalone/automl-utils).

* `dask_utils`. Utilities for working with [dask](https://dask.pydata.org/en/latest/).
    For example, this include functions for easily specifying connection, etc.,
    information for a dask cluster from the command line.
    
* `external_sparse_matrix_list`. A class which wraps a list of sparse matrices
    and seamlessly handles reading and writing them to- and from-disk in
    standard, compressed text formats, such as [Matrix Market](http://math.nist.gov/MatrixMarket/formats.html).

* `incremental_gaussian_estimator`. A class which calculates sample mean and
    variance from univariate online (that is, streaming) observations.

* `math_utils`. Utilities for working with matrices, as well as other
    math utilities. In particular, it includes implementations of a few
    sophisticated statistical algorithms, including:
    
    * Univariate Gaussian KL-divergence
    * A Bayesian test to determine if two proportions differ significantly
    * A Bayesian test to determine if the means of two populations differ
        significantly (that is, something like a Bayesian t-test)
        
* `missing_data_utils`. Utilities for removing data according to different
    missingness mechanisms, including missing at random (MAR), missing
    completely at random (MCAR), and not missing at random (NMAR).

* `mpl_utils`. Utilities for manipulating object-oriented matplotlib plots, that
    is, those which uses `Axes` objects.
    
* `nan_standard_scaler`. An `sklearn` transformer which scales features by the
    observed mean and standard deviation of the training data; in contrast to
    the normal `StandardScaler`, this class ignores `nan`s, `inf`s and other
    typically-problematic values.
    
* `nan_nearest_neighbors`. A simple k-NN algorithm which handles features with
    missing values represented as `np.nan`s.
    
* `pandas_utils`. Utilities for working with pandas data frames, such as
    seamless file-IO for a variety of formats like parquet, hdf5 and excel.
    
* `sparse_vector`. A class which wraps a `scipy.sparse_matrix` to reduce the
    notational burden for working with sparse vectors.
    
#### Domain-specific utilities

* `physionet_utils`. Utilities for working with the [MIMIC](https://mimic.physionet.org/)
    clinical care database as well as other datasets published by `physionet`,
    such as the [`Computing in Cardiology Challenge`](https://www.physionet.org/challenge/2012/).

#### General utilities
* `latex`. Utilities for programmatically creating latex documents.

* `logging_utils`. Utilities for easily controlling logging behavior from the
    command line.
    
* `mysql_db`. Utilities for interacting with MySQL databases.
    
* `parallel`. Utilities for parallel processing, especially focusing on data
    frames and iterators. These functions are largely wrappers around
    [`joblib`](https://pythonhosted.org/joblib/); they tend to be more
    light-weight than dask-based solutions, but they are generally only useful
    for embarrassingly parallel loops.
    
* `shell_utils`. Utilities for interacting with a shell. Many of these functions
    wrap the `os` and/or `subprocess` modules with things like checks that
    required input and output files exist.
    
* `ssh_utils`. Utilities for distributing jobs across a cluster, etc., using
    password-less SSH.

* `utils`. Utilities for working with built-in python types, simple file system
    operations, as well as a variety of other utilities. Many of the functions
    in other modules began here and were migrated after a sufficient number of
    related functions had been added. Thus, many functions from this module
    issue deprecation warnings when called.
    
#### Slurm command line utilities

* `call-program`. A wrapper which calls whatever program it is given. This is
    useful for calling binary programs with `call-sbatch`.

* `call-sbatch`. A wrapper around Slurm's [`sbatch`](https://slurm.schedmd.com/sbatch.html)
    command with reasonable defaults.
    
* `scancel-range`. A script which calls [`scancel`](https://slurm.schedmd.com/scancel.html)
    on a range of job id's.
    
* `slurm`. Utilities for adding slurm options to any script.


#### Other small programs

* `pickle-stan`. A simple script to compile and pickle a
    [Stan](http://mc-stan.org/) model.
    
* `test-gzip`. A simple script which reads and writes a gzipped file to and from
    disk many times. It verifies the integrity of the file after each iteration,
    so it can be useful to diagnose problems with incomplete files being
    written to disk (looking at you, [BeeGFS](https://www.beegfs.io/content/)).