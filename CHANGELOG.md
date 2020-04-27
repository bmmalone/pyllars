# Change Log
All notable changes to this repo will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/), 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [1.0.4] - In progress
### Updated
- xgb_utils to be compatible with `xgboost > 1.0`.

## [1.0.3] - 2020-03-27
### Added
- Helpers for controlling logging options when a command line is not
    available, such as in notebooks or from test cases
- Helpers for pytorch, including early stopping based on validation set
    performance

### Updated
- Binary ML metrics to include precision-recall curves
- Optional progress bar when collecting dask results

### Fixed
- Safely load YAML config files

## [1.0.2] - 2019-06-25
### Added
- Additional documentation
- Helper for plotting binary prediction scores
- precision@k metric for ML ranking tasks
- Helper to extract dask command line options as an array. Presumably
    these would be used in `subprocess.call` or for similar uses.
- Helper to plot mean ROC curves, including shading for the standard
    deviation

### Updated
- Helper for plotting ROC curves to handle arrays of colors
- Binary classification metrics helper to include points needed for
    plotting ROC curves (`sklearn.metrics.roc_curve`)

### Fixed
- `logging_utils` to properly quote extracted command line options.
- Missing import in `validation_utils`

## [1.0.1] - 2019-04-08
### Added
- Helper for `xgboost`
- Additional documentation

## [1.0.0] - 2019-02-11
This version change significantly rearranges the package structure. In particular,
the functionality of the large `utils` and `math_utils` modules has been split
into more specific modules (some of which already existed). Deprecated functionality
has been removed, and sphinx documentation has been added.

Additionally, the package was renamed from `pymisc-utils` to `pyllars` to help
avoid confusion between the older API and the updated one.

This update also adds Travis CI, Coveralls, Read the Docs, and PyPI support for
the project.

The tests are very minimal, and it is expected that some import errors have
not yet been resolved.

## [0.99.xyz]
These versions are technical changes between the 0.2.xx and 1.xx code base.
Please ignore these.

## [0.2.11] - 2018-12-22
### Added
- Helpers for checking and collecting dask futures
- Validation helpers for non-pydata types
- Simple BoW and numeric feature handler
- Helpers for standard plots
- Helpers for MyGene.py
- Helpers for working with the Gene Ontology
- Helper to create scaler from means and standard deviations

### Updated
- k-fold splitter to include validation set
- Transparent file opening for compressed files
- Split out machine learning helpers into a separate module
- Split out statistical helpers into a separate module

### Fixed
- Missing data one hot encoder to handle sparse inputs

## [0.2.10] - 2018-06-06
### Added
- Followup table construction for MIMIC
- Helpers for MIMIC waveform database
- Additional validation helpers

### Updated
- Dataset manager to optionally encode the target variable

### Fixed
- SCIP output parsing for instances where SCIP crashed

## [0.2.9] - 2018-04-18
### Added
- Helper to estimate categorical variable MLEs
- Utilities for working with the SCIP solver
- Helper to check if class attributes have been initialized similar to
    `check_is_fitted` in sklearn
- Standard validation helpers (`validation_utils`)
- Helper to calculate many regression, multi-class classification metrics

## [0.2.8] - 2018-03-12
### Updated
- Cross-validation helper to work with unsupervised learning
- Documentation of all modules to use docstrings

### Removed
- mysql helpers
- automl helpers. These are now available in the [automl-utils](https://github.com/bmmalone/automl-utils)
    package.

## [0.2.7] - 2018-03-08
### Added
- Helper for creating chunks of groups from a data frame. This utility can
    make submitting jobs to dask and other parallel schedulers more efficient.

### Removed
- Dependency on pystan and the `pickle-stan` script. There were no other uses
    of stan within this package.


## [0.2.6] - 2018-03-02
### Added
- Utility for supressing pystan (or other compiled function) output. This
    addition is motivated by [an rpbp issue](https://github.com/dieterich-lab/rp-bp/issues/10),
    and the solution is basically [copied from facebook's prophet](https://github.com/facebook/prophet/issues/223#issuecomment-326455744).
- DatasetManager class to ease reading data and preparing it for sklearn
- Several changes to the ML helper transformers to make working with mixed
    data sets (that is, those with categorical and numerical features) easier
- Several sklearn transformers which are robust to missing data and preserve
    the missing data (i.e., `np.nan`s) so downstream processing can account for
    the missing values appropriately.
- Utilities for working with text (`misc.nlp_utils`, 
    `misc.incremental_count_vectorizer`)

## [0.2.5] - 2017-10-26
### Added
- Utilities for working with the [Computing in Cardiology Challenge 2012
    dataset](https://physionet.org/challenge/2012/)
- Minor extensions and fixes for `AutoSklearnWrapper`
- Helper for drawing rectangles in matplotlib
- Multiclass AUC calculation from [Hand and Till, 2001], [Provost and Domingos, 2000]
- Multicolumn label encoder helper
- Utility to add MCAR, MAR and NMAR missing values to a data matrix
- Simple kNN implementation which is robust against missing values

### Updated
- Confusion matrix plotting utility to work on axes objects and to be more
    configurable

- `fastparquet` imports are now immediately before their use. This is related
  to [Issue #4](https://github.com/bmmalone/pymisc-utils/issues/4)

### Deprecated
- `automl_utils`. The functionality is now available in the 
    [`automl-utils` package](https://github.com/bmmalone/automl-utils).

## [0.2.4] - 2017-08-31
### Added
- Helper to collect sklearn classification metrics
- Classification helpers to `automl_utils`
- Utilities for working with the mimic dataset
- Helpers for working with [`ASlibScenario`s](https://github.com/mlindauer/ASlibScenario)
- Helper for listing subdirectories in `utils`
- Brief description of all modules to the readme

### Updated
- `AutoSklearnWrapper` to clearly used autosklearn only during training;
    otherwise, it behaves as a normal ensemble.

### Removed
- `external_sparse_pickle_list`
- `visualize-roc`

### Fixed
- Parquet reader to handle (ignore) multi-indexes

### Deprecated
- All classes in `column_selector`

## [0.2.3] - 2017-07-27
### Added
- Helper to remove negative values from an np.array

### Updated
- Support for parquet files in `pandas_utils`

## [0.2.2] - 2017-07-24
### Added
- A helper for controlling dask clusters

## [0.2.1] - 2017-07-01
### Added
- A helper `shell_utils.download_file` to download external files and save them
  to a local location.
  
### Updated
- Various pandas utilities to "deprecated" status
  
### Removed
- Deprecated shell interaction utilities from `misc.utils`
- Unused pandas utilities

### Fixed
- Missing logger for `pandas_utils`. See 
  [Issue #1](https://github.com/bmmalone/pymisc-utils/issues/1) for more
  details.

## [0.2.0] - 2017-05-31
This is a new version which moves the project from Bitbucket to GitHub.
Additionally, the bioinformatics utilities (`bio_utils`) have been completely
removed. They will be added to a new
[`pybio-utils`](https://github.com/bmmalone/pybio-utils) repository.

## [0.1.6] - 2017-05-10
### Fixed
- Missing import in counting alignments for bam files

## [0.1.5] - 2017-05-09
### Updated
- `get-read-length-distribution` script to handle bam, fasta and fastq files.
  See [the bio docs](docs/bio.md#get-read-length-distributions) for more
  details.

## [0.1.4] - 2017-05-09
### Removed
- bed, bam and gtf helpers from `bio.py`. These had already been deprecated for
  quite some time.

## [0.1.3] - 2017-03-30
### Added
- Script to remove duplicate entries from bed files. See
  [the bio docs](docs/bio.md#merge-bed12-files-and-remove-duplicate-entries)
  for more details.

## [0.1.2] and previous versions

The initial versions have not been documented in the change log.


