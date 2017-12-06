# Change Log
All notable changes to this repo will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/), 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.2.6] - In progress
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


