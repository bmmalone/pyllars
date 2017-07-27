# Change Log
All notable changes to this repo will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/), 
and this project adheres to [Semantic Versioning](http://semver.org/).

## [0.2.3] - In progress
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


