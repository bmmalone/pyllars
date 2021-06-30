# pyllars

This project contains supporting utilities for Python 3.

[![Build Status](https://travis-ci.org/bmmalone/pyllars.svg)](https://travis-ci.org/bmmalone/pyllars)
[![Coverage Status](https://coveralls.io/repos/github/bmmalone/pyllars/badge.svg?branch=dev)](https://coveralls.io/github/bmmalone/pyllars?branch=dev)
[![Documentation Status](https://readthedocs.org/projects/pyllars/badge/?version=latest)](https://pyllars.readthedocs.io/en/latest/?badge=latest)

**Installation**

This package is available on PyPI.

```
pip3 install pyllars
```

Alternatively, the package can be installed from source.

```
git clone https://github.com/bmmalone/pyllars
cd pyllars
pip3 install .
```

(The "period" at the end is required.)

If possible, I recommend installing inside a virtual environment or with conda. See 
[here](http://www.simononsoftware.com/virtualenv-tutorial-part-2/>), for example.

Please see [the documentation](https://pyllars.readthedocs.io/en/latest/index.html)
for more details.

**pytorch and ray installation**

The `pytorch.torch` submodule requires `pytorch` and `ray-tune`. Due to the
various configuration options (CPU vs. GPU, etc.), the `pyllars` installation
does not attempt to install these dependencies; they need to be installed
manually, though this can be done after installing `pyllars` with no problems.
I suggest using these within an anaconda or similar environment. Please see the
official documentation for installing [`pytorch`](https://pytorch.org/) and
[`ray-tune`](https://anaconda.org/conda-forge/ray-tune) for details.