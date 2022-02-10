Introduction to pyllars
***********************

This package contains many supporting utilities I find useful for Python 3.

Installation
------------

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

**pytorch and ray installation**

The `pytorch.torch` submodule requires ``pytorch`` and ``ray-tune``. Due to the
various configuration options (CPU vs. GPU, etc.), the ``pyllars`` installation
does not attempt to install these dependencies; they need to be installed
manually, though this can be done after installing `pyllars` with no problems.
I suggest using these within an anaconda or similar environment. Please see the
official documentation for installing [`pytorch`](https://pytorch.org/) and
[`ray-tune`](https://anaconda.org/conda-forge/ray-tune) for details.

History
-------

This project was called ``pymisc-utils``. Due to significant changes in the API
when moving from version ``0.2.11`` to version ``1.0.0``, the name was also
changed to avoid confusion.

The new name is also more fun... "pyllars",
"**supporting** utilities"... get it? I'm here all week, folks. Try the veal.