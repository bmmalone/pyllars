from setuptools import find_packages, setup
from setuptools.command.install import install as _install
from setuptools.command.develop import develop as _develop

import importlib

###
# Console scripts
###
slurm_console_scripts = [
    'call-sbatch=misc.slurm_utils.call_sbatch:main',
    'scancel-range=misc.slurm_utils.scancel_range:main',
    'call-program=misc.slurm_utils.call_program:main'
]

console_scripts = slurm_console_scripts

###
# Dependencies
###
install_requires = [
    'cython',
    'numpy',
    'scipy',
    'statsmodels',
    'matplotlib',
    'matplotlib_venn',
    'pandas',
    'sklearn',
    'sklearn_pandas',
    'more_itertools',
    'networkx>=2.0',
    'docopt',
    'tqdm',
    'joblib',
    'xlrd',
    'openpyxl',
    'graphviz',
    'pydot',
    'tables',
    'paramiko',
    'spur',
    'six',
    'nltk',
    'dask[complete]',
]

tests_require = [
    'nose',
]

parquet_require = [
    'fastparquet'
]

all_require = tests_require + parquet_require

extras = {
    'test': tests_require,
    'parquest': parquet_require,
    'all': all_require
}



def _post_install(self):
    import site
    importlib.reload(site)

    # already download the nltk resources used in nlp_utils
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

class my_install(_install):
    def run(self):
        _install.run(self)
        _post_install(self)

class my_develop(_develop):  
    def run(self):
        _develop.run(self)
        _post_install(self)


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='misc',
        version='0.3.0',
        description="This package contains python3 utilities I find useful.",
        long_description=readme(),
        keywords="utilities",
        url="https://github.com/bmmalone/pymisc-utils",
        author="Brandon Malone",
        author_email="bmmalone@gmail.com",
        license='MIT',
        packages=find_packages(),
        install_requires=install_requires,
        include_package_data=True,
        cmdclass={'install': my_install,  # override install
                  'develop': my_develop   # develop is used for pip install -e .
        },
        test_suite='nose.collector',
        tests_require=tests_require,
        extras_require=extras,
        entry_points = {
            'console_scripts': console_scripts
        },
        zip_safe=False
        )
