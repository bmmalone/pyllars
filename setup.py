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
    'dask[complete]',
    'docopt',
    'graphviz',
    'joblib',
    'matplotlib',
    'matplotlib_venn',
    'more_itertools',
    'networkx>=2.0',
    'nltk',
    'numpy',
    'openpyxl',
    'pandas',
    'paramiko',
    'pydot',
    'scipy',
    'six',
    'sklearn',
    'sklearn_pandas',
    'spur',
    'statsmodels',
    'tables',
    'tqdm',
    'xlrd',
]

setup_requires = [
    'pytest-runner'
]

tests_require = [
    'pytest',
    'coverage',
    'pytest-cov',
    'coveralls',
]

parquet_requires = [
    'fastparquet'
]

bio_requires = [
    'goatools',
    'mygene',
    'biopython',
]

docs_require = [
    'sphinx',
    'sphinx_rtd_theme',
]

all_requires = (
    tests_require + 
    parquet_requires + 
    bio_requires + 
    setup_requires + 
    docs_require
)

extras = {
    'test': tests_require,
    'parquest': parquet_requires,
    'bio': bio_requires,
    'all': all_requires,
    'setup': setup_requires,
    'docs': docs_require
}



def _post_install(self):
    import site
    importlib.reload(site)

    # already download the nltk resources used in nlp_utils
    
    nltk_spec = importlib.util.find_spec("nltk")
    nltk_found = nltk_spec is not None
    
    if nltk_found:
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
        setup_requires=setup_requires,
        install_requires=install_requires,
        include_package_data=True,
        cmdclass={'install': my_install,  # override install
                  'develop': my_develop   # develop is used for pip install -e .
        },
        tests_require=tests_require,
        extras_require=extras,
        entry_points = {
            'console_scripts': console_scripts
        },
        zip_safe=False
        )
