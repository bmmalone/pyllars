from setuptools import find_packages, setup


other_console_scripts = [
    'call-sbatch=misc.call_sbatch:main',
    'scancel-range=misc.scancel_range:main',
    'test-gzip=misc.test_gzip:main',
    'call-program=misc.call_program:main'
]

install_requires = [
    'cython',
    'numpy',
    'scipy',
    'statsmodels',
    'matplotlib',
    'matplotlib_venn',
    'pandas',
    'sklearn',
    'fastparquet',
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

extras = {
    'test': tests_require,
}


# previously, there were other types of scripts
console_scripts = other_console_scripts

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='misc',
        version='0.2.8',
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
        test_suite='nose.collector',
        tests_require=tests_require,
        extras_require=extras,
        entry_points = {
            'console_scripts': console_scripts
        },
        zip_safe=False
        )
