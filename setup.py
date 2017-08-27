from setuptools import find_packages, setup

stan_console_scripts = [
    'pickle-stan=misc.pickle_stan:main [stan]'
]

other_console_scripts = [
    'visualize-roc=misc.visualize_roc:main',
    'call-sbatch=misc.call_sbatch:main',
    'scancel-range=misc.scancel_range:main',
    'test-gzip=misc.test_gzip:main',
    'call-program=misc.call_program:main'
]

console_scripts = stan_console_scripts + other_console_scripts

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='misc',
        version='0.2.3',
        description="This package contains python3 utilities I find useful.",
        long_description=readme(),
        keywords="utilities",
        url="",
        author="Brandon Malone",
        author_email="bmmalone@gmail.com",
        license='MIT',
        packages=find_packages(),
        install_requires = [
            'cython',
            'numpy',
            'scipy',
            'statsmodels',
            'matplotlib',
            'pandas',
            'fastparquet',
            'docopt',
            'tqdm',
            'joblib',
            'xlrd',
            'openpyxl',
            'graphviz'
        ],
        extras_require = {
            'hdf5': ['tables'],
            'ssh': ['paramiko', 'spur'],
            'stan': ['pystan'],
        },
        include_package_data=True,
        test_suite='nose.collector',
        tests_require=['nose'],
        entry_points = {
            'console_scripts': console_scripts
        },
        zip_safe=False
        )
