"""
This module contains helpers for using dask: https://dask.pydata.org/en/latest/
"""

import logging
logger = logging.getLogger(__name__)

import collections
import tqdm

def connect(args):
    """ Connect to the dask cluster specifed by the arguments in args

    Specifically, this function uses args.cluster_location to determine whether
    to start a dask.distributed.LocalCluster (in case args.cluster_location is
    "LOCAL") or to (attempt to) connect to an existing cluster (any other
    value).

    If a local cluster is started, it will use a number of worker processes
    equal to args.num_cpus. Each process will use args.num_threads_per_cpu
    threads. The scheduler for the local cluster will listen to a random port.

    Parameters
    ----------
    args: argparse.Namespace
        A namespace containing

    Returns
    -------
    client: dask.distributed.Client
        The client for the dask connection

    cluster: dask.distributed.LocalCluster or None
        If a local cluster is started, the reference to the local cluster
        object is returned. Otherwise, None is returned.
    """

    from dask.distributed import Client as DaskClient
    from dask.distributed import LocalCluster as DaskCluster

    client = None
    cluster = None

    if args.cluster_location == "LOCAL":
        
        msg = "[dask_utils]: starting local dask cluster"
        logger.info(msg)

        cluster = DaskCluster(
            n_workers=args.num_cpus,
            processes=True,
            threads_per_worker=args.num_threads_per_cpu
        )

        client = DaskClient(cluster)

    else:
        msg = "[dask_utils]: attempting to connect to dask cluster: {}"
        msg = msg.format(args.cluster_location)
        logger.info(msg)

        client = DaskClient(address=args.cluster_location)


        if args.client_restart:
            msg = "[dask_utils]: restarting client"
            logger.info(msg)
            client.restart()

    return client, cluster
        

def add_dask_options(parser, num_cpus=1, num_threads_per_cpu=1,
        cluster_location="LOCAL"):

    """ Add options for connecting to and/or controlling a local dask cluster

    Parameters
    ----------
    parser: argparse.ArgumentParser
        The parser to which the options will be added

    other parameters: defaults to use for the options

    Returns
    -------
    None, but a "dask cluster options" group is added to the parser
    """
    dask_options = parser.add_argument_group("dask cluster options")

    dask_options.add_argument('--cluster-location', help="The address for the "
        "cluster scheduler. This should either be \"LOCAL\" or the address "
        "and port of the scheduler. If \"LOCAL\" is given, then a "
        "dask.distributed.LocalCluster will be started.", 
        default=cluster_location)

    dask_options.add_argument('--num-cpus', help="The number of CPUs to use "
        "for a local cluster", type=int, default=num_cpus)

    dask_options.add_argument('--num-threads-per-cpu', help="The number of "
        "threads to allocate for each core. So the total number of threads "
        "for a local cluster will be (args.num_cpus * "
        "args.num_threads_per_cpu).", type=int, default=num_threads_per_cpu)

    dask_options.add_argument('--client-restart', help="If this flag is "
        "given, then the \"restart\" function will be called on the client "
        "after establishing the connection to the cluster",
        action='store_true')

def add_dask_values_to_args(args, num_cpus=1, num_threads_per_cpu=1,
        cluster_location="LOCAL", client_restart=False):

    """ Add the options for a dask cluster to the given argparse namespace

    This function is mostly intended as a helper for use in ipython notebooks.
    """
    args.num_cpus = num_cpus
    args.num_threads_per_cpu = num_threads_per_cpu
    args.cluster_location = cluster_location
    args.client_restart = client_restart

def get_joblib_parallel_backend_context_manager(args):
    """ Get the appropriate context manager for joblib
    """

    from joblib import parallel_backend
    import distributed.joblib

    # first, check if we asked to restart the client
    if args.client_restart:
        # connect handles the call to restart
        client, cluster = connect(args)

    if args.cluster_location == "LOCAL":
        backend_args = ['multiprocessing']
        backend_kwargs = {}
    else:
        backend_args = ['dask.distributed']
        backend_kwargs = {'scheduler_host': args.cluster_location}

    backend = parallel_backend(*backend_args, **backend_kwargs)
    return backend

###
#   Helpers to submit arbitrary jobs to a dask cluster
###

def apply_iter(it, client, func, *args, return_futures=False,
        progress_bar=True, **kwargs):
    """ Call `func` on each item in `it`.

    Additionally, `args` and `kwargs` are passed to the function call.

    Parameters
    ----------
    it: an iterable
        The sequence of inputs for `func`

    client: distributed.client.Client
        A dask client

    func: function pointer
        The function to apply to each row in `data_frame`

    args, kwargs
        The other arguments to pass to `func`

    return_futures: bool
        Whether to wait for the results (`False`, the default) or return a
        list of dask futures (when `True`). If a list of futures is returned,
        the `result` method should be called on each of them at some point
        before attempting to use the results.

    progress_bar: bool
        Whether to show a progress bar when waiting for results. The parameter
        is only relevant when `return_futures` is `False`.

    Returns
    -------
    results: list
        Either the result of each function call or a future which will give
        the result, depending on the value of `return_futures`
    """
    msg = ("[dask_utils.parallel] submitting jobs to cluster")
    logger.debug(msg)

    if progress_bar:
        it = tqdm.tqdm(it)


    ret_list = [
        client.submit(func, *(i, *args), **kwargs) for i in it
    ]

    if return_futures:
        return ret_list

    msg = ("[dask_utils.parallel] collecting results from cluster")
    logger.debug(msg)
    
    # add a progress bar if we asked for one
    if progress_bar:
        ret_list = tqdm.tqdm(ret_list)

    ret_list = [r.result() for r in ret_list]
    return ret_list


def apply_df(data_frame, client, func, *args, return_futures=False,
        progress_bar=True, **kwargs):
    """ Call `func` on each row in `data_frame`.

    Additionally, `args` and `kwargs` are passed to the function call.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        A data frame

    client: distributed.client.Client
        A dask client

    func: function pointer
        The function to apply to each row in `data_frame`

    args, kwargs
        The other arguments to pass to `func`

    return_futures: bool
        Whether to wait for the results (`False`, the default) or return a
        list of dask futures (when `True`). If a list of futures is returned,
        the `result` method should be called on each of them at some point
        before attempting to use the results.

    progress_bar: bool
        Whether to show a progress bar when waiting for results. The parameter
        is only relevant when `return_futures` is `False`.

    Returns
    -------
    results: list
        Either the result of each function call or a future which will give
        the result, depending on the value of `return_futures`
    """

    if len(data_frame) == 0:
        return []

    it = data_frame.iterrows()
    if progress_bar:
        it = tqdm.tqdm(it, total=len(data_frame))

    ret_list = [
        client.submit(func, *(row[1], *args), **kwargs) 
            for row in it
    ]

    if return_futures:
        return ret_list

    # add a progress bar if we asked for one
    if progress_bar:
        ret_list = tqdm.tqdm(ret_list, total=len(data_frame))

    ret_list = [r.result() for r in ret_list]
    return ret_list


def apply_groups(groups, client, func, *args, return_futures=False,
        progress_bar=True, **kwargs):
    """ Call `func` on each group in `groups`.

    Additionally, `args` and `kwargs` are passed to the function call.

    Parameters
    ----------
    groups: pandas.DataFrameGroupBy
        The result of a call to `groupby`on a data frame

    client: distributed.client.Client
        A dask client

    func: function pointer
        The function to apply to each row in `data_frame`

    args, kwargs
        The other arguments to pass to `func`

    return_futures: bool
        Whether to wait for the results (`False`, the default) or return a
        list of dask futures (when `True`). If a list of futures is returned,
        the `result` method should be called on each of them at some point
        before attempting to use the results.

    progress_bar: bool
        Whether to show a progress bar when waiting for results. The parameter
        is only relevant when `return_futures` is `False`.

    Returns
    -------
    results: list
        Either the result of each function call or a future which will give
        the result, depending on the value of `return_futures`
    """

    if len(groups) == 0:
        return []

    it = groups
    if progress_bar:
        it = tqdm.tqdm(it)

    ret_list = [
        client.submit(func, *(group, *args), **kwargs) 
            for name, group in it
    ]

    if return_futures:
        return ret_list

    # add a progress bar if we asked for one
    if progress_bar:
        ret_list = tqdm.tqdm(ret_list)

    ret_list = [r.result() for r in ret_list]
    return ret_list
    
def check_status(f_list):
    """ Collect the status counts of a list of futures
    
    This is primarily intended to check the status of jobs submitted with the
    various `apply` functions when `return_futures` is `True`.
    
    Parameters
    ----------
    f_list: list of dask futures (distributed.client.Future)
    
    Returns
    -------
    status_counter: collections.Counter
        The number of futures with each status
    """
    counter = collections.Counter([f.status for f in f_list])
    return counter
   
def collect_results(f_list, finished_only=True):
    """ Collect the results from a list of futures
    
    By default, only results from finished tasks will be collected. Thus, the
    function is (more or less) non-blocking.
    
    Parameters
    ----------
    f_list: list of dask futures (distributed.client.Future)

    finished_only: bool
        Whether to collect only results for jobs whose status is 'finished'
    
    Returns
    -------
    results: list
        The results for each (finished, if specified) task
    """

    if finished_only:
        ret = [f.result() for f in f_list if f.status == 'finished']
    else:
        ret = [f.result() for f in f_list]

    return ret

def cancel_all(f_list, pending_only=True):
    """ Cancel all (pending) tasks in the list

    By default, only pending tasks are cancelled.

    Parameters
    ----------
    f_list : list of dask futures (distributed.client.Future)

    pending_only : bool
        Whether to cancel only tasks whose status is 'pending'

    Returns
    -------
    None, but the specified tasks are cancelled
    """
    if pending_only:
        for f in f_list:
            if f.status == 'pending':
                f.cancel()
    else:
        for f in f_list:
            f.cancel()

###
#   A simple wrapper to submit an sklearn pipeline to a dask cluster for fitting
###

class dask_pipeline:
    """ This class is a simple wrapper to submit an sklearn pipeline to a dask
    cluster for fitting.

    Example usage::

        my_pipeline = sklearn.pipeline.Pipeline(steps)
        d_pipeline = dask_pipeline(my_pipeline, dask_client)
        d_pipeline_fit = d_pipeline.fit(X, y)
        pipeline_fit = d_pipeline_fit.collect_results()
    """
    def __init__(self, pipeline, dask_client):
        self.pipeline = pipeline
        self.dask_client = dask_client

    def fit(self, X, y):
        self.d_fit = self.dask_client.submit(self.pipeline.fit, X, y)
        return self

    def collect_results(self):
        self.pipeline_fit = self.d_fit.result()

        # and clean up
        del self.d_fit
        del self.dask_client

        return self.pipeline_fit

