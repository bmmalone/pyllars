"""
This module contains helpers for using dask: https://dask.pydata.org/en/latest/
"""

import logging
logger = logging.getLogger(__name__)

from typing import Callable, Iterable, List

import argparse
import collections
import dask.distributed
import pandas as pd
import shlex
import tqdm
import typing

def connect(args:argparse.Namespace) -> typing.Tuple[dask.distributed.Client, typing.Optional[dask.distributed.LocalCluster]]:
    """ Connect to the dask cluster specifed by the arguments in `args`

    Specifically, this function uses args.cluster_location to determine whether
    to start a dask.distributed.LocalCluster (in case args.cluster_location is
    "LOCAL") or to (attempt to) connect to an existing cluster (any other
    value).

    If a local cluster is started, it will use a number of worker processes
    equal to args.num_procs. Each process will use args.num_threads_per_proc
    threads. The scheduler for the local cluster will listen to a random port.

    Parameters
    ----------
    args: argparse.Namespace
        A namespace containing the following fields:
        
        * cluster_location
        * client_restart
        * num_procs
        * num_threads_per_proc

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
            n_workers=args.num_procs,
            processes=True,
            threads_per_worker=args.num_threads_per_proc
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
        

def add_dask_options(
        parser:argparse.ArgumentParser,
        num_procs:int=1,
        num_threads_per_proc:int=1,
        cluster_location:str="LOCAL") -> None:

    """ Add options for connecting to and/or controlling a local dask cluster

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to which the options will be added

    num_procs : int
        The default number of processes for a local cluster
        
    num_threads_per_proc : int
        The default number of threads for each process for a local cluster
        
    cluster_location : str
        The default location of the cluster 

    Returns
    -------
    None : None
        A "dask cluster options" group is added to the parser
    """
    dask_options = parser.add_argument_group("dask cluster options")

    dask_options.add_argument('--cluster-location', help="The address for the "
        "cluster scheduler. This should either be \"LOCAL\" or the address "
        "and port of the scheduler. If \"LOCAL\" is given, then a "
        "dask.distributed.LocalCluster will be started.", 
        default=cluster_location)

    dask_options.add_argument('--num-procs', help="The number of processes to use "
        "for a local cluster", type=int, default=num_procs)

    dask_options.add_argument('--num-threads-per-proc', help="The number of "
        "threads to allocate for each process. So the total number of threads "
        "for a local cluster will be (args.num_procs * "
        "args.num_threads_per_cpu).", type=int, default=num_threads_per_proc)

    dask_options.add_argument('--client-restart', help="If this flag is "
        "given, then the \"restart\" function will be called on the client "
        "after establishing the connection to the cluster",
        action='store_true')

def add_dask_values_to_args(
        args:argparse.Namespace,
        num_procs:int=1,
        num_threads_per_proc:int=1,
        cluster_location:str="LOCAL",
        client_restart:bool=False) -> None:
    """ Add the options for a dask cluster to the given argparse namespace

    This function is mostly intended as a helper for use in ipython notebooks.

    Parameters
    ----------
    args : argparse.Namespace
        The namespace on which the arguments will be set

    num_procs : int
        The number of processes for a local cluster
        
    num_threads_per_proc : int
        The number of threads for each process for a local cluster
        
    cluster_location : str
        The location of the cluster
        
    client_restart : bool
        Whether to restart the client after connection

    Returns
    -------
    None : None
        The respective options will be set on the namespace
    """
    args.num_procs = num_procs
    args.num_threads_per_proc = num_threads_per_proc
    args.cluster_location = cluster_location
    args.client_restart = client_restart
    

def get_dask_cmd_options(args:argparse.Namespace) -> List[str]:
    """ Extract the flags and options specified for dask from
    the parsed arguments.
    
    Presumably, these were added with `add_dask_options`. This function
    returns the arguments as an array. Thus, they are suitable for use
    with `subprocess.run` and similar functions.
    
    Parameters
    -----------
    args : argparse.Namespace
        The parsed arguments
        
    Returns
    -------
    dask_options : typing.List[str]
        The list of dask options and their values.
    """
    
    args_dict = vars(args)

    # first, pull out the text arguments
    dask_options = [
        'num_procs',
        'num_threads_per_proc',
        'cluster_location'
    ]

    # create a list of command line arguments
    ret = []
    
    for o in dask_options:
        arg = str(args_dict[o])
        if len(arg) > 0:
            ret.append('--{}'.format(o.replace('_', '-')))
            ret.append(arg)
    
    if args.client_restart:
        ret.append("--client-restart")
        
    ret = [shlex.quote(c) for c in ret]
        
    return ret

###
#   Helpers to submit arbitrary jobs to a dask cluster
###

def apply_iter(
        it:Iterable,
        client:dask.distributed.Client,
        func:Callable,
        *args,
        return_futures:bool=False,
        progress_bar:bool=True,
        priority:int=0,
        **kwargs) -> List:
    """ Distribute calls to `func` on each item in `it` across `client`.

    Parameters
    ----------
    it : typing.Iterable
        The inputs for `func`

    client : dask.distributed.Client
        A dask client

    func : typing.Callable
        The function to apply to each item in `it`

    args
        Positional arguments to pass to `func`
    
    kwargs
        Keyword arguments to pass to `func`

    return_futures : bool
        Whether to wait for the results (`False`, the default) or return a
        list of dask futures (when `True`). If a list of futures is returned,
        the `result` method should be called on each of them at some point
        before attempting to use the results.

    progress_bar : bool
        Whether to show a progress bar when waiting for results. The parameter
        is only relevant when `return_futures` is `False`.
        
    priority : int
        The priority of the submitted tasks. Please see the dask documentation
        for more details: http://distributed.readthedocs.io/en/latest/priority.html

    Returns
    -------
    results: typing.List
        Either the result of each function call or a future which will give
        the result, depending on the value of `return_futures`
    """
    msg = ("[dask_utils.apply_iter] submitting jobs to cluster")
    logger.debug(msg)

    if progress_bar:
        it = tqdm.tqdm(it)


    ret_list = [
        client.submit(func, *(i, *args), **kwargs, priority=priority) for i in it
    ]

    if return_futures:
        return ret_list

    msg = ("[dask_utils.apply_iter] collecting results from cluster")
    logger.debug(msg)
    
    # add a progress bar if we asked for one
    if progress_bar:
        ret_list = tqdm.tqdm(ret_list)

    ret_list = [r.result() for r in ret_list]
    return ret_list


def apply_df(
        data_frame:pd.DataFrame,
        client:dask.distributed.Client,
        func:typing.Callable,
        *args,
        return_futures:bool=False,
        progress_bar:bool=True,
        priority:int=0,
        **kwargs) -> List:
    """ Distribute calls to `func` on each row in `data_frame` across `client`.

    Additionally, `args` and `kwargs` are passed to the function call.

    Parameters
    ----------
    data_frame: pandas.DataFrame
        A data frame

    client: dask.distributed.Client
        A dask client

    func: typing.Callable
        The function to apply to each row in `data_frame`

    args
        Positional arguments to pass to `func`
    
    kwargs
        Keyword arguments to pass to `func`

    return_futures: bool
        Whether to wait for the results (`False`, the default) or return a
        list of dask futures (when `True`). If a list of futures is returned,
        the `result` method should be called on each of them at some point
        before attempting to use the results.

    progress_bar: bool
        Whether to show a progress bar when waiting for results. The parameter
        is only relevant when `return_futures` is `False`.
        
    priority : int
        The priority of the submitted tasks. Please see the dask documentation
        for more details: http://distributed.readthedocs.io/en/latest/priority.html

    Returns
    -------
    results: typing.List
        Either the result of each function call or a future which will give
        the result, depending on the value of `return_futures`
    """

    if len(data_frame) == 0:
        return []

    it = data_frame.iterrows()
    if progress_bar:
        it = tqdm.tqdm(it, total=len(data_frame))

    ret_list = [
        client.submit(func, *(row[1], *args), **kwargs, priority=priority) 
            for row in it
    ]

    if return_futures:
        return ret_list

    # add a progress bar if we asked for one
    if progress_bar:
        ret_list = tqdm.tqdm(ret_list, total=len(data_frame))

    ret_list = [r.result() for r in ret_list]
    return ret_list


def apply_groups(
        groups:pd.core.groupby.DataFrameGroupBy,
        client:dask.distributed.client.Client,
        func:typing.Callable,
        *args,
        return_futures:bool=False,
        progress_bar:bool=True,
        priority:int=0,
        **kwargs) -> typing.List:
    """ Distribute calls to `func` on each group in `groups` across `client`.

    Additionally, `args` and `kwargs` are passed to the function call.

    Parameters
    ----------
    groups: pandas.DataFrameGroupBy
        The result of a call to `groupby` on a data frame

    client: distributed.Client
        A dask client

    func: typing.Callable
        The function to apply to each group in `groups`

    args
        Positional arguments to pass to `func`
    
    kwargs
        Keyword arguments to pass to `func`

    return_futures: bool
        Whether to wait for the results (`False`, the default) or return a
        list of dask futures (when `True`). If a list of futures is returned,
        the `result` method should be called on each of them at some point
        before attempting to use the results.

    progress_bar: bool
        Whether to show a progress bar when waiting for results. The parameter
        is only relevant when `return_futures` is `False`.
        
    priority : int
        The priority of the submitted tasks. Please see the dask documentation
        for more details: http://distributed.readthedocs.io/en/latest/priority.html

    Returns
    -------
    results: typing.List
        Either the result of each function call or a future which will give
        the result, depending on the value of `return_futures`.
    """

    if len(groups) == 0:
        return []

    it = groups
    if progress_bar:
        it = tqdm.tqdm(it)

    ret_list = [
        client.submit(func, *(group, *args), **kwargs, priority=priority) 
            for name, group in it
    ]

    if return_futures:
        return ret_list

    # add a progress bar if we asked for one
    if progress_bar:
        ret_list = tqdm.tqdm(ret_list)

    ret_list = [r.result() for r in ret_list]
    return ret_list
    
def check_status(f_list:Iterable[dask.distributed.client.Future]) -> collections.Counter:
    """ Collect the status counts of a list of futures
    
    This is primarily intended to check the status of jobs submitted with the
    various `apply` functions when `return_futures` is `True`.
    
    Parameters
    ----------
    f_list: typing.List[dask.distributed.client.Future]
        The list of futures
    
    Returns
    -------
    status_counter: collections.Counter
        The number of futures with each status
    """
    counter = collections.Counter([f.status for f in f_list])
    return counter
   
def collect_results(
        f_list:Iterable[dask.distributed.client.Future],
        finished_only:bool=True,
        progress_bar:bool=False) -> List:
    """ Collect the results from a list of futures
    
    By default, only results from finished tasks will be collected. Thus, the
    function is (more or less) non-blocking.
    
    Parameters
    ----------
    f_list: typing.List[dask.distributed.client.Future]
        The list of futures

    finished_only: bool
        Whether to collect only results for jobs whose status is 'finished'

    progress_bar : bool
        Whether to show a progress bar when waiting for results. The parameter
        is only relevant when `return_futures` is `False`.
        
    Returns
    -------
    results: typing.List
        The results for each (finished, if specified) task
    """
    
    if progress_bar:
        f_list = tqdm.tqdm(f_list)

    if finished_only:
        ret = [f.result() for f in f_list if f.status == 'finished']
    else:
        ret = [f.result() for f in f_list]

    return ret

def cancel_all(f_list:Iterable[dask.distributed.client.Future], pending_only=True) -> None:
    """ Cancel all (pending) tasks in the list

    By default, only pending tasks are cancelled.

    Parameters
    ----------
    f_list : Iterable[dask.distributed.client.Future]
        The list of futures

    pending_only : bool
        Whether to cancel only tasks whose status is 'pending'

    Returns
    -------
    None : None
        The specified tasks are cancelled.
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

    Examples
    --------
    
    .. code-block:: python

        my_pipeline = sklearn.pipeline.Pipeline(steps)
        d_pipeline = dask_pipeline(my_pipeline, dask_client)
        d_pipeline_fit = d_pipeline.fit(X, y)
        pipeline_fit = d_pipeline_fit.collect_results()
    """
    def __init__(self, pipeline, dask_client):
        self.pipeline = pipeline
        self.dask_client = dask_client

    def fit(self, X, y):
        """ Submit the call to `fit` of the underlying pipeline to `dask_client`
        """
        self.d_fit = self.dask_client.submit(self.pipeline.fit, X, y)
        return self

    def collect_results(self):
        """ Collect the "fit" pipeline from `dask_client`. Then, cleanup the
        references to the future and client.
        """
        self.pipeline_fit = self.d_fit.result()

        # and clean up
        del self.d_fit
        del self.dask_client

        return self.pipeline_fit

