###
#   This module contains helpers for using dask:
#       https://dask.pydata.org/en/latest/
###

import logging
logger = logging.getLogger(__name__)

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

    N.B. Unlike with the `parallel` module functions, it does not make sense

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

    ret_list = [
        client.submit(func, *(i, *args), **kwargs) for i in it
    ]

    if return_futures:
        return ret_list

    # add a progress bar if we asked for one
    if progress_bar:
        import tqdm
        ret_list = tqdm.tqdm(ret_list)

    ret_list = [r.result() for r in ret_list]
    return ret_list


def apply_df(data_frame, client, func, *args, return_futures=False,
        progress_bar=True, **kwargs):
    """ Call `func` on each row in `data_frame`.

    Additionally, `args` and `kwargs` are passed to the function call.

    N.B. Unlike with the `parallel` module functions, it does not make sense

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

    ret_list = [
        client.submit(func, *(row[1], *args), **kwargs) 
            for row in data_frame.iterrows()
    ]

    if return_futures:
        return ret_list

    # add a progress bar if we asked for one
    if progress_bar:
        import tqdm
        ret_list = tqdm.tqdm(ret_list)

    ret_list = [r.result() for r in ret_list]
    return ret_list

