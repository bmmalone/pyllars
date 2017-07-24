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
        
        msg = "Starting local dask cluster"
        logger.info(msg)

        cluster = DaskCluster(
            n_workers=args.num_cpus,
            processes=True,
            threads_per_worker=args.num_threads_per_cpu
        )

        client = DaskClient(cluster)

    else:
        msg = "Attempting to connect to dask cluster: {}"
        msg = msg.format(args.cluster_location)
        logger.info(msg)

        client = DaskClient(address=args.cluster_location)

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

def add_dask_values_to_args(args, num_cpus=1, num_threads_per_cpu=1,
        cluster_location="LOCAL"):

    """ Add the options for a dask cluster to the given argparse namespace

    This function is mostly intended as a helper for use in ipython notebooks.
    """
    args.num_cpus = num_cpus
    args.num_threads_per_cpu = num_threads_per_cpu
    args.cluster_location = cluster_location
