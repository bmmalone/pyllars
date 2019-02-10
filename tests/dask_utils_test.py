""" Tests for the `pyllars.dask_utils` module.
"""

import pyllars.dask_utils as dask_utils

import argparse
import pytest

@pytest.fixture
def local_args():
    """ Create an argparse namespace to create a local dask cluster.
    """
    args = argparse.Namespace()
    args.num_procs = 1
    args.num_threads_per_proc = 1
    args.cluster_location = "LOCAL"
    args.client_restart = False
    
    return args

def test_add_dask_values_to_args(local_args):
    """ Test adding values to an argparse namespace.
    """
    args = argparse.Namespace()
    dask_utils.add_dask_values_to_args(args)
    
    assert args == local_args

def test_connect_local(local_args):
    """ Test creating a local dask server.
    """
    dask_client, dask_cluster = dask_utils.connect(local_args)
    
    # check that we have a single worker
    assert len(dask_client.scheduler_info()['workers']) == 1
    
    # also make sure we can submit a job
    expected_output = 3
    
    f = dask_client.submit(max, 3, 2)
    actual_output = f.result()
    
    assert expected_output == actual_output
    
    # now, close the connection
    dask_client.close()
    
    