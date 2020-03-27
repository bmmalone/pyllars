""" This modules contains utilities for working with pytorch.

The utilities in this module make relatively strong assumptions about the
structure of the objects they take as inputs. Please check the documentation
carefully.

It is generally designed to be compatible with [Tune](https://ray.readthedocs.io/en/latest/tune.html).
"""
import logging
logger = logging.getLogger(__name__)

try:
    import torch
except ModuleNotFoundError as mnfe:
    msg = ("Please install pytorch before using torch_utils. pyllars does not "
        "include pytorch by default, so it must be installed using `[torch]`.")
    raise ModuleNotFoundError(msg)

import json
import os
import pathlib
import random as rn

import numpy as np
import ray.tune
import torch.nn as nn

from typing import Optional, Sequence, Union
NetType = Union[nn.Module, nn.DataParallel]

###
# File IO
###

def save_model(model:ray.tune.Trainable, checkpoint_dir:str) -> str:
    """ Save `model` and its optimizer and configuration to `checkpoint_dir`
    
    This function assumes that `model` has a member named `net` which is the
    actual network to be saved. Additionally, `model` should have a member
    named `optimizer` which has its optimizer, and a member named `config`
    which has necessary configuration information.

    This function creates three files in `checkpoint_dir`:

    * checkpoint.pt : the network parameters (in torch format)
    * optim.pt : the optimizer state (in torch format)
    * params.json : the configuration (in json format)

    Parameters
    ----------
    model : ray.tune.Trainable
        The model. See the description for assumptions on its members.

    checkpoint_dir : str
        The path to the checkpoint directory

    Returns
    -------
    model_path : str
        The path to the saved model.
    """
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    
    path = str(checkpoint_dir / 'checkpoint.pt')
    net = get_net(model)
    torch.save(net.state_dict(), path)
    
    optim_path = str(checkpoint_dir / 'optim.pt')
    torch.save({'optimizer': model.optimizer.state_dict()}, optim_path)

    params_path = str(checkpoint_dir / 'params.json')
    with open(params_path, 'w') as f:
        json.dump(model.config, f, indent=4)

    return path

def restore_model(model:ray.tune.Trainable, checkpoint_dir:str) -> None:
    """ Restore the model saved to `checkpoint_dir`

    This function assumes that `model` has the following methods:
    * `get_network_type`: this returns the type of network used by this model.
        In particular, the network type should have a constructor which accepts
        a single `config` argument.

    * `get_optimizer_type`: returns the type of the optimizer used by the model

    * `log`: logs a string message

    It also assumed that `model` has the following members:

    * `config`: the configuration options for the model, network, etc.

    * `device`: a torch.device on which the model will be run
    """

    checkpoint_dir = pathlib.Path(checkpoint_dir)

    checkpoint_path = str(checkpoint_dir / 'checkpoint.pt')
    state_dict = torch.load(checkpoint_path)

    msg = "restoring the network"
    model.log(msg)
    net_type = model.get_network_type()
    net = net_type(model.config)
    net.load_state_dict(state_dict)

    net = send_network_to_device(net, model.device)
    model.net = net


    msg = "restoring the optimizer"
    model.log(msg)

    #TODO: what if the optimizer has hyperparameters (learning rate, etc.)
    optimizer_type = model.get_optimizer_type()
    optimizer = optimizer_type(model.net.parameters())

    optim_path = str(checkpoint_dir / 'optim.pt')
    optimizer_state_dict = torch.load(optim_path)['optimizer']
    optimizer.load_state_dict(optimizer_state_dict)
    model.optimizer = optimizer


###
# Device communication
###
def get_device(device_name:Optional[str]=None) -> torch.device:
    """ Based on `device_name` and the computing environment, create a device

    Parameters
    ----------
    device_name : typing.Optional[str]
        The name of the device. If not given, then the name will be guessed
        based on whether CUDA is available or not

    Returns
    -------
    device : torch.device
        The device
    """
    if device_name is None:
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"

    msg = "[torch_utils.get_device] creating device: '{}'".format(device_name)
    logger.info(msg)
    device = torch.device(device_name)
    return device

def send_network_to_device(net:nn.Module, device:torch.device) -> NetType:
    """ Send `net` to `device`

    If there are multiple CUDA devices available, then the network will first
    be wrapped with `nn.DataParallel`.

    Parameters
    ----------
    net : nn.Module
        The network

    device : torch.device
        The device

    Returns
    -------
    net_on_device : typing.Union[torch.nn.DataParallel, torch.nn.Module]
        The network on the device, possibly wrapped in a DataParallel
    """
    if torch.cuda.device_count() > 1:
        #net = nn.DataParallel(net)
        pass
        
    net = net.to(device=device)

    return net

def send_data_to_device(
        *data:Sequence[torch.TensorType],
        device:torch.device) -> Sequence[torch.TensorType]:
    """ Send all of the tensors in `data` to `device`

    Parameters
    ----------
    *data : typing.Sequence[torch.TensorType]
        The data elemnts

    device : torch.device
        The device
    """
    def _to_device(d, device):
        try:
            d = d.to(device)
        except:
            # then it isn't a tensor
            pass

        return d

    data = [
        _to_device(d, device) for d in data
    ]

    return data

def retrieve_data_from_device(
        *data:Sequence[torch.TensorType]) -> Sequence[np.ndarray]:
    """ Detach all of the tensors in `data` to numpy arrays

    Parameters
    ----------
    *data : typing.Sequence[torch.TensorType]
        The data elemnts. Presumably, these have been sent to a GPU or other
        device using `send_data_to_device` or similar.

    Returns
    -------
    *data : typing.Sequence[torch.TensorType]
        The data elements, but retrieved from the device. Data elements which
        were not already on some device or cannot be converted to numpy arrays
        will be unchanged.
    """
    def _from_device(d):
        try:
            # break the steps down for debugging in case something fails
            d_from_device = d.detach()
            d_cpu = d_from_device.cpu()
            d_numpy = d_cpu.numpy()

            d = d_numpy
        except:
            # then some step failed. just return the original
            pass

        return d

    data = [
        _from_device(d) for d in data
    ]

    return data

def get_net(model:ray.tune.Trainable) -> nn.Module:
    """ Extract the torch network from `model`

    This function assumes that `model` has a member named `net`. That member is
    either directly the network or an instance of `nn.DataParallel`. In case
    `model.net` is a DataParallel instance, then the network will be extracted
    from that.

    The function then verifies that the network is an instance of `nn.Module`;
    otherwise, an exception is raised.

    Parameters
    ----------
    model : ray.tune.Trainable
        The model. See the description for assumptions on its members.

    Returns
    -------
    net : torch.nn.Module
        The network
    """
    net = model.net

    if isinstance(net, nn.DataParallel):
        net = net.module

    if not isinstance(net, nn.Module):
        msg = "Invalid net object: {}".format(net)
        raise ValueError(msg)
    
    return net

###
# Other pytorch utilities
###
def initialize(gpu:str="0", seed:int=8675309) -> None:
    """ Initial various environment variables and random seeds

    In particular, this sets the following environment variables:

    * CUDA_DEVICE_ORDER
    * CUDA_VISIBLE_DEVICES
    * PYTHONHASSEED

    ... and the following random seeds:
    * random.seed
    * numpy.random.seed
    * torch.manual_seed

    ... and the following torch options:
    * torch.backends.cudnn.deterministic
    * torch.backends.cudnn.benchmark
    """
    # so the IDs match nvidia-smi
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['PYTHONHASHSEED'] = '0'

    # set the various random seeds
    rn.seed(seed)    
    np.random.seed(seed)
    torch.manual_seed(seed)

    # and other options
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def tensor_list_to_numpy(tensor_list:Sequence[torch.TensorType]) -> np.ndarray:
    """ Convert the list of tensors to a numpy array

    Parameters
    ----------
    tensor_list : typing.Sequence[torch.TensorType]
        A list of tensors. For example, this may be built up during a prediction
        loop.

    Returns
    -------
    array : numpy.ndarray
        The tensors as a numpy array
    """
    array = np.array([ 
        t.numpy() for t in tensor_list 
    ])

    return array