import matplotlib.pyplot as plt
import numpy as np
import random

import torch

GLOBAL_DEVICE = None

def mse(x: torch.Tensor, y: torch.Tensor):
    return torch.mean(torch.square(x - y))

def get_device():
    global GLOBAL_DEVICE
    return GLOBAL_DEVICE

def set_device(device: str):
    global GLOBAL_DEVICE
    GLOBAL_DEVICE = device

def check_device(device: str):
    if device is None:
        print("Device is not set, defaulting to CPU")
        set_device('cpu')
        
    global GLOBAL_DEVICE
    return GLOBAL_DEVICE

def gpu_setup(verbose: bool=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose:
        print('Using %s device' % device)

    global GLOBAL_DEVICE
    set_device(device)

    return device

def detach(T: torch.Tensor):
    return T.to('cpu').detach().numpy()

def autodiff(u: torch.Tensor, X: torch.Tensor, allow_unused: bool=False):
    return torch.autograd.grad(
        u, X,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
        allow_unused=allow_unused
    )[0]

# write a seed everything function
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def T_(T: torch.Tensor, shape: tuple[int, int]=(-1, 1)):
    """ Condition a tensor T. Returns a reshaped tensor on the device
    which requires grad. Defualt shape is (-1, 1).

    Example: Instead of

        T = torch.linspace(-1, 1, 100).reshape((-1, 1))
        T.requires_grad_()
        T = T.to(GLOBAL_DEVICE)

    we have
    
        T = _T(torch.linspace(-1, 1, 100))

    Attributes:
    ===========
    T: The tensor to be conditioned.
    shape: The shape the tensor should be in.

    Returns:
    ========
    A conditioned version of the tensor T.
    """
    global GLOBAL_DEVICE
    device = check_device(GLOBAL_DEVICE)
    
    if shape is not None:
        T = T.reshape(shape)
    T.requires_grad_()
    return T.to(device)