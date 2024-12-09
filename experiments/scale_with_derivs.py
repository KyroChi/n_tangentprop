import datetime
import jsonargparse
import gc
import os
import time
import torch
import random

from tqdm.auto import tqdm

from ntp.common import (
    autodiff, 
    get_device,
    gpu_setup, 
    seed_everything, 
    set_device
)
from ntp.models import DenseNet

def run_experiment(
    n_iters: int = 100,
    network_width: int = 24,
    network_depth: int = 3,
    batch_size: int = 256,
    n_derivs: int = 9,
    device: str = None,
    save_dir: str = './_runs/scale_with_derivs/',
    random_seed: int = 42,
):
    seed_everything(random_seed)

    if device is None:
        gpu_setup()
    else:
        set_device(device)

    device = get_device()

    derivs = list(range(n_derivs + 1))

    os.makedirs(save_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    model = DenseNet(
        n_layers=network_depth,
        l_width=network_width,
    ).to(device)

    output_data = {
        'info': {
            'network_width': network_width, 
            'network_depth': network_depth, 
            'n_iters': n_iters, 
            'n_params': sum(p.numel() for p in model.parameters())
        }
    }

    for d in derivs:
        output_data[d] = {
            'forward_time': {'ntp': [], 'ad': []}, 
            'backward_time': {'ntp': [], 'ad': []}, 
            'total_time': {'ntp': [], 'ad': []}
        }

    tests = [(d, m) for d in derivs for _ in range(n_iters) for m in ['ntp', 'ad']]
    random.shuffle(tests)

    x = torch.randn(batch_size, 1, device=device).requires_grad_(True)

    torch.backends.cudnn.benchmark = True

    for d, mode in tqdm(tests):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        start = time.perf_counter()

        if mode == 'ntp':
            y = model(x, n_derivs=d)
        else:
            y = model(x, n_derivs=0)
            y_tensor = torch.zeros((d + 1, batch_size, 1), device=device)
            y_tensor[0] = y[0]
            for i in range(1, d + 1):
                y_tensor[i] = autodiff(y_tensor[i - 1], x)
            y = y_tensor
        f_time = time.perf_counter() - start

        assert y.shape[0] == d + 1, f'{len(y)} != {d + 1}'

        loss = torch.sum(y)

        start_b = time.perf_counter()
        loss.backward()
        end_b = time.perf_counter()

        b_time = end_b - start_b
        t_time = end_b - start

        for p in model.parameters():
            p.grad = None

        output_data[d]['forward_time'][mode].append(f_time)
        output_data[d]['backward_time'][mode].append(b_time)
        output_data[d]['total_time'][mode].append(t_time)

    torch.save(output_data, os.path.join(save_dir, f'{now}.pt'))

if __name__ == "__main__":
    cli = jsonargparse.CLI(run_experiment)