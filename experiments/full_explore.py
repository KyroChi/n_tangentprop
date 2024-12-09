import datetime
import gc
import os
import time
import torch
import random

from tqdm.auto import tqdm

from ntp.common import autodiff
from ntp.models import DenseNet

n_iters = 100
device = 'cuda'

global_now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

network_depths = [3, 5, 7]
network_widths = [24, 32, 48, 64, 76, 88, 104, 128]
n_derivs = [1, 3, 6, 8]
batch_sizes = [2**5, 2**6, 2**8, 2**10, 2**12]
modes = ['ntp', 'ad']

experiments = {}
experiment_tuples = []
for depth in network_depths:
    for width in network_widths:
        for batch_size in batch_sizes:
            for deriv in n_derivs:
                for mode in modes:
                    experiments[(depth, width, batch_size, deriv, mode)] = {
                        'ttime': [],
                        'ftime': [],
                        'btime': []
                    }
                    for _ in range(n_iters):
                        experiment_tuples.append((depth, width, batch_size, deriv, mode))

random.shuffle(experiment_tuples)

for depth, width, batch_size, deriv, mode in (pbar := tqdm(experiment_tuples)):
    pbar.set_postfix_str(f"depth={depth}, width={width}, batch_size={batch_size}, deriv={deriv}, mode={mode}")
    model = DenseNet(
        n_layers=depth,
        l_width=width,
    ).to(device)

    x = torch.randn((batch_size, 1)).to(device).requires_grad_()
    gc.collect()
    torch.cuda.synchronize()
    start = time.perf_counter()

    if mode == 'ad':
        y_derivs = [model(x, n_derivs=0)]
        for _ in range(deriv):
            y_derivs.append(autodiff(y_derivs[-1], x))
        loss = torch.sum(sum(y_derivs))
    elif mode == 'ntp':
        y_derivs = model(x, n_derivs=deriv)
        loss = torch.sum(y_derivs)
    
    f_time = time.perf_counter() - start

    now = time.perf_counter()
    loss.backward()
    end_time = time.perf_counter()

    b_time = end_time - now
    t_time = end_time - start

    key = (depth, width, batch_size, deriv, mode)

    experiments[key]['ftime'].append(f_time)
    experiments[key]['btime'].append(b_time)
    experiments[key]['ttime'].append(t_time)

    os.makedirs('_runs/full_explore', exist_ok=True)
    torch.save(experiments, f'_runs/full_explore/{global_now}_full_search.pt')
