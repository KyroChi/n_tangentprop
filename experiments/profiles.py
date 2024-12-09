import datetime
import jsonargparse
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from ntp.common import autodiff, gpu_setup, mse, seed_everything, T_
from ntp.models import DenseNet, ParameterWrapper

# Get the functioning LBFGS module
import os
import sys
module_path = os.path.abspath(os.path.join('/home/kyle/new_thesis/PyTorch_LBFGS/functions'))
if module_path not in sys.path:
    sys.path.append(module_path)
from LBFGS import FullBatchLBFGS

class ProfileModel(nn.Module):
    """
    Compute -sin(pi / 4 x) + (model(x) - model(-x)) and it's derivatives
    """
    def __init__(self, base_model: nn.Module):
        super(ProfileModel, self).__init__()
        self.base_model = base_model

    def forward(
        self, 
        X: torch.Tensor, 
        n_derivs: int=0,
        mode: str='ad'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode == 'ad':
            U, r = self.base_model(X, n_derivs=0)
            U = [U]
            for _ in range(n_derivs):
                U.append(autodiff(U[-1], X))
        elif mode == 'ntp':
            U, r = self.base_model(X, n_derivs=n_derivs)
        else:
            raise ValueError(f"Mode {mode} not recognized. Acceptable modes are 'ad' and 'ntp'.")
        
        out = []

        for i, u in enumerate(U):
            if i % 2 == 0:
                # out.append(
                #     (-math.pi / 4)**(i / 2) * -torch.sin(math.pi / 4 * X) + u#(u - torch.flip(u, dims=[1]))
                # )
                out.append(u)
            else:
                # out.append(
                #     (-math.pi / 4)**((i - 1) / 2) * torch.cos(math.pi / 4 * X) + u#(u - torch.flip(u, dims=[1]))
                # )
                out.append(u)

        return out, r

def burgers_residual(
    X: torch.Tensor,
    r: float,
    U: list[torch.Tensor],
    derivs: list[int],
) -> torch.Tensor:
    """
    args:
    =====
        X: The input values
        r: The value of \lambda
        U: Model outputs
        derivs: Which derivatives to enforce the residual for. 

    return:
    =======
        derivs: The derivatives of the residual corresponding to the input derivs.
    """
    assert len(derivs) >= 1, f"Must provide derivatives."
    assert sum([0 if d >= 0 else 1 for d in derivs]) == 0, f"Derivatives cannot be negative."
    assert len(U) >= max(derivs) + 2, f"U did not supply enough derivatives for the requested derivs."

    out = [None for _ in derivs]

    for i, d in enumerate(derivs):
        if d == 0:
            out[i] = -r * U[0] + ( (1 + r) * X + U[0] ) * U[1]
        elif d == 1:
            out[i] = U[1] + (1 + r) * X * U[2] + U[0] * U[2] + U[1]**2
        elif d == 2:
            out[i] = U[2] + (1 + r) * U[2] + (1 + r) * X * U[3] + U[1] * U[2] + U[0] * U[3] + 2 * U[1] * U[2]
        elif d == 3:
            out[i] = U[3] + (1 + r) * U[3] + (1 + r) * U[3] + (1 + r) * X * U[4] + U[2]**2 + U[1] * U[3] + U[1] * U[3] + U[0] * U[4] + 2 * U[2]**2 + 2 * U[1] * U[3]
        elif d > 3:
            out[i] = U[3] + (1 + r) * U[3] + (1 + r) * U[3] + (1 + r) * X * U[4] + U[2]**2 + U[1] * U[3] + U[1] * U[3] + U[0] * U[4] + 2 * U[2]**2 + 2 * U[1] * U[3]
        else:
            # We use the following formula for the n-th derivative of the residual (m >= 3)
            # ( (3m - 1) / 2 + (m + 1) U[1] ) U[m] + ( (1 + r) X + U[0])U[m + 1] = - \sum_{k=2}^{m-1}(m\choose{k}) U[k]U[m - k + 1]
            out[i] = ( ((1 + r) * d - r) + (d + 1) * U[1] ) * U[d]
            out[i] += ((1 + r) * X + U[0]) * U[d + 1]
            for k in range(2, d):
                out[i] += math.comb(d, k) * U[k] * U[d - k + 1]

    return out

def loss_fn(
    X: torch.Tensor,
    X_local: torch.Tensor,
    model: nn.Module,
    n_derivs: int,
    deriv_mode: str,
) -> tuple[torch.Tensor, float]:
    _U, r = model(X, n_derivs=2, mode=deriv_mode)

    residuals = burgers_residual(
        X, r, _U, [0, 1]
    )

    bndry = torch.square(_U[0][0, 0] - 1) + torch.square(_U[0][-1, 0] + 1)
    targets = [bndry] + [0.3 * mse(res, torch.zeros_like(res)) for res in residuals]

    _U, _ = model(X_local, n_derivs=n_derivs + 1, mode=deriv_mode)
    
    residuals = burgers_residual(
        X_local, r, _U, [n_derivs]
    )
    targets += [0.2 * mse(residuals[0], 0.)]
    
    return sum(targets), r

def run_experiment(
    profile_id: int,
    mode: str,
    r: float=0.5,
    n_layers: int=3,
    l_width: int=24,
    n_adam_epochs: int=10000,
    n_lbfgs_epochs: int=20000,
    X_samples: int=301,
    X_local_samples: int=31,
    lr=3.5e-4
) -> None:
    """
    args:
    =====
        profile_id: The id of the profile. 0 is stable, 1, 2, ... are the unstable profiles.
        mode: 'ad' for autodifferentiation or 'ntp' for n-TangentProp
    """
    assert mode == 'ad' or mode == 'ntp', f"Acceptable modes are 'ad' or 'ntp', got {mode} instead."
    assert profile_id >= 0, f"profile_id must be greater than or equal to zero. Got {profile_id}."

    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    seed_everything(42)

    device = gpu_setup()
    torch.set_default_dtype(torch.float64)

    print('Using device:', device)
    print('   Precision:', torch.get_default_dtype())

    k = profile_id

    model = ProfileModel(ParameterWrapper(
        DenseNet(
            n_layers=n_layers,
            l_width=l_width,
            scale=1.15,
        ),
        activation=nn.Tanh,
        l_width=5,
        l_bound=1/(2 * k + 3),
        u_bound=1/(2 * k + 1)
    )).to(device)

    opt1 = torch.optim.Adam(model.base_model.parameters(), lr=lr, foreach=False)
    opt2 = FullBatchLBFGS(model.base_model.parameters(), lr=1, history_size=30, line_search='Wolfe', debug=False)

    X = T_(torch.linspace(-2., 2., X_samples))
    X_local = T_(torch.linspace(-0.075, 0.075, X_local_samples))

    times = []
    losses = {
        'loss': [],
        'r': [],
    }

    start = time.perf_counter()
    for _ in (pbar := tqdm(range(1, n_adam_epochs + 1))):
        opt1.zero_grad()
        loss, r = loss_fn(
            X, 
            X_local, 
            model, 
            n_derivs=2 * k + 3, 
            deriv_mode=mode
        )
        loss.backward()
        opt1.step()

        losses['loss'].append(loss.item())
        losses['r'].append(r.item())

        pbar.set_postfix_str(
            f"Loss: {loss.item():.5e}, r: {r.item():.6e}"
        )

        times.append(time.perf_counter() - start)

    for _ in (pbar := tqdm(range(1, n_lbfgs_epochs + 1))):
        def closure():
            opt2.zero_grad()
            loss, _ = loss_fn(
                X, 
                X_local, 
                model, 
                n_derivs=2 * k + 3, 
                deriv_mode=mode
            )
            return loss

        loss, _, _, _, _, _, _, _ = opt2.step({'closure': closure})
        loss, r = loss_fn(
            X, 
            X_local, 
            model, 
            n_derivs=2 * k + 3, 
            deriv_mode=mode
        )

        losses['loss'].append(loss.item())
        losses['r'].append(r.item())

        pbar.set_postfix_str(
            f"Loss: {loss.item():.5e}, r: {r.item():.6e}"
        )

        times.append(time.perf_counter() - start)

    base_path = "_runs/ss_burgers"
    os.makedirs(base_path, exist_ok=True)

    torch.save(times, f"{base_path}/{now}_{mode}_times.pt")
    torch.save(model, f"{base_path}/{now}_{mode}_model.pt")
    torch.save(losses, f"{base_path}/{now}_{mode}_losses.pt")

    out = model(X, n_derivs=2 * k + 3, mode=mode)
    torch.save(out, f"{base_path}/{now}_{mode}_out.pt")

if __name__ == "__main__":
    cli = jsonargparse.CLI(run_experiment)