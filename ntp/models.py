import torch
import torch.nn as nn

import torch.nn.functional as F

from ntp.bell_coef import bell_coef
from ntp.common import T_

from typing import List, Tuple, Dict

@torch.jit.script
def multi_deriv_forward(
    intermediates: torch.Tensor,
    xs: torch.Tensor,
    x:torch.Tensor,
    out_dim: int,
    bell_coef:Dict[int, List[Tuple[float, List[int]]]],
    n_derivs:int=0,
) -> torch.Tensor:
    new_xs = torch.zeros((n_derivs + 1, x.shape[0], out_dim), device=x.device, dtype=x.dtype)
    new_xs[0] = intermediates[0]

    for i in range(1, n_derivs + 1):
        for coef, exponents in bell_coef[i]:
            q = 0
            new_term = torch.ones_like(xs[0])
            for m, j in enumerate(exponents):
                if j != 0:
                    new_term = new_term * torch.pow(xs[m + 1], j)
                    q += j
            new_xs[i] = new_xs[i] + coef * new_term * intermediates[q]

    return new_xs


class ParameterWrapper(nn.Module):
    def __init__(self, 
                 base_model: nn.Module, 
                 l_width: int, 
                 activation: nn.Module, 
                 l_bound: float=1/3,
                 u_bound: float=1.):
        super(ParameterWrapper, self).__init__()

        layers = [
            nn.Linear(1, l_width),
            activation(),
            nn.Linear(l_width, 1),
            activation()
        ]

        self.stack = nn.Sequential(*layers)
        self.base_model = base_model

        self.l_bound = l_bound
        self.u_bound = u_bound
        self.zero = T_(torch.tensor(0.))

    def forward(self, x: torch.Tensor, **kwargs):
        r = (self.u_bound - self.l_bound) / 2 * (self.stack(self.zero) + 1) + self.l_bound
        return self.base_model(x, **kwargs), r

class TanhDiff:
    @staticmethod
    @torch.jit.script
    def eval(
        x: torch.Tensor,
        n_derivs: int=0,
    ) -> torch.Tensor:
        tanh = torch.tanh(x)
        sech2 = 1 - tanh**2

        out = torch.zeros((n_derivs + 1, x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        out[0] = tanh

        for i in range(1, n_derivs + 1):
            if i == 1:
                out[i] = sech2
            elif i == 2:
                out[i] = - 2 * tanh * sech2
            elif i == 3:
                out[i] = 4 * tanh**2 * sech2 - 2 * sech2**2
            elif i == 4:
                out[i] = - 8 * tanh**3 * sech2 + 16 * tanh * sech2**2
            elif i == 5:
                out[i] = 16 * tanh**4 * sech2 - 88 * tanh**2 * sech2**2 + 16 * sech2**3
            elif i == 6:
                out[i] = - 32 * tanh**5 * sech2 + 416 * tanh**3 * sech2**2 - 272 * tanh * sech2**3
            elif i == 7:
                out[i] = 64 * tanh**6 * sech2 - 1824 * tanh**4 * sech2**2 + 2880 * tanh**2 * sech2**3 - 272 * sech2**4
            elif i == 8:
                out[i] = - 128 * tanh**7 * sech2 + 7680 * tanh**5 * sech2**2 - 24576 * tanh**3 * sech2**3 + 7936 * tanh * sech2**4
            elif i == 9:
                out[i] = 256 * tanh**8 * sech2 - 31616 * tanh**6 * sech2**2 + 185856 * tanh**4 * sech2**3 - 137216 * tanh**2 * sech2**4 + 7936 * sech2**5
            elif i == 10:
                out[i] = - 512 * tanh**9 * sech2 + 128512 * tanh**7 * sech2**2 - 1304832 * tanh**5 * sech2**3 + 1841152 * tanh**3 * sech2**4 - 353792 * tanh * sech2**5
            else:
                raise ValueError(f"Derivative {i} not implemented")
            
        return out


class DenseNet(nn.Module):
    def __init__(
        self,
        n_layers: int,
        l_width: int,
        activation: nn.Module=TanhDiff,
        in_dim: int=1,
        scale: float=1.
    ) -> None:
        super(DenseNet, self).__init__()

        layers = [
            nn.Linear(in_dim, l_width),
        ]

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(l_width, l_width))

        layers.append(nn.Linear(l_width, 1))

        self.stack = nn.Sequential(*layers)
        self.scale = scale

        self.a = activation

    def forward(
        self, 
        x: torch.Tensor, 
        n_derivs: int=0
    ) -> torch.Tensor:
        if n_derivs == 0:
            x = self.stack[0](x)
            for _, layer in enumerate(self.stack[1:]):
                x = layer(self.a.eval(x=x, n_derivs=0)[0])
            return (self.scale * x).unsqueeze(0)

        input_derivs = [x]
        if n_derivs >= 1:
            input_derivs.append(torch.ones_like(x))
        for i in range(2, n_derivs + 1):
            input_derivs.append(torch.zeros_like(x))

        
        xs = torch.zeros((n_derivs + 1, x.shape[0], self.stack[0].weight.shape[-2]), device=x.device, dtype=x.dtype)
        xs[0] = self.stack[0](input_derivs[0])
        xs[1] = F.linear(input_derivs[1], self.stack[0].weight)
        for i in range(2, n_derivs + 1):
            xs[i] = F.linear(input_derivs[i], self.stack[0].weight)

        for layer in self.stack[1:]:
            intermediates = self.a.eval(x=xs[0], n_derivs=n_derivs)
            xs = multi_deriv_forward(intermediates, xs, x, layer.weight.shape[-1], bell_coef, n_derivs=n_derivs)

            if layer.weight.shape[-1] != layer.weight.shape[-2]:
                xs_out = torch.zeros((n_derivs + 1, x.shape[0], layer.weight.shape[-1]), device=x.device, dtype=x.dtype)
            else:
                xs_out = xs

            xs_out = F.linear(xs.clone(), layer.weight)
            xs_out[0, ...] = xs_out[0, ...] + layer.bias

            xs = xs_out

        return self.scale * xs