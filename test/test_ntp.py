import torch

from ntp.models import DenseNet, TanhDiff
from ntp.common import autodiff, T_

def test_tanh_diff():
    # We need torch.float64 to maintain desired precision past 9 derivatives
    n_derivs = 10
    X = T_(torch.randn(100, 1, dtype=torch.float64))

    tanh_out = TanhDiff.eval(X, n_derivs=n_derivs)

    tanh_out_torch = [torch.tanh(X)]
    for _ in range(n_derivs):
        tanh_out_torch.append(autodiff(tanh_out_torch[-1], X))

    tanh_out_torch = torch.stack(tanh_out_torch, dim=0)

    assert torch.allclose(tanh_out, tanh_out_torch, atol=1e-5), f"max abs diff: {torch.max(torch.abs(tanh_out - tanh_out_torch))}"

def test_derivatives_match():
    X = T_(torch.randn(100, 1, dtype=torch.float64))
    n_derivs = 3

    model = DenseNet(n_derivs, 10, TanhDiff).to(torch.float64)

    ntp_u = model(X, n_derivs=n_derivs)
    ad_u = model(X, n_derivs=0)

    assert ntp_u.shape == torch.Size((n_derivs + 1, 100, 1))
    assert ad_u.shape == torch.Size((1, 100, 1))

    assert torch.allclose(ntp_u[0, ...], ad_u, atol=1e-5), f"zero deriv max abs diff: {torch.max(torch.abs(ntp_u[0, ...] - ad_u))}"

    ad_u_tensor = torch.zeros((n_derivs + 1, 100, 1), dtype=torch.float64)
    ad_u_tensor[0, ...] = ad_u

    for i in range(1, n_derivs + 1):
        ad_u_tensor[i, ...] = autodiff(ad_u_tensor[i - 1, ...], X)

    assert torch.allclose(ntp_u, ad_u_tensor, atol=1e-5), f"all derivs max abs diff: {torch.max(torch.abs(ntp_u - ad_u_tensor))}"