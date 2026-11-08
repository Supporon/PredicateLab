"""R06: independent numerical checks for the audited LUSI quadratic loss.

Reference oracle (does not call WeightedMSELoss):
    L = mean(e * ((tau_hat I + tau P) @ e))
    dL/de = 2/n (tau_hat I + tau P) @ e, when P is fixed from X.

Covers finite-difference / autograd gradients, a constant descriptor (P=0),
rank-one Phi_ones, and a random DSL descriptor. It also tests the batch-row
permutation equivariance required of a per-sample predicate; phi_cycle violates
that property because it treats shuffled batch position as time.

Usage: python -I r06_lusi_math.py <src_copy>
"""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(src))
sys.path.insert(0, str(src / "TSIL-main"))

import torch
from predicatelab.domains.tsil.dsl import compile_descriptor
from src.model_backbone_phitp import WeightedMSELoss
from src.temporal_predicates import get_temporal_I_P


def manual(out, target, p, tau_hat):
    e = (out - target).reshape(-1, 1)
    ident = torch.eye(len(e), dtype=e.dtype)
    m = tau_hat * ident + (1.0 - tau_hat) * p
    return torch.mean(e * (m @ e)), 2.0 / len(e) * (m @ e)


def finite_diff(fn, x, eps=1e-4):
    ans = torch.empty_like(x)
    with torch.no_grad():
        for i in range(x.numel()):
            a, b = x.clone(), x.clone()
            a[i] += eps; b[i] -= eps
            ans[i] = (fn(a) - fn(b)) / (2 * eps)
    return ans

# Actual loss vs independently calculated quadratic form / derivative
B, T, F = 9, 20, 8
torch.manual_seed(734)
x = torch.randn(B, T, F)
target = torch.randn(B)
out = torch.randn(B, requires_grad=True)
tauhat = torch.tensor(0.5)
desc = {"family": "volatility", "params": {"window": 10}}
fn = compile_descriptor(desc, seq_len=T, feature_dim=F)
I, P = fn(x)
actual = WeightedMSELoss(phi_fn=fn)("external", x, None, out, target, tauhat, 1-tauhat)["total"]
ref, grad_ref = manual(out, target, P, tauhat)
actual.backward()
grad_auto = out.grad.detach().clone()
grad_fd = finite_diff(lambda z: manual(z, target, P, tauhat)[0], out.detach())

# Degenerate constant DSL descriptor must generate P=0, not Phi_ones
z = torch.ones(B, T, F)
_, p_zero = fn(z)
# Phi_ones separately (all entries 1/B)
_, p_ones = WeightedMSELoss().get_ones_I_P(B, torch.device("cpu"))

# Required equivariance of a relation induced by sample values, versus phi_cycle.
# Choose a batch with a nonzero batch-FFT frequency: a generic random batch
# often yields period=1, which makes the position codes constant by coincidence.
cycle_x = torch.stack([torch.sin(torch.linspace(0, 2 * torch.pi * 4, T)).unsqueeze(1).repeat(1, F)
                       + 0.01 * j for j in range(B)])
permutation = torch.tensor([4, 1, 8, 0, 7, 2, 6, 3, 5])
_, p_cycle = get_temporal_I_P("phi_cycle", cycle_x)
_, p_cycle_perm = get_temporal_I_P("phi_cycle", cycle_x[permutation])
expected = p_cycle[permutation][:, permutation]
_, p_vol = get_temporal_I_P("phi_volatility", x)
_, p_vol_perm = get_temporal_I_P("phi_volatility", x[permutation])

result = {
  "quadratic_loss": {
    "actual": float(actual.detach()), "independent_reference": float(ref.detach()),
    "absolute_difference": float(abs(actual.detach()-ref.detach())),
    "max_abs_autograd_vs_reference_gradient": float((grad_auto-grad_ref.reshape(-1)).abs().max()),
    "max_abs_finite_difference_vs_reference_gradient": float((grad_fd-grad_ref.reshape(-1)).abs().max()),
  },
  "degeneracy": {
    "constant_dsl_P_max_abs": float(p_zero.abs().max()),
    "ones_control_P_diag": float(torch.diagonal(p_ones).mean()),
    "ones_control_P_offdiag": float(p_ones[0,1]),
  },
  "row_permutation_equivariance": {
    "phi_cycle_max_abs_difference": float((p_cycle_perm-expected).abs().max()),
    "phi_volatility_max_abs_difference": float((p_vol_perm-p_vol[permutation][:, permutation]).abs().max()),
    "criterion": "0 (within float tolerance) is required for a predicate based only on sample values, independent of arbitrary batch order",
  },
}
print(json.dumps(result, indent=2))
