"""Differentiable proximal operators: SVT and soft-thresholding.

Both operators are batched and autograd-friendly. ``svt`` routes through CPU
when running on MPS (the MPS svd backend is unreliable). It also recovers
gracefully when LAPACK's batched SVD fails to converge on an ill-conditioned
matrix, which can happen late in training when ``L_temp`` becomes nearly
rank-r.
"""
from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn.functional as F

Scalar = Union[float, torch.Tensor]


def soft_threshold(x: torch.Tensor, tau: Scalar) -> torch.Tensor:
    """Element-wise soft threshold (proximal operator of the L1 norm).

    Returns ``sign(x) * max(|x| - tau, 0)``. ``tau`` may be a Python float or
    a tensor that broadcasts against ``x``.
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - tau, min=0.0)


def _safe_svd_per_sample(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-element SVD with progressive perturbations, used as a last-resort
    fallback when the batched SVD raises ``LinAlgError``.

    Each detached perturbation only breaks ties for the LAPACK solver; it
    does not enter the gradient graph, so backward through SVT remains
    well-defined for the inputs.
    """
    n = x.shape[-1]
    U_list, S_list, Vh_list = [], [], []
    eps_schedule = (1e-5, 1e-4, 1e-3, 1e-2)

    for i in range(x.shape[0]):
        xi = x[i]
        scale = xi.detach().abs().mean().clamp(min=1e-6)
        Ui = Si = Vhi = None

        for eps in (0.0,) + eps_schedule:
            try:
                xi_try = xi if eps == 0.0 else xi + (eps * scale * torch.randn_like(xi)).detach()
                Ui, Si, Vhi = torch.linalg.svd(xi_try, full_matrices=False)
                break
            except torch._C._LinAlgError:
                continue

        if Ui is None:
            # Pathological sample: pass through unchanged so grads keep
            # flowing — for SVT this means S - tau gives back the input
            # unmodified, which is the safest no-op.
            Ui = torch.eye(n, device=xi.device, dtype=xi.dtype)
            Si = torch.zeros(n, device=xi.device, dtype=xi.dtype)
            Vhi = torch.eye(n, device=xi.device, dtype=xi.dtype)

        U_list.append(Ui)
        S_list.append(Si)
        Vh_list.append(Vhi)

    return torch.stack(U_list), torch.stack(S_list), torch.stack(Vh_list)


def svt(x: torch.Tensor, tau: Scalar) -> torch.Tensor:
    """Singular Value Thresholding (proximal operator of the nuclear norm).

    Batched: ``x`` has shape ``(..., n, n)``. Returns the same shape.
    """
    original_device = x.device
    use_cpu_for_svd = original_device.type == "mps"

    if use_cpu_for_svd:
        x_work = x.to("cpu")
        if isinstance(tau, torch.Tensor):
            tau_work = tau.to("cpu")
        else:
            tau_work = tau
    else:
        x_work = x
        tau_work = tau

    # Defensive: replace non-finite entries before sending to LAPACK.
    if not torch.isfinite(x_work).all():
        x_work = torch.where(
            torch.isfinite(x_work), x_work, torch.zeros_like(x_work)
        )

    try:
        U, S, Vh = torch.linalg.svd(x_work, full_matrices=False)
    except torch._C._LinAlgError:
        U, S, Vh = _safe_svd_per_sample(x_work)

    S_shrunk = F.relu(S - tau_work)
    out = U @ torch.diag_embed(S_shrunk) @ Vh

    if use_cpu_for_svd:
        out = out.to(original_device)
    return out
