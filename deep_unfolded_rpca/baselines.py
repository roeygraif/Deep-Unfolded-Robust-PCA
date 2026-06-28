"""Non-learned baselines: truncated SVD and classical Robust PCA."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch

from .operators import soft_threshold, svt


def _svd_on_safe_device(D: torch.Tensor):
    """Run SVD on CPU when on MPS, then return results on the original device."""
    original_device = D.device
    work = D.cpu() if original_device.type == "mps" else D
    U, S, Vh = torch.linalg.svd(work, full_matrices=False)
    return U, S, Vh, original_device


def truncated_svd_baseline(
    D: torch.Tensor, rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Best rank-r approximation of D as L_hat; S_hat = D - L_hat.

    Batched: ``D`` has shape ``(B, n, n)``.
    """
    U, S, Vh, original_device = _svd_on_safe_device(D)
    U_r = U[..., :, :rank]
    S_r = S[..., :rank]
    Vh_r = Vh[..., :rank, :]
    L_hat = U_r @ torch.diag_embed(S_r) @ Vh_r
    if L_hat.device != original_device:
        L_hat = L_hat.to(original_device)
    S_hat = D - L_hat
    return L_hat, S_hat


def classical_rpca(
    D: torch.Tensor,
    alpha: float = 1.0,
    tau_L: Optional[float] = None,
    tau_S: float = 0.1,
    n_iter: int = 200,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Proximal-gradient classical Robust PCA.

    Iterates::

        R       = L + S - D
        L_next  = SVT(L - alpha * R, tau_L)
        S_next  = soft_threshold(S - alpha * R, tau_S)

    Default ``tau_L = 1/sqrt(n)`` is the standard nuclear-norm regularizer
    scale. Runs under ``torch.no_grad`` since this baseline is non-learned.
    """
    if tau_L is None:
        n = D.shape[-1]
        tau_L = 1.0 / math.sqrt(n)

    L = torch.zeros_like(D)
    S = torch.zeros_like(D)

    with torch.no_grad():
        for _ in range(n_iter):
            R = L + S - D
            L = svt(L - alpha * R, tau_L)
            S = soft_threshold(S - alpha * R, tau_S)
    return L, S


def tune_classical_rpca(
    D: torch.Tensor,
    L_true: torch.Tensor,
    S_true: torch.Tensor,
    alpha: float = 1.0,
    n_iter: int = 200,
    tau_L_grid: Optional[List[float]] = None,
    tau_S_grid: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Small grid search over ``(tau_L, tau_S)`` on a validation batch.

    Returns the pair that minimises ``L_rel_err + S_rel_err``. Used so the
    classical baseline is honestly tuned rather than strawmanned.
    """
    from .metrics import relative_frobenius_error

    n = D.shape[-1]
    if tau_L_grid is None:
        tau_L_grid = [
            0.5 / math.sqrt(n),
            1.0 / math.sqrt(n),
            2.0 / math.sqrt(n),
            4.0 / math.sqrt(n),
        ]
    if tau_S_grid is None:
        tau_S_grid = [0.05, 0.1, 0.2, 0.3]

    best: Optional[Dict[str, float]] = None
    for tL in tau_L_grid:
        for tS in tau_S_grid:
            L_hat, S_hat = classical_rpca(
                D, alpha=alpha, tau_L=tL, tau_S=tS, n_iter=n_iter
            )
            err = (
                relative_frobenius_error(L_hat, L_true).item()
                + relative_frobenius_error(S_hat, S_true).item()
            )
            if best is None or err < best["score"]:
                best = {"tau_L": float(tL), "tau_S": float(tS), "score": float(err)}
    assert best is not None
    return best
