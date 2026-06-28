"""Evaluation metrics for L/S recovery."""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import torch


def relative_frobenius_error(
    X_hat: torch.Tensor, X: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    """Per-sample relative Frobenius error, returned as a scalar mean.

    ``X_hat``, ``X`` have shape ``(B, n, n)``.
    """
    num = torch.linalg.norm(X_hat - X, dim=(-2, -1))
    den = torch.linalg.norm(X, dim=(-2, -1)).clamp_min(eps)
    return (num / den).mean()


def sparse_support_f1(
    S_hat: torch.Tensor,
    S: torch.Tensor,
    threshold: float = 0.1,
) -> Dict[str, float]:
    """Precision, recall, F1 on the sparse support over the full batch.

    Predicted support: ``|S_hat| > threshold``. True support: ``S != 0``.
    """
    pred = S_hat.abs() > threshold
    true = S != 0

    tp = int((pred & true).sum().item())
    fp = int((pred & ~true).sum().item())
    fn = int((~pred & true).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}


def tune_f1_threshold(
    method: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    D_val: torch.Tensor,
    S_val: torch.Tensor,
    candidates: List[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5),
) -> float:
    """Pick the |S_hat|-threshold that maximises F1 on a validation batch.

    Different methods produce S_hat with different "softness", so a single
    fixed threshold systematically favours methods with sharper outputs.
    Tuning per method on a held-out batch makes the F1 comparison fair.
    """
    with torch.no_grad():
        _, S_hat = method(D_val)
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = sparse_support_f1(S_hat, S_val, threshold=t)["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return float(best_t)
