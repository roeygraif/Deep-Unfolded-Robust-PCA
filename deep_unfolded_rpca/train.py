"""Training and evaluation loops."""
from __future__ import annotations

import time
from copy import deepcopy
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .metrics import relative_frobenius_error


def _device_sync(device: torch.device) -> None:
    """Synchronize the active device, so ``time.perf_counter`` is meaningful."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _supervised_loss(
    L_hat: torch.Tensor,
    S_hat: torch.Tensor,
    L_true: torch.Tensor,
    S_true: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    return F.mse_loss(L_hat, L_true) + beta * F.mse_loss(S_hat, S_true)


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int,
    lr: float,
    beta: float,
    device: torch.device,
    log_every: int = 5,
    cosine_lr: bool = True,
    lr_min_ratio: float = 1.0 / 30.0,
) -> Tuple[Dict[str, list], Dict[str, torch.Tensor]]:
    """Train ``model`` with Adam. Returns ``(history, best_state_dict)``.

    Loss = ``MSE(L_hat, L) + beta * MSE(S_hat, S)``. The model is restored to
    the best validation-loss checkpoint before returning. With ``cosine_lr``
    the learning rate is annealed from ``lr`` to ``lr * lr_min_ratio``.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * lr_min_ratio
        )
        if cosine_lr
        else None
    )

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            D = batch["D"].to(device)
            L_true = batch["L"].to(device)
            S_true = batch["S"].to(device)

            L_hat, S_hat = model(D)
            loss = _supervised_loss(L_hat, S_hat, L_true, S_true, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        train_loss = sum(train_losses) / len(train_losses)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                D = batch["D"].to(device)
                L_true = batch["L"].to(device)
                S_true = batch["S"].to(device)
                L_hat, S_hat = model(D)
                val_loss = _supervised_loss(L_hat, S_hat, L_true, S_true, beta)
                val_losses.append(val_loss.item())
        val_loss = sum(val_losses) / len(val_losses)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())

        if epoch == 1 or epoch == n_epochs or epoch % log_every == 0:
            print(
                f"  epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f}"
            )

    model.load_state_dict(best_state)
    return history, best_state


def evaluate_method(
    method_name: str,
    method_callable: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    dataset: Dataset,
    device: torch.device,
    threshold: float = 0.1,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate any decomposition method uniformly.

    ``method_callable(D) -> (L_hat, S_hat)``. Reports per-sample mean L and S
    relative Frobenius error, sparse-support precision/recall/F1, and runtime
    per sample. Used identically for SVD, classical RPCA, and the unfolded
    network so the comparison is apples-to-apples.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    L_err_weighted_sum = 0.0
    S_err_weighted_sum = 0.0
    f1_tp = 0
    f1_fp = 0
    f1_fn = 0
    total_time = 0.0
    n_samples_total = 0

    for batch in loader:
        D = batch["D"].to(device)
        L_true = batch["L"].to(device)
        S_true = batch["S"].to(device)
        bsz = D.shape[0]

        _device_sync(device)
        t0 = time.perf_counter()
        L_hat, S_hat = method_callable(D)
        _device_sync(device)
        elapsed = time.perf_counter() - t0

        total_time += elapsed
        n_samples_total += bsz

        L_err_weighted_sum += relative_frobenius_error(L_hat, L_true).item() * bsz
        S_err_weighted_sum += relative_frobenius_error(S_hat, S_true).item() * bsz

        pred = S_hat.abs() > threshold
        true = S_true != 0
        f1_tp += int((pred & true).sum().item())
        f1_fp += int((pred & ~true).sum().item())
        f1_fn += int((~pred & true).sum().item())

    L_err = L_err_weighted_sum / n_samples_total
    S_err = S_err_weighted_sum / n_samples_total
    precision = f1_tp / max(f1_tp + f1_fp, 1)
    recall = f1_tp / max(f1_tp + f1_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "method": method_name,
        "L_rel_err": L_err,
        "S_rel_err": S_err,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "runtime_per_sample_s": total_time / n_samples_total,
        "total_time_s": total_time,
        "n_samples": n_samples_total,
    }
