"""Experiment 1 (main): train DeepUnfoldedRPCA and benchmark all 3 methods.

Settings: n=32, rank=4, sparsity=5%, sigma=0.05.
Outputs: training_history.json, model_best.pt, results.csv (overwritten),
         classical_params.json, and plots 1-5.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_unfolded_rpca import (
    DeepUnfoldedRPCA,
    RPCADataset,
    classical_rpca,
    evaluate_method,
    get_device,
    set_seed,
    train_model,
    truncated_svd_baseline,
    tune_classical_rpca,
    tune_f1_threshold,
)
from deep_unfolded_rpca.plots import plot_method_bar, plot_training_curves


# ---- configuration -------------------------------------------------------
# Values below were chosen by an iterative dev sweep (see outputs/iter*_log.txt
# in earlier runs): K=10 layers and beta=2.5 hit the sweet spot of L vs S
# accuracy on n=32; cosine LR + 80 epochs was needed for the loss to plateau.
SEED = 42
N = 32
RANK = 4
SPARSITY = 0.05
SIGMA = 0.05
K_LAYERS = 10
BATCH_SIZE = 32
N_EPOCHS = 80
LR = 3e-3
BETA = 2.5
N_TRAIN = 1500
N_VAL = 200
N_TEST = 300
N_ITER_CLASSICAL = 200
# --------------------------------------------------------------------------


def main() -> None:
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)

    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    print("Generating datasets...")
    train_set = RPCADataset(N_TRAIN, N, RANK, SPARSITY, SIGMA, seed=SEED)
    val_set = RPCADataset(N_VAL, N, RANK, SPARSITY, SIGMA, seed=SEED + 1)
    test_set = RPCADataset(N_TEST, N, RANK, SPARSITY, SIGMA, seed=SEED + 2)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepUnfoldedRPCA(n_layers=K_LAYERS, n=N).to(device)

    print(f"Training {K_LAYERS}-layer DeepUnfoldedRPCA for {N_EPOCHS} epochs...")
    t0 = time.perf_counter()
    history, _ = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=N_EPOCHS,
        lr=LR,
        beta=BETA,
        device=device,
    )
    train_time = time.perf_counter() - t0
    print(f"Training done in {train_time:.1f}s")

    torch.save(model.state_dict(), outputs / "model_best.pt")
    with open(outputs / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, outputs / "plot_1_training_curves.png")

    print("Tuning classical RPCA on validation set...")
    val_D = val_set.D.to(device)
    val_L = val_set.L.to(device)
    val_S = val_set.S.to(device)
    best = tune_classical_rpca(val_D, val_L, val_S, n_iter=N_ITER_CLASSICAL)
    print(f"  best classical: tau_L={best['tau_L']:.4f}, tau_S={best['tau_S']:.4f}")

    def svd_method(D):
        return truncated_svd_baseline(D, rank=RANK)

    def classical_method(D):
        return classical_rpca(
            D,
            alpha=1.0,
            tau_L=best["tau_L"],
            tau_S=best["tau_S"],
            n_iter=N_ITER_CLASSICAL,
        )

    model.eval()

    def unfolded_method(D):
        with torch.no_grad():
            return model(D)

    methods = [
        ("Truncated SVD", svd_method),
        ("Classical RPCA", classical_method),
        ("Deep-unfolded RPCA", unfolded_method),
    ]

    print("Tuning F1 threshold per method on validation set...")
    thresholds = {name: tune_f1_threshold(fn, val_D, val_S) for name, fn in methods}
    for name, t in thresholds.items():
        print(f"  {name}: threshold={t}")

    with open(outputs / "classical_params.json", "w") as f:
        json.dump(
            {
                "tau_L": best["tau_L"],
                "tau_S": best["tau_S"],
                "thresholds": thresholds,
            },
            f,
        )

    print("Evaluating on test set...")
    results = []
    for name, fn in methods:
        r = evaluate_method(
            name, fn, test_set, device=device, threshold=thresholds[name]
        )
        r["experiment"] = "exp1_main"
        r["sigma"] = SIGMA
        r["sparsity"] = SPARSITY
        r["threshold"] = thresholds[name]
        results.append(r)
        print(
            f"  {name}: L_err={r['L_rel_err']:.4f}, S_err={r['S_rel_err']:.4f}, "
            f"F1={r['f1']:.4f}, t/sample={r['runtime_per_sample_s']*1000:.2f}ms"
        )

    df = pd.DataFrame(results)
    df.to_csv(outputs / "results.csv", index=False)

    names = [r["method"] for r in results]
    plot_method_bar(
        names,
        [r["L_rel_err"] for r in results],
        "L relative Frobenius error",
        "Low-rank recovery error",
        outputs / "plot_2_L_error.png",
    )
    plot_method_bar(
        names,
        [r["S_rel_err"] for r in results],
        "S relative Frobenius error",
        "Sparse recovery error",
        outputs / "plot_3_S_error.png",
    )
    plot_method_bar(
        names,
        [r["f1"] for r in results],
        "F1 score",
        "Sparse support F1",
        outputs / "plot_4_f1.png",
    )
    plot_method_bar(
        names,
        [r["runtime_per_sample_s"] * 1000 for r in results],
        "Runtime per sample (ms)",
        "Inference runtime (log scale)",
        outputs / "plot_5_runtime.png",
        log_y=True,
        value_fmt="{:.2f}",
    )

    print(f"Results saved to {outputs / 'results.csv'}")
    print("Plots 1-5 saved.")


if __name__ == "__main__":
    main()
