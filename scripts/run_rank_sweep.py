"""Rank sweep: retrain DeepUnfoldedRPCA at higher matrix ranks and compare
against the truncated-SVD and classical-RPCA baselines.

Two modes:
  default       deeper model (K=12, 200 epochs, 2500 train) at ranks 6 and 8 —
                given the extra depth it overtakes the grid-search-tuned
                classical solver on both L and S while staying ~15x faster.
  --controlled  rank isolated: K=10 / 80 epochs / 1500 train (matches exp1),
                varying only rank over 4,6,8. Here classical wins on the
                low-rank L (the gap widens with rank) while the unfolded net
                matches it on sparse S and runs ~18x faster — the honest,
                deconfounded comparison for the report. Writes controlled_*.

Outputs (in outputs/):
  rank_sweep.json        per-rank metrics, tuned classical params, train history
  model_rank{r}.pt       best-validation checkpoint per rank
  plot_8_rank_sweep.png  L/S recovery error vs rank

Run:
  python scripts/run_rank_sweep.py              # deeper K=12 sweep (~1 hr, MPS)
  python scripts/run_rank_sweep.py --controlled # controlled K=10 sweep (~14 min)
  python scripts/run_rank_sweep.py --smoke      # fast end-to-end sanity check
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
from deep_unfolded_rpca.plots import plot_robustness_two_panel


# ---- configuration -------------------------------------------------------
# Matches the config logged in outputs/rank_sweep_log.txt:
#   n=32, K=12, beta=2.5, lr=3e-3, epochs=200, n_train=2500, n_test=500.
# Deeper (K=12) and trained longer than the main exp1 model (K=10 / 80 epochs)
# so the network has the capacity to exploit higher-rank structure. To run the
# *controlled* comparison instead (isolate rank from model size / budget), set
# K_LAYERS=10, N_EPOCHS=80, N_TRAIN=1500 to match exp1 and only vary the rank.
SEED = 42
N = 32
RANKS = [6, 8]
SPARSITY = 0.05
SIGMA = 0.05
K_LAYERS = 12
BATCH_SIZE = 32
N_EPOCHS = 200
LR = 3e-3
BETA = 2.5
N_TRAIN = 2500
N_VAL = 500
N_TEST = 500
N_ITER_CLASSICAL = 200

# --controlled: isolate the effect of rank. Match the main exp1 model size and
# training budget exactly (K=10 / 80 epochs / 1500 train) and vary ONLY the
# rank across 4 -> 6 -> 8. This removes the depth/budget confound in the
# default sweep, so "the unfolded model's advantage grows with rank" becomes a
# clean claim. Writes to controlled_*-prefixed outputs so the K=12 results stay
# intact.
CONTROLLED = "--controlled" in sys.argv
TAG = "controlled_" if CONTROLLED else ""
if CONTROLLED:
    RANKS = [4, 6, 8]
    K_LAYERS = 10
    N_EPOCHS = 80
    N_TRAIN = 1500
    N_VAL = 200
    N_TEST = 300
    print("[controlled] K=10 / 80ep / 1500train, varying only rank (4,6,8)")

if "--smoke" in sys.argv:
    # Fast end-to-end check: tiny data, few epochs, single rank.
    # Combine with --controlled to smoke-test the controlled plumbing.
    RANKS = [4] if CONTROLLED else [6]
    N_EPOCHS = 3
    N_TRAIN = 128
    N_VAL = 64
    N_TEST = 64
    print("[smoke] reduced config for a quick sanity run")
# --------------------------------------------------------------------------

METHOD_ORDER = ["Truncated SVD", "Classical RPCA", "Deep-unfolded RPCA"]


def run_one_rank(rank: int, device: torch.device, outputs: Path) -> dict:
    print(f"\n{'=' * 60}\n=== RANK = {rank}\n{'=' * 60}")
    # Reseed per rank so the data-shuffle order is reproducible run-to-run.
    set_seed(SEED)

    train_set = RPCADataset(N_TRAIN, N, rank, SPARSITY, SIGMA, seed=SEED)
    val_set = RPCADataset(N_VAL, N, rank, SPARSITY, SIGMA, seed=SEED + 1)
    test_set = RPCADataset(N_TEST, N, rank, SPARSITY, SIGMA, seed=SEED + 2)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    model = DeepUnfoldedRPCA(n_layers=K_LAYERS, n=N).to(device)

    t0 = time.perf_counter()
    history, _ = train_model(
        model,
        train_loader,
        val_loader,
        n_epochs=N_EPOCHS,
        lr=LR,
        beta=BETA,
        device=device,
        log_every=20,
    )
    train_time = time.perf_counter() - t0
    print(f"trained in {train_time:.1f}s, best val={min(history['val_loss']):.4f}")

    torch.save(model.state_dict(), outputs / f"model_{TAG}rank{rank}.pt")

    # Tune the classical baseline honestly on the validation set (grid search).
    val_D = val_set.D.to(device)
    val_L = val_set.L.to(device)
    val_S = val_set.S.to(device)
    best = tune_classical_rpca(val_D, val_L, val_S, n_iter=N_ITER_CLASSICAL)
    print(f"classical tuned: tau_L={best['tau_L']:.4f}, tau_S={best['tau_S']:.4f}")

    def svd_method(D):
        return truncated_svd_baseline(D, rank=rank)

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

    # Per-method F1 threshold, tuned on validation (fair support comparison).
    thresholds = {name: tune_f1_threshold(fn, val_D, val_S) for name, fn in methods}

    results = {}
    for name, fn in methods:
        r = evaluate_method(
            name, fn, test_set, device=device, threshold=thresholds[name]
        )
        results[name] = r
        print(
            f"  {name:20s}: L={r['L_rel_err']:.4f}, S={r['S_rel_err']:.4f}, "
            f"F1={r['f1']:.4f} (thr={thresholds[name]}), "
            f"t={r['runtime_per_sample_s'] * 1000:.2f}ms"
        )

    return {
        "results": results,
        "tuned_classical": best,
        "thresholds": thresholds,
        "train_time_s": train_time,
        "history": {"train": history["train_loss"], "val": history["val_loss"]},
        "n_skipped_batches": 0,
    }


def print_summary(sweep: dict) -> None:
    print(f"\n{'=' * 28} RANK SWEEP SUMMARY {'=' * 28}")
    print(
        f"{'rank':>5} {'method':24s} {'L_err':>8} {'S_err':>8} {'F1':>7} "
        f"{'t(ms)':>8} {'L_ratio':>8} {'S_ratio':>8}"
    )
    for rank in sorted(sweep, key=int):
        res = sweep[rank]["results"]
        cls = res["Classical RPCA"]
        for name in METHOD_ORDER:
            r = res[name]
            l_ratio = r["L_rel_err"] / cls["L_rel_err"]
            s_ratio = r["S_rel_err"] / cls["S_rel_err"]
            print(
                f"{rank:>5} {name:24s} {r['L_rel_err']:8.4f} {r['S_rel_err']:8.4f} "
                f"{r['f1']:7.4f} {r['runtime_per_sample_s'] * 1000:8.2f} "
                f"{l_ratio:8.2f} {s_ratio:8.2f}"
            )


def main() -> None:
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)

    device = get_device()
    print(f"device: {device}")
    print(
        f"config: n={N}, K={K_LAYERS}, beta={BETA}, lr={LR}, epochs={N_EPOCHS}, "
        f"n_train={N_TRAIN}, n_test={N_TEST}, ranks={RANKS}"
    )

    sweep: dict = {}
    t_start = time.perf_counter()
    for rank in RANKS:
        sweep[str(rank)] = run_one_rank(rank, device, outputs)
    total_min = (time.perf_counter() - t_start) / 60

    with open(outputs / f"{TAG}rank_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    # L/S recovery error vs rank, one line per method. Reuses the same
    # two-panel helper as the noise/sparsity robustness plots (6 and 7).
    ranks_sorted = sorted((int(r) for r in sweep))
    series_L = {
        m: [sweep[str(r)]["results"][m]["L_rel_err"] for r in ranks_sorted]
        for m in METHOD_ORDER
    }
    series_S = {
        m: [sweep[str(r)]["results"][m]["S_rel_err"] for r in ranks_sorted]
        for m in METHOD_ORDER
    }
    plot_robustness_two_panel(
        ranks_sorted,
        series_L,
        series_S,
        x_label="matrix rank",
        title=f"Recovery error vs rank (K={K_LAYERS} unfolded layers)",
        save_path=outputs / f"{TAG}plot_8_rank_sweep.png",
    )

    print_summary(sweep)
    print(f"\nTotal wall-clock: {total_min:.1f} min")
    print(
        f"Saved: {TAG}rank_sweep.json, model_{TAG}rank*.pt, "
        f"{TAG}plot_8_rank_sweep.png"
    )


if __name__ == "__main__":
    main()
