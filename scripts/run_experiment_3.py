"""Experiment 3: sparsity robustness across sparsity in {0.02, 0.05, 0.10}.

Loads the model trained in experiment 1 and the tuned classical params, then
evaluates all three methods on a fresh test set per sparsity level. Produces
plot 7 and appends rows to results.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_unfolded_rpca import (
    DeepUnfoldedRPCA,
    RPCADataset,
    classical_rpca,
    evaluate_method,
    get_device,
    set_seed,
    truncated_svd_baseline,
)
from deep_unfolded_rpca.plots import plot_robustness_two_panel


SEED = 42
N = 32
RANK = 4
SIGMA = 0.05
K_LAYERS = 10
SPARSITIES = [0.02, 0.05, 0.10]
N_TEST = 300
N_ITER_CLASSICAL = 200


def main() -> None:
    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)

    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}")

    model = DeepUnfoldedRPCA(n_layers=K_LAYERS, n=N).to(device)
    state = torch.load(outputs / "model_best.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    with open(outputs / "classical_params.json") as f:
        cp = json.load(f)
    thresholds = cp.get("thresholds", {})
    print(f"Classical params: tau_L={cp['tau_L']:.4f}, tau_S={cp['tau_S']:.4f}")
    print(f"Thresholds (from exp1): {thresholds}")

    rows = []
    series_L = {"Truncated SVD": [], "Classical RPCA": [], "Deep-unfolded RPCA": []}
    series_S = {"Truncated SVD": [], "Classical RPCA": [], "Deep-unfolded RPCA": []}

    for sparsity in SPARSITIES:
        print(f"\nsparsity = {sparsity}")
        test_set = RPCADataset(
            N_TEST, N, RANK, sparsity, SIGMA, seed=SEED + 200 + int(sparsity * 1000)
        )

        def svd_method(D):
            return truncated_svd_baseline(D, rank=RANK)

        def classical_method(D):
            return classical_rpca(
                D,
                alpha=1.0,
                tau_L=cp["tau_L"],
                tau_S=cp["tau_S"],
                n_iter=N_ITER_CLASSICAL,
            )

        def unfolded_method(D):
            with torch.no_grad():
                return model(D)

        for name, fn in [
            ("Truncated SVD", svd_method),
            ("Classical RPCA", classical_method),
            ("Deep-unfolded RPCA", unfolded_method),
        ]:
            t = thresholds.get(name, 0.1)
            r = evaluate_method(name, fn, test_set, device=device, threshold=t)
            r["experiment"] = "exp3_sparsity"
            r["sigma"] = SIGMA
            r["sparsity"] = sparsity
            r["threshold"] = t
            rows.append(r)
            series_L[name].append(r["L_rel_err"])
            series_S[name].append(r["S_rel_err"])
            print(
                f"  {name}: L_err={r['L_rel_err']:.4f}, S_err={r['S_rel_err']:.4f}, "
                f"F1={r['f1']:.4f}"
            )

    plot_robustness_two_panel(
        SPARSITIES,
        series_L,
        series_S,
        x_label="Sparsity (fraction of nonzero entries)",
        title="Robustness to sparsity level",
        save_path=outputs / "plot_7_sparsity_robustness.png",
    )

    csv_path = outputs / "results.csv"
    df_new = pd.DataFrame(rows)
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_old = df_old[~df_old["experiment"].astype(str).str.startswith("exp3")]
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(csv_path, index=False)
    print(f"\nAppended {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
