"""Build deep_unfolded_rpca.ipynb — the public, runnable notebook deliverable.

Regenerate with:  python scripts/build_notebook.py

The notebook reproduces the main K=10 comparison *live* (mirrors
run_experiment_1.py exactly: seeds 42/43/44, final-epoch weights, grid-tuned
classical baseline, per-method F1 thresholds) and loads the rank-sweep results
produced by scripts/run_rank_sweep.py. Building it from a script guarantees
valid notebook JSON and keeps the artifact regenerable.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deep_unfolded_rpca.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text: str) -> None:
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------- 0. intro
md(r"""
# Deep-Unfolded Robust PCA

**Model Based Deep Learning (361.2.2320) — course project.**

We take the classical Robust-PCA proximal-gradient solver, **unroll** it into a
small `K`-layer neural network, and *learn* its per-layer step sizes and
shrinkage thresholds end-to-end on synthetic data. This notebook reproduces the
project's headline results:

1. the data model `D = L + S + N` and a visual decomposition,
2. training the `K = 10` unfolded network,
3. the main comparison vs. truncated SVD and a **grid-search-tuned** classical RPCA,
4. how the methods scale with matrix **rank** (a *controlled* comparison),
5. what extra network **depth** (`K = 12`) buys.

Sections 1–4 run live (a few minutes). Sections 5–6 load results produced by
`scripts/run_rank_sweep.py`.
""")

md(r"""
## 0. Setup

Run this first. Locally (from the repo root) it just puts the package on the
path; on Colab it clones the repo and installs the dependencies.
""")

code(r"""
import os, sys, json

# Make the deep_unfolded_rpca package importable; clone the repo on Colab.
if not os.path.isdir("deep_unfolded_rpca"):
    !git clone https://github.com/roeygraif/Deep-Unfolded-Robust-PCA.git
    os.chdir("Deep-Unfolded-Robust-PCA")
    !pip install -q -r requirements.txt
sys.path.insert(0, os.getcwd())

import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from deep_unfolded_rpca import (
    DeepUnfoldedRPCA, RPCADataset, classical_rpca, evaluate_method,
    get_device, set_seed, train_model, truncated_svd_baseline,
    tune_classical_rpca, tune_f1_threshold,
)

device = get_device()
print("device:", device, "| torch:", torch.__version__)
""")

# ---------------------------------------------------------------- 1. data
md(r"""
## 1. The data model

We observe a matrix $D \in \mathbb{R}^{n\times n}$ that is a sum of three parts:

$$ D \;=\; \underbrace{L}_{\text{low-rank}} \;+\; \underbrace{S}_{\text{sparse}} \;+\; \underbrace{N}_{\text{dense noise}} $$

- **L** — low-rank structure (e.g. a static video background),
- **S** — sparse outliers (e.g. moving objects / anomalies),
- **N** — small dense Gaussian noise, $N \sim \mathcal{N}(0,\sigma^2)$.

The task is to recover `L` and `S` from `D`. Below is one synthetic sample
(`n = 32`, rank 4, 5% sparse, σ = 0.05).
""")

code(r"""
set_seed(42)
sample = RPCADataset(n_samples=1, n=32, rank=4, sparsity=0.05, sigma=0.05, seed=0)
D0, L0, S0 = sample.D[0], sample.L[0], sample.S[0]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
for a, M, t in zip(ax, [D0, L0, S0], ["D = observed", "L = low-rank", "S = sparse"]):
    im = a.imshow(M.numpy(), cmap="coolwarm")
    a.set_title(t); a.axis("off")
    fig.colorbar(im, ax=a, fraction=0.046)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 2. method
md(r"""
## 2. Classical Robust PCA, and unrolling it

Classical RPCA solves a convex relaxation
$\min_{L,S}\; \tfrac12\|D-L-S\|_F^2 + \lambda_L\|L\|_* + \lambda_S\|S\|_1$
with **proximal gradient** (~200 iterations):

```
R  =  L + S − D
L  ←  SVT(L − α·R, τ_L)              # singular-value soft-threshold  (nuclear-norm prox)
S  ←  soft_threshold(S − α·R, τ_S)   # entrywise soft-threshold        (ℓ1 prox)
```

`SVT` shrinks each singular value by `τ_L`; `soft_threshold` shrinks each entry
by `τ_S`.

**Deep unfolding** turns each iteration into one network *layer* and makes
$(\alpha_k,\tau_L^k,\tau_S^k)$ **learnable** per layer (kept positive via
softplus). With `K = 10` layers the network is ~20× shallower than the
200-iteration solver, but its parameters are trained on data rather than
grid-searched — the core model-based-deep-learning idea.
""")

# ---------------------------------------------------------------- 3. train
md(r"""
## 3. Train the K = 10 unfolded network

Main setting (identical to `run_experiment_1.py`): `n=32, rank=4, sparsity=5%,
σ=0.05`, 1500 train / 200 val / 300 test samples, 80 epochs, loss
$\mathrm{MSE}(\hat L, L) + \beta\,\mathrm{MSE}(\hat S, S)$ with β = 2.5. Seeded
for reproducibility; expect a few minutes. (Set `QUICK = True` for a ~1-minute
smoke run that will *not* match the reported numbers.)
""")

code(r"""
QUICK = False  # True -> fast smoke run (won't reproduce the report numbers)

set_seed(42)
N, RANK, K = 32, 4, 10
n_train, n_epochs = (256, 15) if QUICK else (1500, 80)

train_set = RPCADataset(n_train, N, RANK, 0.05, 0.05, seed=42)
val_set   = RPCADataset(200,     N, RANK, 0.05, 0.05, seed=43)
test_set  = RPCADataset(300,     N, RANK, 0.05, 0.05, seed=44)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)

model = DeepUnfoldedRPCA(n_layers=K, n=N).to(device)
# `history, _`: like run_experiment_1.py we evaluate the final-epoch weights.
history, _ = train_model(model, train_loader, val_loader,
                         n_epochs=n_epochs, lr=3e-3, beta=2.5, device=device)

plt.figure(figsize=(6, 3.4))
plt.plot(history["train_loss"], label="train")
plt.plot(history["val_loss"], label="val")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log")
plt.legend(); plt.title(f"Training curve (K={K})"); plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 4. compare
md(r"""
## 4. Main comparison

We grid-search-tune the classical baseline on the **validation** set, tune a
per-method F1 threshold (fair support comparison), then evaluate all three
methods on the held-out test set.
""")

code(r"""
val_D, val_L, val_S = val_set.D.to(device), val_set.L.to(device), val_set.S.to(device)
best = tune_classical_rpca(val_D, val_L, val_S, n_iter=200)
print("classical tuned:", {k: round(float(v), 4) for k, v in best.items() if k != 'score'})

def svd_method(D):       return truncated_svd_baseline(D, rank=RANK)
def classical_method(D): return classical_rpca(D, alpha=1.0,
                                               tau_L=best["tau_L"], tau_S=best["tau_S"], n_iter=200)
model.eval()
def unfolded_method(D):
    with torch.no_grad():
        return model(D)

methods = [("Truncated SVD", svd_method),
           ("Classical RPCA", classical_method),
           ("Deep-unfolded RPCA", unfolded_method)]
thresholds = {name: tune_f1_threshold(fn, val_D, val_S) for name, fn in methods}

rows = []
for name, fn in methods:
    r = evaluate_method(name, fn, test_set, device=device, threshold=thresholds[name])
    rows.append({"method": name, "L rel err": r["L_rel_err"], "S rel err": r["S_rel_err"],
                 "F1": r["f1"], "ms/sample": r["runtime_per_sample_s"] * 1000})

df = pd.DataFrame(rows).set_index("method")
df["speedup x"] = (df.loc["Classical RPCA", "ms/sample"] / df["ms/sample"])
df.round({"L rel err": 4, "S rel err": 4, "F1": 4, "ms/sample": 2, "speedup x": 1})
""")

md(r"""
The unfolded network lands within ~1.15× of the tuned classical solver on `L`
and ~1.10× on `S`, with an F1 gap of about a point — while running **~18×
faster** per sample. That is the headline trade-off: ≈10 learned layers approach
the accuracy of ≈200 grid-search-tuned iterations at a fraction of the compute.
""")

md(r"""
### Qualitative recovery on one test sample
""")

code(r"""
D1 = test_set.D[:1].to(device)
L1_true, S1_true = test_set.L[0], test_set.S[0]
with torch.no_grad():
    L_unf, S_unf = model(D1)
L_cls, S_cls = classical_rpca(D1, alpha=1.0, tau_L=best["tau_L"], tau_S=best["tau_S"], n_iter=200)

grid = [
    [(D1[0], "D (observed)"), (L1_true, "L true"), (L_unf[0], "L̂ unfolded"), (L_cls[0], "L̂ classical")],
    [None,                    (S1_true, "S true"), (S_unf[0], "Ŝ unfolded"), (S_cls[0], "Ŝ classical")],
]
fig, ax = plt.subplots(2, 4, figsize=(13, 6.4))
for r in range(2):
    for c in range(4):
        cell = grid[r][c]
        if cell is None:
            ax[r, c].axis("off"); continue
        M, t = cell
        ax[r, c].imshow(M.detach().cpu().numpy(), cmap="coolwarm")
        ax[r, c].set_title(t); ax[r, c].axis("off")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------- 5. rank
md(r"""
## 5. How the methods scale with matrix rank — *controlled* comparison

Here we change **only the rank** of `L` (4 → 6 → 8) while holding the unfolded
model and its training budget fixed at the main setting (`K=10`, 80 epochs, 1500
train); the classical baseline is grid-search-tuned at each rank. This isolates
the effect of rank from model size. Produced by
`python scripts/run_rank_sweep.py --controlled`.
""")

code(r"""
from IPython.display import Image

ctrl = "outputs/controlled_rank_sweep.json"
if os.path.exists(ctrl):
    sweep = json.load(open(ctrl))
    rows = []
    for rk in sorted(sweep, key=int):
        for m in ["Truncated SVD", "Classical RPCA", "Deep-unfolded RPCA"]:
            r = sweep[rk]["results"][m]
            rows.append({"rank": int(rk), "method": m,
                         "L rel err": round(r["L_rel_err"], 4),
                         "S rel err": round(r["S_rel_err"], 4),
                         "F1": round(r["f1"], 4),
                         "ms/sample": round(r["runtime_per_sample_s"] * 1000, 2)})
    display(pd.DataFrame(rows).set_index(["rank", "method"]))
    if os.path.exists("outputs/controlled_plot_8_rank_sweep.png"):
        display(Image("outputs/controlled_plot_8_rank_sweep.png"))
else:
    print("Not found. Generate with:  python scripts/run_rank_sweep.py --controlled")
""")

md(r"""
With model size held fixed, the unfolded net **trails classical on the low-rank
`L` at every rank, and the gap widens with rank** (≈1.15× → 1.46× → 1.55×): ten
unrolled layers (≈10 iterations) cannot resolve a higher-rank subspace as well
as classical's ~200 iterations, and training had already converged — a capacity
ceiling, not undertraining. On the **sparse `S` it overtakes classical at rank 6
and 8** (its relative `S` advantage *grows* with rank), while staying ~18× faster
with F1 within ~3 points. So the honest claim is *parity on sparse recovery at a
large speedup*, not a uniform win.
""")

# ---------------------------------------------------------------- 6. depth
md(r"""
## 6. What more depth buys (`K = 12`) — a capacity study

**This is not a controlled rank comparison** — it changes depth *and* training
budget together (12 layers, 200 epochs, 2500 train). It answers a different
question: *can* the unrolled approach beat classical at high rank given enough
layers? Produced by `python scripts/run_rank_sweep.py` (default mode).
""")

code(r"""
deep = "outputs/rank_sweep.json"
if os.path.exists(deep):
    sweep = json.load(open(deep))
    rows = []
    for rk in sorted(sweep, key=int):
        for m in ["Classical RPCA", "Deep-unfolded RPCA"]:
            r = sweep[rk]["results"][m]
            rows.append({"rank": int(rk),
                         "method": m if m != "Deep-unfolded RPCA" else "Deep-unfolded RPCA (K=12)",
                         "L rel err": round(r["L_rel_err"], 4),
                         "S rel err": round(r["S_rel_err"], 4),
                         "F1": round(r["f1"], 4),
                         "ms/sample": round(r["runtime_per_sample_s"] * 1000, 2)})
    display(pd.DataFrame(rows).set_index(["rank", "method"]))
else:
    print("Not found. Generate with:  python scripts/run_rank_sweep.py")
""")

md(r"""
With the extra depth the unfolded model surpasses classical on **both**
components at rank 8 (~40% lower `L`, ~50% lower `S`) while still ~15× faster:
the rank-induced `L` gap from section 5 closes once the network is deep enough to
emulate more iterations — *capacity ≈ iteration budget*.
""")

# ---------------------------------------------------------------- 7. takeaways
md(r"""
## 7. Takeaways

- **Unrolling buys speed.** A `K=10` learned network matches a 200-iteration
  grid-search-tuned classical solver on **sparse recovery and support** at
  **~18× lower runtime** — ≈10 learned layers ≈ ≈200 hand-set iterations.
- **At fixed depth, low-rank recovery is the limitation**, and it worsens as the
  intrinsic rank grows: 10 layers can't emulate enough iterations to pin down a
  higher-rank subspace.
- **Capacity ≈ iteration budget.** Adding depth (`K=12`) closes that gap and lets
  the learned model beat classical on both components even at rank 8.
- **Next step:** apply the unrolled solver to real **video background
  subtraction** (background = low-rank `L`, moving objects = sparse `S`) and
  measure the speedup vs. classical RPCA on real frames.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {OUT}  ({len(cells)} cells)")
