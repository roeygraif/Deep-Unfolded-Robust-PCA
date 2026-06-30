# Deep-Unfolded Robust PCA for Low-Rank + Sparse Matrix Decomposition

Course project for **Model Based Deep Learning (361.2.2320)**, BGU.

We replace the iterative Robust-PCA solver with a small **deep-unfolded** network
whose `K` layers correspond to `K` proximal-gradient iterations and whose step
sizes and shrinkage thresholds are **learned** end-to-end on synthetic data.

## 1. Motivation

Many real-world datasets can be modeled as the superposition of a structured
low-rank component, sparse outliers, and dense noise:

- **video surveillance:** static background = low-rank, moving objects = sparse
- **recommender systems:** latent user/item factors = low-rank, corrupt ratings = sparse
- **network monitoring:** regular traffic = low-rank, intrusions = sparse outliers

The job is to **decompose** an observed matrix into these components.

## 2. Mathematical System Model

We observe `D ∈ R^{n×n}` formed as

```
D = L + S + N
```

where

- `L` is the **low-rank** structured component
- `S` is the **sparse** anomaly/outlier component
- `N ~ N(0, σ² I)` is dense Gaussian noise

We want to estimate `L` and `S` from `D`.

## 3. Classical Robust PCA

The convex relaxation of the rank/sparsity problem is

```
min_{L,S}  0.5 · ||D − L − S||_F²  +  λ_L · ||L||_*  +  λ_S · ||S||_1
```

where `||·||_*` is the **nuclear norm** (sum of singular values, encouraging
low rank) and `||·||_1` is the **entry-wise L1 norm** (encouraging sparsity).

We solve it with **proximal gradient** (about 200 iterations):

```
R       = L + S − D
L_next  = SVT(L − α · R, τ_L)              # proximal of nuclear norm
S_next  = soft_threshold(S − α · R, τ_S)   # proximal of L1
```

`SVT` shrinks each singular value of its argument by `τ_L`. `soft_threshold`
shrinks each entry by `τ_S`.

## 4. Model-based DL: Deep Unfolding

Each classical iteration becomes one **layer** of a neural network whose
parameters `(α_k, τ_L^k, τ_S^k)` are **learned** per layer. With `K = 10`
layers the unfolded net is *much shallower* than the classical solver
(~200 iters), yet matches its sparse-recovery accuracy (and exceeds it given
more depth) because the parameters are trained on data rather than tuned by
grid search. Positivity is enforced via `softplus`
on raw learnable scalars.

This is the core "model-based deep learning" idea: take a domain-specific
solver, unroll it, and learn its hyper-parameters end-to-end with backprop.

## How to run

**Quickest (recommended):** open [`deep_unfolded_rpca.ipynb`](deep_unfolded_rpca.ipynb)
and *Run All*. It trains the `K = 10` network and reproduces the main results in
a few minutes, then shows the rank and depth studies. It also runs on Google
Colab — the first cell clones this repo and installs the dependencies.

To run the full experiment suite from the command line instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train + evaluate everything (≤ 1 hour on a MacBook)
python scripts/run_all.py
```

Or run experiments individually:

```bash
python scripts/run_experiment_1.py    # main: train + eval all 3 methods
python scripts/run_experiment_2.py    # noise robustness
python scripts/run_experiment_3.py    # sparsity robustness
```

Outputs land in `outputs/`:

- `results.csv` — tidy results table
- `REPORT.md` — auto-generated tables + discussion
- `plot_*.png` — 7 plots, ready to drop into slides
- `model_best.pt`, `training_history.json`

## Project structure

```
deep_unfolded_rpca/   # library
├── data.py           # synthetic data generation
├── operators.py      # SVT, soft-threshold (MPS-safe SVD)
├── baselines.py      # truncated SVD, classical RPCA
├── model.py          # DeepUnfoldedRPCA (K-layer unrolled network)
├── train.py          # training + evaluation loops
├── metrics.py        # relative error, sparse support F1
├── plots.py          # matplotlib helpers (no seaborn)
└── utils.py          # device + seeding

scripts/              # experiment runners (call into the package)
outputs/              # results (a curated slice is committed; rest is runtime)
```

## Methods compared

| Method                  | What it does                                          |
| ----------------------- | ----------------------------------------------------- |
| **Truncated SVD**       | Best rank-`r` approximation; treats outliers as noise |
| **Classical RPCA**      | Proximal gradient with grid-search-tuned `τ_L, τ_S` (200 iters) |
| **Deep-unfolded RPCA**  | K=10 unrolled layers with learned `(α_k, τ_L^k, τ_S^k)` |

## Metrics

- **Relative Frobenius error** for `L` and `S`: `||X̂ − X||_F / ||X||_F`
- **Sparse support F1**: precision/recall on `|Ŝ| > threshold` vs. `S ≠ 0`
- **Runtime per sample** on the test set

## Notes

- Random seed `42` throughout for reproducibility.
- MPS users: SVD is automatically routed through CPU inside `svt()` to dodge
  the known MPS svd numerical/backward issues.
- All settings (matrix size, K layers, batch size, etc.) live at the top of
  each `scripts/run_experiment_*.py` for easy tweaking.
