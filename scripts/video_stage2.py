"""Stage 2: train the deep-unfolded model on SYNTHETIC video-shaped data, then
run it AND classical RPCA + truncated SVD on the REAL escalator clip. Reports
the timing comparison + background/foreground agreement, and saves a
side-by-side decomposition figure.

  python scripts/video_stage2.py            # full (~15 min on M4)
  python scripts/video_stage2.py --smoke    # fast sanity run
"""
import sys
import time
sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from deep_unfolded_rpca import (
    DeepUnfoldedRPCA, classical_rpca, truncated_svd_baseline,
    get_device, set_seed, train_model,
)
from deep_unfolded_rpca.metrics import relative_frobenius_error
from deep_unfolded_rpca.video import VideoRPCADataset, load_escalator_matrix

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

# ---- config ----
DS = 64
T = 100                 # train + test at the same frame count (scale match)
K = 15                  # unrolled layers
BG_RANK, FG_DENSITY, FG_MAG, SIGMA = 2, 0.08, 0.4, 0.02
INIT_TAU_L, INIT_TAU_S = 15.0, 0.05
N_TRAIN, N_VAL, BATCH, EPOCHS, LR, BETA = 128, 32, 4, 40, 3e-3, 8.0
CLASSICAL_TAU_L, CLASSICAL_TAU_S, N_ITER = 10.0, 0.05, 200
SVD_RANK = 2

if "--smoke" in sys.argv:
    T, K, N_TRAIN, N_VAL, EPOCHS = 60, 8, 8, 4, 1
    print("[smoke] tiny config")

P = DS * DS
device = get_device()
set_seed(42)
print(f"device {device} | matrix {P}x{T}, K={K}")

# ---- train on synthetic video-shaped data ----
train_set = VideoRPCADataset(N_TRAIN, P, T, BG_RANK, FG_DENSITY, FG_MAG, SIGMA, seed=42)
val_set = VideoRPCADataset(N_VAL, P, T, BG_RANK, FG_DENSITY, FG_MAG, SIGMA, seed=10_000)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH, shuffle=False)

model = DeepUnfoldedRPCA(n_layers=K, n=T, init_tau_L=INIT_TAU_L, init_tau_S=INIT_TAU_S)
t0 = time.perf_counter()
history, _ = train_model(model, train_loader, val_loader, n_epochs=EPOCHS,
                         lr=LR, beta=BETA, device=device, log_every=5)
print(f"trained in {(time.perf_counter() - t0) / 60:.1f} min, "
      f"best val {min(history['val_loss']):.5f}")

# synthetic val recovery quality
model.eval()
vb = next(iter(DataLoader(val_set, batch_size=16)))
with torch.no_grad():
    Lh, Sh = model(vb["D"].to(device))
print(f"[synthetic val] L_rel={relative_frobenius_error(Lh, vb['L'].to(device)).item():.4f} "
      f"S_rel={relative_frobenius_error(Sh, vb['S'].to(device)).item():.4f}")

# ---- real escalator: run all three on CPU for a fair timing comparison ----
D, _ = load_escalator_matrix("data/escalator_data.mat", ds=DS, n_frames=T)
model_cpu = model.to("cpu").eval()
Db = D.unsqueeze(0)


def timed(fn, reps=3):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    return out, (time.perf_counter() - t0) / reps


with torch.no_grad():
    (Lu, Su), tu = timed(lambda: tuple(x.squeeze(0) for x in model_cpu(Db)))
    (Lc, Sc), tc = timed(lambda: tuple(x.squeeze(0) for x in classical_rpca(
        Db, alpha=1.0, tau_L=CLASSICAL_TAU_L, tau_S=CLASSICAL_TAU_S, n_iter=N_ITER)))
    (Ls, Ss), ts = timed(lambda: tuple(x.squeeze(0) for x in truncated_svd_baseline(
        Db, rank=SVD_RANK)))

print("\n=== REAL ESCALATOR (64x64 x %d frames) — timing on CPU ===" % T)
print(f"  Classical RPCA ({N_ITER} iters): {tc * 1000:8.1f} ms")
print(f"  Deep-unfolded  ({K} layers)    : {tu * 1000:8.1f} ms   ->  {tc / tu:5.1f}x faster")
print(f"  Truncated SVD  (rank {SVD_RANK})      : {ts * 1000:8.1f} ms")
print(f"\n  agreement (no GT on real video):")
print(f"    unfolded vs classical:  L {relative_frobenius_error(Lu, Lc).item():.3f}  "
      f"S {relative_frobenius_error(Su, Sc).item():.3f}")
print(f"    SVD      vs classical:  L {relative_frobenius_error(Ls, Lc).item():.3f}  "
      f"S {relative_frobenius_error(Ss, Sc).item():.3f}")

# ---- side-by-side decomposition figure ----
def img(col):
    return col.reshape(DS, DS).detach().numpy()

methods = [("Classical", Lc, Sc), ("Unfolded", Lu, Su), ("SVD", Ls, Ss)]
show = [int(T * f) for f in (0.25, 0.55, 0.85)]
ncol = 1 + 2 * len(methods)
fig, ax = plt.subplots(len(show), ncol, figsize=(2.0 * ncol, 2.2 * len(show)))
for r, t in enumerate(show):
    ax[r, 0].imshow(img(D[:, t]), cmap="gray")
    ax[r, 0].set_ylabel(f"frame {t}", fontsize=9)
    if r == 0:
        ax[r, 0].set_title("original")
    for mi, (name, L, S) in enumerate(methods):
        ax[r, 1 + 2 * mi].imshow(img(L[:, t]), cmap="gray")
        ax[r, 2 + 2 * mi].imshow(np.abs(img(S[:, t])), cmap="hot")
        if r == 0:
            ax[r, 1 + 2 * mi].set_title(f"{name}\nbg L")
            ax[r, 2 + 2 * mi].set_title(f"{name}\nfg |S|")
    for c in range(ncol):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
plt.tight_layout()
out = OUT / "video_stage2_compare.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
torch.save(model_cpu.state_dict(), OUT / "model_video_unfolded.pt")
print("saved", out, "and model_video_unfolded.pt")
