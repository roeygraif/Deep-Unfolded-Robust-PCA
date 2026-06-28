"""Stage 1 video demo / tuning: classical RPCA + truncated SVD on the escalator
clip. Builds D = (pixels x frames) from data/escalator_data.mat downsampled to
DS x DS, sweeps the classical tau_L, and saves a side-by-side figure
(original | background L | foreground |S|) so we can pick a clean separation."""
import sys
sys.path.insert(0, ".")
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import scipy.io as sio
import matplotlib.pyplot as plt

from deep_unfolded_rpca import classical_rpca, truncated_svd_baseline

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
DS, FRAME = 64, 100

mat = sio.loadmat("data/escalator_data.mat")
H, W = int(mat["m"][0, 0]), int(mat["n"][0, 0])
X = mat["X"].astype(np.float32) / 255.0
T = X.shape[1]
frames = torch.from_numpy(X.reshape(H, W, T, order="F")).permute(2, 0, 1).unsqueeze(1)
D = (F.interpolate(frames, size=(DS, DS), mode="bilinear", align_corners=False)
     .reshape(T, DS * DS).t().contiguous())            # (P, T)
P = D.shape[0]
print(f"D = {P} x {T}  ({H}x{W} -> {DS}x{DS})")
sv = torch.linalg.svdvals(D)
print("D top-10 singular values:", [round(float(s), 1) for s in sv[:10]])

def im(col):
    return col.reshape(DS, DS).numpy()

configs = []
for tL in [10.0, 20.0, 30.0, 50.0]:
    L, S = classical_rpca(D.unsqueeze(0), alpha=1.0, tau_L=tL, tau_S=0.05, n_iter=100)
    L, S = L.squeeze(0), S.squeeze(0)
    svL = torch.linalg.svdvals(L)
    rank = int((svL > 0.01 * svL[0]).sum())
    spars = float((S.abs() > 0.05).float().mean())
    print(f"  classical tau_L={tL:4}: rank(L)~{rank:3d}, frac|S|>0.05={spars:.3f}")
    configs.append((f"classical tL={tL}", L, S))

Lsvd, Ssvd = truncated_svd_baseline(D.unsqueeze(0), rank=3)
configs.append(("SVD rank-3", Lsvd.squeeze(0), Ssvd.squeeze(0)))

n = len(configs)
fig, ax = plt.subplots(n, 3, figsize=(8, 2.4 * n))
for r, (name, L, S) in enumerate(configs):
    ax[r, 0].imshow(im(D[:, FRAME]), cmap="gray"); ax[r, 0].set_ylabel(name, fontsize=9)
    ax[r, 1].imshow(im(L[:, FRAME]), cmap="gray")
    ax[r, 2].imshow(np.abs(im(S[:, FRAME])), cmap="hot")
    for c in range(3):
        ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
    if r == 0:
        ax[r, 0].set_title("original"); ax[r, 1].set_title("background L"); ax[r, 2].set_title("foreground |S|")
plt.tight_layout()
out = OUT / "video_stage1_tune.png"
plt.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
