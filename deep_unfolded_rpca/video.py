"""Video background subtraction: synthetic video-shaped training data + a
loader for the real escalator clip.

For background subtraction the data matrix stacks ``T`` vectorized frames as
columns: ``D in R^{P x T}`` with ``P = pixels``. The static background is the
same scene in every column (low-rank) and moving objects are sparse. The
synthetic generator mimics that structure *and scale* (a near-rank-1 background
plus sparse foreground) so a model trained on it transfers to real footage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def generate_video_sample(
    P: int,
    T: int,
    bg_rank: int,
    fg_density: float,
    fg_mag: float,
    sigma: float,
    rng: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """One synthetic (D, L, S) of shape (P, T): a near-rank-1 background that
    looks like a repeated frame (plus small slow drift), sparse foreground, and
    small dense noise. Values are kept roughly in [0, 1] like real grayscale."""
    base = 0.2 + 0.6 * torch.rand(P, 1, generator=rng)        # [0.2, 0.8] base frame
    L = base.repeat(1, T)                                     # rank-1 background
    if bg_rank > 1:                                           # small slow drift
        U = torch.randn(P, bg_rank - 1, generator=rng)
        Vt = torch.randn(bg_rank - 1, T, generator=rng)
        L = L + 0.02 * (U @ Vt)
    mask = (torch.rand(P, T, generator=rng) < fg_density).float()
    sign = torch.where(torch.rand(P, T, generator=rng) < 0.5, -1.0, 1.0)
    mag = fg_mag * (0.5 + torch.rand(P, T, generator=rng))    # ~[0.5, 1.5] * fg_mag
    S = mask * sign * mag
    N = torch.randn(P, T, generator=rng) * sigma
    return {"D": L + S + N, "L": L, "S": S}


class VideoRPCADataset(Dataset):
    """Synthetic video-shaped (D, L, S) matrices, generated on the fly (these
    matrices are large, so the corpus is not pre-materialized). Sample ``idx``
    is deterministic via ``seed + idx``."""

    def __init__(
        self,
        n_samples: int,
        P: int,
        T: int,
        bg_rank: int,
        fg_density: float,
        fg_mag: float,
        sigma: float,
        seed: int,
    ):
        self.n_samples = n_samples
        self.P, self.T = P, T
        self.bg_rank, self.fg_density, self.fg_mag = bg_rank, fg_density, fg_mag
        self.sigma, self.seed = sigma, seed

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + idx)
        return generate_video_sample(
            self.P, self.T, self.bg_rank, self.fg_density, self.fg_mag, self.sigma, rng
        )


def load_escalator_matrix(
    path: Union[str, Path], ds: int = 64, n_frames: Union[int, None] = None
) -> Tuple[torch.Tensor, int]:
    """Load ``escalator_data.mat`` into a grayscale matrix ``D`` of shape
    ``(ds*ds, n_frames)`` in [0, 1], each column a bilinearly-downsampled frame.
    Returns ``(D, ds)``."""
    import scipy.io as sio

    mat = sio.loadmat(str(path))
    H, W = int(mat["m"][0, 0]), int(mat["n"][0, 0])
    X = mat["X"].astype(np.float32) / 255.0
    T = X.shape[1] if n_frames is None else min(n_frames, X.shape[1])
    frames = torch.from_numpy(X[:, :T].reshape(H, W, T, order="F"))
    frames = frames.permute(2, 0, 1).unsqueeze(1)            # (T, 1, H, W)
    frames = F.interpolate(frames, size=(ds, ds), mode="bilinear", align_corners=False)
    D = frames.reshape(T, ds * ds).t().contiguous()          # (ds*ds, T)
    return D, ds
