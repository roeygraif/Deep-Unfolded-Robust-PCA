"""Synthetic data generation: D = L + S + N.

L is a random low-rank matrix, S is sparse with random sign and amplitude,
N is i.i.d. Gaussian noise. ``RPCADataset`` pre-generates the full corpus on
CPU using a fixed seed so runs are reproducible bit-for-bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch.utils.data import Dataset


@dataclass
class DataConfig:
    n: int = 32
    rank: int = 4
    sparsity: float = 0.05
    sigma: float = 0.05


def generate_sample(
    n: int,
    rank: int,
    sparsity: float,
    sigma: float,
    rng: torch.Generator,
) -> Dict[str, torch.Tensor]:
    """Generate one sample (L, S, N, D) of shape (n, n).

    Scaling: U, V ~ N(0, 1), L = (U @ V.T) / sqrt(n) so typical |L_ij| ~ 0.35
    for n=32, r=4. Sparse entries have magnitude in [0.5, 1.5] with random
    sign — large enough relative to L and noise to be detectable.
    """
    U = torch.randn(n, rank, generator=rng)
    V = torch.randn(n, rank, generator=rng)
    L = (U @ V.T) / math.sqrt(n)

    mask = (torch.rand(n, n, generator=rng) < sparsity).float()
    sign = torch.where(
        torch.rand(n, n, generator=rng) < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    magnitude = 0.5 + torch.rand(n, n, generator=rng)
    S = mask * sign * magnitude

    N = torch.randn(n, n, generator=rng) * sigma

    D = L + S + N
    return {"L": L, "S": S, "N": N, "D": D}


class RPCADataset(Dataset):
    """Pre-generated dataset of (D, L, S) triples on CPU.

    The full corpus is materialized up front from a fixed seed for full
    reproducibility. Move tensors to the device inside the training loop.
    """

    def __init__(
        self,
        n_samples: int,
        n: int,
        rank: int,
        sparsity: float,
        sigma: float,
        seed: int,
    ):
        self.n_samples = n_samples
        self.n = n
        self.rank = rank
        self.sparsity = sparsity
        self.sigma = sigma

        rng = torch.Generator(device="cpu")
        rng.manual_seed(seed)

        D_all = torch.empty(n_samples, n, n)
        L_all = torch.empty(n_samples, n, n)
        S_all = torch.empty(n_samples, n, n)
        for i in range(n_samples):
            sample = generate_sample(n, rank, sparsity, sigma, rng)
            D_all[i] = sample["D"]
            L_all[i] = sample["L"]
            S_all[i] = sample["S"]

        self.D = D_all
        self.L = L_all
        self.S = S_all

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"D": self.D[idx], "L": self.L[idx], "S": self.S[idx]}
