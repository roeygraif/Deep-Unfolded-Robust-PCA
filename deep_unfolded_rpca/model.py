"""Deep-unfolded Robust PCA: K-layer unrolled proximal-gradient solver."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import soft_threshold, svt


def _inv_softplus(y: float) -> float:
    """Inverse of softplus: log(exp(y) - 1). Requires y > 0."""
    return math.log(math.expm1(y))


class DeepUnfoldedRPCA(nn.Module):
    """K-layer unrolled RPCA solver with learnable per-layer parameters.

    Each layer applies one classical RPCA proximal-gradient step::

        R       = L + S - D
        L_next  = SVT(L - alpha_k * R, tau_L_k)
        S_next  = soft_threshold(S - alpha_k * R, tau_S_k)

    Parameters ``alpha_k``, ``tau_L_k``, ``tau_S_k`` are learned per layer
    and constrained to be positive via ``softplus`` on raw scalars. Initial
    values are chosen so layer 0 reproduces the classical solver, giving
    the network a useful warm start.
    """

    def __init__(
        self,
        n_layers: int = 8,
        n: int = 32,
        init_alpha: float = 1.0,
        init_tau_L: float | None = None,
        init_tau_S: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n = n

        # Default tau_L = 1/sqrt(n) suits the synthetic square case; pass a
        # larger value for video-scale matrices, whose singular values are far
        # bigger (a near-rank-1 background dominates), so the threshold must be
        # in that range to separate background from foreground.
        if init_tau_L is None:
            init_tau_L = 1.0 / math.sqrt(n)

        raw_alpha = _inv_softplus(init_alpha)
        raw_tau_L = _inv_softplus(init_tau_L)
        raw_tau_S = _inv_softplus(init_tau_S)

        self._alpha = nn.Parameter(torch.full((n_layers,), raw_alpha))
        self._tau_L = nn.Parameter(torch.full((n_layers,), raw_tau_L))
        self._tau_S = nn.Parameter(torch.full((n_layers,), raw_tau_S))

    def step_params(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return positive ``(alpha, tau_L, tau_S)`` tensors of shape ``(K,)``."""
        return (
            F.softplus(self._alpha),
            F.softplus(self._tau_L),
            F.softplus(self._tau_S),
        )

    def forward(self, D: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unfold K iterations of RPCA proximal gradient on D.

        ``D`` has shape ``(B, n, n)``. Returns ``(L_hat, S_hat)``.
        """
        L = torch.zeros_like(D)
        S = torch.zeros_like(D)
        alpha, tau_L, tau_S = self.step_params()

        for k in range(self.n_layers):
            R = L + S - D
            L_temp = L - alpha[k] * R
            S_temp = S - alpha[k] * R
            L = svt(L_temp, tau_L[k])
            S = soft_threshold(S_temp, tau_S[k])

        return L, S
