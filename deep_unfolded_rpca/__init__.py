"""Deep-unfolded Robust PCA: low-rank + sparse + noise decomposition."""
from .baselines import classical_rpca, truncated_svd_baseline, tune_classical_rpca
from .data import RPCADataset, generate_sample
from .metrics import relative_frobenius_error, sparse_support_f1, tune_f1_threshold
from .model import DeepUnfoldedRPCA
from .operators import soft_threshold, svt
from .train import evaluate_method, train_model
from .utils import get_device, set_seed

__all__ = [
    "RPCADataset",
    "generate_sample",
    "DeepUnfoldedRPCA",
    "classical_rpca",
    "truncated_svd_baseline",
    "tune_classical_rpca",
    "svt",
    "soft_threshold",
    "relative_frobenius_error",
    "sparse_support_f1",
    "tune_f1_threshold",
    "train_model",
    "evaluate_method",
    "get_device",
    "set_seed",
]
