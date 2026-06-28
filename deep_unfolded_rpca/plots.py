"""Matplotlib plot helpers (no seaborn)."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


_METHOD_COLORS = {
    "Truncated SVD": "#7f7f7f",
    "Classical RPCA": "#1f77b4",
    "Deep-unfolded RPCA": "#d62728",
}


def plot_training_curves(history: Dict[str, list], save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="train", marker="o", markersize=3)
    ax.plot(epochs, history["val_loss"], label="validation", marker="s", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_method_bar(
    method_names: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    save_path: Path,
    log_y: bool = False,
    value_fmt: str = "{:.4g}",
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [_METHOD_COLORS.get(name, "#2ca02c") for name in method_names]
    bars = ax.bar(method_names, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            value_fmt.format(v),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_robustness_two_panel(
    x_values: List[float],
    series_L: Dict[str, List[float]],
    series_S: Dict[str, List[float]],
    x_label: str,
    title: str,
    save_path: Path,
) -> None:
    """Two side-by-side subplots: L error and S error vs the swept axis."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    markers = ["o", "s", "^"]

    for ax, series, ylabel in [
        (axes[0], series_L, "L relative error"),
        (axes[1], series_S, "S relative error"),
    ]:
        for (name, vals), marker in zip(series.items(), markers):
            color = _METHOD_COLORS.get(name)
            ax.plot(x_values, vals, marker=marker, label=name, color=color)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
