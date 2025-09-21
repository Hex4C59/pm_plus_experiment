#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Visualization helpers for training and evaluation curves.

This module provides simple helpers to plot training losses and arbitrary
metrics. Functions accept Python lists or 1-D tensors and save PNG images.
A small CLI reads a CSV file (such as produced by Logger) and plots
selected columns.

Example :
    >>> from src.utils.visualization import plot_losses
    >>> plot_losses([0.5,0.4],[0.45,0.42],"runs/exp1/loss.png")
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"


import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt


def _ensure_list(x: Optional[Iterable[float]]) -> List[float]:
    """Return list copy of iterable or empty list if None."""
    if x is None:
        return []
    return list(float(v) for v in x)


def plot_losses(
    train_losses: Iterable[float],
    val_losses: Optional[Iterable[float]],
    save_path: str,
) -> None:
    """
    Plot training and validation loss curves and save to PNG.

    Args:
        train_losses: Sequence of training losses per epoch/step.
        val_losses: Optional sequence of validation losses.
        save_path: Destination image file path.

    Raises:
        IOError: If saving the figure fails.

    """
    t = _ensure_list(train_losses)
    v = _ensure_list(val_losses)
    epochs = list(range(1, len(t) + 1))
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, t, label="train_loss", marker="o")
    if v:
        plt.plot(list(range(1, len(v) + 1)), v, label="val_loss", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
        plt.savefig(str(out), dpi=150)
        plt.close()
    except Exception as exc:
        raise IOError(f"Failed to save plot: {exc}") from exc


def plot_metrics(
    metrics: Dict[str, Iterable[float]],
    save_path: str,
) -> None:
    """
    Plot multiple metric curves on the same axes and save to PNG.

    Args:
        metrics: Mapping from metric name to sequence of values.
        save_path: Destination image file path.

    Raises:
        IOError: If saving the figure fails.

    """
    plt.figure(figsize=(6, 4))
    for name, seq in metrics.items():
        vals = _ensure_list(seq)
        plt.plot(range(1, len(vals) + 1), vals, label=name, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    plt.grid(True)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        plt.tight_layout()
        plt.savefig(str(out), dpi=150)
        plt.close()
    except Exception as exc:
        raise IOError(f"Failed to save plot: {exc}") from exc


def plot_from_csv(
    csv_path: str,
    columns: List[str],
    save_path: str,
) -> None:
    """
    Read specified columns from CSV and plot them.

    Args:
        csv_path: Path to CSV file produced by Logger.
        columns: List of column names to plot (must exist in CSV).
        save_path: Destination image path.

    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If requested columns are not present.

    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    with open(p, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    # validate columns
    for c in columns:
        if c not in rows[0]:
            raise ValueError(f"Column '{c}' not found in CSV")
    series: Dict[str, List[float]] = {c: [] for c in columns}
    for r in rows:
        for c in columns:
            val = r.get(c, "")
            try:
                series[c].append(float(val) if val != "" else float("nan"))
            except Exception:
                series[c].append(float("nan"))
    plot_metrics(series, save_path)


def _cli() -> None:
    """CLI to plot columns from a CSV file to PNG."""
    import argparse

    parser = argparse.ArgumentParser(description="Plot CSV columns to PNG")
    parser.add_argument("csv", type=str, help="CSV path (e.g., runs/exp1/metrics.csv)")
    parser.add_argument(
        "--cols", type=str, required=True, help="Comma-separated column names to plot"
    )
    parser.add_argument("out", type=str, help="Output image path (PNG)")
    args = parser.parse_args()
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    plot_from_csv(args.csv, cols, args.out)


if __name__ == "__main__":
    _cli()
