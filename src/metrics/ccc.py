#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Concordance Correlation Coefficient (CCC) metric implementations.

This module provides functions and stateful metric classes to compute the
Concordance Correlation Coefficient. It contains:
 - concordance_ccc: a stateless function computing CCC for two tensors.
 - CCCMetric: a streaming/stateful metric that accumulates sums and can
   compute CCC over multiple batches without storing all samples.
 - CombinedCCC: convenience helper to compute weighted CCC for two
   dimensions (valence and arousal).

Example :
    >>> import torch
    >>> from src.metrics.ccc import concordance_ccc, CCCMetric
    >>> p = torch.tensor([0.1, 0.2, 0.3])
    >>> t = torch.tensor([0.0, 0.1, 0.4])
    >>> print(concordance_ccc(p, t))
    >>> m = CCCMetric()
    >>> m.update(p, t); print(m.compute())
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"


from typing import Optional, Tuple

import torch
from torch import Tensor

from .base import BaseMetric


def concordance_ccc(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute concordance correlation coefficient between pred and target.

    Args:
        pred: Prediction tensor of shape (N,) or (N,1).
        target: Target tensor of same shape as pred.
        eps: Small epsilon to avoid division by zero.

    Returns:
        Scalar tensor with CCC in [-1, 1].

    Raises:
        ValueError: If pred and target shapes do not match.

    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    p = pred.view(-1).to(dtype=torch.float32)
    t = target.view(-1).to(dtype=torch.float32)
    if p.numel() == 0:
        return torch.tensor(0.0, dtype=torch.float32)
    mean_p = torch.mean(p)
    mean_t = torch.mean(t)
    var_p = torch.var(p, unbiased=False)
    var_t = torch.var(t, unbiased=False)
    cov = torch.mean((p - mean_p) * (t - mean_t))
    ccc = (2.0 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + eps)
    return ccc


class CCCMetric(BaseMetric):
    """
    Streaming Concordance Correlation Coefficient metric.

    This implementation accumulates sufficient statistics (sums and sums of
    squares) to compute mean, variance and covariance without storing all
    predictions and targets.

    Attributes:
        n: Number of accumulated samples.
        sum_pred: Sum of predictions.
        sum_target: Sum of targets.
        sum_pred_sq: Sum of squared predictions.
        sum_target_sq: Sum of squared targets.
        sum_prod: Sum of element-wise product pred * target.
        eps: Numerical stability constant.

    """

    def __init__(self, name: str = "ccc", eps: float = 1e-8) -> None:
        """Create a CCCMetric instance."""
        super().__init__(name=name)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self.n = 0  # type: int
        self.sum_pred = 0.0
        self.sum_target = 0.0
        self.sum_pred_sq = 0.0
        self.sum_target_sq = 0.0
        self.sum_prod = 0.0

    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Optional[Tensor] = None,
    ) -> None:
        """
        Accumulate batch preds and targets.

        Args:
            preds: Tensor shape (B,) or (B,1).
            targets: Tensor shape matching preds.
            mask: Optional boolean mask (B,) where True indicates valid.

        Raises:
            ValueError: If shapes mismatch.

        """
        if preds.shape != targets.shape:
            raise ValueError("preds and targets must have same shape")
        p = preds.view(-1).to(dtype=torch.float64)
        t = targets.view(-1).to(dtype=torch.float64)
        if mask is not None:
            m = mask.view(-1).to(dtype=torch.bool)
            p = p[m]
            t = t[m]
        if p.numel() == 0:
            return
        # convert to Python floats for stable accumulation in double
        self.n += int(p.numel())
        self.sum_pred += float(p.sum().item())
        self.sum_target += float(t.sum().item())
        self.sum_pred_sq += float((p * p).sum().item())
        self.sum_target_sq += float((t * t).sum().item())
        self.sum_prod += float((p * t).sum().item())

    def compute(self) -> Tensor:
        """
        Compute CCC from accumulated statistics.

        Returns:
            Scalar tensor containing CCC.

        Raises:
            RuntimeError: If no samples accumulated.

        """
        if self.n == 0:
            return torch.tensor(0.0, dtype=torch.float32)
        n = float(self.n)
        mean_p = self.sum_pred / n
        mean_t = self.sum_target / n
        var_p = self.sum_pred_sq / n - mean_p * mean_p
        var_t = self.sum_target_sq / n - mean_t * mean_t
        cov = self.sum_prod / n - mean_p * mean_t
        denom = var_p + var_t + (mean_p - mean_t) ** 2 + float(self.eps)
        ccc_val = (2.0 * cov) / denom
        return torch.tensor(ccc_val, dtype=torch.float32)


class CombinedCCC:
    """
    Compute weighted CCC for valence and arousal pairs.

    This helper uses two CCCMetric or computes directly from tensors.
    The combined score is: alpha * ccc_arousal + (1-alpha) * ccc_valence.

    Attributes:
        alpha: Weight for arousal CCC.

    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8) -> None:
        """Create a CombinedCCC helper."""
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self.eps = float(eps)

    def compute_from_tensors(
        self, preds: Tensor, targets: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute valence/arousal CCCs and combined score from tensors.

        Args:
            preds: Tensor shape (B,2) or dict-like with columns [valence,arousal].
            targets: Tensor shape (B,2) matching preds.
            mask: Optional boolean mask (B,).

        Returns:
            Tuple (ccc_valence, ccc_arousal, combined) as tensors.

        """
        if preds.dim() != 2 or preds.size(1) != 2:
            raise ValueError("preds must be shape (B,2)")
        if targets.dim() != 2 or targets.size(1) != 2:
            raise ValueError("targets must be shape (B,2)")
        pv = preds[:, 0:1]
        pa = preds[:, 1:2]
        tv = targets[:, 0:1]
        ta = targets[:, 1:2]
        if mask is not None:
            m = mask.view(-1).to(dtype=torch.bool)
            pv = pv[m]
            pa = pa[m]
            tv = tv[m]
            ta = ta[m]
        ccc_v = concordance_ccc(pv, tv, eps=self.eps)
        ccc_a = concordance_ccc(pa, ta, eps=self.eps)
        combined = self.alpha * ccc_a + (1.0 - self.alpha) * ccc_v
        return ccc_v, ccc_a, combined
