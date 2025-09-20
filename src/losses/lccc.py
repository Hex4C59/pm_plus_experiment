#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Combined Concordance Correlation Coefficient (CCC) loss.

This module implements a combined CCC-based loss for valence and arousal.
The loss value is computed as:
    L_ccc = 1 - (alpha * ccc_arousal + (1 - alpha) * ccc_valence)
where ccc_* are concordance correlation coefficients for each target and
alpha is a weighting scalar (default 0.5). The loss returns a dict with
the scalar `loss` and per-dimension CCC values useful for logging.

Example :
    >>> import torch
    >>> from src.losses.lccc import LCCCLoss
    >>> preds = {"valence": torch.randn(8,1), "arousal": torch.randn(8,1)}
    >>> targets = {"valence": torch.randn(8,1), "arousal": torch.randn(8,1)}
    >>> loss_fn = LCCCLoss(alpha=0.5)
    >>> out = loss_fn(preds, targets)
    >>> print(out["loss"].item(), out["ccc_valence"].item(), out["ccc_arousal"].item())
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

from __future__ import annotations

from typing import Dict, Optional, Union

import torch
from torch import Tensor

from src.losses.base import BaseLoss


def _ensure_tensor(x: Union[Tensor, float, int]) -> Tensor:
    """Convert a numeric value to a tensor if necessary."""
    if isinstance(x, Tensor):
        return x
    return torch.tensor(x, dtype=torch.float32)


def concordance_ccc(pred: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Compute concordance correlation coefficient between two 1-D tensors.

    Args:
        pred: Prediction tensor of shape (B, 1) or (B,) on CPU or device.
        target: Target tensor of same shape as pred.
        eps: Small epsilon to avoid division by zero.

    Returns:
        Scalar tensor containing CCC in range [-1, 1].

    Raises:
        ValueError: If input shapes do not match or have incorrect dims.

    """
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    # Flatten to (B,)
    p = pred.view(-1).to(dtype=torch.float32)
    t = target.view(-1).to(dtype=torch.float32)
    if p.numel() == 0:
        return torch.tensor(0.0, device=p.device)
    mean_p = torch.mean(p)
    mean_t = torch.mean(t)
    var_p = torch.var(p, unbiased=False)
    var_t = torch.var(t, unbiased=False)
    cov = torch.mean((p - mean_p) * (t - mean_t))
    ccc = (2.0 * cov) / (var_p + var_t + (mean_p - mean_t) ** 2 + eps)
    return ccc


class LCCCLoss(BaseLoss):
    """
    Combined CCC loss for valence and arousal regression.

    Attributes:
        alpha: Weight for arousal CCC (float in [0,1]).
        eps: Numerical stability constant.

    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8, name: str = "lccc"):
        """
        Initialize the LCCC loss.

        Args:
            alpha: Weighting of arousal CCC (alpha in [0,1]). Default 0.5.
            eps: Small epsilon for numerical stability.
            name: Optional name for the loss instance.

        """
        super().__init__(name=name)
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(
        self,
        preds: Union[Dict[str, Tensor], Tensor],
        targets: Union[Dict[str, Tensor], Tensor],
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute the combined CCC loss.

        Args:
            preds: Predictions. Either a dict with keys 'valence' and
                'arousal' mapping to tensors of shape (B,1) or a tensor
                of shape (B,2) with columns [valence, arousal].
            targets: Targets in the same structure as preds.
            mask: Optional boolean mask (B,) to indicate valid samples.
                If provided, masked-out entries will be ignored in CCC
                computation.

        Returns:
            Dictionary with keys:
              - 'loss': scalar tensor for optimization
              - 'ccc_valence': scalar tensor
              - 'ccc_arousal': scalar tensor

        Raises:
            ValueError: If inputs are malformed.

        """
        # Extract valence/arousal tensors
        if isinstance(preds, dict):
            if "valence" not in preds or "arousal" not in preds:
                raise ValueError("preds dict must contain 'valence' and 'arousal'")
            pv = preds["valence"]
            pa = preds["arousal"]
        else:
            if preds.dim() != 2 or preds.size(1) != 2:
                raise ValueError("preds tensor must have shape (B,2)")
            pv = preds[:, 0:1]
            pa = preds[:, 1:2]

        if isinstance(targets, dict):
            if "valence" not in targets or "arousal" not in targets:
                raise ValueError("targets dict must contain 'valence' and 'arousal'")
            tv = targets["valence"]
            ta = targets["arousal"]
        else:
            if targets.dim() != 2 or targets.size(1) != 2:
                raise ValueError("targets tensor must have shape (B,2)")
            tv = targets[:, 0:1]
            ta = targets[:, 1:2]

        # Optionally apply mask by filtering valid indices
        if mask is not None:
            if mask.dim() > 1:
                mask = mask.view(-1)
            valid_idx = mask.nonzero(as_tuple=False).view(-1)
            if valid_idx.numel() == 0:
                # No valid samples: return zero loss but valid tensors
                zero = torch.tensor(0.0, dtype=torch.float32)
                return {"loss": zero, "ccc_valence": zero, "ccc_arousal": zero}
            pv = pv[valid_idx]
            pa = pa[valid_idx]
            tv = tv[valid_idx]
            ta = ta[valid_idx]

        # compute CCC per dimension
        ccc_v = concordance_ccc(pv, tv, eps=self.eps)
        ccc_a = concordance_ccc(pa, ta, eps=self.eps)

        # combined scalar (we want to maximize CCC, so loss = 1 - weighted_ccc)
        weighted = self.alpha * ccc_a + (1.0 - self.alpha) * ccc_v
        loss = 1.0 - weighted

        # ensure tensors are same dtype/device
        loss = loss.to(dtype=torch.float32)
        ccc_v = ccc_v.to(dtype=torch.float32)
        ccc_a = ccc_a.to(dtype=torch.float32)

        return {"loss": loss, "ccc_valence": ccc_v, "ccc_arousal": ccc_a}
