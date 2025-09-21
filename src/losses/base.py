#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Abstract loss base class for SER experiments.

This module provides a lightweight abstract base class for loss objects
used in training and evaluation. Loss implementations should inherit from
BaseLoss and implement the `forward` method which computes a loss scalar
and may return auxiliary statistics for logging.

Example :
    >>> import torch
    >>> from src.losses.base import BaseLoss
    >>> class DummyLoss(BaseLoss):
    ...     def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask=None):
    ...         return {"loss": ((preds - targets) ** 2).mean()}
    >>> loss = DummyLoss()
    >>> out = loss(torch.randn(4,1), torch.randn(4,1))
    >>> print(out["loss"].item())
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"


import abc
from typing import Any, Dict, Optional

from torch import Tensor, nn


class BaseLoss(nn.Module, abc.ABC):
    """
    Abstract base loss class.

    Implementations must override forward to compute the loss and may return
    auxiliary metrics for logging.

    Attributes:
        name: Human readable loss name.

    """

    name: str

    def __init__(self, name: str = "base_loss") -> None:
        """Initialize the base loss with a name."""
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def forward(
        self,
        preds: Any,
        targets: Any,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Compute loss and optional diagnostics.

        Args:
            preds: Model predictions. Concrete implementations decide the
                expected structure (tensor or dict).
            targets: Ground-truth targets. Structure should match preds.
            mask: Optional boolean or byte mask (B, T) to ignore padded
                frames or invalid entries.

        Returns:
            A dictionary containing at least the key "loss" with a scalar
            tensor value. Additional items may include per-component
            metrics for logging.

        Raises:
            NotImplementedError: If the method is not implemented by subclass.

        """
        raise NotImplementedError
