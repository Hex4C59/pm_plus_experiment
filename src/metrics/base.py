#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Base metric interfaces for PM+ experiments.

This module defines an abstract BaseMetric class used by concrete metric
implementations. BaseMetric provides a minimal stateful interface with
update/reset/compute methods to support streaming or batched evaluation.
Concrete metrics should inherit from BaseMetric and implement the three
abstract methods.

Example :
    >>> from src.metrics.base import BaseMetric
    >>> class Dummy(BaseMetric):
    ...     def __init__(self): super().__init__("dummy")
    ...     def update(self, preds, targets, mask=None): pass
    ...     def compute(self): return 0.0
    ...     def reset(self): pass
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"


import abc
from typing import Optional

import torch


class BaseMetric(abc.ABC):
    """
    Abstract base class for stateful evaluation metrics.

    Attributes:
        name: Human readable metric name.

    """

    name: str

    def __init__(self, name: str = "base_metric") -> None:
        """Initialize the metric with a name."""
        self.name = name

    @abc.abstractmethod
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Update internal state with a batch of predictions and targets.

        Args:
            preds: Prediction tensor, shape (B,) or (B,1).
            targets: Target tensor, shape matching preds.
            mask: Optional boolean mask (B,) indicating valid entries.

        Raises:
            ValueError: If shapes are incompatible.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self) -> torch.Tensor:
        """
        Compute and return the metric value from accumulated state.

        Returns:
            A scalar torch.Tensor containing the metric.

        Raises:
            NotImplementedError: If not implemented by subclass.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the internal accumulation state."""
        raise NotImplementedError
