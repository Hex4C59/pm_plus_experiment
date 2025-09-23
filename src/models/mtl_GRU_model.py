#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Multi-task GRU model for emotion recognition.

This module provides a small, configurable multi-task GRU-based model that
produces continuous regression outputs (e.g., V/A/D or VAD) and an optional
classification head. The model is intended as a drop-in baseline for
frame-level features (shape: B x T x D). It exposes a from_config helper
that facilitates construction from experiment configuration dictionaries.

Example :
    >>> from src.models.mtl_GRU_model import BaselineEmotionGRU
    >>> import torch
    >>> model = BaselineEmotionGRU.from_config({
    ...     "input_size": 768,
    ...     "output_size": 3,
    ...     "num_classes": 7,
    ...     "hidden_size": 128,
    ... })
    >>> x = torch.randn(4, 120, 768)
    >>> out = model(x)
    >>> print(out["regression"].shape, "regression present")
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-23"

from typing import Dict, Optional

from torch import Tensor, nn

try:
    from .base_model import BaseModel
except Exception:  # pragma: no cover - fallback if import context differs
    BaseModel = nn.Module  # type: ignore


class GRUModel(BaseModel):
    """
    GRU model for emotion regression with optional classification head.

    This class implements a simple GRU encoder followed by a projection
    to a shared embedding and separate regression/classification heads.

    Attributes:
        gru: GRU encoder module (batch_first=True).
        embedding: Linear layer that projects GRU hidden to embedding.
        regression_head: Linear layer producing continuous outputs.
        classification_head: Optional linear layer for classification.
        use_classification: Flag enabling classification head.

    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_classes: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.2,
        use_classification: bool = True,
    ) -> None:
        """
        Initialize the GRU baseline.

        Args:
            input_size: Dimensionality of input frame features.
            output_size: Dimensionality of regression output (e.g., 3 for VAD).
            num_classes: Number of classes for classification head.
            hidden_size: Hidden size of the GRU (per direction if bidir).
            num_layers: Number of stacked GRU layers.
            embedding_dim: Size of shared embedding before heads.
            dropout: Dropout probability between GRU layers.
            use_classification: Whether to include classification head.

        """
        super().__init__()
        self.use_classification = bool(use_classification)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=float(dropout) if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.embedding = nn.Linear(hidden_size, embedding_dim)
        self.regression_head = nn.Linear(embedding_dim, output_size)

        self.classification_head: Optional[nn.Linear]
        if self.use_classification:
            self.classification_head = nn.Linear(embedding_dim, num_classes)
        else:
            self.classification_head = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier uniform and zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Run a forward pass.

        The model pools the sequence by taking the last GRU output along time
        dimension and then applies a projection and heads.

        Args:
            x: Input tensor of shape (B, T, D).

        Returns:
            A dict containing:
            - "regression": Tensor of shape (B, output_size).
            - "classification": Tensor of shape (B, num_classes) if enabled.

        Raises:
            ValueError: If input tensor rank is not 3.

        """
        if x.dim() != 3:
            raise ValueError("Input tensor must be 3D (B, T, D)")

        outputs, _ = self.gru(x)  # outputs: (B, T, hidden_size)
        last_step = outputs[:, -1, :]  # (B, hidden_size)
        embedding = self.embedding(last_step)
        regression_output = self.regression_head(embedding)

        result: Dict[str, Tensor] = {"regression": regression_output}

        if self.use_classification and self.classification_head is not None:
            classification_output = self.classification_head(embedding)
            result["classification"] = classification_output

        return result

    @classmethod
    def from_config(cls, cfg: Dict[str, object]) -> "GRUModel":
        """
        Build an instance from a configuration dictionary.

        Args:
            cfg: Configuration dict with keys:
                - input_size (int)
                - output_size (int)
                - num_classes (int, optional)
                - hidden_size (int, optional)
                - num_layers (int, optional)
                - embedding_dim (int, optional)
                - dropout (float, optional)
                - use_classification (bool, optional)

        Returns:
            An initialized BaselineEmotionGRU instance.

        Raises:
            KeyError: If required keys are missing in cfg.

        """
        return cls(
            input_size=int(cfg["input_size"]),
            output_size=int(cfg["output_size"]),
            num_classes=int(cfg.get("num_classes", 7)),
            hidden_size=int(cfg.get("hidden_size", 128)),
            num_layers=int(cfg.get("num_layers", 2)),
            embedding_dim=int(cfg.get("embedding_dim", 128)),
            dropout=float(cfg.get("dropout", 0.2)),
            use_classification=bool(cfg.get("use_classification", True)),
        )
