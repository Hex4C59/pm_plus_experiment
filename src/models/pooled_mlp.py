#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Pooled MLP regression model for valence and arousal prediction.

This module implements a simple pooled-features + MLP model that accepts
frame-level features (e.g., wav2vec2 hidden states) and predicts two
continuous values: valence and arousal. It pools temporal frames per
utterance using configurable pooling (mean, mean+std, mean+max) to obtain a
fixed-size embedding, then applies a shared MLP trunk and two separate
regression heads.

Usage:
    >>> import torch
    >>> from src.models.pooled_mlp import PooledMLPRegressor
    >>> model = PooledMLPRegressor.from_config({
    ...     "input_dim": 768,
    ...     "hidden_dims": [512, 256],
    ...     "dropout": 0.2,
    ...     "pooling": "mean_std"
    ... })
    >>> x = torch.randn(4, 120, 768)  # (batch, time, dim)
    >>> out = model(x)
    >>> out["valence"].shape, out["arousal"].shape
    (torch.Size([4, 1]), torch.Size([4, 1]))

Example :
    >>> # Create and run a forward pass
    >>> model = PooledMLPRegressor(input_dim=768, hidden_dims=(256,), dropout=0.1)
    >>> feat = torch.randn(2, 100, 768)
    >>> preds = model(feat)
    >>> print(preds["valence"].shape, preds["arousal"].shape)
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

from typing import Sequence

from torch import nn

from src.models.base_model import BaseModel


class PooledMLPRegressor(BaseModel):
    """
    Pooled features + MLP regressor with two heads.

    The model pools variable-length frame features into a fixed vector and
    passes it through a shared MLP trunk followed by two independent linear
    heads producing scalar outputs for valence and arousal.

    Attributes :
        input_dim: Dimension of input frame features.
        pooling: Pooling method name ('mean', 'mean_std', 'mean_max').
        trunk: Shared MLP trunk (nn.Sequential).
        head_valence: Linear head predicting valence (output dim 1).
        head_arousal: Linear head predicting arousal (output dim 1).
        dropout: Dropout module used between trunk layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (256, 128),
        pooling: str = "mean",
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU(),
        name: str = "pooled_mlp",
    ) -> None:
        """
        Initialize the PooledMLPRegressor.

        Args:
            input_dim: Dimension of per-frame features (D).
            hidden_dims: Sequence of trunk hidden layer sizes.
            pooling: Pooling method: 'mean', 'mean_std', or 'mean_max'.
            dropout: Dropout probability between trunk layers.
            activation: Activation module to use between layers.
            name: Optional model name for checkpointing.

        """
        super().__init__(name=name)
        self.input_dim = input_dim
        self.pooling = pooling
        self.dropout = dropout
        trunk_in_dim = input_dim