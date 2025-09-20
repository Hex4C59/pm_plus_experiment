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

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from src.models.base_model import BaseModel


def _ensure_batch_dim(x: Tensor) -> Tuple[Tensor, bool]:
    """
    Ensure input tensor has a batch dimension.

    Args:
        x: Input tensor of shape (T, D) or (B, T, D).

    Returns:
        A tuple (tensor_with_batch, was_1d_input) where was_1d_input
        indicates the original input had no batch dimension.

    """
    if x.dim() == 2:
        return x.unsqueeze(0), True
    return x, False


def _compute_mask_from_lengths(
    lengths: Optional[Tensor], max_len: int
) -> Optional[Tensor]:
    """
    Create boolean mask (B, T) from lengths tensor.

    Args:
        lengths: Optional int tensor of shape (B,) with valid frame counts.
        max_len: Maximum time length T.

    Returns:
        Bool mask or None if lengths is None.

    """
    if lengths is None:
        return None
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    return mask


def pooled_statistics(
    features: Tensor, mask: Optional[Tensor], method: str = "mean"
) -> Tensor:
    """
    Pool temporal features into a fixed-size embedding.

    Supported methods: "mean", "mean_std", "mean_max".

    Args:
        features: Tensor of shape (B, T, D).
        mask: Optional boolean mask of shape (B, T) where True marks valid
            frames. If None, all frames are valid.
        method: Pooling method name.

    Returns:
        Pooled tensor of shape (B, P) where P depends on method.

    Raises:
        ValueError: If an unsupported pooling method is requested.

    """
    if mask is None:
        valid = features
        lengths = features.new_full((features.size(0),), features.size(1))
    else:
        mask_f = mask.unsqueeze(-1).to(features.dtype)
        valid = features * mask_f
        lengths = mask.sum(dim=1).clamp_min(1).to(features.dtype)

    mean = valid.sum(dim=1) / lengths.unsqueeze(1)

    if method == "mean":
        return mean

    if method == "mean_std":
        # compute variance with mask-aware second moment
        sq = (valid * valid).sum(dim=1) / lengths.unsqueeze(1)
        var = sq - mean * mean
        std = var.clamp_min(0.0).sqrt()
        return torch.cat([mean, std], dim=1)

    if method == "mean_max":
        # masked max: set invalid frames to very small value
        if mask is None:
            _max, _ = features.max(dim=1)
        else:
            neg_inf = torch.finfo(features.dtype).min
            masked = features.masked_fill(~mask.unsqueeze(-1), neg_inf)
            _max, _ = masked.max(dim=1)
        return torch.cat([mean, _max], dim=1)

    raise ValueError(f"Unsupported pooling method: {method}")


class PooledMLPRegressor(BaseModel):
    """
    Pooled features + MLP regressor with two heads.

    The model pools variable-length frame features into a fixed vector and
    passes it through a shared MLP trunk followed by two independent linear
    heads producing scalar outputs for valence and arousal.

    Attributes:
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
        self.dropout = nn.Dropout(dropout)
        trunk_in_dim = input_dim
        if pooling == "mean_std" or pooling == "mean_max":
            trunk_in_dim = input_dim * 2

        trunk_layers: List[nn.Module] = []
        prev = trunk_in_dim
        for h in hidden_dims:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(activation)
            trunk_layers.append(self.dropout)
            prev = h
        # final trunk layer outputs embedding for heads
        self.trunk = nn.Sequential(*trunk_layers)
        self.head_valence = nn.Linear(prev, 1)
        self.head_arousal = nn.Linear(prev, 1)

        # Initialize weights with a simple scheme
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier uniform (KISS)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        pooling: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        """
        Run a forward pass.

        Args:
            features: Input tensor of shape (B, T, D) or (T, D).
            lengths: Optional int tensor (B,) giving valid frame counts;
                used to construct a mask. If None, all frames are valid.
            pooling: Optional override for the instance pooling method.

        Returns:
            A dict with keys:
                'valence': Tensor of shape (B, 1)
                'arousal': Tensor of shape (B, 1)

        Raises:
            ValueError: If pooling method is unsupported.

        """
        x, was_1d = _ensure_batch_dim(features)
        if x.dim() != 3:
            raise ValueError("features must have shape (B, T, D) or (T, D)")

        b, t, d = x.size()
        use_pool = pooling or self.pooling
        mask = _compute_mask_from_lengths(lengths, t) if lengths is not None else None

        pooled = pooled_statistics(x, mask, method=use_pool)
        emb = self.trunk(pooled)
        val = self.head_valence(emb)
        aro = self.head_arousal(emb)

        if was_1d:
            val = val.squeeze(0)
            aro = aro.squeeze(0)
        return {"valence": val, "arousal": aro}

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "PooledMLPRegressor":
        """
        Build the model from a configuration dictionary.

        Args:
            cfg: Configuration dict containing keys:
                - input_dim (int): frame feature dim
                - hidden_dims (Sequence[int], optional)
                - pooling (str, optional)
                - dropout (float, optional)

        Returns:
            Initialized PooledMLPRegressor instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return cls(
            input_dim=int(cfg["input_dim"]),
            hidden_dims=tuple(cfg.get("hidden_dims", (256, 128))),
            pooling=cfg.get("pooling", "mean"),
            dropout=float(cfg.get("dropout", 0.1)),
            name=cfg.get("name", "pooled_mlp"),
        )
