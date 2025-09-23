#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Self-attentive pooling + BiLSTM regressor for valence and arousal.

This module implements a BiLSTM encoder over frame-level features and
a self-attentive pooling layer that produces a fixed-size utterance
embedding. The embedding is passed through a small MLP trunk and two
separate linear heads predicting continuous valence and arousal scores.

The design supports variable-length sequences via a lengths mask and
provides a from_config constructor to simplify integration with the
existing experiment configs.

Example :
    >>> import torch
    >>> from src.models.attn_bilstm import SelfAttentiveBiLSTMRegressor
    >>> cfg = {"input_dim":768, "lstm_hidden":256, "lstm_layers":1,
    ...        "mlp_hidden":[256], "dropout":0.2}
    >>> model = SelfAttentiveBiLSTMRegressor.from_config(cfg)
    >>> x = torch.randn(4, 120, 768)  # (B, T, D)
    >>> lengths = torch.randint(80, 120, (4,))
    >>> out = model(x, lengths=lengths)
    >>> out["valence"].shape, out["arousal"].shape
    (torch.Size([4, 1]), torch.Size([4, 1]))
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-21"

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from src.models.base_model import BaseModel


def _ensure_batch_dim(x: Tensor) -> Tuple[Tensor, bool]:
    """Ensure input has batch dim; return (tensor, was_batched_flag)."""
    if x.dim() == 2:
        return x.unsqueeze(0), True
    return x, False


def _compute_mask_from_lengths(
    lengths: Optional[Tensor], max_len: int
) -> Optional[Tensor]:
    """Create boolean mask (B, T) from lengths tensor or return None."""
    if lengths is None:
        return None
    device = lengths.device
    idx = torch.arange(max_len, device=device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def _masked_softmax(
    scores: torch.Tensor, mask: Optional[torch.Tensor], dim: int = 1
) -> torch.Tensor:
    """
    Apply softmax over scores with optional boolean mask.

    The mask may be provided in several shapes:
      - (B, T) boolean mask
      - (B, T, 1) boolean mask
      - (T,)  lengths-like mask (broadcasted across batch)
    This function normalizes the mask to match the scores shape and
    applies a large negative value to masked positions before softmax.

    Args:
        scores: Tensor of shape (B, T) or (..., T) over which to softmax.
        mask: Optional boolean Tensor indicating valid positions.
        dim: Dimension over which to apply softmax.

    Returns:
        Tensor with the same shape as scores containing softmaxed values.

    Raises:
        ValueError: If mask cannot be broadcast to scores shape.

    """
    if mask is None:
        return torch.softmax(scores, dim=dim)

    # Ensure mask is boolean and on same device as scores.
    mask = mask.to(device=scores.device, dtype=torch.bool)

    # Collapse trailing singleton dims like (B, T, 1) -> (B, T)
    if mask.dim() == scores.dim() and mask.size(-1) == 1 and scores.dim() >= 2:
        mask = mask.squeeze(-1)

    # If mask is 1-D (T,), broadcast to (B, T)
    if mask.dim() == 1 and scores.dim() >= 2:
        mask = mask.unsqueeze(0).expand(scores.size(0), -1)

    # Try to expand mask to scores shape if possible
    try:
        if mask.shape != scores.shape:
            mask = mask.expand_as(scores)
    except Exception:
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} is not compatible with "
            f"scores shape {tuple(scores.shape)}"
        )

    neg_inf = -1e9
    scores = scores.masked_fill(~mask, neg_inf)
    return torch.softmax(scores, dim=dim)


class SelfAttentiveBiLSTMRegressor(BaseModel):
    """
    BiLSTM encoder with self-attentive pooling and MLP heads.

    Attributes:
        lstm: Bidirectional LSTM encoder producing contextual frames.
        attn_proj: Linear projection for attention scoring.
        trunk: Shared MLP trunk applied to pooled embedding.
        head_valence: Linear head producing valence scalar.
        head_arousal: Linear head producing arousal scalar.

    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int = 256,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        attn_dim: int = 128,
        mlp_hidden: Sequence[int] = (256,),
        dropout: float = 0.1,
        name: str = "attn_bilstm",
    ) -> None:
        """
        Initialize the SelfAttentiveBiLSTMRegressor.

        Args:
            input_dim: Dimension of per-frame features (D).
            lstm_hidden: Hidden size for the LSTM (per direction).
            lstm_layers: Number of stacked LSTM layers.
            bidirectional: Whether LSTM is bidirectional.
            attn_dim: Internal dimension for attention projection.
            mlp_hidden: Sequence of hidden sizes for shared MLP trunk.
            dropout: Dropout probability used in trunk and between LSTM layers.
            name: Optional model name.

        """
        super().__init__(name=name)
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.bidirectional = bidirectional
        self.dropout = float(dropout)

        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=self.dropout if lstm_layers > 1 else 0.0,
        )

        enc_dim = lstm_hidden * self.num_directions
        # attention: project encoder outputs -> attn_dim -> score
        self.attn_proj = nn.Sequential(
            nn.Linear(enc_dim, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

        # MLP trunk after pooling
        trunk_layers: List[nn.Module] = []
        prev = enc_dim
        for h in mlp_hidden:
            trunk_layers.append(nn.Linear(prev, h))
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Dropout(self.dropout))
            prev = h
        self.trunk = nn.Sequential(*trunk_layers)

        # final heads
        self.head_valence = nn.Linear(prev, 1)
        self.head_arousal = nn.Linear(prev, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize linear layers with Xavier uniform and zero biases."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        features: Tensor,
        lengths: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass: encode, self-attentive pool, and predict.

        Args:
            features: Tensor of shape (B, T, D) or (T, D).
            lengths: Optional int tensor (B,) with frame counts.
            return_attn: If True include attention weights in output.

        Returns:
            Dict with keys 'valence' and 'arousal' (each (B,1)). If
            return_attn is True an 'attn' key with (B, T) weights is
            also returned.

        Raises:
            ValueError: If input tensor has unsupported dimensions.

        """
        x, was_1d = _ensure_batch_dim(features)
        if x.dim() != 3:
            raise ValueError("features must have shape (B, T, D) or (T, D)")
        b, t, d = x.size()

        # Pack/Unpack to handle variable lengths efficiently
        if lengths is not None:
            # sort by decreasing lengths for pack_padded_sequence
            lengths_sorted, idx_sort = torch.sort(lengths, descending=True)
            _, idx_unsort = torch.sort(idx_sort)
            x_sorted = x[idx_sort]
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted.cpu(), batch_first=True, enforce_sorted=True
            )
            packed_out, _ = self.lstm(packed)
            out_sorted, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=t
            )
            enc_out = out_sorted[idx_unsort]
            mask = _compute_mask_from_lengths(lengths, t)
        else:
            enc_out, _ = self.lstm(x)
            mask = None

        # attention scores and masked softmax
        scores = self.attn_proj(enc_out).squeeze(-1)  # (B, T)
        attn_weights = _masked_softmax(scores, mask, dim=1)  # (B, T)

        # weighted sum -> pooled embedding
        pooled = torch.sum(enc_out * attn_weights.unsqueeze(-1), dim=1)  # (B, enc_dim)

        emb = self.trunk(pooled) if len(self.trunk) > 0 else pooled
        val = self.head_valence(emb)
        aro = self.head_arousal(emb)

        if was_1d:
            val = val.squeeze(0)
            aro = aro.squeeze(0)

        out: Dict[str, Tensor] = {"valence": val, "arousal": aro}
        if return_attn:
            out["attn"] = attn_weights
        return out

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "SelfAttentiveBiLSTMRegressor":
        """
        Construct model from experiment configuration dictionary.

        Args:
            cfg: Configuration containing keys:
                - input_dim (int)
                - lstm_hidden (int, optional)
                - lstm_layers (int, optional)
                - bidirectional (bool, optional)
                - attn_dim (int, optional)
                - mlp_hidden (sequence, optional)
                - dropout (float, optional)

        Returns:
            Initialized SelfAttentiveBiLSTMRegressor instance.

        Raises:
            KeyError: If required keys are missing.

        """
        return cls(
            input_dim=int(cfg["input_dim"]),
            lstm_hidden=int(cfg.get("lstm_hidden", 256)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
            bidirectional=bool(cfg.get("bidirectional", True)),
            attn_dim=int(cfg.get("attn_dim", 128)),
            mlp_hidden=tuple(cfg.get("mlp_hidden", (256,))),
            dropout=float(cfg.get("dropout", 0.1)),
            name=cfg.get("name", "attn_bilstm"),
        )
