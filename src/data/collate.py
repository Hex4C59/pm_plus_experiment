#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Collate utilities for padded batching of sequence features.

The pad_collate_fn pads a list of items produced by the dataset into a batch
dictionary that contains:
  - features: Tensor[B, T_max, D]
  - lengths: LongTensor[B]
  - valence: FloatTensor[B,1] or None
  - arousal: FloatTensor[B,1] or None
  - speakers, audio_files, paths: lists of metadata strings

Example :
    >>> batch = [ds[i] for i in range(4)]
    >>> batch_collated = pad_collate_fn(batch)
    >>> print(batch_collated["features"].shape)
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

from typing import Dict, List

import torch
from torch import Tensor


def pad_collate_fn(
    batch: List[Dict[str, object]], pad_value: float = 0.0
) -> Dict[str, object]:
    """
    Pad variable-length feature sequences into a batched tensor.

    Args:
        batch: List of sample dicts returned by PooledFeatureDataset.
        pad_value: Value used for padding.

    Returns:
        A dictionary with batched tensors and metadata.

    Raises:
        ValueError: If batch is empty or contains malformed entries.

    """
    if not batch:
        raise ValueError("Received empty batch for collate")

    feats: List[Tensor] = [b["feature"] for b in batch]
    lengths = torch.tensor([int(b["length"]) for b in batch], dtype=torch.long)
    B = len(feats)
    D = int(feats[0].size(1))
    T_max = int(lengths.max().item())

    out = feats[0].new_full((B, T_max, D), float(pad_value))
    for i, f in enumerate(feats):
        t = int(f.size(0))
        out[i, :t] = f

    valence_tensor = None
    arousal_tensor = None
    if "valence" in batch[0] and batch[0]["valence"] is not None:
        vals = [float(b["valence"]) if b["valence"] is not None else 0.0 for b in batch]
        valence_tensor = torch.tensor(vals, dtype=torch.float32).unsqueeze(1)
    if "arousal" in batch[0] and batch[0]["arousal"] is not None:
        ars = [float(b["arousal"]) if b["arousal"] is not None else 0.0 for b in batch]
        arousal_tensor = torch.tensor(ars, dtype=torch.float32).unsqueeze(1)

    return {
        "features": out,
        "lengths": lengths,
        "valence": valence_tensor,
        "arousal": arousal_tensor,
        "speakers": [b.get("speaker") for b in batch],
        "audio_files": [b.get("audio_file") for b in batch],
        "paths": [b.get("path") for b in batch],
    }
