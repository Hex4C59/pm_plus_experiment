#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Data module exports for PM+ experiments.

This module re-exports dataset and data-loading utilities used by training
and evaluation scripts. It keeps the package surface small and stable so
other modules import from src.data rather than deep paths.

Example :
    >>> from src.data import PooledFeatureDataset, make_dataloader
    >>> ds = PooledFeatureDataset(features_root="data/processed/features",
    ...                           split="train", annotations=None)
    >>> dl = make_dataloader(features_root="data/processed/features",
    ...                      split="train", batch_size=8)
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

from .collate import pad_collate_fn  # noqa: E402
from .dataset import PooledFeatureDataset  # noqa: E402
from .loader import make_dataloader  # noqa: E402

__all__ = ["PooledFeatureDataset", "pad_collate_fn", "make_dataloader"]
