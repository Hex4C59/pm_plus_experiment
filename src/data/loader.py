#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
DataLoader factory for pooled feature datasets.

This module provides a convenience function to create a PyTorch DataLoader
for PooledFeatureDataset with the pad_collate_fn collator.

Example :
    >>> dl = make_dataloader(features_root="data/processed/features",
    ...                      split="train", batch_size=16, num_workers=4)
    >>> batch = next(iter(dl))
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

from typing import Optional

from torch.utils.data import DataLoader

from .collate import pad_collate_fn
from .dataset import PooledFeatureDataset


def make_dataloader(
    features_root: str,
    split: str,
    model_folder: Optional[str] = None,
    annotations: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for pooled feature datasets.

    Args:
        features_root: Root directory containing feature folders.
        split: Split name to load.
        model_folder: Optional model-specific folder name.
        annotations: Optional annotation file path for labels.
        batch_size: Batch size.
        shuffle: Whether to shuffle dataset.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory in DataLoader.

    Returns:
        Configured torch.utils.data.DataLoader.

    """
    dataset = PooledFeatureDataset(
        features_root=features_root,
        split=split,
        model_folder=model_folder,
        annotations=annotations,
        return_labels=(annotations is not None),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=pad_collate_fn,
    )
