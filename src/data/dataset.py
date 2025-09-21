#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Pooled feature dataset for PM+ experiments.

This module implements PooledFeatureDataset which reads per-utterance
feature files saved as .pt (dict with key "feature") and optionally loads
annotation labels. The dataset returns per-item dictionaries suitable for
the collate function provided in src.data.collate.

Example :
    >>> ds = PooledFeatureDataset("data/processed/features", "train",
    ...                           model_folder="wav2vec2-l12",
    ...                           annotations="data/annotations/label.xlsx")
    >>> item = ds[0]
    >>> print(item["feature"].shape, item["valence"], item["arousal"])
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"


from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch import Tensor


class PooledFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset that yields per-utterance feature tensors and optional labels.

    Attributes:
        features_root: Root folder containing model-specific feature folders.
        split: Split name to load, e.g. "train" or "test".
        model_folder: Specific model subfolder under features_root. If None,
            the first child directory under features_root is used.
        exts: Allowed feature file extensions.
        samples: Ordered list of feature file paths.
        label_map: Mapping from audio filename or stem to (valence, arousal).

    """

    def __init__(
        self,
        features_root: str,
        split: str,
        model_folder: Optional[str] = None,
        exts: Optional[Sequence[str]] = None,
        annotations: Optional[str] = None,
        return_labels: bool = True,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            features_root: Root directory where feature folders live.
            split: Name of the split subfolder to read.
            model_folder: Optional model-specific folder name.
            exts: Allowed file extensions, default ['.pt'].
            annotations: Optional path to Excel/CSV with audio-level labels.
            return_labels: Whether to attempt to return valence/arousal.

        Raises:
            FileNotFoundError: If feature directory or annotations file not found.
            ValueError: If annotations miss required columns.

        """
        self.features_root = Path(features_root)
        self.split = split
        exts_list = list(exts) if exts is not None else [".pt"]
        self.exts = {e if e.startswith(".") else f".{e}" for e in exts_list}
        # Resolve model folder
        base = self._resolve_model_folder(model_folder)
        self.split_dir = base / split
        if not self.split_dir.exists() or not self.split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {self.split_dir}")

        # Gather feature files
        self.samples: List[Path] = sorted(
            p
            for p in self.split_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in self.exts
        )

        # Load annotations if provided
        self.label_map: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        if annotations and return_labels:
            ann_path = Path(annotations)
            if not ann_path.exists():
                raise FileNotFoundError(f"Annotations file not found: {ann_path}")
            df = self._read_annotations(ann_path)
            self._build_label_map(df)

    def _resolve_model_folder(self, model_folder: Optional[str]) -> Path:
        """
        Return the Path of model folder to use.

        Args:
            model_folder: Optional requested model folder name.

        Returns:
            Path to the resolved model folder.

        Raises:
            FileNotFoundError: If no candidate model folder exists.

        """
        if model_folder:
            cand = self.features_root / model_folder
            if not cand.exists():
                raise FileNotFoundError(f"Model folder not found: {cand}")
            return cand
        # pick the first child folder
        children = [p for p in self.features_root.iterdir() if p.is_dir()]
        if not children:
            raise FileNotFoundError(f"No model folders under: {self.features_root}")
        return sorted(children)[0]

    def _read_annotations(self, path: Path) -> pd.DataFrame:
        """
        Read an annotations file (Excel or CSV) into a DataFrame.

        Args:
            path: Path to the annotations file.

        Returns:
            pandas.DataFrame with annotations.

        Raises:
            ValueError: If file format is unsupported.

        """
        if path.suffix.lower() in {".xls", ".xlsx"}:
            return pd.read_excel(path, engine="openpyxl")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported annotation file format: {path.suffix}")

    def _build_label_map(self, df: pd.DataFrame) -> None:
        """
        Build mapping from audio file names/stems to (valence, arousal).

        Args:
            df: DataFrame containing at least 'audio_file' column and optional
                'v_value'/'a_value' columns.

        Raises:
            ValueError: If required columns are missing.

        """
        cols = {c.strip(): c for c in df.columns.astype(str)}
        if "audio_file" not in cols:
            raise ValueError("Annotations must contain column 'audio_file'")
        vcol = cols.get("v_value", None)
        acol = cols.get("a_value", None)
        for _, row in df.iterrows():
            fn = str(row[cols["audio_file"]]).strip()
            v = (
                float(row[vcol])
                if vcol is not None and not pd.isna(row[vcol])
                else None
            )
            a = (
                float(row[acol])
                if acol is not None and not pd.isna(row[acol])
                else None
            )
            key1 = fn
            key2 = Path(fn).stem
            self.label_map[key1] = (v, a)
            self.label_map[key2] = (v, a)

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        """
        Return a single sample dictionary.

        The returned dict contains:
            - feature: Tensor[T, D]
            - length: int (T)
            - valence: Optional[float]
            - arousal: Optional[float]
            - speaker: str
            - audio_file: str
            - path: str

        Args:
            index: Sample index.

        Returns:
            Sample dictionary.

        Raises:
            ValueError: If feature data is malformed.

        """
        p = self.samples[index]
        data = torch.load(str(p), map_location="cpu")
        feat = data.get("feature") if isinstance(data, dict) else data
        if not isinstance(feat, Tensor):
            raise ValueError(f"No tensor 'feature' in {p}")
        # Ensure shape (T, D)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        if feat.dim() == 3 and feat.size(0) == 1:
            feat = feat.squeeze(0)
        speaker = p.parent.name
        audio_file = f"{p.stem}.wav"
        length = int(feat.size(0))
        v = a = None
        # lookup labels by filename or stem
        if self.label_map:
            v_a = self.label_map.get(p.name) or self.label_map.get(p.stem)
            if v_a:
                v, a = v_a
        return {
            "feature": feat,
            "length": length,
            "valence": v,
            "arousal": a,
            "speaker": speaker,
            "audio_file": audio_file,
            "path": str(p),
        }
