#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Extract wav2vec2 features for the PM+_split dataset and save as .pt files.

This script extracts hidden states from a local wav2vec2 checkpoint for all
audio files organized under a split directory tree such as:
  data/processed/PM+_split/train/<speaker>/*.wav
  data/processed/PM+_split/test/<speaker>/*.wav

For each split it walks recursively, extracts features from a chosen
transformer layer, and saves feature files preserving the speaker-level
subdirectory structure:
  <out_root>/<model_name>-l<layer>/<split>/<speaker>/<utterance>.pt

The script supports chunked processing for long waveforms, multiple audio
extensions, device selection, and optional skipping of existing feature files.

Example:
    >>> python scripts/preprocess/extract_wav2vec2.py \\
    ...   --ckpt_dir pretrained/wav2vec2-base-100h \\
    ...   --data_root data/processed/PM+_split \\
    ...   --out_root data/processed/features \\
    ...   --layer 12 --device cuda --gpu 0

"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

import argparse
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model

AUDIO_EXTS: Set[str] = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


class Wav2Vec2Extractor:
    """
    Lightweight wav2vec2 feature extractor.

    Attributes:
        model: Loaded HuggingFace Wav2Vec2Model instance.
        device: Torch device used for inference.
        sample_rate: Target sample rate (16000).
        max_chunk: Maximum raw samples per forward pass.
        model_name: Basename of the checkpoint directory.

    """

    def __init__(self, ckpt_dir: str, device: str, max_chunk: int = 1_600_000):
        """
        Initialize model and device.

        Args:
            ckpt_dir: Directory containing HuggingFace checkpoint files.
            device: Torch device (e.g., 'cpu' or 'cuda:0').
            max_chunk: Max raw samples per forward (default ~100s @16kHz).

        Raises:
            FileNotFoundError: When required checkpoint files are missing.

        """
        cfg_path = Path(ckpt_dir) / "config.json"
        bin_path = Path(ckpt_dir) / "pytorch_model.bin"
        if not cfg_path.exists() or not bin_path.exists():
            raise FileNotFoundError("Missing config.json or pytorch_model.bin")

        config = Wav2Vec2Config.from_pretrained(ckpt_dir)
        self.model = Wav2Vec2Model.from_pretrained(
            ckpt_dir, config=config, local_files_only=True
        )
        self.model.eval()
        self.device = torch.device(device)
        self.model.to(self.device)  # type: ignore

        self.sample_rate = 16_000
        self.max_chunk = max_chunk
        self.model_name = Path(ckpt_dir).name

    def read_audio(self, path: str) -> np.ndarray:
        """
        Read audio, convert to mono and resample to 16 kHz.

        Args:
            path: Audio file path.

        Returns:
            Float32 mono waveform resampled to 16 kHz.

        Raises:
            RuntimeError: If reading fails.

        """
        wav, sr = sf.read(path, always_2d=False)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        wav = wav.astype(np.float32, copy=False)
        if sr != self.sample_rate:
            dur = wav.shape[0] / float(sr)
            tgt_len = int(dur * self.sample_rate)
            idx = np.linspace(0, wav.shape[0] - 1, tgt_len, dtype=np.float32)
            left = np.floor(idx).astype(np.int64)
            right = np.minimum(left + 1, wav.shape[0] - 1)
            frac = idx - left
            wav = (1.0 - frac) * wav[left] + frac * wav[right]
        return wav

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardize waveform to zero mean and unit variance.

        Args:
            x: Tensor with shape (1, T).

        Returns:
            Normalized tensor with same shape.

        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def extract(self, wav_path: str, layer: int = 12) -> torch.Tensor:
        """
        Extract hidden states from a specified transformer layer.

        Args:
            wav_path: Path to a wav file.
            layer: Transformer layer index (1..L). Out of range uses last.

        Returns:
            Feature tensor with shape (T', D) on CPU.

        """
        wav = self.read_audio(wav_path)
        t = torch.from_numpy(wav).to(self.device, dtype=torch.float32)[None, :]
        t = self._norm(t)

        chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for s in range(0, t.size(1), self.max_chunk):
                x = t[:, s : s + self.max_chunk]
                out = self.model(x, output_hidden_states=True)
                hs = out.hidden_states
                idx = min(layer, len(hs) - 1)
                chunks.append(hs[idx])

        feat = torch.cat(chunks, dim=1).squeeze(0)
        return feat.cpu()

    def save_feature(self, feat: torch.Tensor, save_path: str) -> None:
        """
        Save feature tensor to a .pt file.

        Args:
            feat: Feature tensor (T', D).
            save_path: Path without extension; '.pt' will be appended.

        Returns:
            None

        """
        out = {"feature": feat.contiguous()}
        torch.save(out, f"{save_path}.pt")


def process_split(
    extractor: Wav2Vec2Extractor,
    split: str,
    split_dir: str,
    out_root: str,
    layer: int,
    exts: Set[str],
) -> Tuple[int, int]:
    """
    Process a dataset split directory recursively and save features.

    This function walks split_dir recursively, finds audio files with allowed
    extensions, extracts features, and writes them under:
      <out_root>/<model>-l<layer>/<split>/<relative_path_without_ext>.pt

    Args:
        extractor: Feature extractor instance.
        split: Split name (e.g., 'train' or 'test').
        split_dir: Directory path containing speakers subdirs.
        out_root: Root directory to save features.
        layer: Transformer layer index.
        exts: Allowed file extensions (lowercase, with dot).

    Returns:
        Tuple of (processed_count, total_count).

    """
    split_path = Path(split_dir)
    feat_dirname = f"{extractor.model_name}-l{layer}"
    base_out = Path(out_root) / feat_dirname / split
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"Saving features to: {base_out}")
    wav_files = sorted(
        [p for p in split_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    )

    total = len(wav_files)
    done = 0

    for i, wav_path in enumerate(wav_files, 1):
        try:
            rel = wav_path.relative_to(split_path)
        except Exception:
            # Fallback to name only if relative computation fails.
            rel = Path(wav_path.name)
        out_subdir = base_out / rel.parent
        out_subdir.mkdir(parents=True, exist_ok=True)
        save_base = out_subdir / rel.stem
        if (save_base.with_suffix(".pt")).exists():
            done += 1
            continue
        feat = extractor.extract(str(wav_path), layer=layer)
        extractor.save_feature(feat, str(save_base))
        done += 1
        if i % 100 == 0:
            print(f"[{split}] {i}/{total}")

    return done, total


def parse_exts(exts_arg: str) -> Set[str]:
    """
    Parse comma-separated extensions into a set with leading dots.

    Args:
        exts_arg: Comma-separated extensions like 'wav,flac'.

    Returns:
        Set of normalized extensions (e.g., {'.wav', '.flac'}).

    """
    exts = {("." + e.strip().lower().lstrip(".")) for e in exts_arg.split(",")}
    return exts or AUDIO_EXTS


def main() -> None:
    """CLI entry for batch feature extraction for PM+_split."""
    parser = argparse.ArgumentParser(
        description="Extract wav2vec2 features for PM+_split directory tree"
    )
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root folder containing split subdirs (e.g. data/processed/PM+_split)",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root where feature folders will be created",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument(
        "--max_chunk", type=int, default=1_600_000, help="Max samples per forward"
    )
    parser.add_argument(
        "--exts",
        type=str,
        default="wav",
        help="Comma-separated audio extensions to process (default: wav)",
    )
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    extractor = Wav2Vec2Extractor(
        ckpt_dir=args.ckpt_dir, device=device, max_chunk=args.max_chunk
    )

    exts = parse_exts(args.exts)
    print(f"Using model: {extractor.model_name}")
    print(f"Extracting layer: {args.layer}")
    print(f"Audio extensions: {sorted(exts)}")
    print(f"Selected device: {device}")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Process each immediate child directory under data_root as a split
    total_done = 0
    total_all = 0
    for split_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        split_name = split_dir.name
        done, tot = process_split(
            extractor,
            split_name,
            str(split_dir),
            args.out_root,
            args.layer,
            exts,
        )
        print(f"{split_name}: {done}/{tot}")
        total_done += done
        total_all += tot

    print(f"All done: {total_done}/{total_all}")


if __name__ == "__main__":
    main()
