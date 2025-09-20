#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch extraction of HuBERT features for speech emotion recognition.

This module provides a command-line interface to extract hidden states from
HuBERT models for a dataset of audio files. Features are saved in .pt format
for downstream emotion regression/classification tasks. Supports chunked
processing for long audio and flexible layer selection.

Example :
    >>> python scripts/extract_hubert.py \
            --ckpt_dir pretrain_model/hubert-base-100h \
            --data_root data/raw \
            --out_root data/processed/features \
            --layer 12 \
            --device cuda \
            --gpu_id 1
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-11-15"

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import HubertConfig, HubertModel


class HubertExtractor:
    """
    Lightweight HuBERT feature extractor.

    Detailed description of the class.

    Attributes :
        model (HubertModel): HuggingFace HuBERT model.
        device (torch.device): Device for computation.
        sample_rate (int): Expected sample rate (Hz).
        max_chunk (int): Max samples per forward pass.
        model_name (str): Pretrained model folder name.
    """

    def __init__(self, ckpt_dir: str, device: torch.device, max_chunk: int = 1_600_000):
        cfg_path = Path(ckpt_dir) / "config.json"
        bin_path = Path(ckpt_dir) / "pytorch_model.bin"
        if not cfg_path.exists() or not bin_path.exists():
            raise FileNotFoundError("Missing config.json or pytorch_model.bin")

        config = HubertConfig.from_pretrained(ckpt_dir)
        self.model = HubertModel.from_pretrained(
            ckpt_dir, config=config, local_files_only=True
        )
        self.model.eval()
        self.device = device
        self.model.to(self.device)  # type: ignore

        self.sample_rate = 16_000
        self.max_chunk = max_chunk
        self.model_name = Path(ckpt_dir).name

    def read_audio(self, path: str) -> np.ndarray:
        """
        Read an audio file, convert to mono, resample if needed.

        Args :
            path (str): Audio file path.

        Returns :
            np.ndarray: Float32 mono waveform at 16 kHz.

        Raises :
            RuntimeError: When reading audio fails.
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

        Args :
            x (torch.Tensor): Tensor shape (1, T).

        Returns :
            torch.Tensor: Normalized tensor (1, T).

        Raises :
            None
        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def extract(self, wav_path: str, layer: int = 12) -> torch.Tensor:
        """
        Extract hidden states from the specified transformer layer.

        Args :
            wav_path (str): Path to a wav file.
            layer (int): Transformer layer index (1..num_hidden_layers).

        Returns :
            torch.Tensor: Tensor of shape (T', D).

        Raises :
            None
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
        Save features to a .pt file.

        Args :
            feat (torch.Tensor): Feature tensor (T', D).
            save_path (str): Path without extension.

        Returns :
            None

        Raises :
            None
        """
        out = {"feature": feat.contiguous()}
        torch.save(out, f"{save_path}.pt")


def process_split(
    extractor: HubertExtractor,
    split: str,
    wav_dir: str,
    out_root: str,
    layer: int,
) -> Tuple[int, int]:
    """
    Process a dataset split directory.

    Args :
        extractor (HubertExtractor): Feature extractor instance.
        split (str): Split name ('train'|'validation'|'test').
        wav_dir (str): Directory containing wav files.
        out_root (str): Root dir to save features.
        layer (int): Transformer layer index.

    Returns :
        Tuple[int, int]: (processed_count, total_count).

    Raises :
        None
    """
    wav_dir_p = Path(wav_dir)
    feat_dirname = f"{extractor.model_name}-l{layer}"
    out_dir = Path(out_root) / feat_dirname / split
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving features to: {out_dir}")

    wav_files = sorted([p for p in wav_dir_p.iterdir() if p.suffix == ".wav"])
    total = len(wav_files)
    done = 0

    for i, wav_path in enumerate(wav_files, 1):
        name = wav_path.stem
        save_path = out_dir / name
        if (save_path.with_suffix(".pt")).exists():
            done += 1
            continue
        feat = extractor.extract(str(wav_path), layer=layer)
        extractor.save_feature(feat, str(save_path))
        done += 1
        if i % 100 == 0:
            print(f"[{split}] {i}/{total}")

    return done, total


def main() -> None:
    """
    CLI entry for batch HuBERT feature extraction.

    Detailed description of the function.

    Args :
        None

    Returns :
        None

    Raises :
        SystemExit: When input dirs are invalid.
    """
    parser = argparse.ArgumentParser(description="Extract HuBERT features")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpu_id", type=int, default=0, help="GPU device id (default: 0)"
    )
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--max_chunk", type=int, default=1_600_000)
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    extractor = HubertExtractor(
        ckpt_dir=args.ckpt_dir, device=device, max_chunk=args.max_chunk
    )

    print(f"Using model: {extractor.model_name}")
    print(f"Extracting layer: {args.layer}")
    print(f"Output folder will be: {extractor.model_name}-l{args.layer}")
    print(f"Using device: {device}")

    splits = ["test", "train", "validation"]
    total_done = 0
    total_all = 0
    for sp in splits:
        wav_dir = str(Path(args.data_root) / sp)
        if not Path(wav_dir).exists():
            print(f"Skip: {wav_dir} (not found)")
            continue
        done, tot = process_split(extractor, sp, wav_dir, args.out_root, args.layer)
        print(f"{sp}: {done}/{tot}")
        total_done += done
        total_all += tot

    print(f"All done: {total_done}/{total_all}")


if __name__ == "__main__":
    main()
