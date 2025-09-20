#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch extraction of wav2vec2 features for speech emotion recognition.

This module provides a command-line interface to extract hidden states from
wav2vec2 models for a dataset of audio files. Features are saved in .pt format
for downstream emotion regression/classification tasks. Supports chunked
processing for long audio and flexible layer selection.

Example :
    >>> python scripts/extract_wav2vec2.py \
            --ckpt_dir pretrain_model/wav2vec2-base-100h \
            --data_root data/raw \
            --out_root data/processed/features \
            --layer 12
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
from transformers import Wav2Vec2Config, Wav2Vec2Model


class Wav2Vec2Extractor:
    """
    Lightweight wav2vec2 feature extractor.

    Args:
        ckpt_dir: Directory containing HuggingFace 'config.json' and 'pytorch_model.bin'
        device: Torch device ('cuda' or 'cpu')
        max_chunk: Max raw samples per forward (e.g., 1_600_000 ≈ 100 s @ 16 kHz)

    Returns:
        None

    Raises:
        FileNotFoundError: When files are missing

    Examples:
        >>> ext = Wav2Vec2Extractor("pretrain_model/wav2vec2-base-100h", "cpu")
        >>> x = np.zeros(16000, dtype=np.float32)
        >>> t = torch.from_numpy(x)[None, :]
        >>> with torch.no_grad():
        ...     y = ext.model(t.to(ext.device), output_hidden_states=True)
        ...     _ = y.last_hidden_state

    """

    def __init__(self, ckpt_dir: str, device: str, max_chunk: int = 1_600_000):
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

        # 获取预训练模型文件夹名称
        self.model_name = Path(ckpt_dir).name

    def read_audio(self, path: str) -> np.ndarray:
        """
        Read an audio file, convert to mono, resample if needed.

        Args:
            path: Audio file path

        Returns:
            Float32 mono waveform at 16 kHz

        Raises:
            RuntimeError: When reading audio fails

        Examples:
            >>> # doctest: +SKIP
            >>> import numpy as np
            >>> ext = Wav2Vec2Extractor("ckpt", "cpu")
            >>> wav = ext.read_audio("a.wav")
            >>> isinstance(wav, np.ndarray)
            True

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
            x: Tensor shape (1, T)

        Returns:
            Normalized tensor (1, T)

        Raises:
            None

        Examples:
            >>> import torch
            >>> ext = Wav2Vec2Extractor("ckpt", "cpu")
            >>> y = ext._norm(torch.ones(1, 4))
            >>> y.shape
            torch.Size([1, 4])

        """
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def extract(self, wav_path: str, layer: int = 12) -> torch.Tensor:
        """
        Extract hidden states from the specified transformer layer.

        Note: HuggingFace returns 'hidden_states' with length L+1.
        Index 0 is the output before the first transformer block.
        Here, 'layer' starts from 1..L, and out-of-range falls back to last.

        Args:
            wav_path: Path to a wav file
            layer: Transformer layer index (1..num_hidden_layers)

        Returns:
            Tensor of shape (T', 768) for base model

        Raises:
            None

        Examples:
            >>> # doctest: +SKIP
            >>> ext = Wav2Vec2Extractor("ckpt", "cpu")
            >>> feats = ext.extract("a.wav", layer=12)
            >>> feats.ndim
            2

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

        Args:
            feat: Feature tensor (T', D)
            save_path: Path without extension

        Returns:
            None

        Raises:
            None

        Examples:
            >>> import torch, tempfile
            >>> ext = Wav2Vec2Extractor("ckpt", "cpu")
            >>> tmp = tempfile.NamedTemporaryFile(delete=True).name
            >>> ext.save_feature(torch.zeros(2, 3), tmp)  # doctest: +ELLIPSIS

        """
        out = {"feature": feat.contiguous()}
        torch.save(out, f"{save_path}.pt")


def process_split(
    extractor: Wav2Vec2Extractor,
    split: str,
    wav_dir: str,
    out_root: str,
    layer: int,
) -> Tuple[int, int]:
    """
    Process a dataset split directory.

    Args:
        extractor: Feature extractor instance
        split: Split name ('train'|'validation'|'test')
        wav_dir: Directory containing wav files
        out_root: Root dir to save features
        layer: Transformer layer index

    Returns:
        Tuple of (processed_count, total_count)

    Raises:
        None

    Examples:
        >>> # doctest: +SKIP
        >>> proc, total = process_split(ext, "train", "wav/train", "feat", 12)

    """
    wav_dir_p = Path(wav_dir)
    # 使用预训练模型文件夹名称+层数作为输出文件夹名
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
    CLI entry for batch feature extraction.

    Args:
        None

    Returns:
        None

    Raises:
        SystemExit: When input dirs are invalid

    Examples:
        >>> # doctest: +SKIP
        >>> main()

    """
    parser = argparse.ArgumentParser(description="Extract wav2vec2 features")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--gpu", type=int, default=0, help="Specify which GPU to use (default: 0)"
    )
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--max_chunk", type=int, default=1_600_000)
    args = parser.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"

    extractor = Wav2Vec2Extractor(
        ckpt_dir=args.ckpt_dir, device=device, max_chunk=args.max_chunk
    )

    print(f"Using model: {extractor.model_name}")
    print(f"Extracting layer: {args.layer}")
    print(f"Output folder will be: {extractor.model_name}-l{args.layer}")
    print(
        f"Selected device: {device} (GPU index: {args.gpu if device == 'cuda' else 'N/A'})"
    )

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
