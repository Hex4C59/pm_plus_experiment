#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Checkpoint utilities: atomic save and robust load helpers.

This module provides save_checkpoint and load_checkpoint helpers that
support atomic writes (via temporary file + rename) and optional optimizer
state handling. It uses torch.save/torch.load when torch is available but
falls back to generic pickle for non-torch state.

Example :
    >>> from src.utils.checkpoint import save_checkpoint, load_checkpoint
    >>> ckpt = {"model_state": {"w": 1}, "epoch": 1}
    >>> save_checkpoint(ckpt, "runs/exp1/epoch1.pt")
    >>> loaded = load_checkpoint("runs/exp1/epoch1.pt")
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"


import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


def save_checkpoint(state: Dict[str, Any], path: str, overwrite: bool = False) -> str:
    """
    Save checkpoint atomically to disk.

    Args:
        state: Checkpoint dict (model state, optimizer state, metadata).
        path: Destination path for the checkpoint file.
        overwrite: If True, overwrite existing file.

    Returns:
        Path to written checkpoint file as string.

    Raises:
        FileExistsError: If file exists and overwrite is False.
        IOError: If write fails.

    """
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint exists: {dst}")
    tmp = dst.with_suffix(".tmp")
    try:
        if torch is not None:
            torch.save(state, tmp)
        else:
            with open(tmp, "wb") as f:
                pickle.dump(state, f)
        # atomic replace
        os.replace(tmp, dst)
    except Exception as exc:
        # cleanup tmp file on error
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise IOError(f"Failed to save checkpoint: {exc}") from exc
    return str(dst)


def load_checkpoint(path: str, map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    Load checkpoint saved by save_checkpoint.

    Args:
        path: Path to checkpoint file.
        map_location: Torch map_location (e.g., 'cpu' or {'cuda:0':'cpu'}).

    Returns:
        The loaded checkpoint dict.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
        RuntimeError: If loading fails.

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    try:
        if torch is not None:
            ckpt = torch.load(str(p), map_location=map_location)
        else:
            with open(p, "rb") as f:
                ckpt = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint: {exc}") from exc
    return ckpt


def _cli() -> None:
    """CLI for quick save/load smoke test of the checkpoint utilities."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Checkpoint save/load test")
    parser.add_argument("out", type=str, help="Output checkpoint path")
    parser.add_argument("--meta", type=str, default='{"epoch":1}', help="JSON metadata")
    args = parser.parse_args()
    meta = json.loads(args.meta)
    state = {"meta": meta}
    path = save_checkpoint(state, args.out, overwrite=True)
    print(f"Saved checkpoint to {path}")
    loaded = load_checkpoint(path, map_location="cpu")
    print("Loaded keys:", list(loaded.keys()))


if __name__ == "__main__":
    _cli()
