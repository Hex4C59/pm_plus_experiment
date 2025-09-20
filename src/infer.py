#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Run inference and evaluate CCC on pooled-MLP regression models.

This script loads a trained PooledMLPRegressor and runs inference on a
feature split (e.g., test). If annotations are provided, it computes CCC
for valence and arousal. It writes per-utterance predictions to a CSV.

Example :
    >>> python -m src.infer \
    ...   --config configs/experiments/pooled_mlp.yaml \
    ...   --ckpt runs/pooled_mlp_v1/best.pt --split test \
    ...   --device cuda:0 --out_csv runs/pooled_mlp_v1/test_preds.csv
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

from __future__ import annotations

import argparse
import csv
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.loader import make_dataloader
from src.metrics.ccc import CCCMetric
from src.models.pooled_mlp import PooledMLPRegressor
from src.utils.checkpoint import load_checkpoint

# ------------------------- Configuration utilities -------------------------


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If YAML parsing fails.

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update dictionary dst with src values.

    Args:
        dst: Destination dictionary to be updated.
        src: Source dictionary providing override values.

    Returns:
        Updated destination dictionary.

    """
    for k, v in src.items():
        if (
            isinstance(v, dict)
            and k in dst
            and isinstance(dst[k], dict)
            and dst[k] is not v
        ):
            deep_update(dst[k], v)
        else:
            dst[k] = deepcopy(v)
    return dst


def load_config_with_includes(config_path: str) -> Dict[str, Any]:
    """
    Load a config with include and overrides resolution.

    The top-level config may contain:
      - include: list of file paths (relative to this config) to merge
      - overrides: dict to deep-override included/base configs

    Args:
        config_path: Path to the experiment YAML.

    Returns:
        A merged configuration dictionary.

    """
    base_dir = Path(config_path).parent
    cfg = load_yaml(config_path)
    merged: Dict[str, Any] = {}
    includes = cfg.get("include", []) or []
    for rel in includes:
        inc_path = str((base_dir / rel).resolve())
        inc_cfg = load_yaml(inc_path)
        deep_update(merged, inc_cfg)
    top_cfg = {k: v for k, v in cfg.items() if k not in ("include", "overrides")}
    deep_update(merged, top_cfg)
    overrides = cfg.get("overrides", {}) or {}
    deep_update(merged, overrides)
    return merged


# ----------------------------- Data / Model IO -----------------------------


def build_dataloader_from_cfg(
    data_cfg: Dict[str, Any],
    split: str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """
    Create a DataLoader for the given split from config.

    Args:
        data_cfg: Data configuration dictionary.
        split: Split name, e.g., "test" or "val".
        batch_size: Mini-batch size for inference.
        num_workers: Number of worker processes.

    Returns:
        A configured DataLoader instance.

    Raises:
        KeyError: If required fields are missing in config.

    """
    features_root = data_cfg["features"]
    annotations = data_cfg.get("annotations", None)
    model_folder = data_cfg.get("model_folder", None)
    return make_dataloader(
        features_root=features_root,
        split=split,
        model_folder=model_folder,
        annotations=annotations,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def load_model_from_ckpt(
    model_cfg: Dict[str, Any],
    ckpt_path: str,
    device: torch.device,
) -> PooledMLPRegressor:
    """
    Load model and weights from checkpoint.

    Args:
        model_cfg: Model configuration dictionary for instantiation.
        ckpt_path: Path to a checkpoint file with 'model_state'.
        device: Torch device to place the model.

    Returns:
        A PooledMLPRegressor with weights loaded on device.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        KeyError: If 'model_state' missing in checkpoint.

    """
    model = PooledMLPRegressor.from_config(model_cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, map_location=str(device))
    state = ckpt.get("model_state")
    if state is None:
        raise KeyError("Checkpoint missing key 'model_state'")
    model.load_state_dict(state)
    model.eval()
    return model


# ----------------------------- Inference logic -----------------------------


@torch.no_grad()
def run_inference(
    model: PooledMLPRegressor,
    loader: DataLoader,
    device: torch.device,
    out_csv: str,
) -> Dict[str, float]:
    """
    Run inference, save predictions, and compute optional CCC.

    Args:
        model: Trained model in evaluation mode.
        loader: DataLoader iterating over the split.
        device: Torch device for computation.
        out_csv: Destination CSV path for predictions.

    Returns:
        A dictionary with keys:
            - 'count': number of utterances processed
            - 'ccc_valence': CCC if labels available, else NaN
            - 'ccc_arousal': CCC if labels available, else NaN

    Raises:
        IOError: If writing the output CSV fails.

    """
    rows: List[Dict[str, Any]] = []
    metric_v = CCCMetric(name="ccc_valence")
    metric_a = CCCMetric(name="ccc_arousal")
    have_labels = False

    for batch in loader:
        feats = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        audio_files: List[str] = batch.get("audio_files", [])
        speakers: List[str] = batch.get("speakers", [])
        paths: List[str] = batch.get("paths", [])

        preds = model(feats, lengths=lengths)
        pv = preds["valence"].detach().cpu().view(-1)
        pa = preds["arousal"].detach().cpu().view(-1)

        val_t = batch.get("valence", None)
        aro_t = batch.get("arousal", None)
        if val_t is not None and aro_t is not None:
            have_labels = True
            tv = val_t.view(-1).to(dtype=torch.float32).cpu()
            ta = aro_t.view(-1).to(dtype=torch.float32).cpu()
            metric_v.update(pv, tv)
            metric_a.update(pa, ta)

        for i in range(pv.numel()):
            row: Dict[str, Any] = {
                "audio_file": audio_files[i] if i < len(audio_files) else "",
                "speaker": speakers[i] if i < len(speakers) else "",
                "path": paths[i] if i < len(paths) else "",
                "val_pred": float(pv[i].item()),
                "aro_pred": float(pa[i].item()),
            }
            if have_labels:
                row["val_target"] = float(val_t[i].item()) if val_t is not None else ""
                row["aro_target"] = float(aro_t[i].item()) if aro_t is not None else ""
            rows.append(row)

    # Write CSV atomically
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    fieldnames = [
        "audio_file",
        "speaker",
        "path",
        "val_pred",
        "aro_pred",
        "val_target",
        "aro_target",
    ]
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                # ensure keys exist (empty if missing)
                rec = {k: r.get(k, "") for k in fieldnames}
                writer.writerow(rec)
        os.replace(tmp, out_path)
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise IOError(f"Failed to write CSV: {exc}") from exc

    # Metrics
    ccc_v = float("nan")
    ccc_a = float("nan")
    if have_labels:
        ccc_v = float(metric_v.compute().item())
        ccc_a = float(metric_a.compute().item())

    return {"count": len(rows), "ccc_valence": ccc_v, "ccc_arousal": ccc_a}


# ----------------------------------- Main -----------------------------------


def main() -> None:
    """CLI entry point for inference and CCC evaluation."""
    parser = argparse.ArgumentParser(
        description="Run inference with PooledMLPRegressor and evaluate CCC."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML (supports include/overrides).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path containing model_state.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to run inference on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string like 'cuda:0' or 'cpu' (auto if None).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV path for predictions "
        "(default: runs/<exp_id>/<split>_preds.csv).",
    )
    args = parser.parse_args()

    cfg = load_config_with_includes(args.config)
    project = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    exp_cfg = cfg.get("experiment", {}) or {}

    # Device
    device_str = args.device or project.get("device") or "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Experiment directory for default outputs (place under runs/infer)
    output_root = project.get("output_root", "runs")
    exp_id = exp_cfg.get("id", "infer_run")
    exp_dir = Path(output_root) / "infer" / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save merged config and manifest for reproducibility
    _save_run_config(exp_dir=str(exp_dir), merged_cfg=cfg)
    _write_manifest(
        exp_dir=str(exp_dir),
        meta={
            "type": "infer",
            "config": args.config,
            "exp_id": exp_id,
            "device": device_str,
            "split": args.split,
            "ckpt": args.ckpt,
        },
    )

    # Build loader and model
    loader = build_dataloader_from_cfg(
        data_cfg=data_cfg,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    model = load_model_from_ckpt(
        model_cfg=model_cfg, ckpt_path=args.ckpt, device=device
    )

    # Output CSV path
    out_csv = args.out_csv or str(exp_dir / f"{args.split}_preds.csv")

    # Run inference
    stats = run_inference(model=model, loader=loader, device=device, out_csv=out_csv)

    # Report
    print(
        f"[Inference] Done. N={stats['count']}, "
        f"CCC(valence)={stats['ccc_valence']}, "
        f"CCC(arousal)={stats['ccc_arousal']}. CSV: {out_csv}"
    )


# ----------------------------- Run artifacts --------------------------------


def _save_run_config(exp_dir: str, merged_cfg: Dict[str, Any]) -> None:
    """
    Save merged configuration as YAML and JSON in the experiment directory.

    Args:
        exp_dir: Output experiment directory.
        merged_cfg: Fully merged configuration dictionary.

    Returns:
        None

    Raises:
        IOError: If writing config files fails.

    """
    out_dir = Path(exp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = out_dir / "config.yaml"
    json_path = out_dir / "config.json"
    try:
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged_cfg, f, sort_keys=False, allow_unicode=True)
        with open(json_path, "w", encoding="utf-8") as f:
            import json

            json.dump(merged_cfg, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise IOError(f"Failed to write run config: {exc}") from exc


def _write_manifest(exp_dir: str, meta: Dict[str, Any]) -> None:
    """
    Write a lightweight manifest.json in the experiment directory.

    Args:
        exp_dir: Output experiment directory.
        meta: Dictionary with basic run metadata.

    Returns:
        None

    Raises:
        IOError: If writing the manifest fails.

    """
    import json
    import sys

    out_path = Path(exp_dir) / "manifest.json"
    payload = {
        "cmd": " ".join(sys.argv),
        **meta,
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        raise IOError(f"Failed to write manifest: {exc}") from exc


if __name__ == "__main__":
    main()
