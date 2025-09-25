#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Train a pooled-MLP regressor using CCC-based loss for continuous emotion.

This module implements the end-to-end training workflow for a pooled MLP
regressor that consumes pre-extracted frame-level features (e.g.,
wav2vec2 hidden states) and predicts continuous valence and arousal
values. It provides utilities to load and merge YAML configs (with
include/overrides), build dataloaders, construct the model and optimizer,
run training with gradient accumulation and optional gradient clipping,
evaluate using CCC metrics, perform early stopping, save best/last
checkpoints, and emit CSV logging and optional PNG plots for analysis.

The module is designed for reproducible experiments: it writes the final
merged configuration and a small manifest into the experiment output
directory and exposes a command line interface for all important
parameters.

Example :
    >>> python -m src.train \\
    ...   --config configs/experiments/pooled_mlp.yaml \\
    ...   --device cuda:0 --exp_dir runs/train/pooled_mlp_v1 --plot
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"


import argparse
import csv
import json
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.loader import make_dataloader
from src.losses.lccc import LCCCLoss
from src.metrics.ccc import CCCMetric
from src.models import get_model_class
from src.utils.checkpoint import load_checkpoint, save_checkpoint
from src.utils.logger import Logger
from src.utils.visualization import plot_losses, plot_metrics

# ------------------------- Configuration utilities -------------------------


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return a dictionary.

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

    After merging, expand placeholders of the form ${some.path} using
    values from the merged config (dotted lookup) or environment vars.

    Args:
        config_path: Path to the experiment YAML.

    Returns:
        A merged configuration dictionary with placeholders resolved.

    Raises:
        FileNotFoundError: If an included file does not exist.
        KeyError: If a placeholder cannot be resolved from config or env.

    """
    import re

    base_dir = Path(config_path).parent
    cfg = load_yaml(config_path)
    merged: Dict[str, Any] = {}
    includes = cfg.get("include", []) or []
    for rel in includes:
        inc_path = str((base_dir / rel).resolve())
        inc_cfg = load_yaml(inc_path)
        deep_update(merged, inc_cfg)
    # merge top config (without include/overrides)
    top_cfg = {k: v for k, v in cfg.items() if k not in ("include", "overrides")}
    deep_update(merged, top_cfg)
    # apply overrides if present
    overrides = cfg.get("overrides", {}) or {}
    deep_update(merged, overrides)

    # --- placeholder resolution utilities ---
    placeholder_pattern = re.compile(r"\$\{([^}]+)\}")

    def _lookup_dotted(d: Dict[str, Any], dotted: str) -> Optional[str]:
        """Lookup dotted path in dict and return string value or None."""
        cur: Any = d
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        # convert non-str to str for substitution
        if cur is None:
            return None
        if isinstance(cur, (str, int, float, bool)):
            return str(cur)
        # for other types (list/dict), return JSON-ish string
        try:
            import json

            return json.dumps(cur, ensure_ascii=False)
        except Exception:
            return str(cur)

    def _resolve(obj: Any) -> Any:
        """Recursively resolve placeholders in obj using merged dict and env."""
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        if isinstance(obj, str):
            # repeatedly replace until no placeholders remain
            s = obj
            max_iters = 10
            for _ in range(max_iters):
                matches = list(placeholder_pattern.finditer(s))
                if not matches:
                    break
                new_s = s
                for m in matches:
                    key = m.group(1)
                    replacement = _lookup_dotted(merged, key)
                    if replacement is None:
                        replacement = os.environ.get(key)
                    if replacement is None:
                        raise KeyError(f"Cannot resolve placeholder '{key}' in config")
                    # replace this occurrence
                    new_s = new_s.replace(m.group(0), replacement)
                if new_s == s:
                    break
                s = new_s
            return s
        return obj

    # perform resolution in merged config
    merged = _resolve(merged)
    return merged


# ----------------------- Training/evaluation helpers -----------------------


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value.

    Returns:
        None

    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # no-op on CPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(
    data_cfg: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation/test dataloaders.

    This function looks for split folders 'train' and one of {'val','test'}.
    If 'val' exists under features root, it is used for validation; otherwise
    'test' is used.

    Args:
        data_cfg: Data configuration dictionary.
        batch_size: Batch size for both train and val loaders.
        num_workers: Number of DataLoader workers.
        pin_memory: Whether to pin memory in the DataLoader workers (useful when
            using CUDA to speed up host-to-device transfers).

    Returns:
        Tuple of (train_loader, val_loader, val_split_name).

    Raises:
        FileNotFoundError: If required split directories are not found.

    """
    features_root = data_cfg["features"]
    annotations = data_cfg.get("annotations", None)
    model_folder = data_cfg.get("model_folder", None)


    train_loader = make_dataloader(
        features_root=features_root,
        split="train",
        model_folder=model_folder,
        annotations=annotations,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_dataloader(
        features_root=features_root,
        split="test",
        model_folder=model_folder,
        annotations=annotations,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def build_model(model_cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """
    Create model from configuration and move to device.

    Args:
        model_cfg: Model configuration dictionary.
        device: Torch device.

    Returns:
        Initialized model on the target device.

    Raises:
        KeyError: If required keys are missing.

    """
    model_type = model_cfg.get("type", "pooled_mlp")

    # Obtain model class from registry, then instantiate using from_config
    # (or fall back to direct construction) for consistent behavior.
    model_cls = get_model_class(model_type)
    if hasattr(model_cls, "from_config"):
        model = model_cls.from_config(model_cfg)
    else:
        model = model_cls(**model_cfg)
    return model.to(device)


def build_optimizer(
    optim_cfg: Dict[str, Any],
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """
    Create optimizer from configuration.

    Args:
        optim_cfg: Optimizer configuration dictionary.
        model: Model whose parameters are to be optimized.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer name is unsupported.

    """
    name = (optim_cfg.get("name") or "adamw").lower()
    lr = float(optim_cfg.get("lr", 1e-4))
    wd = float(optim_cfg.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(optim_cfg.get("momentum", 0.9))
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=wd
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optim_cfg: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> Optional[torch.optim.lr_scheduler.LambdaLR]:
    """
    Create a step-wise scheduler with linear warmup then constant LR.

    Args:
        optim_cfg: Optimizer configuration including scheduler settings.
        optimizer: Optimizer to schedule.
        total_steps: Total training steps (epochs * steps_per_epoch).

    Returns:
        A LambdaLR scheduler or None if not requested.

    """
    sched_cfg = optim_cfg.get("scheduler", None)
    if not sched_cfg:
        return None
    name = (sched_cfg.get("name") or "linear").lower()
    if name != "linear":
        return None
    warmup = int(sched_cfg.get("warmup_steps", 0))
    warmup = max(0, warmup)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(max(1, warmup))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


@dataclass
class EarlyStoppingState:
    """
    Container for early stopping state.

    Attributes:
        best_score: Best monitored score so far.
        patience_counter: Number of epochs without improvement.
        best_state_dict: Snapshot of the best model weights.
        improved: Whether the last epoch improved the best score.

    """

    best_score: float
    patience_counter: int
    best_state_dict: Optional[Dict[str, Any]]
    improved: bool


def init_early_stopping(monitor: str, mode: str) -> EarlyStoppingState:
    """
    Initialize early stopping state based on mode.

    Args:
        monitor: Name of the metric monitored (unused here).
        mode: 'min' or 'max' determining improvement direction.

    Returns:
        Initialized EarlyStoppingState.

    """
    if mode == "min":
        best = float("inf")
    else:
        best = float("-inf")
    return EarlyStoppingState(
        best_score=best, patience_counter=0, best_state_dict=None, improved=False
    )  # noqa: E501


def is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    """
    Determine if the current score improves over best.

    Args:
        current: Current metric value.
        best: Best metric value so far.
        mode: 'min' for lower-is-better, 'max' otherwise.
        min_delta: Minimum change to qualify as improvement.

    Returns:
        True if improved, False otherwise.

    """
    if mode == "min":
        return current < (best - min_delta)
    return current > (best + min_delta)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    loss_fn: LCCCLoss,
    accumulation_steps: int = 1,
    max_grad_norm: Optional[float] = None,
    epoch: int = 0,
    show_progress: bool = True,
) -> Tuple[float, int]:
    """
    Train model for a single epoch and show progress with tqdm.

    Args:
        model: Model to train.
        loader: Training DataLoader.
        device: Device for computation.
        optimizer: Optimizer instance.
        loss_fn: LCCC loss function.
        accumulation_steps: Gradient accumulation steps.
        max_grad_norm: Optional gradient norm clipping.
        epoch: Current epoch number used for progress description.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Tuple of (average_loss, num_steps).

    Raises:
        ValueError: If annotations are missing in a training batch.

    """
    model.train()
    total_loss = 0.0
    steps = 0
    optimizer.zero_grad(set_to_none=True)

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=f"Train Epoch {epoch}", unit="batch", leave=False)

    last_loss_value = None
    for i, batch in enumerate(iterator, 1):
        feats = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        val_t = batch.get("valence")
        aro_t = batch.get("arousal")
        if val_t is None or aro_t is None:
            raise ValueError("Annotations (valence/arousal) are required for training")
        val_t = val_t.to(device)
        aro_t = aro_t.to(device)

        preds = model(feats, lengths=lengths)
        targets = {"valence": val_t, "arousal": aro_t}
        out = loss_fn(preds, targets)
        loss = out["loss"] / float(accumulation_steps)
        loss.backward()

        if max_grad_norm is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm)

        if i % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
            total_loss += float(out["loss"].detach().cpu().item())
        last_loss_value = out["loss"].detach().cpu().item()

        if show_progress:
            iterator.set_postfix({"loss": f"{last_loss_value:.4f}"})

    # handle last partial accumulation as a step for reporting
    if (i % accumulation_steps) != 0:
        steps += 1
        total_loss += float(last_loss_value)

    avg_loss = total_loss / max(1, steps)
    return avg_loss, steps


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    loss_fn: LCCCLoss,
    epoch: int = 0,
    show_progress: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set and show progress with tqdm.

    Args:
        model: Trained model in eval mode.
        loader: DataLoader for validation/test set.
        device: Torch device.
        loss_fn: LCCC loss for reporting val_loss.
        epoch: Current epoch number for progress description.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Dictionary with avg loss and CCC metrics:
        {'val_loss','ccc_valence','ccc_arousal'}.

    Raises:
        ValueError: If validation batch lacks valence/arousal targets.

    """
    model.eval()
    total_loss = 0.0
    steps = 0
    metric_v = CCCMetric(name="ccc_valence")
    metric_a = CCCMetric(name="ccc_arousal")

    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=f"Eval Epoch {epoch}", unit="batch", leave=False)

    for batch in iterator:
        feats = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        val_t = batch.get("valence")
        aro_t = batch.get("arousal")
        if val_t is None or aro_t is None:
            raise ValueError("Validation requires valence and arousal targets")
        val_t = val_t.to(device)
        aro_t = aro_t.to(device)

        preds = model(feats, lengths=lengths)
        targets = {"valence": val_t, "arousal": aro_t}
        out = loss_fn(preds, targets)

        total_loss += float(out["loss"].detach().cpu().item())
        steps += 1

        metric_v.update(preds["valence"].detach().cpu(), val_t.detach().cpu())
        metric_a.update(preds["arousal"].detach().cpu(), aro_t.detach().cpu())

        if show_progress:
            iterator.set_postfix({"val_loss": f"{total_loss / steps:.4f}"})

    avg_loss = total_loss / max(1, steps)
    ccc_v = float(metric_v.compute().item())
    ccc_a = float(metric_a.compute().item())
    return {"val_loss": avg_loss, "ccc_valence": ccc_v, "ccc_arousal": ccc_a}


# ----------------------------------- Main -----------------------------------


def main() -> None:
    """CLI entry point for training PooledMLPRegressor on CCC loss."""
    parser = argparse.ArgumentParser(
        description="Train pooled-MLP regressor with CCC loss on features"
    )
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g., 'cuda:0' or 'cpu')",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help="Experiment directory (default from config experiment.id)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path (optional)",
    )
    parser.add_argument("--plot", action="store_true", help="Plot curves at the end")
    parser.add_argument(
        "--no_val",
        action="store_true",
        help="Disable validation during training (use infer.py for test).",
    )
    args = parser.parse_args()

    cfg = load_config_with_includes(args.config)
    project = cfg.get("project", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    optim_cfg = cfg.get("optim", {})

    seed = int(project.get("seed", 42))
    set_seed(seed)

    # Device selection
    device_str = args.device or project.get("device") or "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Experiment directory (place under runs/train by default)
    exp_id = (cfg.get("experiment", {}) or {}).get("id", "default_exp")
    output_root = project.get("output_root", "runs")
    default_dir = Path(output_root) / "train" / exp_id
    exp_dir = args.exp_dir or str(default_dir)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    # Save merged config and manifest for reproducibility
    _save_run_config(exp_dir=exp_dir, merged_cfg=cfg)
    _write_manifest(
        exp_dir=exp_dir,
        meta={
            "type": "train",
            "config": args.config,
            "exp_id": exp_id,
            "device": device_str,
            "seed": seed,
        },
    )

    # Logger
    logger = Logger(exp_dir, filename="metrics.csv", console=True)

    # Data and loaders
    batch_size = int(train_cfg.get("batch_size", 32))

    num_workers = int(project.get("num_workers", 4))
    pin_memory = bool(project.get("pin_memory", True))
    if args.no_val:
        # Only prepare train loader; skip building/using val loader.
        train_loader = make_dataloader(
            features_root=data_cfg["features"],
            split="train",
            model_folder=data_cfg.get("model_folder", None),
            annotations=data_cfg.get("annotations", None),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = None
        # disable early stopping and related mechanisms
        es_enabled = False

        # Record explicitly that validation is disabled both to console
        # and to the CSV metrics file for reproducibility.
        try:
            logger.log({"validation_status": "disabled"})
        except Exception:
            # Fallback to console if CSV logging fails.
            print("[Logger] Failed to write validation disabled note to metrics.csv")
        print("[Info] Validation disabled during training; use infer.py for test")
    else:
        train_loader, val_loader = build_dataloaders(
            data_cfg,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    # Model, optimizer, scheduler, loss
    model = build_model(model_cfg, device)
    optimizer = build_optimizer(optim_cfg, model)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(train_cfg.get("epochs", 50))
    scheduler = build_scheduler(optim_cfg, optimizer, total_steps=total_steps)
    loss_alpha = float(cfg.get("loss", {}).get("alpha", 0.5))
    loss_fn = LCCCLoss(alpha=loss_alpha)

    # Early stopping settings
    es_cfg = train_cfg.get("early_stopping", {}) or {}
    es_enabled = bool(es_cfg.get("enabled", True))
    monitor = str(es_cfg.get("monitor", "val_loss"))
    mode = str(es_cfg.get("mode", "min"))
    patience = int(es_cfg.get("patience", 8))
    min_delta = float(es_cfg.get("min_delta", 0.0))
    restore_best = bool(es_cfg.get("restore_best", True))
    es_state = init_early_stopping(monitor, mode)

    # Resume if requested
    start_epoch = 1
    best_ckpt_path = str(Path(exp_dir) / "best.pt")
    last_ckpt_path = str(Path(exp_dir) / "last.pt")
    if args.resume:
        ckpt = load_checkpoint(args.resume, map_location=device_str)
        model.load_state_dict(ckpt.get("model_state", {}))
        if "optim_state" in ckpt:
            optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "best_score" in ckpt:
            es_state.best_score = float(ckpt["best_score"])

    # Training loop
    epochs = int(train_cfg.get("epochs", 50))
    accum_steps = int(train_cfg.get("accumulation_steps", 1))
    eval_interval = int(train_cfg.get("eval_interval", 1))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.0)) or None

    tr_losses: List[float] = []
    va_losses: List[float] = []
    ccc_v_hist: List[float] = []
    ccc_a_hist: List[float] = []

    for epoch in range(start_epoch, epochs + 1):
        avg_loss, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accumulation_steps=accum_steps,
            max_grad_norm=max_grad_norm,
            epoch=epoch,
            show_progress=True,
        )
        tr_losses.append(avg_loss)

        # Scheduler step per epoch (linear warmup is per-step; we step here too)
        if scheduler is not None:
            scheduler.step()

        metrics: Dict[str, float] = {}
        # Only run evaluation when a val loader is available
        if (not args.no_val) and ((epoch % eval_interval) == 0 or epoch == epochs):
            eval_out = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                loss_fn=loss_fn,
                epoch=epoch,
                show_progress=True,
            )
            va_losses.append(eval_out["val_loss"])
            ccc_v_hist.append(eval_out["ccc_valence"])
            ccc_a_hist.append(eval_out["ccc_arousal"])
            metrics = eval_out

            # Early stopping check (es_enabled may have been disabled when --no_val)
            if es_enabled:
                current = metrics.get(monitor, float("inf"))
                if is_improvement(current, es_state.best_score, mode, min_delta):
                    es_state.best_score = current
                    es_state.patience_counter = 0
                    es_state.best_state_dict = deepcopy(model.state_dict())
                    es_state.improved = True
                    # Save best checkpoint
                    best_payload = {
                        "epoch": epoch,
                        "model_state": es_state.best_state_dict,
                        "optim_state": optimizer.state_dict(),
                        "best_score": es_state.best_score,
                        "monitor": monitor,
                        "mode": mode,
                        "metrics": metrics,
                    }
                    save_checkpoint(best_payload, best_ckpt_path, overwrite=True)
                else:
                    es_state.patience_counter += 1
                    es_state.improved = False

        # Save last checkpoint every epoch
        last_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "best_score": es_state.best_score,
        }
        save_checkpoint(last_payload, last_ckpt_path, overwrite=True)

        # Log row
        row: Dict[str, Any] = {"epoch": epoch, "train_loss": avg_loss}
        row.update(metrics)
        # mark whether this epoch produced the best monitored metric
        row["is_best"] = bool(es_state.improved)
        if row["is_best"]:
            try:
                best_json = Path(exp_dir) / "best_metrics.json"
                best_csv = Path(exp_dir) / "best_metrics.csv"

                # add timestamp for reproducibility
                row_with_time = {**row, "time": time.time()}

                # write JSON (overwrites previous best)
                with open(best_json, "w", encoding="utf-8") as f_j:
                    json.dump(row_with_time, f_j, indent=2, ensure_ascii=False)

                # write CSV (single-row, overwrite)
                with open(best_csv, "w", newline="", encoding="utf-8") as f_c:
                    fieldnames = list(row_with_time.keys())
                    writer = csv.DictWriter(f_c, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row_with_time)
            except Exception as exc:
                print(f"[WARN] Failed to write best_metrics files: {exc}")
        logger.log(row)

        # Early stopping termination
        if es_enabled and es_state.patience_counter >= patience:
            print(
                f"[EarlyStopping] No improvement in {patience} evals. Stopping at "
                f"epoch {epoch}."
            )
            break

    # Optionally restore best weights
    if es_enabled and restore_best and es_state.best_state_dict is not None:
        model.load_state_dict(es_state.best_state_dict)

    # Final plots
    if args.plot:
        try:
            plot_losses(
                train_losses=tr_losses,
                val_losses=va_losses if va_losses else None,
                save_path=str(Path(exp_dir) / "loss_curves.png"),
            )
            plot_metrics(
                metrics={
                    "ccc_valence": ccc_v_hist,
                    "ccc_arousal": ccc_a_hist,
                },
                save_path=str(Path(exp_dir) / "ccc_curves.png"),
            )
        except Exception as exc:
            print(f"[Plot] Failed to save plots: {exc}")

    logger.close()
    print(f"Training finished. Artifacts saved in: {exp_dir}")


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
