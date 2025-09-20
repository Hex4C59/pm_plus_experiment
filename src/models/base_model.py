#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Abstract base model for SER experiments.

This module defines an abstract, torch.nn.Module based base class that
standardizes model interface used across training and evaluation code.
It requires derived classes to implement initialization, forward pass,
and a simple factory constructor. Utility helpers for saving and loading
checkpoints are provided.

Example :
    >>> from src.models.base_model import BaseModel
    >>> class MyModel(BaseModel):
    ...     def __init__(self, input_dim: int, out_dim: int) -> None:
    ...         super().__init__()
    ...         self.net = torch.nn.Linear(input_dim, out_dim)
    ...     def forward(self, inputs: torch.Tensor, **kwargs):
    ...         return {"pred": self.net(inputs)}
    ...     @classmethod
    ...     def from_config(cls, cfg: dict):
    ...         return cls(cfg["input_dim"], cfg["out_dim"])
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

from __future__ import annotations  # type: ignore

import abc
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


class BaseModel(nn.Module, abc.ABC):
    """
    Abstract base model defining a consistent model interface.

    Derived classes must implement the forward method and a from_config
    constructor. This class also provides lightweight checkpoint helpers.

    Attributes:
        name: Human readable model name.

    """

    name: str

    def __init__(self, name: str = "base_model") -> None:
        """
        Initialize common base attributes.

        Args:
            name: Short name for the model instance used in logging and
                checkpoint filenames.

        """
        super().__init__()
        self.name = name

    @abc.abstractmethod
    def forward(self, inputs: torch.Tensor, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Run a forward pass and return a dict of outputs.

        Implementations should return a dictionary that contains model
        predictions and optionally intermediate tensors used by losses.

        Args:
            inputs: Input tensor, typically audio features or raw tensors.
            **kwargs: Optional keyword arguments used by the specific model.

        Returns:
            A mapping of output names to tensors.

        Raises:
            RuntimeError: For invalid input shapes or states.

        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BaseModel":
        """
        Create a model instance from a configuration mapping.

        This constructor allows external code to instantiate models using
        a simple config dict.

        Args:
            cfg: Configuration dictionary with required hyperparameters.

        Returns:
            An initialized model instance.

        """
        raise NotImplementedError

    def save_checkpoint(
        self,
        path: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save model state_dict and optional optimizer state.

        Args:
            path: Optional file path to save the checkpoint. If None, a
                default name '{model_name}.pt' in the current directory is
                used.
            optimizer: Optional optimizer whose state_dict will be saved.
            extra: Optional mapping with extra metadata to include.

        Returns:
            Path of the written checkpoint file.

        Raises:
            IOError: If saving fails due to I/O error.

        """
        ckpt_path = Path(path) if path else Path(f"{self.name}.pt")
        payload: Dict[str, Any] = {"model_state": self.state_dict()}
        if optimizer is not None:
            payload["optim_state"] = optimizer.state_dict()
        if extra is not None:
            payload["extra"] = extra
        try:
            torch.save(payload, ckpt_path)
        except Exception as exc:
            raise IOError(f"Failed to save checkpoint: {exc}") from exc
        return ckpt_path

    def load_checkpoint(
        self, path: str, map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint and apply model state_dict.

        Args:
            path: Path to checkpoint file.
            map_location: Optional device mapping for torch.load.

        Returns:
            The loaded checkpoint dictionary.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If state_dict cannot be loaded.

        """
        ckpt_path = Path(path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        map_loc = map_location or ("cpu")
        ckpt = torch.load(ckpt_path, map_location=map_loc)
        if "model_state" not in ckpt:
            # Backward compatibility: allow raw state_dict files.
            state = ckpt if isinstance(ckpt, dict) else {}
        else:
            state = ckpt["model_state"]
        try:
            self.load_state_dict(state)
        except Exception as exc:
            raise RuntimeError(f"Failed to load state_dict: {exc}") from exc
        return ckpt

    def export_config(self, cfg: Dict[str, Any], path: str) -> None:
        """
        Export a config dict as a JSON file next to a checkpoint.

        Args:
            cfg: Configuration dictionary to save.
            path: File path to write the JSON config.

        Raises:
            IOError: If writing the file fails.

        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            out_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception as exc:
            raise IOError(f"Failed to write config: {exc}") from exc
