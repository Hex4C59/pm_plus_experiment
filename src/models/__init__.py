#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Model registry for the project.

Provides a simple MODEL_REGISTRY mapping model string keys to model
classes and a small factory helper `get_model` that constructs a model
from a configuration dictionary.

Example :
    >>> from src.models import get_model
    >>> cfg = {"input_size":768, "output_size":2, "hidden_size":128}
    >>> model = get_model("mtl_gru", cfg)
"""

from typing import Any, Callable, Dict, Optional, Type

# import model classes (best-effort; keep fallbacks)
try:
    from .attn_bilstm import SelfAttentiveBiLSTMRegressor  # type: ignore
except Exception:
    SelfAttentiveBiLSTMRegressor = None  # type: ignore

try:
    from .mtl_GRU_model import GRUModel  # type: ignore
except Exception:
    GRUModel = None  # type: ignore

# pooled_mlp class name may vary; try common names
PooledMLP = None
try:
    from .pooled_mlp import PooledMLPRegressor as PooledMLP  # type: ignore
except Exception:
    try:
        from .pooled_mlp import PooledMLP as PooledMLP  # type: ignore
    except Exception:
        PooledMLP = None  # type: ignore

# registry mapping keys used in configs -> class
MODEL_REGISTRY: Dict[str, Type] = {}
if SelfAttentiveBiLSTMRegressor is not None:
    MODEL_REGISTRY["attn_bilstm"] = SelfAttentiveBiLSTMRegressor
if GRUModel is not None:
    MODEL_REGISTRY["gru"] = GRUModel
if PooledMLP is not None:
    MODEL_REGISTRY["pooled_mlp"] = PooledMLP


def get_model_class(name: str, cfg: Optional[Dict[str, Any]] = None) -> Any:
    """
    Return a model class or an instantiated model.

    If `cfg` is None this function returns the registered model class so
    callers can instantiate it themselves. If `cfg` is provided the
    function will attempt to construct and return an instance by calling
    the class's `from_config` method (preferred) or by invoking the
    class constructor with `**cfg`.

    Args:
        name: Registry key of the model (e.g. "gru", "attn_bilstm").
        cfg: Optional configuration dict for instantiation. If omitted,
            the raw model class is returned.

    Returns:
        The model class (when cfg is None) or an instantiated model object.

    Raises:
        KeyError: If requested model name is not registered.
        RuntimeError: If instantiation with the provided cfg fails.

    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not found in MODEL_REGISTRY. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    cls = MODEL_REGISTRY[key]

    # If caller only wants the class, return it.
    if cfg is None:
        return cls

    # Prefer `from_config` factory when instantiating.
    if hasattr(cls, "from_config"):
        return cls.from_config(cfg)  # type: ignore
    try:
        return cls(**cfg)  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate model '{name}': {exc}") from exc
