#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Model registry for dynamic model instantiation.

This module provides a registry system to dynamically instantiate models
based on configuration. It allows users to specify model types in their
experiment configurations without hardcoding model classes in the training script.

Example :
    >>> from src.models import get_model_class
    >>> model_cls = get_model_class("pooled_mlp")
    >>> model = model_cls.from_config(config)
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-22"

from typing import Any, Dict, Type

from src.models.base_model import BaseModel

# Registry to store model classes
_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}


def register_model(name: str, model_class: Type[BaseModel]) -> None:
    """
    Register a model class in the registry.

    Args:
        name: String identifier for the model type.
        model_class: The model class to register.

    Returns:
        None

    """
    _MODEL_REGISTRY[name] = model_class


def get_model_class(name: str) -> Type[BaseModel]:
    """
    Retrieve a model class from the registry.

    Args:
        name: String identifier for the model type.

    Returns:
        The registered model class.

    Raises:
        KeyError: If the model type is not registered.

    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Model type '{name}' is not registered. Available types: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name]


def list_available_models() -> list:
    """
    List all registered model types.

    Returns:
        List of available model type names.

    """
    return list(_MODEL_REGISTRY.keys())


# Import and register models
from src.models.attn_bilstm import SelfAttentiveBiLSTMRegressor
from src.models.pooled_mlp import PooledMLPRegressor

# Register models
register_model("pooled_mlp", PooledMLPRegressor)
register_model("attn_bilstm", SelfAttentiveBiLSTMRegressor)
