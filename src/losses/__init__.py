#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Losses package exports.

Expose common loss implementations for training scripts.
"""

from .base import BaseLoss  # noqa: F401
from .lccc import LCCCLoss  # noqa: F401

__all__ = ["BaseLoss", "LCCCLoss"]
