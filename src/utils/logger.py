#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Lightweight experiment logger that writes metrics to console and CSV.

This module provides a simple Logger class to record scalar metrics or
arbitrary key-value entries for training/validation. Each call to log()
appends a timestamped row to a CSV file and optionally prints a
one-line summary to stdout. The CSV is created atomically on first write
and supports incremental updates.

Example :
    >>> from src.utils.logger import Logger
    >>> logger = Logger("runs/exp1")
    >>> logger.log({"epoch": 1, "loss": 0.5, "val_loss": 0.6})
    >>> logger.close()
"""

__author__ = "Liu Yang"
__copyright__ = "Copyright 2025, AIMSL"
__license__ = "MIT"
__maintainer__ = "Liu Yang"
__email__ = "yang.liu6@siat.ac.cn"
__last_updated__ = "2025-09-20"

import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable


class Logger:
    """
    Simple logger that writes rows to CSV and optionally to console.

    Attributes:
        exp_dir: Experiment directory where CSV is stored.
        csv_path: Full path to CSV file.
        console: Whether to print logs to stdout.
        _lock: Threading lock for safe concurrent writes.
        _writer: CSV DictWriter instance (created on first write).
        _file: Open file handle for CSV.
        _fieldnames: Ordered list of CSV columns.

    """

    def __init__(
        self,
        exp_dir: str,
        filename: str = "metrics.csv",
        console: bool = True,
    ) -> None:
        """
        Create a Logger.

        Args:
            exp_dir: Directory to store CSV file.
            filename: CSV filename under exp_dir.
            console: If True, print a compact summary to stdout.

        """
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.exp_dir / filename
        self.console = console
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        self._fieldnames = None

    def _init_writer(self, row: Dict[str, object]) -> None:
        """
        Initialize CSV writer and file handle using row keys.

        Args:
            row: A sample dictionary whose keys define CSV columns.

        """
        self._fieldnames = ["time"] + list(row.keys())
        tmp_path = self.csv_path.with_suffix(".tmp")
        f = open(tmp_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(f, fieldnames=self._fieldnames)
        self._writer.writeheader()
        f.flush()
        os.fsync(f.fileno())
        f.close()
        # rename tmp to actual file atomically
        os.replace(tmp_path, self.csv_path)
        # open for append
        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)

    def log(self, entry: Dict[str, object]) -> None:
        """
        Append a new metric row.

        Args:
            entry: Mapping of metric names to scalar values or JSON-serial-
                izable objects.

        Raises:
            ValueError: If entry is empty.

        """
        if not entry:
            raise ValueError("entry must be a non-empty dict")
        with self._lock:
            if self._writer is None:
                self._init_writer(entry)
            # ensure same columns: missing keys will be empty
            record = {"time": time.time()}
            for k in self._fieldnames[1:]:
                val = entry.get(k, "")
                # convert nested structures to JSON string
                if isinstance(val, (dict, list, tuple)):
                    val = json.dumps(val, ensure_ascii=False)
                record[k] = val
            self._writer.writerow(record)
            self._file.flush()
            try:
                os.fsync(self._file.fileno())
            except Exception:
                pass
            if self.console:
                # compact console output
                items = ", ".join(f"{k}={record[k]}" for k in self._fieldnames[1:])
                print(
                    f"[LOG] {time.strftime('%Y-%m-%d %H:%M:%S')} {items}",
                    file=sys.stdout,
                )

    def close(self) -> None:
        """Close underlying CSV file handle if open."""
        with self._lock:
            if self._file is not None:
                try:
                    self._file.close()
                except Exception:
                    pass
                self._file = None
                self._writer = None

    def iter_csv(self) -> Iterable[Dict[str, str]]:
        """
        Yield rows from the CSV file as dictionaries.

        Returns:
            Iterator of dict rows. If file missing, yields nothing.

        """
        if not self.csv_path.exists():
            return iter(())
        f = open(self.csv_path, "r", newline="", encoding="utf-8")
        reader = csv.DictReader(f)
        for row in reader:
            yield row
        f.close()


def _cli() -> None:
    """CLI helper to append one JSON-formatted entry to a CSV."""
    import argparse

    parser = argparse.ArgumentParser(description="Append JSON entry to metrics CSV")
    parser.add_argument("exp_dir", type=str, help="Experiment directory")
    parser.add_argument(
        "--entry",
        type=str,
        required=True,
        help='JSON string for the entry, e.g. \'{"epoch":1,"loss":0.5}\'',
    )
    parser.add_argument(
        "--no_console", action="store_true", help="Disable console print"
    )
    args = parser.parse_args()
    logger = Logger(args.exp_dir, console=not args.no_console)
    entry = json.loads(args.entry)
    logger.log(entry)
    logger.close()


if __name__ == "__main__":
    _cli()
