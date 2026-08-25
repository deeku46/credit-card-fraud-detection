"""
utils.py
--------
Small shared helpers used across the pipeline: logging setup and seeding.
"""

import logging
import random

import numpy as np


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger so every module prints consistently."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def set_seed(seed: int) -> None:
    """Seed every RNG we touch so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)

from contextlib import contextmanager
import time


@contextmanager
def timer(label: str, logger=None):
    """Context manager that measures wall-clock time for a code block.
    Used to report inference latency, one of the required operational metrics.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        msg = f"[{label}] took {elapsed:.4f}s"
        if logger:
            logger.info(msg)
        else:
            print(msg)