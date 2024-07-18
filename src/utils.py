import logging
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
FORMAT = "[%(levelname)s] %(name)s.%(funcName)s - %(message)s"

logging.basicConfig(format=FORMAT)


def get_logger(
    name=__name__,
    log_level=logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Initializes command line logger."""

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if log_file is not None:
        formatter = logging.Formatter(FORMAT)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def load_config():

    with open(ROOT / "config.yaml", "r") as f:
        config = yaml.safe_load(f)

    return config


def make_delayed(signal: np.ndarray, delays: np.ndarray, circpad=False) -> np.ndarray:
    """
    Create delayed versions of the 2-D signal.

    Parameters
    -----------
    signal : np.ndarray
        2-D array of shape (n_samples, n_features)
    delays : np.ndarray
        1-D array of delays to apply to the signal
        can be positive or negative; negative values advance the signal (shifting it backward)
    circpad : bool
        If True, use circular padding for delays
        If False, use zero padding for delays

    Returns
    --------
    np.ndarray
        2-D array of shape (n_samples, n_features * n_delays)
    """

    delayed_signals = []
    n_samples, n_features = signal.shape

    for delay in delays:
        delayed_signal = np.zeros_like(signal)
        if circpad:
            delayed_signal = np.roll(signal, delay, axis=0)
        else:
            if delay > 0:
                delayed_signal[delay:, :] = signal[:-delay, :]
            elif delay < 0:
                delayed_signal[:delay, :] = signal[-delay:, :]
            else:
                delayed_signal = signal.copy()
        delayed_signals.append(delayed_signal)

    return np.hstack(delayed_signals)
