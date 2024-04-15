import numpy as np
from scipy.signal import hilbert, resample

from utils import get_logger

log = get_logger(__name__)


def get_envelope(signal: np.ndarray) -> np.ndarray:
    """Compute the audio envelope"""
    log.info("Computing envelope.")
    return np.abs(hilbert(signal))  # type: ignore


def trim(signal: np.ndarray, sfreq: int, duration: float = 10.0) -> np.ndarray:
    log.info(f"Trimming beginning and end by {duration}s.")
    return signal[int(sfreq * duration) : int(-sfreq * duration)]


def downsample(signal: np.ndarray, factor: float) -> np.ndarray:
    num = int(signal.shape[0] / factor)
    log.info(f"Downsampling to {num} samples.")
    return resample(signal, num=num)  # type: ignore


# TODO: band pass filtering
