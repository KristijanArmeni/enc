"""
module for loading fMRI data and features and the like
"""

from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
from scipy.io import wavfile

from .utils import get_logger, load_config

WAV_DIR = "stimuli"

log = get_logger(__name__)


def load_wav(story: str) -> Tuple[int, np.ndarray]:
    """Load wav file. Return ndarray with shape [samples, channels]."""
    config = load_config()

    wav_path = Path(config["DATA_DIR"], WAV_DIR, f"{story}.wav")
    sample_rate, wav = wavfile.read(wav_path)
    log.info(
        f"{story}.wav"
        f" | channels: {wav.shape[1]}"
        f" | length {wav.shape[0] / sample_rate}s"
    )
    return sample_rate, wav


def load_textgrid(story: str):
    pass


def load_fmri(story: str, subject: str) -> np.ndarray:
    config = load_config()
    subject_dir = Path(config["DATA_DIR"], f"derivative/preprocessed_data/{subject}")
    resp_path = Path(subject_dir, f"{story}.hf5")
    hf = h5py.File(resp_path, "r")
    # shape (time, voxels)
    return np.array(hf["data"][:])  # type: ignore
