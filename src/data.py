"""
module for loading fMRI data and features and the like
"""
import os

import h5py
import numpy as np

from utils import load_config


def load_wav(story: str):
    pass


def load_textgrid(story: str):
    pass


def load_fmri(story: str, subject: str):
    config = load_config()
    subject_dir = os.path.join(
        config["DATA_DIR"], f"derivative/preprocessed_data/{subject}"
    )
    resp_path = os.path.join(subject_dir, f"{story}.hf5")
    hf = h5py.File(resp_path, "r")
    return np.array(hf["data"][:])  # shape (time, voxels)
