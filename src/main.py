import os
from pathlib import Path

import cortex
import numpy as np
from matplotlib import pyplot as plt

from data import load_fmri, load_wav
from features import downsample, get_envelope, trim
from regression import cross_validation_ridge_regression, score_correlation
from utils import get_logger, load_config

log = get_logger(__name__)
cfg = load_config()

DATADIR = Path(cfg["DATA_DIR"])
STORIES = cfg["STORIES"]
WAV_DIR = "stimuli"
CACHE_DIR = Path(cfg["CACHE_DIR"])


def load_envelope_data(story: str, subject: str, tr_len: float = 2.0) -> np.ndarray:
    path_cache_x = Path(CACHE_DIR, "envelope_data", f"{subject}_{story}_{tr_len}_X.npy")
    if Path.exists(path_cache_x):
        log.info(f"Loading from cache: {path_cache_x}")
        return np.load(path_cache_x)

    log.info(f"No data found in cache: {path_cache_x}")

    sfreq, wav_data = load_wav(story)

    fmri_sfreq = 1 / tr_len
    scaling_factor = sfreq / fmri_sfreq

    wav_data = np.mean(wav_data, axis=1)

    X_data = downsample(trim(get_envelope(wav_data), sfreq), scaling_factor)
    X_data = X_data[:, np.newaxis]

    os.makedirs(Path(CACHE_DIR, "envelope_data"), exist_ok=True)
    np.save(path_cache_x, X_data)
    log.info("Cached results.")
    return X_data


def do_envelope_regression():
    alpha = 1.0
    n_splits = 5

    X_data_list = []
    y_data_list = []
    for story_id in [0, 1, 2, 3, 4]:
        story = STORIES[story_id]
        subject = "UTS02"
        tr_len = 2.0

        X_data = load_envelope_data(story, subject, tr_len)
        y_data = load_fmri(story, subject)

        X_data_list.append(X_data)
        y_data_list.append(y_data)

    # from here: regression function

    mean_scores, all_scores, all_weights = cross_validation_ridge_regression(
        X_data_list,
        y_data_list,
        n_splits=n_splits,
        alpha=alpha,
        score_fct=score_correlation,
    )

    plt.plot(mean_scores)
    plt.show()

    vol_data = cortex.Volume(np.array(mean_scores), "UTS02", "UTS02_auto")
    cortex.webshow(vol_data)


if __name__ == "__main__":
    do_envelope_regression()
