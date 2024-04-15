from pathlib import Path

import cortex
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

from data import load_fmri, load_wav
from features import downsample, get_envelope, trim
from utils import get_logger, load_config

log = get_logger(__name__)
cfg = load_config()

DATADIR = Path(cfg["DATA_DIR"])
STORIES = cfg["STORIES"]
WAV_DIR = "stimuli"


def do_envelope_regression():
    story = STORIES[0]
    subject = "UTS02"
    tr_len = 2.0
    alpha = 1.0

    sfreq, wav_data = load_wav(story)
    frmi_data = load_fmri(story, subject)

    fmri_sfreq = 1 / tr_len
    scaling_factor = sfreq / fmri_sfreq

    wav_data = np.mean(wav_data, axis=1)

    X_data = downsample(trim(get_envelope(wav_data), sfreq), scaling_factor)

    clf = Ridge(alpha=alpha)
    clf.fit(X_data[:, np.newaxis], frmi_data)

    y_hat = clf.predict(X_data[:, np.newaxis])
    coefs = [np.corrcoef(y1, y2)[0, 1] for y1, y2 in zip(frmi_data.T, y_hat.T)]

    plt.plot(coefs)
    plt.show()

    vol_data = cortex.Volume(np.array(coefs), "UTS02", "UTS02_auto")
    cortex.webshow(vol_data)


if __name__ == "__main__":
    do_envelope_regression()
