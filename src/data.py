"""
module for loading fMRI data and features and the like
"""

import re
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.io import wavfile

from utils import get_logger, load_config

log = get_logger(__name__)
cfg = load_config()

DATADIR = Path(cfg["DATA_DIR"])
STORIES = cfg["STORIES"]
WAV_DIR = "stimuli"
EMBEDDINGS_FILE = Path(DATADIR, "derivative", "english1000sm.hf5")


def load_wav(story: str) -> Tuple[int, np.ndarray]:
    """Load wav file. Return ndarray with shape [samples, channels]."""

    wav_path = Path(DATADIR, WAV_DIR, f"{story}.wav")
    sample_rate, wav = wavfile.read(wav_path)
    log.info(
        f"{story}.wav"
        f" | channels: {wav.shape[1]}"
        f" | length {wav.shape[0] / sample_rate}s"
    )
    return sample_rate, wav


def load_textgrid(story: str):

    textgrid_dir = DATADIR / "derivative" / "TextGrids"
    fn = textgrid_dir / f"{story}.TextGrid"

    with open(fn, "r") as f:
        lines = [e.strip("\n").strip() for e in f.readlines()]

    # find mathc using re.match
    start = float(lines[3].strip("xmin = ").strip())
    stop = float(lines[4].strip("xmax = ").strip())

    assert stop > start

    # find interval tiers
    interval_tiers = []
    tier_names = {}
    tier_n = []
    tier_starts = []
    for i, line in enumerate(lines):
        if re.match(r'class = "IntervalTier"', line):
            interval_tiers.append(i)
        if re.match(r"name = ", line):
            name = line.split('"')[1]
            tier_names[name] = i
        if re.match(r"intervals: size = ", line):
            tier_n.append(int(line.split(" ")[-1]))
            tier_starts.append(i + 1)

    # find the tier with the words
    phone_start, word_start = tier_starts
    phone_stop = phone_start + tier_n[0] * 4
    word_stop = word_start + tier_n[1] * 4
    phone_tier = np.array(lines[phone_start:phone_stop])
    word_tier = np.array(lines[word_start:word_stop])

    out = {n: None for n in tier_names.keys()}

    def find_tiers(tier_array):
        starttimes = tier_array[1::4]
        stoptimes = tier_array[2::4]
        textstrings = tier_array[3::4]

        starttimes = [float(e.strip("xmin = ").strip()) for e in starttimes]
        stoptimes = [float(e.strip("xmax = ").strip()) for e in stoptimes]
        textstrings = [e.strip("text = ").strip('"') for e in textstrings]

        assert (
            len(starttimes) == len(stoptimes) == len(textstrings)
        ), f"{len(starttimes)}, {len(stoptimes)}, {len(textstrings)}"

        return starttimes, stoptimes, textstrings

    phones_start, phones_stop, phones = find_tiers(phone_tier)
    phone_dict = {"start": phones_start, "stop": phones_stop, "text": phones}

    words_start, words_stop, words = find_tiers(word_tier)
    word_dict = {"start": words_start, "stop": words_stop, "text": words}

    colnames = ["start", "stop", "text"]
    out["phone"] = pd.DataFrame(phone_dict, columns=colnames)
    out["word"] = pd.DataFrame(word_dict, columns=colnames)

    return out


def load_fmri(story: str, subject: str) -> np.ndarray:
    """Load fMRI data. Return ndarray with shape [time, voxels]."""

    subject_dir = Path(DATADIR, f"derivative/preprocessed_data/{subject}")
    resp_path = Path(subject_dir, f"{story}.hf5")
    hf = h5py.File(resp_path, "r")
    log.info(
        f"{story}.hf5"
        f" | {subject}"
        f" | time: {hf['data'].shape[0]}"
        f" | voxels: {hf['data'].shape[1]}"
    )
    return np.array(hf["data"][:])  # type: ignore
