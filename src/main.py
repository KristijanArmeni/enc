import argparse
import os
from pathlib import Path
from typing import Optional, Union

import cortex
import numpy as np
from matplotlib import pyplot as plt

from data import load_fmri, load_wav
from features import downsample, get_embeddings, get_envelope, trim
from regression import cross_validation_ridge_regression, score_correlation
from utils import get_logger, lanczosinterp2D, load_config, make_delayed

log = get_logger(__name__)
cfg = load_config()

DATADIR = Path(cfg["DATA_DIR"])
STORIES = cfg["STORIES"]
WAV_DIR = "stimuli"
CACHE_DIR = Path(cfg["CACHE_DIR"])


def load_envelope_data(
    story: str,
    tr_len: float,
    y_data: np.ndarray,
    use_cache: bool = True,
) -> np.ndarray:
    path_cache_x = Path(CACHE_DIR, "envelope_data", f"{story}_{tr_len}_X.npy")
    if Path.exists(path_cache_x) and use_cache:
        log.info(f"Loading from cache: {path_cache_x}")
        return np.load(path_cache_x)
    elif use_cache:
        log.info(f"No data found in cache: {path_cache_x}")

    n_trs = y_data.shape[0]

    sfreq, wav_data = load_wav(story)

    wav_data = np.mean(wav_data, axis=1)

    X_envelope = get_envelope(wav_data)
    X_trimmed = trim(X_envelope, sfreq)
    X_data = downsample(X_trimmed, sfreq, tr_len, n_trs)
    X_data = X_data[:, np.newaxis]

    if use_cache:
        os.makedirs(Path(CACHE_DIR, "envelope_data"), exist_ok=True)
        np.save(path_cache_x, X_data)
        log.info("Cached results.")
    return X_data


def load_sm1000_data(
    story: str, tr_len: float, y_data: np.ndarray, start_trim: float = 10.0
) -> np.ndarray:

    data, starts, stops = get_embeddings(story)

    # get 'mean word position'
    t_word = (starts + stops) / 2

    n_trs = y_data.shape[0]

    # get starting word index for each tr
    # first 10 seconds of the fMRI data is trimmed
    data_word_starts = [
        sum(t_word < idx_tr * tr_len + start_trim) for idx_tr in range(n_trs + 1)
    ]

    X_data = np.empty((n_trs, data.shape[1]))
    for idx_tr, (idx_start, idx_end) in enumerate(
        zip(data_word_starts[:-1], data_word_starts[1:])
    ):
        if idx_start == idx_end:
            # no word in tr
            X_data[idx_tr] = 0
        else:
            # average words occuring in tr
            X_data[idx_tr] = np.mean(data[idx_start:idx_end], axis=0)

    return X_data


def downsample_embeddings_lanczos(
    embeddings: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    n_trs: int,
    tr_len: float,
    start_trim: float = 10.0,
) -> np.ndarray:
    word_times = (starts + stops) / 2
    tr_times = np.arange(n_trs) * tr_len + start_trim
    downsampled_embeddings = lanczosinterp2D(embeddings, word_times, tr_times, window=3)
    return downsampled_embeddings


def do_regression(
    predictor: str = "embeddings",
    n_stories: Optional[int] = None,
    subject: str = "UTS02",
    tr_len: float = 2.0,
    n_delays: int = 4,
    interpolation: str = "lanczos",
    use_cache: bool = True,
    show_results: bool = True,
    shuffle: bool = False,
    stories: list[str] = STORIES,
) -> tuple[
    np.ndarray, list[np.ndarray], list[np.ndarray], list[Union[float, np.ndarray]]
]:
    """Runs regression.

    Parameters
    ----------
    predictor : {"envelope", "embeddings", "embeddings_huth"}
        Which predictor to use for the regression. "envelope": audio envelope that
        participants heard. "embeddings": embeddings computed from the words that
        participants heard. "embeddings_huth": embeddings which were used in the
        huth paper (https://www.nature.com/articles/s41597-023-02437-z)
    n_stories: int, optional
        The , if this is `None` will use all stories.
    subject : {"UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"}, default="UTS02"
        Subject identifier
    tr_len: float, default=2.0
        Length of tr-windows used to sample fMRI data.
    n_delays: int, default=4
        How many delays are used to model the HRF, which is modeled by adding
        a shifted set of duplicated features for each delay. `n_delays=5` implies
        that the the features of the stimulus are shifted concatinated 5 times
        to training/testing data.
    interpolation: {"lanczos", "average"}, default="lanczos"
        Whether to use lanczos interpolation or just average the words within a TR.
        Only applies to the 'embeddings' predictor.
    use_cache: bool, default=True
        Whether the cache is used for `envelope` features.
    show_results: bool, default=True
        Create a plot showing the results in pycortex.
    shuffle: bool, default=False
        Whether to shuffle the predictors (features).
    stories: list[str] = STORIES,
    """

    if n_stories is None:
        n_stories = len(stories)
    else:
        stories = stories[:n_stories]
    n_splits = n_stories

    X_data_list = []
    y_data_list = []
    for story in stories:

        y_data = load_fmri(story, subject)
        if shuffle:
            y_data = np.random.permutation(y_data)

        if predictor == "embeddings":
            if interpolation == "lanczos":
                data, starts, stops = get_embeddings(story)
                X_data = downsample_embeddings_lanczos(
                    data, starts, stops, y_data.shape[0], tr_len
                )
            elif interpolation == "average":
                X_data = load_sm1000_data(story, tr_len, y_data)
            X_data = make_delayed(X_data, np.arange(1, n_delays + 1), circpad=False)

        elif predictor == "envelope":
            X_data = load_envelope_data(story, tr_len, y_data, use_cache)
            X_data = make_delayed(X_data, np.arange(1, n_delays + 1), circpad=False)

        elif predictor == "embeddings_huth":
            X_data = np.load(f"/Volumes/opt/enc/data/embeddings_huth/{story}.npy")
            X_data = make_delayed(X_data, np.arange(1, n_delays + 1), circpad=False)

        assert (
            X_data.shape[0] == y_data.shape[0]
        ), f"ERROR loading {story}: X {X_data.shape} and y {y_data.shape} do not match up"
        X_data_list.append(X_data)
        y_data_list.append(y_data)

    # from here: regression function

    (
        mean_scores,
        all_scores,
        all_weights,
        best_alphas,
    ) = cross_validation_ridge_regression(
        X_data_list, y_data_list, n_splits=n_splits, score_fct=score_correlation
    )

    if show_results:
        plt.plot(mean_scores)
        plt.show()
    log.info(f"Mean correlation (averages across splits) (r): {mean_scores.mean()}")
    log.info(f"Max  correlation (averaged across splits) (r): {mean_scores.max()}")

    if show_results:
        vol_data = cortex.Volume(
            mean_scores, subject, f"{subject}_auto", vmin=0, vmax=0.5, cmap="inferno"
        )

        cortex.quickshow(vol_data)
        plt.title(f"{subject} {predictor} performance.")
        plt.show()
        plt.savefig(os.path.join("data", f"{predictor}_{subject}_{n_stories}.png"))
        # save the plot
        _ = cortex.quickflat.make_png(
            os.path.join("data", f"{predictor}_{subject}_{n_splits}.png"),
            vol_data,
            recache=False,
        )
        # without print statement the plot does not show up.
        print("Done")
    return mean_scores, all_scores, all_weights, best_alphas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to run a ")
    parser.add_argument(
        "predictor",
        choices=["embeddings", "envelope", "embeddings_huth"],
        help="The predictor",
    )
    parser.add_argument(
        "--n_stories",
        default=5,
        type=int,
        help="How many stories are used. Averages LOO results.",
    )
    parser.add_argument(
        "--subject",
        default="UTS02",
        type=str,
        help="Subject identifier in the downloaded dataset.",
    )
    parser.add_argument(
        "--not_use_cache",
        action="store_true",
        help="Disable cache for feature commputation.",
    )
    parser.add_argument(
        "--interpolation_method",
        default="lanczos",
        choices=["lanczos", "average"],
        help="Interpolation method for embeddings.",
    )
    parser.add_argument(
        "--n_delays",
        default=4,
        type=int,
        help="Delays to model the HRF.",
    )

    args = parser.parse_args()
    do_regression(
        predictor=args.predictor,
        n_stories=args.n_stories,
        subject=args.subject,
        use_cache=not args.not_use_cache,
        n_delays=args.n_delays,
    )
