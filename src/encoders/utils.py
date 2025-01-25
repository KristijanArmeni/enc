import logging
import os
from datetime import datetime
import time
from pathlib import Path
from random import choices
from typing import List, Optional, Union

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent.parent
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


log = get_logger(__name__)


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


def sinc(f_c, t):
    """
    Sin function with cutoff frequency f_c.

    Parameters
    -----------
    f_c : float
        Cutoff frequency
    t : np.ndarray or float
        Time

    Returns
    --------
    np.ndarray or float
        Sin function with cutoff frequency f_c
    """
    return np.sin(np.pi * f_c * t) / (np.pi * f_c * t)


def lanczosfun(f_c, t, a=3):
    """
    Lanczos function with cutoff frequency f_c.

    Parameters
    -----------
    f_c : float
        Cutoff frequency
    t : np.ndarray or float
        Time
    a : int
        Number of lobes (window size), typically 2 or 3; only signals within the window will have non-zero weights.

    Returns
    --------
    np.ndarray or float
        Lanczos function with cutoff frequency f_c
    """
    val = sinc(f_c, t) * sinc(f_c, t / a)
    val[t == 0] = 1.0
    val[np.abs(t * f_c) > a] = 0.0

    return val


def lanczosinterp2D(signal, oldtime, newtime, window=3, cutoff_mult=1.0):
    """
    Lanczos interpolation for 2D signals; interpolates [signal] from [oldtime] to [newtime], assuming that the rows of [signal] correspond to [oldtime]. Returns a new signal with rows corresponding to [newtime] and the same number of columns as [signal].

    Parameters
    -----------
    signal : np.ndarray
        2-D array of shape (n_samples, n_features)
    oldtime : np.ndarray
        1-D array of old time points
    newtime : np.ndarray
        1-D array of new time points
    window : int
        Number of lobes (window size) for the Lanczos function
    cutoff_mult : float
        Multiplier for the cutoff frequency

    Returns
    --------
    np.ndarray
        2-D array of shape (len(newtime), n_features)
    """
    # Find the cutoff frequency
    f_c = 1 / (np.max(np.abs(np.diff(newtime)))) * cutoff_mult
    # Build the Lanczos interpolation matrix
    interp_matrix = np.zeros((len(newtime), len(oldtime)))
    for i, t in enumerate(newtime):
        interp_matrix[i, :] = lanczosfun(f_c, t - oldtime, a=window)
    # Interpolate the signal
    newsignal = np.dot(interp_matrix, signal)

    return newsignal


def check_make_dirs(
    paths: Union[str, List[str]],
    verbose: bool = True,
    isdir: bool = False,
) -> None:
    """Create base directories for given paths if they do not exist.

    Parameters
    ----------
    paths: List[str] | str
        A path or list of paths for which to check the basedirectories
    verbose: bool, default=True
        Whether to log the output path
    isdir: bool, default=False
        Treats given path(s) as diretory instead of only checking the basedir.
    """

    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if isdir and path != "" and not os.path.exists(path):
            os.makedirs(path)
        elif os.path.dirname(path) != "" and not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if verbose:
            log.info(f"Output path: {path}")


def create_run_folder_name() -> str:
    """Returns the name of the run folder including
    a random id and the current date."""

    date = datetime.today().strftime("%Y-%m-%d_%H-%M")
    rand_num = "".join(choices("0123456789", k=6))
    return f"{date}_{rand_num}"


def counter(iterable, countevery=100, total=None, logger=logging.getLogger("counter")):
    """Logs a status and timing update to [logger] every [countevery] draws from [iterable].
    If [total] is given, log messages will include the estimated time remaining.
    """
    start_time = time.time()

    ## Check if the iterable has a __len__ function, use it if no total length is supplied
    if total is None:
        if hasattr(iterable, "__len__"):
            total = len(iterable)

    for count, thing in enumerate(iterable):
        yield thing

        if not count % countevery:
            current_time = time.time()
            rate = float(count + 1) / (current_time - start_time)

            if rate > 1:  ## more than 1 item/second
                ratestr = "%0.2f items/second" % rate
            else:  ## less than 1 item/second
                ratestr = "%0.2f seconds/item" % (rate**-1)

            if total is not None:
                remitems = total - (count + 1)
                remtime = remitems / rate
                timestr = ", %s remaining" % time.strftime(
                    "%H:%M:%S", time.gmtime(remtime)
                )
                itemstr = "%d/%d" % (count + 1, total)
            else:
                timestr = ""
                itemstr = "%d" % (count + 1)

            formatted_str = "%s items complete (%s%s)" % (itemstr, ratestr, timestr)
            if logger is None:
                print(formatted_str)
            else:
                logger.info(formatted_str)


def mult_diag(d, mtx, left=True):
    """Multiply a full matrix by a diagonal matrix.
    This function should always be faster than dot.

    Input:
      d -- 1D (N,) array (contains the diagonal elements)
      mtx -- 2D (N,N) array

    Output:
      mult_diag(d, mts, left=True) == dot(diag(d), mtx)
      mult_diag(d, mts, left=False) == dot(mtx, diag(d))

    By Pietro Berkes
    From http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026807.html
    """
    if left:
        return (d * mtx.T).T
    else:
        return d * mtx
