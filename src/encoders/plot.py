import os
from pathlib import Path
from typing import Dict, Optional

import cortex
import matplotlib.pyplot as plt
import numpy as np
from cortex import svgoverlay
from scipy.stats import bootstrap, sem, trim_mean

from encoders.utils import ROOT, check_make_dirs, load_config

os.environ["PATH"] = load_config()["INKSCAPE_PATH"] + ":" + os.environ["PATH"]
svgoverlay.INKSCAPE_VERSION = str(load_config()["INKSCAPE_VERSION"])


def plot_voxel_performance(
    subject: str,
    scores: np.ndarray,
    show_plot: bool = False,
    ax=None,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = 0.5,
    cmap: str = "inferno",
):
    """Plots the given scores for each voxel.

    Parameters
    ----------
    subject : {"UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"}
        Subject identifier
    scores : np.ndarray
        The scores for each voxel
    save_path : str or None
        Path to save the plot. If `None`, plot will not be saved.
    show_plot : bool, optional
        Whether to show the plot or not.
    title : str, optional
        Title for the plot. If `None`, no title will be shown.
    vmin : float or None, optional
        Minimal value on color map. If `None`, pycortex chooses the value automatically.
    vmax : float or None, optional
        Maximal value on color map. If `None`, pycortex chooses the value automatically.
    cmap : str, optional
        Color map for plot.
    ax : matplotlib.Axis,
    """

    vol_data = cortex.Volume(
        scores, subject, f"{subject}_auto", vmin=vmin, vmax=vmax, cmap=cmap
    )
    # cortex.quickshow(vol_data)

    _ = cortex.quickflat.make_figure(
        braindata=vol_data,
        fig=ax,
        recache=False,
    )

    return None


def plot_aggregate_results(
    results: Dict[str, Dict],
    save_path: Optional[str],
    show_plot: bool = False,
    title: Optional[str] = None,
):
    """Plots a line plot with aggregate results over the amount of training stories.

    Parameters
    ----------
    results : Dict
        The scores for each voxel
    save_path : str or None
        Path to save the plot. If `None`, plot will not be saved.
    show_plot : bool, optional
        Whether to show the plot or not.
    title : str, optional
        Title for the plot. If `None`, no title will be shown.
    """

    fig, ax = plt.subplots()

    for predictor, predictor_results in results.items():
        x = []
        y = []
        for n_stories, value in predictor_results.items():
            x.append(int(n_stories))
            y.append(value)

        ax.plot(x, y, label=predictor)

    plt.xlabel("N stories")
    plt.ylabel("Correlation")
    plt.legend(loc="upper left")

    if title is not None:
        ax.title(title)
    if show_plot:
        plt.show()

    if save_path is not None:
        check_make_dirs(save_path)
        fig.savefig(save_path)


def load_data(datapath, n_stories, condition):
    fn = Path(datapath, n_stories, condition, "scores_mean.npy")
    return np.load(fn)


def get_perc(x1, perc):
    return x1[x1 > np.percentile(x1, perc)]


def trim_mean_wrapper(x):
    return trim_mean(x, proportiontocut=0.1)


def get_ci(x):

    ci = bootstrap(
        data=(x,), statistic=trim_mean_wrapper, n_resamples=500, confidence_level=0.95
    ).confidence_interval

    return (ci.low.item(), ci.high.item())


def get_yerr(mean, ci):
    return (mean - ci[0], ci[1] - mean)


def plot_means(data_tuple):

    data1 = data_tuple[0]
    data2 = data_tuple[1]

    fig, ax = plt.subplots(1, 2, figsize=(5, 4), layout="constrained", sharey=True)

    ax[0].boxplot(data1, flierprops={"color": "gray"})
    ax[1].boxplot(data2)

    fig.supylabel("Mean test-set corr.\n(across top 1% voxels)", ha="center")
    fig.supxlabel("N train stories", ha="center")

    ax[0].set_title("Original")
    ax[1].set_title("Shuffled")

    for a in ax:
        a.set_xticks([1, 2, 3], [1, 5, 12])
        a.grid(visible=True, ls="--", alpha=0.6)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    return fig


n_stories = [1, 5, 12]


# load the data
def load_data_wrapper(ds: str, which: str):

    datadir = Path(ds, which, "UTS02")
    rho_orig, rho_shuf = {}, {}
    for n in n_stories:
        rho_orig[str(n)] = load_data(
            datapath=datadir, n_stories=str(n), condition="not_shuffled"
        )
        rho_shuf[str(n)] = load_data(
            datapath=datadir, n_stories=str(n), condition="shuffled"
        )

    return rho_orig, rho_shuf


def make_brain_plots(scores_dict):

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")

    titles = {
        "1": "1 Training story",
        "5": "5 Training stories",
        "12": "12 Training stories",
    }
    for i, n in enumerate(n_stories):

        plot_voxel_performance(
            scores=scores_dict[str(n)], subject="UTS02", vmin=0, vmax=0.5, ax=ax[i]
        )

        ax[i].set_title(titles[str(n)])

    cbar = ax[1].images[0].colorbar
    cbar.ax.set_xlabel("Test-set correlation", fontsize=12)

    return fig


def make_figure(run_dir, which):

    rho_orig, _ = load_data_wrapper(ds=run_dir, which=which)

    fig = make_brain_plots(scores_dict=rho_orig)

    return fig


if __name__ == "__main__":

    cfg = load_config()
    RUNS_DIR = cfg["RUNS_DIR"]
    ds = RUNS_DIR + "/2024-11-06_15-34_361506"
    SAVEPATH = Path(ROOT, "fig")

    fig1 = make_figure(run_dir=ds, which="embeddings")
    fn1 = str(SAVEPATH / "embedding_performance.svg")
    fig1.savefig(fn1)
    fig1.savefig(fn1.replace(".svg", ".png"), bbox_inches="tight", dpi=300)

    fig2 = make_figure(run_dir=ds, which="envelope")
    fig2.suptitle(
        "Envelope encoding model performance with increasing training data", fontsize=14
    )
    fn2 = str(SAVEPATH / "envelope_performance.svg")
    fig2.savefig(fn2)
    fig2.savefig(fn2.replace(".svg", ".png"), bbox_inches="tight", dpi=300)

    out = load_data_wrapper(ds=ds, which="embeddings")

    orig = out[0]
    shuf = out[1]
    y = np.zeros(len(orig))

    y_orig = np.array([np.mean(data) for data in orig.values()])
    y_orig_sem = np.array([sem(data) for data in orig.values()])
    y_shuf = np.array([np.mean(data) for data in shuf.values()])

    fig, ax = plt.subplots()

    x = ["1", "5", "12"]

    ax.plot(x, y_orig, "-o")

    # plot sem
    ax.fill_between(x, y1=y_orig + y_orig_sem, y2=y_orig - y_orig_sem, alpha=0.3)

    # ax.plot(["1", "5", "12"], y_shuf)

    ax.set_xlabel("N Stories")
    ax.set_ylabel("Regression performance")

    ax.grid(visible=True, lw=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.show()
