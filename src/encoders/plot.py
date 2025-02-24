import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import cortex
import matplotlib.pyplot as plt
import numpy as np
from cortex import svgoverlay
from scipy.stats import bootstrap, sem, trim_mean

from encoders.utils import ROOT, check_make_dirs, get_logger, load_config

log = get_logger(__name__)

INKSCAPE_PATH = load_config().get("INKSCAPE_PATH")
if not Path(INKSCAPE_PATH).exists():
    log.critical(
        "INKSCAPE_PATH not valid. Install inkscape, and place path to"
        " excecutale into the config as INKSCAPE_PATH."
    )
    import sys

    sys.exit(-1)

os.environ["PATH"] = INKSCAPE_PATH + ":" + os.environ["PATH"]
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
        ax.title(title)  # type: ignore
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


n_stories = [1, 12, 20]


# load the data
def load_data_wrapper(ds: str, n_stories, which: str):
    datadir = Path(ds, which, "UTS02")
    rho_orig = {}
    for n in n_stories:
        rho_orig[str(n)] = load_data(
            datapath=datadir, n_stories=str(n), condition="not_shuffled"
        )
        # rho_shuf[str(n)] = load_data(
        #    datapath=datadir, n_stories=str(n), condition="shuffled"
        # )

    return rho_orig


def make_brain_plots(scores_dict):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")

    for i, items in enumerate(scores_dict.items()):
        n, data = items
        plot_voxel_performance(scores=data, subject="UTS02", vmin=0, vmax=0.5, ax=ax[i])

        ax[i].set_title(
            f"{n} Training story" if int(n) == 1 else f"{n} Training stories"
        )

    cbar = ax[1].images[0].colorbar
    cbar.ax.set_xlabel("Test-set correlation", fontsize=12)

    return fig


def make_figure(run_dir, which):
    rho_orig = load_data_wrapper(ds=run_dir, which=which, n_stories=[1, 12, 20])

    fig = make_brain_plots(scores_dict=rho_orig)

    return fig


def plot_training_curve(run_dir):
    # load embedding model performance
    embeds = load_data_wrapper(
        ds=run_dir, which="embeddings", n_stories=[1, 3, 5, 7, 9, 11, 12, 15, 20]
    )

    # load audio envelope model performance
    audio = load_data_wrapper(
        ds=run_dir, which="envelope", n_stories=[1, 3, 5, 7, 9, 11, 12, 15, 20]
    )

    # load the performance scores and average across cortex shape = (n_stories,)
    y_embeds = np.array([np.nanmean(data) for data in embeds.values()])
    y_audio = np.array([np.nanmean(data) for data in audio.values()])

    # figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # ["1", "3", "5", ...]
    x = embeds.keys()

    ax.plot(x, y_embeds, "-o", label="Word embeddings")
    ax.plot(x, y_audio, "-o", label="Audio envelope")

    ax.legend(title="Predictor")
    ax.set_title("Model performance with increasing training set size (UTS02)")

    ax.set_xlabel("N Stories")
    ax.set_ylabel("Mean test-set performance\n(average across all voxels)")
    ax.grid(visible=True, lw=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "plot.py",
        description="Plot the replication figures and save as .pdf and .png",
    )

    parser.add_argument("run_dir", help="folder with results for the run to be ploted")
    parser.add_argument("save_path", help="path to where the figures are saved")

    args = parser.parse_args()

    cfg = load_config()

    SAVEPATH = Path(args.save_path)

    # embedding performance
    fig1 = make_figure(run_dir=args.run_dir, which="embeddings")
    fig1.suptitle(
        "Embedding encoding model performance with increasing training data",
        fontsize=14,
    )

    fn1 = str(SAVEPATH / "embeddings_performance.pdf")
    log.info(f"Saving {fn1}")
    fig1.savefig(fn1, bbox_inches="tight", transparent=True)

    fn1_png = fn1.replace(".pdf", ".png")
    log.info(f"Saving {fn1_png}")
    fig1.savefig(fn1_png, bbox_inches="tight", dpi=300)

    # audio envelope model performance
    fig2 = make_figure(run_dir=args.run_dir, which="envelope")
    fig2.suptitle(
        "Envelope encoding model performance with increasing training data", fontsize=14
    )

    fn2 = str(SAVEPATH / "envelope_performance.pdf")
    log.info(f"Saving {fn2}")
    fig2.savefig(fn2, bbox_inches="tight", transparent=True)

    fn2_png = fn2.replace(".pdf", ".png")
    log.info(f"Saving {fn2_png}")
    fig2.savefig(fn2_png, bbox_inches="tight", dpi=300)

    # training curve figure
    fig3 = plot_training_curve(run_dir=args.run_dir)

    fn3 = str(SAVEPATH / "training_curve.pdf")
    log.info(f"Saving {fn3}")
    fig3.savefig(fn3, bbox_inches="tight", transparent=True)

    fn3_png = fn3.replace(".pdf", ".png")
    log.info(f"Saving {fn3_png}")
    fig3.savefig(fn3_png, bbox_inches="tight", dpi=300)
