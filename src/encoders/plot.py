import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import cortex
import matplotlib as mpl
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from cortex import svgoverlay

from encoders.utils import (
    BRAIN_PLOT_ORIG_PNG,
    ROOT,
    TRAIN_CURVE_ORIG_PNG,
    get_logger,
    load_config,
)

log = get_logger(__name__)

# matpltotlib params
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["figure.labelsize"] = 14

SUBJECT_IDS = ["UTS01", "UTS02", "UTS03"]
# this 8-color is also available in seaborn:
# https://seaborn.pydata.org/generated/seaborn.husl_palette.html
# but defined here manually to avoid needing seaborn as dependency
HUSL_PALETTE = [
    "#f77189",
    "#ce9032",
    "#97a431",
    "#32b166",
    "#36ada4",
    "#39a7d0",
    "#a48cf4",
    "#f561dd",
]

# make sure inkscape is installed
INKSCAPE_PATH = load_config().get("INKSCAPE_PATH")
if not Path(INKSCAPE_PATH).exists():
    log.critical(
        "INKSCAPE_PATH not valid. Install inkscape, and place path to"
        " excecutale into the config as INKSCAPE_PATH."
    )
    import sys

    sys.exit(-1)

# add inkscape path to PATH
os.environ["PATH"] = INKSCAPE_PATH + ":" + os.environ["PATH"]
svgoverlay.INKSCAPE_VERSION = str(load_config()["INKSCAPE_VERSION"])


def plot_voxel_performance(
    subject: str,
    scores: np.ndarray,
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

    _ = cortex.quickflat.make_figure(
        braindata=vol_data,
        with_colorbar=False,
        fig=ax,
        recache=False,
    )

    return None


def load_data(datapath, n_stories, condition):
    fn = Path(datapath, n_stories, condition, "scores_mean.npy")
    fn2 = Path(datapath, n_stories, condition, "scores_sem.npy")
    return np.load(fn), np.load(fn2)


# load the data
def load_data_wrapper(
    data_folder: str,
    subject: str = "UTS02",
    n_stories: list[int] = [1],
    which: str = "envelope",
):
    # TEMP FIX, make folders consistent
    datadir = Path(data_folder, subject, which)

    rho_means = {}
    rho_sem = {}
    for n in n_stories:
        mean_data, sem_data = load_data(
            datapath=datadir, n_stories=str(n), condition="not_shuffled"
        )

        rho_means[str(n)] = mean_data
        rho_sem[str(n)] = sem_data

    return rho_means, rho_sem


def make_performance_plots(scores_dict: Dict) -> matplotlib.figure.Figure:
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


def make_brain_fig(data_folder, which):
    rho_means, _ = load_data_wrapper(
        data_folder=data_folder, which=which, n_stories=[1, 11, 20]
    )

    fig = make_performance_plots(scores_dict=rho_means)

    return fig


N_STORIES = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]


def _load_means_sem(data_folder: str):
    data_mean = {
        sub: load_data_wrapper(
            data_folder=str(data_folder),
            subject=sub,
            which="eng1000",
            n_stories=N_STORIES,
        )[0]
        for sub in SUBJECT_IDS
    }

    data_sem = {
        sub: load_data_wrapper(
            data_folder=str(data_folder),
            subject=sub,
            which="eng1000",
            n_stories=N_STORIES,
        )[1]
        for sub in SUBJECT_IDS
    }

    def _data2array_agg(data_dict: Dict, aggfunc) -> np.ndarray:
        out = np.array(
            [
                [aggfunc(data) for data in subject_data.values()]
                for subject_data in data_dict.values()
            ]
        )

        return out

    # load the performance scores and average across cortex shape = (n_stories,)
    y_mean = _data2array_agg(data_dict=data_mean, aggfunc=np.mean)

    y_sem = np.array(
        [[subdict[n].item() for n in subdict.keys()] for subdict in data_sem.values()]
    )

    return y_mean, y_sem


def make_training_curve_fig(
    repro_folder: str,
    repli_folder: str,
) -> matplotlib.figure.Figure:
    y_repro, y_repro_sem = _load_means_sem(data_folder=repro_folder)
    y_repli, y_repli_sem = _load_means_sem(data_folder=repli_folder)

    fig = plt.figure(figsize=(15, 6.5), constrained_layout=True)
    gs = gridspec.GridSpec(
        3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 0.1], figure=fig
    )

    # create axes objects for plotting
    # Create first subplot (for image) without sharing axes
    axes = []
    ax00 = plt.subplot(gs[0, 0])
    axes.append([ax00])

    # Create subplots with shared axes for the other two in the top row
    ax01 = plt.subplot(gs[0, 1])
    ax02 = plt.subplot(gs[0, 2], sharex=ax01, sharey=ax01)
    axes[0].extend([ax01, ax02])

    # Create bottom row subplots
    bottom_row = []
    for j in range(3):
        ax = plt.subplot(gs[1, j])
        bottom_row.append(ax)
    axes.append(bottom_row)

    axes = np.array(axes)

    cax = fig.add_subplot(gs[2, 1])

    for ax in axes[0, :]:
        ax.set_prop_cycle(color=HUSL_PALETTE)

    img = mpimg.imread(TRAIN_CURVE_ORIG_PNG)
    axes[0, 0].imshow(img)
    img2 = mpimg.imread(BRAIN_PLOT_ORIG_PNG)
    axes[1, 0].imshow(img2)

    height, width = img.shape[:2]
    aspect_ratio = height / width

    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # ["1", "3", "5", ...]
    x = N_STORIES

    axes[0, 1].plot(
        x, y_repro.T, "-o", label=[s.replace("UT", "") for s in SUBJECT_IDS]
    )
    axes[0, 2].plot(x, y_repli.T, "-o")

    for i in range(1, 3):
        axes[0, i].set_box_aspect(aspect_ratio)

    # plot standard erros
    for i in range(y_repli_sem.shape[0]):
        axes[0, 1].fill_between(
            x, y1=y_repro[i] - y_repro_sem[i], y2=y_repro[i] + y_repro_sem[i], alpha=0.3
        )
        axes[0, 2].fill_between(
            x, y1=y_repli[i] - y_repli_sem[i], y2=y_repli[i] + y_repli_sem[i], alpha=0.3
        )

    # load the fMRI data
    rho_voxels_repro, _ = load_data_wrapper(
        data_folder=repro_folder,
        which="eng1000",
        n_stories=[25],
    )

    rho_voxels_repli, _ = load_data_wrapper(
        data_folder=repli_folder,
        which="eng1000",
        n_stories=[25],
    )

    for i, items in enumerate(rho_voxels_repro.items()):
        n, data = items
        plot_voxel_performance(
            scores=data, subject="UTS02", vmin=0, vmax=0.5, ax=axes[1, 1]
        )

    for i, items in enumerate(rho_voxels_repli.items()):
        n, data = items
        plot_voxel_performance(
            scores=data, subject="UTS02", vmin=0, vmax=0.5, ax=axes[1, 2]
        )

    cbar = fig.colorbar(
        mappable=axes[1, 1].images[0],
        cax=cax,
        orientation="horizontal",
        shrink=0.5,
    )
    cbar.ax.set_xlabel("Correlation (r)", fontsize=12)

    minor_ticks = np.arange(x[-1])

    for a in axes[0, 1::]:
        a.set_xticks(minor_ticks, minor=True)
        a.set_xticks([0, 5, 10, 15, 20, 25], labels=[0, 5, 10, 15, 20, 25], fontsize=12)
        a.set_xlabel("Number of Training Stories", fontsize=12)
        a.tick_params(axis="both", which="major", labelsize=12)
        a.grid(which="major", visible=True, lw=0.5, alpha=0.7)
        a.grid(which="minor", visible=True, ls="--", lw=0.5, alpha=0.5)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)

    axes[0, 1].legend(title="Participant")
    # axes[0, 0].set_title("Published results\n(10.1038/s41597-023-02437-z)")
    # axes[0, 1].set_title(
    #    "Reproducibility experiment\n(`different-team-same-artifacts`)"
    # )
    # axes[0, 2].set_title(
    #    "Replication experiment\n(`different-team-different-artifacts`)"
    # )
    axes[0, 1].set_ylabel("Mean Correlation (r)", fontsize=12)

    return fig


def plot_repli_comparison(repro_folder: str, repli_folder1: str, repli_folder2: str):
    y_mean, y_sem = _load_means_sem(repro_folder)
    y_mean1, y_sem1 = _load_means_sem(repli_folder1)
    y_mean2, y_sem2 = _load_means_sem(repli_folder2)

    x = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True, layout="constrained")

    for a in ax:
        a.set_prop_cycle(color=HUSL_PALETTE)

    all_data = [
        (y_mean, y_sem),
        (y_mean1, y_sem1),
        (y_mean2, y_sem2),
    ]

    for k, data in enumerate(all_data):
        y_mean, y_sem = data
        ax[k].plot(x, y_mean.T, "-o", label=[s.replace("UT", "") for s in SUBJECT_IDS])

        for i in range(y_sem1.shape[0]):
            ax[k].fill_between(
                x, y1=y_mean[i] - y_sem[i], y2=y_mean[i] + y_sem[i], alpha=0.3
            )

    minor_ticks = np.arange(x[-1])
    for a in ax:
        a.set_xticks(minor_ticks, minor=True)
        a.set_xticks([0, 5, 10, 15, 20, 25], labels=[0, 5, 10, 15, 20, 25])
        a.grid(which="major", visible=True, lw=0.5, alpha=0.7)
        a.grid(which="minor", visible=True, ls="--", lw=0.5, alpha=0.5)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.spines["left"].set_visible(False)
        a.spines["bottom"].set_visible(False)

    # ax[0].set_title("Replication")
    # ax[1].set_title("Reproduction")
    # ax[2].set_title("Reproduction 2")
    ax[0].set_ylabel("Mean correlation (r)")
    fig.supxlabel("Number of training stories")

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "plot.py",
        description="Plot the replication figures and save as .pdf and .png",
    )
    parser.add_argument(
        "repro_folder",
        help="folder with results for the replication experiment to be ploted",
    )
    parser.add_argument(
        "repli_folder",
        help="folder with result for the reproducibility experiment to be ploted",
    )
    parser.add_argument(
        "repli_folder2",
        help="folder with result for the reproducibility experiment to be ploted",
    )
    parser.add_argument("save_path", help="path to where the figures are saved")

    args = parser.parse_args()

    cfg = load_config()

    savepath = Path(args.save_path)

    # TRAINING CURVE FIGURE
    fig1 = make_training_curve_fig(
        repro_folder=args.repro_folder,
        repli_folder=args.repli_folder,
    )

    fn = str(savepath / "training_curve.pdf")
    log.info(f"Saving {fn}")
    fig1.savefig(fn, bbox_inches="tight", transparent=True)

    fn3_png = fn.replace(".pdf", ".png")
    log.info(f"Saving {fn3_png}")
    fig1.savefig(fn3_png, bbox_inches="tight", dpi=300)

    fn_svg = fn.replace(".pdf", ".svg")
    log.info(f"Saving {fn_svg}")
    fig1.savefig(fn_svg, bbox_inches="tight")

    fig2 = plot_repli_comparison(
        repro_folder=args.repro_folder,
        repli_folder1=args.repli_folder,
        repli_folder2=args.repli_folder2,
    )

    fn2 = str(savepath / "repli_vs_repli2.pdf")
    log.info(f"Saving {fn2}")
    fig2.savefig(fn2, bbox_inches="tight", transparent=True)

    fn2_png = fn2.replace(".pdf", ".png")
    log.info(f"Saving {fn2_png}")
    fig2.savefig(fn2_png, bbox_inches="tight", dpi=300)

    fn3_svg = fn2.replace(".pdf", ".svg")
    log.info(f"Saving {fn3_svg}")
    fig2.savefig(fn3_svg, bbox_inches="tight")
