import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import cortex
import matplotlib as mpl
import matplotlib.figure
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from cortex import svgoverlay

from encoders.utils import ROOT, get_logger, load_config

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
    # cortex.quickshow(vol_data)

    _ = cortex.quickflat.make_figure(
        braindata=vol_data,
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


runs_dir = load_config()["RUNS_DIR"]
repro_folder = Path(runs_dir, "2025-03-12_15-25_487099")
repli_folder = Path(runs_dir, "2025-03-06_17-05_265539")


def make_training_curve_fig(
    repli_folder: str, repro_folder: str
) -> matplotlib.figure.Figure:
    # load embedding model performance
    data_repli = {
        sub: load_data_wrapper(
            data_folder=str(repro_folder),
            subject=sub,
            which="embeddings",
            n_stories=[1, 3, 5, 7, 9, 11, 13, 15, 20],
        )[0]
        for sub in SUBJECT_IDS
    }

    data_repro = {
        sub: load_data_wrapper(
            data_folder=str(repli_folder),
            subject=sub,
            which="embeddings",
            n_stories=[1, 3, 5, 7, 9, 11, 13, 15, 20],
        )[0]
        for sub in SUBJECT_IDS
    }

    # data_repli_sem = {sub:
    #    load_data_wrapper(
    #        ds=str(REPLI_FOLDER),
    #        subject=sub,
    #        which="embeddings",
    #        n_stories=[1, 3, 5, 7, 9, 11, 13, 15, 20],
    #    )[1]
    #    for sub in SUBJECT_IDS
    # }

    # data_repro_sem = {sub:
    #    load_data_wrapper(
    #        ds=str(REPRO_FOLDER),
    #        subject=sub,
    #        which="embeddings",
    #        n_stories=[1, 3, 5, 7, 9, 11, 13, 15, 20],
    #    )[1]
    #    for sub in SUBJECT_IDS
    # }

    def _data2array_agg(data_dict: Dict, aggfunc) -> np.ndarray:
        out = np.array(
            [
                [aggfunc(data) for data in subject_data.values()]
                for subject_data in data_dict.values()
            ]
        )

        return out

    # load the performance scores and average across cortex shape = (n_stories,)
    y_repli = _data2array_agg(data_dict=data_repli, aggfunc=np.mean)
    y_repro = _data2array_agg(data_dict=data_repro, aggfunc=np.mean)

    extras = np.array([[np.nan, np.nan, np.nan]]).T
    y_repli = np.hstack([y_repli, extras])
    y_repro = np.hstack([y_repro, extras])

    # y_repli_sem = _data2array_agg(data_dict=data_repli_sem, aggfunc=sem)
    # y_repro_sem = _data2array_agg(data_dict=data_repro_sem, aggfunc=sem)

    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1.2])

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

    for ax in axes[0, :]:
        ax.set_prop_cycle(color=HUSL_PALETTE)

    ORIG_RESULT = Path(ROOT, "src", "encoders", "orig.png")
    img = mpimg.imread(ORIG_RESULT)
    axes[0, 0].imshow(img)
    ORIG_BRAIN = Path(ROOT, "src", "encoders", "brain_orig.png")
    img2 = mpimg.imread(ORIG_BRAIN)
    axes[1, 0].imshow(img2)

    height, width = img.shape[:2]
    aspect_ratio = height / width

    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("")
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    # Force the figure to draw so we can get correct positions
    fig.canvas.draw()

    # Get the positions of both subplots we want to group
    pos_00 = axes[0, 0].get_position()
    pos_10 = axes[1, 0].get_position()

    # Create a rectangle patch that encompasses both subplots
    padding = 0.03
    rect = patches.Rectangle(
        (
            pos_00.x0 - padding,
            pos_10.y0 - padding,
        ),  # Lower left corner (from bottom subplot)
        pos_00.width + 2 * padding,  # Width (use first subplot's width)
        (pos_00.y0 + pos_00.height) - pos_10.y0 + 0.04,  # Total height of both subplots
        linewidth=2,
        edgecolor="#4682B4",  # Steel blue
        facecolor="lightblue",
        alpha=0.2,
        linestyle="--",
        zorder=0,
        transform=fig.transFigure,
    )

    # Add the rectangle to the figure
    fig.add_artist(rect)

    # ["1", "3", "5", ...]
    x = [int(e) for e in list(data_repli["UTS01"].keys())] + [25]

    axes[0, 1].plot(
        x, y_repli.T, "-o", label=[s.replace("UT", "") for s in SUBJECT_IDS]
    )
    axes[0, 2].plot(x, y_repro.T, "-o")
    axes[0, 2].label_outer()

    for i in range(3):
        axes[0, i].set_box_aspect(aspect_ratio)

    # plot standard erros
    # for i in range(y_repli_sem.shape[0]):
    #    ax[0].fill_between(
    #        x,
    #        y1=y_repli[i] - y_repli_sem[i],
    #        y2=y_repli[i] + y_repli_sem[i],
    #        alpha=0.3
    #    )
    #    ax[1].fill_between(
    #        x,
    #        y1=y_repro[i] - y_repro_sem[i],
    #        y2=y_repro[i] + y_repro_sem[i],
    #        alpha=0.3
    #    )

    # load the fMRI data
    rho_voxels_repro, _ = load_data_wrapper(
        data_folder=repro_folder, which="embeddings", n_stories=[20]
    )

    rho_voxels_repli, _ = load_data_wrapper(
        data_folder=repli_folder,
        which="embeddings",
        n_stories=[20],
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

    cbar = axes[1, 1].images[0].colorbar
    cbar.ax.set_xlabel("Test-set correlation (r)", fontsize=12)
    # cbar.ax.set_xticks([0, 0.5], labels=[0, 0.5])

    minor_ticks = np.arange(x[-1])

    for a in axes[0, 1::]:
        a.set_xticks(minor_ticks, minor=True)
        a.set_xticks([1, 5, 10, 15, 20, 25], labels=[1, 5, 10, 15, 20, 25])
        a.grid(which="major", visible=True, lw=0.5, alpha=0.7)
        a.grid(which="minor", visible=True, ls="--", lw=0.5, alpha=0.5)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    axes[0, 1].legend(title="Participant")
    axes[0, 0].set_title("Published results\n(10.1038/s41597-023-02437-z)")
    axes[0, 1].set_title(
        "Reproducibility experiment\n(`different-team-same-artifacts`)"
    )
    axes[0, 2].set_title(
        "Replication experiment\n(`different-team-different-artifacts`)"
    )
    axes[0, 1].set_ylabel("Mean Correlation (r)")

    # fig.supxlabel("Number of Training Stories")

    plt.tight_layout()

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "plot.py",
        description="Plot the replication figures and save as .pdf and .png",
    )

    parser.add_argument(
        "exp1_folder",
        help="folder with results for the replication experiment to be ploted",
    )
    parser.add_argument(
        "exp2_folder",
        help="folder with result for the reproducibility experiment to be ploted",
    )
    parser.add_argument("save_path", help="path to where the figures are saved")

    args = parser.parse_args()

    cfg = load_config()

    savepath = Path(args.save_path)

    # replication figure
    fig1 = make_brain_fig(data_folder=args.exp1_folder, which="embeddings")
    fig1.suptitle(
        "Replication: "
        + "Semantic encoding model performance with increasing training data",
        fontsize=14,
    )

    fn1 = str(savepath / "repli_semantic_performance.pdf")
    log.info(f"Saving {fn1}")
    fig1.savefig(fn1, bbox_inches="tight", transparent=True)

    fn1_png = fn1.replace(".pdf", ".png")
    log.info(f"Saving {fn1_png}")
    fig1.savefig(fn1_png, bbox_inches="tight", dpi=300)

    # REPRODUCIBILITY EXPERIMENT FIGURE
    fig2 = make_brain_fig(data_folder=args.exp2_folder, which="embeddings")
    fig2.suptitle(
        "Reproducibility: "
        + "Semantic encoding model performance with increasing training data",
        fontsize=14,
    )

    fn2 = str(savepath / "repro_semantic_performance.pdf")
    log.info(f"Saving {fn2}")
    fig2.savefig(fn2, bbox_inches="tight", transparent=True)

    fn2_png = fn2.replace(".pdf", ".png")
    log.info(f"Saving {fn2_png}")
    fig2.savefig(fn2_png, bbox_inches="tight", dpi=300)

    # REPRODUCIBILITY EXTENSION FIGURE
    fig2 = make_brain_fig(data_folder=args.exp2_folder, which="envelope")
    fig2.suptitle(
        "Extension: "
        + "Sensory encoding model performance with increasing training data",
        fontsize=14,
    )

    fn2 = str(savepath / "extension_sensory_performance.pdf")
    log.info(f"Saving {fn2}")
    fig2.savefig(fn2, bbox_inches="tight", transparent=True)

    fn2_png = fn2.replace(".pdf", ".png")
    log.info(f"Saving {fn2_png}")
    fig2.savefig(fn2_png, bbox_inches="tight", dpi=300)

    # TRAINING CURVE FIGURE
    fig3 = make_training_curve_fig(
        repli_folder=args.exp1_folder,
        repro_folder=args.exp2_folder,
    )

    fn3 = str(savepath / "training_curve.pdf")
    log.info(f"Saving {fn3}")
    fig3.savefig(fn3, bbox_inches="tight", transparent=True)

    fn3_png = fn3.replace(".pdf", ".png")
    log.info(f"Saving {fn3_png}")
    fig3.savefig(fn3_png, bbox_inches="tight", dpi=300)
