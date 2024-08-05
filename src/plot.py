from typing import Dict, Optional

import cortex
import matplotlib.pyplot as plt
import numpy as np

from utils import check_make_dirs


def plot_voxel_performance(
    subject: str,
    scores: np.ndarray,
    save_path: Optional[str],
    show_plot: bool = False,
    title: Optional[str] = None,
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
    """

    vol_data = cortex.Volume(
        scores, subject, f"{subject}_auto", vmin=vmin, vmax=vmax, cmap=cmap
    )
    cortex.quickshow(vol_data)

    if title is not None:
        plt.title(title)
    if show_plot:
        plt.show()
    if save_path is not None:
        _ = cortex.quickflat.make_png(
            save_path,
            vol_data,
            recache=False,
        )


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


if __name__ == "__main__":
    results_dct = {
        "embeddings": {"1": 0.24799084683845113, "3": 0.19998598781776672},
        "envelope": {"1": 0.328002316334423, "3": 0.41280642600830114},
    }
    plot_aggregate_results(results_dct, "data/plots/example.png", True)
