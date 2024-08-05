import argparse
import json
import os

import numpy as np

from main import do_regression
from plot import plot_aggregate_results
from utils import check_make_dirs

BASE_DIR = "data/mean_scores"
BASE_DIR_RESULTS = "data/results_max"


def run_all(
    subject: str,
    n_delays: int,
    predictor: str = "all",
    plot_results: bool = True,
):
    """Runs multiple encoding models with increasing amounts of training data
    and plots the outputs.

    Parameters
    ----------
    subject : {"UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"}
        Subject identifier
    n_delays : int
        How many delays are used to model the HRF. The HRF is modeled by adding
        a shifted set of duplicated features for each delay. `n_delays=5` implies
        that the the features of the stimulus are shifted concatinated 5 times
        to training/testing data.
    predictor : {"all", "envelope", "embeddings"}
        Run both predictors (default), or only encoding model with the envelope or
        embeddings predictor.
    plot_results : bool, optional
        Whether to plot the results or not.
    """

    if predictor == "all":
        predictors = ["envelope", "embeddings"]
    else:
        predictors = [predictor]

    # envelope
    results_agg = {}
    for current_predictor in predictors:
        for shuffle in [False, True]:
            # Handle output paths
            base_path = f"{BASE_DIR}/{subject}_{current_predictor}_{n_delays}"
            check_make_dirs(BASE_DIR, isdir=True)  # make sure dir exists
            base_path_results = (
                f"{BASE_DIR_RESULTS}/{subject}_{current_predictor}_{n_delays}"
            )
            check_make_dirs(BASE_DIR_RESULTS, isdir=True)

            if shuffle:
                base_path += "_shuffled"
                base_path_results += "_shuffled"

            mean_scores_train1 = do_regression(
                current_predictor,
                n_stories=2,
                subject=subject,
                n_delays=n_delays,
                show_results=False,
                shuffle=shuffle,
            )
            np.save(f"{base_path}_train1.npy", mean_scores_train1)
            mean_scores_train3 = do_regression(
                current_predictor,
                n_stories=4,
                subject=subject,
                n_delays=n_delays,
                show_results=False,
                shuffle=shuffle,
            )
            np.save(f"{base_path}_train3.npy", mean_scores_train3)
            mean_scores_train5 = do_regression(
                current_predictor,
                n_stories=6,
                subject=subject,
                n_delays=n_delays,
                show_results=False,
                shuffle=shuffle,
            )
            np.save(f"{base_path}_train5", mean_scores_train5)
            mean_scores_train10 = do_regression(
                current_predictor,
                n_stories=11,
                subject=subject,
                n_delays=n_delays,
                show_results=False,
                shuffle=shuffle,
            )
            np.save(f"{base_path}_train10.npy", mean_scores_train10)

            results_key = current_predictor
            if shuffle:
                results_key += "_shuffled"
            results_max = {
                results_key: {
                    "1": mean_scores_train1.max(),
                    "3": mean_scores_train3.max(),
                    "5": mean_scores_train5.max(),
                    "10": mean_scores_train10.max(),
                }
            }
            with open(f"{base_path_results}.json", "w") as f_out:
                json.dump(results_max, f_out, indent=4)

            results_agg.update(results_max)

    if plot_results:
        output_path = os.path.join(
            "data",
            "plots",
            f"{subject}_run_{predictor}_delays_{n_delays}.png",
        )
        plot_aggregate_results(results_agg, output_path, show_plot=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "run_all.py",
        description="",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="UTS02",
        help="Subject identifier",
        choices=[
            "UTS01",
            "UTS02",
            "UTS03",
            "UTS04",
            "UTS05",
            "UTS06",
            "UTS07",
            "UTS08",
        ],
    )
    parser.add_argument(
        "--n_delays",
        type=int,
        default=5,
        help="How many delays are used to model the HRF.",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="all",
        choices=["all", "embeddings", "envelope"],
        help="Which predictors to run",
    )
    args = parser.parse_args()
    run_all(args.subject, args.n_delays)
