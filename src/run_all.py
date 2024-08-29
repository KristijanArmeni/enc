import argparse
import json
import os
from collections import defaultdict
from functools import partial

import numpy as np

from main import do_regression
from utils import ROOT, check_make_dirs, create_run_folder_name

RUNS_DIR = ROOT / "data" / "runs"


def run_all(
    subject: str,
    n_delays: int,
    predictor: str = "all",
    n_train_stories_list: list[int] = [1, 4],
):
    """Runs multiple encoding models with increasing amounts of training data
    and plots the outputs.

    The outputs are saved in the `data` dir in following structure
    data/runs/<date>-<id>/<predictor>/<subject>/<n_training_stories>/<shuffle>/<results>
    whereas results are:
        - mean_scores.npy : the mean scores across folds

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
    """

    if predictor == "all":
        predictors = ["envelope", "embeddings"]
    else:
        predictors = [predictor]

    # handle data folder
    folder_name = create_run_folder_name()
    base_dir = os.path.join(RUNS_DIR, folder_name)
    check_make_dirs(base_dir, isdir=True)

    # log all parameters
    # config = {
    #     "subject": subject,
    #     "n_delays": n_delays,
    #     "predictor": predictor,  # command line argument
    #     "predictors": predictors,  # actual predictors ran
    # }

    # aggregate overall max correlations and continously update in json
    results_max_agg = defaultdict(
        partial(defaultdict, partial(defaultdict, dict))
    )  # enables instantiating hierarchy of dicts without manually creating them at each level.
    results_max_path = os.path.join(base_dir, "results_max.json")
    for current_predictor in predictors:
        for shuffle in [False, True]:
            shuffle_str = "shuffled" if shuffle else "not_shuffled"
            for n_train_stories in n_train_stories_list:
                output_dir = os.path.join(
                    base_dir,
                    current_predictor,
                    subject,
                    str(n_train_stories),
                    shuffle_str,
                )
                check_make_dirs(output_dir, verbose=False, isdir=True)
                mean_scores, all_scores, all_weights, best_alphas = do_regression(
                    current_predictor,
                    n_stories=n_train_stories + 1,
                    subject=subject,
                    n_delays=n_delays,
                    show_results=False,
                    shuffle=shuffle,
                )
                np.save(os.path.join(output_dir, "scores_mean.npy"), mean_scores)
                for idx_fold, (scores_fold, weights_fold, best_alpha) in enumerate(
                    zip(all_scores, all_weights, best_alphas)
                ):
                    output_dir_fold = os.path.join(output_dir, f"fold_{idx_fold}")
                    check_make_dirs(output_dir_fold, verbose=False, isdir=True)
                    np.save(
                        os.path.join(output_dir_fold, "scores.npy"),
                        scores_fold,
                    )
                    np.save(
                        os.path.join(output_dir_fold, "weights.npy"),
                        weights_fold,
                    )
                    np.save(
                        os.path.join(output_dir_fold, "best_alphas.npy"),
                        np.array(best_alpha),
                    )

                results_max_agg[current_predictor][subject][n_train_stories][
                    shuffle_str
                ] = mean_scores.max()

                # update results file
                with open(results_max_path, "w") as f_out:
                    json.dump(results_max_agg, f_out, indent=4)


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
    run_all(
        subject=args.subject,
        n_delays=args.n_delays,
        predictor=args.predictor,
    )
