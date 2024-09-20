import argparse
import json
import os
from collections import defaultdict
from functools import partial
from typing import Union

import numpy as np

from main import do_regression
from utils import check_make_dirs, create_run_folder_name, get_logger, load_config

log = get_logger(__name__)

RUNS_DIR = load_config()["RUNS_DIR"]


def run_all(
    subject: Union[str, list[str]] = "UTS02",
    n_train_stories: Union[int, list[int]] = [1, 4],
    predictor: Union[str, list[str]] = "all",
    n_delays: int = 5,
    interpolation: str = "lanczos",
):
    """Runs multiple encoding models with increasing amounts of training data
    and plots the outputs.

    The outputs are saved in the `data` dir in following structure
    data/runs/<date>-<id>/<predictor>/<subject>/<n_training_stories>/<shuffle>/<results>
    whereas results are:
        - mean_scores.npy : the mean scores across folds

    Parameters
    ----------
    subject : str or list of str
        Subject identifier
        Can be one or a list of : {"all", "UTS01", "UTS02", "UTS03", "UTS04", "UTS05", "UTS06", "UTS07", "UTS08"}
    n_train_stories : int or list of int
        Amount of training stories, can be one or multiple amounts.
    predictor : {"all", "envelope", "embeddings"}
        Run both predictors (default), or only encoding model with the envelope or
        embeddings predictor.
    n_delays : int
        How many delays are used to model the HRF. The HRF is modeled by adding
        a shifted set of duplicated features for each delay. `n_delays=5` implies
        that the the features of the stimulus are shifted concatinated 5 times
        to training/testing data.
    interpolation : {"lanczos", "average"}
        Whether to use lanczos interpolation or just average the words within a TR.
        Only applies to the 'embeddings' predictor.
    """

    # put arguments in right format
    if isinstance(subject, str):
        subjects = [subject]
    else:
        subjects = subject
    if "all" in subjects:
        subjects = [
            "UTS01",
            "UTS02",
            "UTS03",
            "UTS04",
            "UTS05",
            "UTS06",
            "UTS07",
            "UTS08",
        ]

    if isinstance(n_train_stories, int):
        n_train_stories_list = [n_train_stories]
    else:
        n_train_stories_list = n_train_stories

    if isinstance(predictor, str):
        predictors = [predictor]
    else:
        predictors = predictor
    if "all" in predictors:
        predictors = ["envelope", "embeddings"]

    # handle data folder
    folder_name = create_run_folder_name()
    base_dir = os.path.join(RUNS_DIR, folder_name)
    check_make_dirs(base_dir, isdir=True)

    # log all parameters
    config = {
        "subject_arg": subject,  # command line arguments
        "subjects": subjects,  # resolved predictors
        "predictor_arg": predictor,
        "predictors": predictors,
        "n_train_stories": n_train_stories,
        "interpolation": interpolation,
        "n_delays": n_delays,
    }
    # update results file
    params_path = os.path.join(base_dir, "params.json")
    with open(params_path, "w") as f_out:
        json.dump(config, f_out, indent=4)
    log.info(f"Written parameters to {params_path}")

    # aggregate overall max correlations and continously update in json
    results_max_agg = defaultdict(
        partial(defaultdict, partial(defaultdict, dict))
    )  # enables instantiating hierarchy of dicts without manually creating them at each level.
    results_max_path = os.path.join(base_dir, "results_max.json")
    for current_predictor in predictors:
        for current_subject in subjects:
            for shuffle in [False, True]:
                shuffle_str = "shuffled" if shuffle else "not_shuffled"
                for current_n_train_stories in n_train_stories_list:
                    output_dir = os.path.join(
                        base_dir,
                        current_predictor,
                        current_subject,
                        str(current_n_train_stories),
                        shuffle_str,
                    )
                    check_make_dirs(output_dir, verbose=False, isdir=True)
                    mean_scores, all_scores, all_weights, best_alphas = do_regression(
                        current_predictor,
                        n_stories=current_n_train_stories + 1,
                        subject=current_subject,
                        n_delays=n_delays,
                        show_results=False,
                        shuffle=shuffle,
                        interpolation=interpolation,
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

                    results_max_agg[current_predictor][current_subject][
                        current_n_train_stories
                    ][shuffle_str] = mean_scores.max()

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
        nargs="+",
        type=str,
        default=["UTS02"],
        help="List of subject identifier. Can be 'all' for all subjects.",
        choices=[
            "all",
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
        "--n_train_stories",
        nargs="+",
        type=int,
        default=[1, 3],
        help="Amount of training stories, can be one or multiple amounts.",
    )
    parser.add_argument(
        "--predictor",
        nargs="+",
        type=str,
        default=["all"],
        choices=["all", "embeddings", "envelope"],
        help="Predictor for the encoding model. Can be 'all' or a combination of predictors.",
    )
    parser.add_argument(
        "--n_delays",
        type=int,
        default=5,
        help="How many delays are used to model the HRF.",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["lanczos", "average"],
        default="lanczos",
        help="Interpolation method used for embeddings predictor.",
    )
    args = parser.parse_args()
    run_all(
        subject=args.subject,
        n_train_stories=args.n_train_stories,
        predictor=args.predictor,
        n_delays=args.n_delays,
        interpolation=args.interpolation,
    )
