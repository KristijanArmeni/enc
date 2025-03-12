import argparse
import json
import logging
import os
import pathlib
import sys
from os.path import dirname, join
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import sem

from encoders.utils import check_make_dirs, create_run_folder_name, load_config
from lebel_encoding.config import EM_DATA_DIR, REPO_DIR
from lebel_encoding.encoding_utils import apply_zscore_and_hrf, get_response
from lebel_encoding.feature_spaces import _FEATURE_CONFIG, get_feature_space
from lebel_encoding.ridge_utils.ridge import bootstrap_ridge

## mofidied from encoding.py; runs replication experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--feature", type=str, default="eng1000")
    parser.add_argument("--n_train_stories", nargs="+", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--trim", type=int, default=5)
    parser.add_argument("--ndelays", type=int, default=4)
    parser.add_argument("--nboots", type=int, default=20)
    parser.add_argument("--chunklen", type=int, default=10)
    parser.add_argument("--nchunks", type=int, default=10)
    parser.add_argument("--singcutoff", type=float, default=1e-10)
    parser.add_argument("--use_corr", action="store_true")
    parser.add_argument("--single_alpha", action="store_true")
    parser.add_argument("--run_folder_name", type=str)
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    RUNS_DIR = load_config()["RUNS_DIR"]

    fs = " ".join(_FEATURE_CONFIG.keys())
    assert args.feature in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs

    train_stories, test_stories = [], []
    with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
        sess_to_story = json.load(f)

    for sess in sess_to_story.keys():
        stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
        train_stories.extend(stories)
        if tstory not in test_stories:
            test_stories.append(tstory)
    assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"
    allstories = list(set(train_stories) | set(test_stories))

    if isinstance(args.n_train_stories, int):
        n_train_stories_list = [args.n_train_stories]
    else:
        n_train_stories_list = args.n_train_stories

    # if run folder name not given, create one
    if not args.run_folder_name:
        run_folder_name = create_run_folder_name()
    else:
        run_folder_name = args.run_folder_name

    run_folder = os.path.join(RUNS_DIR, run_folder_name)
    Path(run_folder).mkdir(parents=True, exist_ok=True)

    for current_n_train_stories in n_train_stories_list:
        # e.g. /UTS01/embeddings/5/not_shuffled/scores_mean.py
        output_dir = os.path.join(
            run_folder,
            args.subject,
            "embeddings",  # only option here
            str(current_n_train_stories),
            "not_shuffled",
        )
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print("Saving encoding model & results to:", output_dir)

        downsampled_feat = get_feature_space(args.feature, allstories)
        print("Stimulus & Response parameters:")
        print("trim: %d, ndelays: %d" % (args.trim, args.ndelays))

        # result arrays
        scores_list = list()
        weights_list = list()
        best_alphas_list = list()

        rng = np.random.default_rng()
        for repeat in range(args.n_repeats):
            print("Repeat %d" % repeat)
            print(train_stories)
            # choose stories to sample for this repeat
            curr_train_stories: list[str] = rng.choice(
                train_stories, size=current_n_train_stories, replace=False
            ).tolist()
            curr_test_stories = test_stories

            # Delayed stimulus
            delRstim = apply_zscore_and_hrf(
                curr_train_stories, downsampled_feat, args.trim, args.ndelays
            )
            print("delRstim: ", delRstim.shape)
            delPstim = apply_zscore_and_hrf(
                curr_test_stories, downsampled_feat, args.trim, args.ndelays
            )
            print("delPstim: ", delPstim.shape)

            # Response
            zRresp = get_response(curr_train_stories, args.subject)
            print("zRresp: ", zRresp.shape)
            zPresp = get_response(curr_test_stories, args.subject)
            print("zPresp: ", zPresp.shape)

            # Ridge
            alphas = np.logspace(1, 3, 10)

            print("Ridge parameters:")
            print(
                "nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s"
                % (
                    args.nboots,
                    args.chunklen,
                    args.nchunks,
                    args.single_alpha,
                    args.use_corr,
                )
            )

            wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
                delRstim,
                zRresp,
                delPstim,
                zPresp,
                alphas,
                args.nboots,
                args.chunklen,
                args.nchunks,
                singcutoff=args.singcutoff,
                single_alpha=args.single_alpha,
                use_corr=args.use_corr,
            )

            scores_list.append(corrs)
            weights_list.append(wt)
            best_alphas_list.append(valphas)

        mean_scores = np.mean(scores_list, axis=0)
        sem_scores = sem(scores_list, axis=0)
        np.save(os.path.join(output_dir, "scores_mean.npy"), mean_scores)
        np.save(os.path.join(output_dir, "scores_sem.npy"), sem_scores)
