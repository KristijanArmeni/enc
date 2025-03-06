import argparse
import json
import logging
import os
import pathlib
import sys
from os.path import dirname, join

import h5py
import numpy as np

from encoders.utils import check_make_dirs, create_run_folder_name, load_config
from lebel_encoding.config import EM_DATA_DIR, REPO_DIR
from lebel_encoding.encoding_utils import *
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
    parser.add_argument("-use_corr", action="store_true")
    parser.add_argument("-single_alpha", action="store_true")
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    globals().update(args.__dict__)

    RUNS_DIR = load_config()["RUNS_DIR"]

    fs = " ".join(_FEATURE_CONFIG.keys())
    assert feature in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs

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

    if isinstance(n_train_stories, int):
        n_train_stories_list = [n_train_stories]
    else:
        n_train_stories_list = n_train_stories

    folder_name = create_run_folder_name()
    base_dir = os.path.join(RUNS_DIR, folder_name)
    for current_n_train_stories in n_train_stories_list:
        output_dir = os.path.join(
            base_dir,
            "replication",
            subject,
            str(current_n_train_stories),
            "not_shuffled",
        )
        check_make_dirs(output_dir, verbose=False, isdir=True)
        print("Saving encoding model & results to:", output_dir)

        downsampled_feat = get_feature_space(feature, allstories)
        print("Stimulus & Response parameters:")
        print("trim: %d, ndelays: %d" % (trim, ndelays))

        # result arrays
        scores_list = list()
        weights_list = list()
        best_alphas_list = list()

        rng = np.random.default_rng()
        for repeat in range(n_repeats):
            print("Repeat %d" % repeat)
            print(train_stories)
            # choose stories to sample for this repeat
            curr_train_stories: list[str] = rng.choice(
                train_stories, size=current_n_train_stories, replace=False
            ).tolist()
            curr_test_stories = test_stories

            # Delayed stimulus
            delRstim = apply_zscore_and_hrf(
                curr_train_stories, downsampled_feat, trim, ndelays
            )
            print("delRstim: ", delRstim.shape)
            delPstim = apply_zscore_and_hrf(
                curr_test_stories, downsampled_feat, trim, ndelays
            )
            print("delPstim: ", delPstim.shape)

            # Response
            zRresp = get_response(curr_train_stories, subject)
            print("zRresp: ", zRresp.shape)
            zPresp = get_response(curr_test_stories, subject)
            print("zPresp: ", zPresp.shape)

            # Ridge
            alphas = np.logspace(1, 3, 10)

            print("Ridge parameters:")
            print(
                "nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s"
                % (nboots, chunklen, nchunks, single_alpha, use_corr)
            )

            wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
                delRstim,
                zRresp,
                delPstim,
                zPresp,
                alphas,
                nboots,
                chunklen,
                nchunks,
                singcutoff=singcutoff,
                single_alpha=single_alpha,
                use_corr=use_corr,
            )

            scores_list.append(corrs)
            weights_list.append(wt)
            best_alphas_list.append(valphas)

        mean_scores = np.mean(scores_list, axis=0)
        np.save(os.path.join(output_dir, "scores_mean.npy"), mean_scores)
