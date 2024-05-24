from typing import Callable, List, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from utils import get_logger, load_config

log = get_logger(__name__)
cfg = load_config()


def score_correlation(y_test, y_predict) -> np.ndarray:
    """Returns the correlations for each voxel given predicted and true data.

    Parameters
    ----------
    y_test : np.ndarray
        shape = (number_trs, n_voxels)
    y_predict : np.ndarray
        shape = (number_trs, n_voxels)

    Returns
    -------
    np.ndarray
        shape = (n_voxels)
    """
    return np.array(
        [np.corrcoef(y1, y2)[0, 1] for y1, y2 in zip(y_test.T, y_predict.T)]
    )


def z_score(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (data - means) / (stds + 1e-6)


def cross_validation_ridge_regression(
    X_data_list: List[np.ndarray],
    y_data_list: List[np.ndarray],
    n_splits: int,
    alpha: float,
    score_fct: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Cross validate ridge regression

    Parameters
    ----------
    X_data_list : List[np.ndarray]
        List of X data as np array for each story
    y_data_list : List[np.ndarray]
        List of fmri data as np array for each story.
        Must be in same order as X_data_list.
    n_splits : int
        Cross validation splits
    alpha : float
        Yeah...
    score_fct : fct(np.ndarray, np.ndarray) -> np.ndarray
        A function taking y_test (shape = (number_trs, n_voxels))
        and y_predict (same shape as y_test) and returning an
        array with an entry for each voxel (shape = (n_voxels))
    """

    kf = KFold(n_splits=n_splits)

    all_scores = []
    all_weights = []
    for fold, (train_indices, test_indices) in enumerate(kf.split(X_data_list)):  # type: ignore
        log.info(f"Fold {fold}")
        X_train_list = [X_data_list[i] for i in train_indices]
        y_train_list = [y_data_list[i] for i in train_indices]
        X_test_list = [X_data_list[i] for i in test_indices]
        y_test_list = [y_data_list[i] for i in test_indices]

        X_train_unnormalized = np.concatenate(X_train_list, axis=0)
        y_train_unnormalized = np.concatenate(y_train_list, axis=0)
        X_test_unnormalized = np.concatenate(X_test_list, axis=0)
        y_test_unnormalized = np.concatenate(y_test_list, axis=0)

        X_means = X_train_unnormalized.mean(axis=0)
        y_means = y_train_unnormalized.mean(axis=0)
        X_stds = X_train_unnormalized.mean(axis=0)
        y_stds = y_train_unnormalized.mean(axis=0)

        X_train = z_score(X_train_unnormalized, X_means, X_stds)
        y_train = z_score(y_train_unnormalized, y_means, y_stds)
        X_test = z_score(X_test_unnormalized, X_means, X_stds)
        y_test = z_score(y_test_unnormalized, y_means, y_stds)

        clf = Ridge(alpha=alpha)
        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        fold_scores = score_fct(y_test, y_predict)

        all_scores.append(fold_scores)
        all_weights.append(clf.coef_)

    mean_scores = np.mean(all_scores, axis=0)

    return mean_scores, all_scores, all_weights
