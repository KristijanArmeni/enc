from typing import Callable, Optional, Union

import numpy as np
from sklearn.linear_model import RidgeCV

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


def ridge_regression(
    train_stories: list[str],
    test_stories: list[str],
    X_data_dict: dict[str, np.ndarray],
    y_data_dict: dict[str, np.ndarray],
    score_fct: Callable[[np.ndarray, np.ndarray], np.ndarray],
    alphas: Optional[np.ndarray] = np.logspace(1, 3, 10),
) -> tuple[np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """Runs CVRidgeRegression on given train stories, returns scores for
    test story, regression weights, and the best alphas.

    Parameters
    ----------
    train_stories: list of str
        List of training stories. Stories must be in X_data_dict.
    test_stories: list of str
        List of testing stories. Stories must be in X_data_dict.
    X_data_dict : dict[str, np.ndarray]
        Dict with story to X_data (features) pairs.
    y_data_dict : dict[str, np.ndarray]
        Dict with story to y_data (fMRI data) pairs.
    score_fct : fct(np.ndarray, np.ndarray) -> np.ndarray
        A function taking y_test (shape = (number_trs, n_voxels))
        and y_predict (same shape as y_test) and returning an
        array with an entry for each voxel (shape = (n_voxels)).
    alphas : np.ndarray or `None`, default = `np.logspace(1, 3, 10)`
        Array of alpha values to optimize over. If `None`, will choose
        default value.

    Returns
    -------
    scores : np.ndarray
        The prediction scores for the test stories.
    weights : np.ndarray
        The regression weights.
    best_alphas : float | np.ndarray
        The best alphas for each voxel.
    """
    if alphas is None:
        alphas = np.logspace(1, 3, 10)

    X_train_list = [X_data_dict[story] for story in train_stories]
    y_train_list = [y_data_dict[story] for story in train_stories]
    X_test_list = [X_data_dict[story] for story in test_stories]
    y_test_list = [y_data_dict[story] for story in test_stories]

    X_train_unnormalized = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test_unnormalized = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    X_means = X_train_unnormalized.mean(axis=0)
    X_stds = X_train_unnormalized.std(axis=0)

    X_train = z_score(X_train_unnormalized, X_means, X_stds)
    X_test = z_score(X_test_unnormalized, X_means, X_stds)

    clf = RidgeCV(alphas=alphas, alpha_per_target=True)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)
    scores = score_fct(y_test, y_predict)

    return scores, clf.coef_, clf.alpha_
