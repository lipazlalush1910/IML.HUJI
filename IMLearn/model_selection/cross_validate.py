from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def getExcludedSet(X: np.ndarray, y: np.ndarray, exclude_index: int, cv: int):
    excluded_X = np.array([])
    excluded_y = np.array([])
    for i in range(cv):
        if i == exclude_index:
            continue
        if excluded_X.size == 0 and excluded_y.size == 0:
            excluded_X = X[i]
            excluded_y = y[i]
        else:
            excluded_X = np.concatenate((excluded_X, X[i]))
            excluded_y = np.concatenate((excluded_y, y[i]))
    return excluded_X, excluded_y



def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds_X = np.array_split(X, cv)
    folds_y = np.array_split(y, cv)  
    train_score = np.zeros(cv)
    validation_score = np.zeros(cv)
    for i in range(cv):
        test_X_i = folds_X[i]
        test_y_i = folds_y[i]

        train_X_i, train_y_i = getExcludedSet(folds_X, folds_y, i, cv)

        estimator.fit(train_X_i, train_y_i)
        train_score[i] = scoring(y, estimator.predict(X))
        validation_score[i] = scoring(test_y_i, estimator.predict(test_X_i))
    return np.mean(train_score), np.mean(validation_score)


