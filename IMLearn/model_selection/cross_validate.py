from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def getFoldedSet(folded_X, folded_y, ind, cv):
    updated_folded_X = np.array([])
    updated_folded_y = np.array([])
    for i in range(cv):
        if i == ind:
            continue
        if updated_folded_X.size == 0 and updated_folded_y.size == 0:
            updated_folded_X = folded_X[i]
            updated_folded_y = folded_y[i]
        else:
            updated_folded_X = np.concatenate((updated_folded_X, folded_X[i]))
            updated_folded_y = np.concatenate((updated_folded_y, folded_y[i]))
    return updated_folded_X, updated_folded_y


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
    folded_X = np.array_split(X, cv)
    folded_y = np.array_split(y, cv)
    train_score = []
    validation_score = []
    for i in range(cv):
        test_set_X = folded_X[i]
        test_set_y = folded_y[i]
        train_set_X, train_set_y = getFoldedSet(folded_X, folded_y, i, cv)
        estimator.fit(train_set_X, train_set_y)
        prediction = estimator.predict(X)
        test_prediction = estimator.predict(test_set_X)
        train_score.append(scoring(y, prediction))
        validation_score.append(scoring(test_set_y, test_prediction))
    validation_score = np.array(validation_score)
    train_score = np.array(train_score)
    return np.sum(train_score) / cv, np.sum(validation_score) / cv
