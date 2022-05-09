from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        minimal_error_threshold = 1
        for i in range(X.shape[1]):
            feature_i = X[:, i]
            sign_threshold, sign_error = self._find_threshold(feature_i, y, 1)
            neg_sign_threshold, neg_sign_error = self._find_threshold(feature_i,y, -1)
            if sign_error < neg_sign_error:
                if sign_error < minimal_error_threshold:
                    self.threshold_ = sign_threshold
                    self.j_ = i
                    minimal_error_threshold = sign_error
                    self.sign_ = 1
            else:
                if neg_sign_error < minimal_error_threshold:
                    self.threshold_ = neg_sign_threshold
                    self.j_ = i
                    minimal_error_threshold = neg_sign_error
                    self.sign_ = -1


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.array([-self.sign_ if x[self.j_] < self.threshold_ else self.sign_ for x in X])

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sotred_index = np.argsort(values)
        sorted_vals = np.sort(values)
        sorted_labels = np.take(labels, sotred_index)
        sign_labels = np.sign(sorted_labels)
        min_thr = np.inf
        min_thr_err = 1
        temp_labels_by_thr = np.ones(values.shape[0]) * sign

        for i in range(sorted_vals.shape[0]):
            temp_err_by_thr = np.sum(np.where(temp_labels_by_thr != sign_labels,
                                              np.abs(sorted_labels), 0)) / len(sorted_vals)
            if temp_err_by_thr < min_thr_err:
                min_thr_err = temp_err_by_thr
                min_thr = sorted_vals[i]
            temp_labels_by_thr[i] = -sign
        return min_thr, min_thr_err


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        predict_y = self._predict(X)
        return np.sum(np.where(predict_y != y, np.abs(y), 0))
