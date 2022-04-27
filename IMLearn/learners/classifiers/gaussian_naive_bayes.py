from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import misclassification_error


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.zeros(self.classes_.shape[0])
        self.mu_ = np.zeros((self.classes_.shape[0], X.shape[1]))
        self.vars_ = np.zeros((self.classes_.shape[0], X.shape[1]))

        for i in range(self.classes_.shape[0]):
            class_name = self.classes_[i]
            class_count = np.count_nonzero(y == class_name)
            class_elements = X[y == class_name]
            self.pi_[i] = class_count / y.shape[0]
            self.mu_[i] = class_elements.mean(axis=0)
            self.vars_[i] = class_elements.var(axis=0, ddof=1)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return np.array([self.classes_[np.argmax(self._calc_sample_likelihood(x))] for x in X])

    def calc_normal_dist(self, sample, class_ind):
        sqrt_factor = 1 / (np.sqrt(2 * np.pi * self.vars_[class_ind]))
        exp_factor = -(((sample - self.mu_[class_ind]) ** 2)) / (2 * self.vars_[class_ind])
        pred_y = sqrt_factor * (np.exp(exp_factor))
        return pred_y

    def _calc_sample_likelihood(self, x):
        all_pred_for_x = []
        for k in range(self.classes_.shape[0]):
            # check which class is the best fitting for the current sample (x)
            pred_y = np.sum(np.log(self.calc_normal_dist(x, k))) + np.log(self.pi_[k])
            all_pred_for_x.append(pred_y)
        return all_pred_for_x

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        return np.array([self._calc_sample_likelihood(x) for x in X])

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
        return misclassification_error(y, self._predict(X))
