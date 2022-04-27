from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import misclassification_error


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

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
        self.cov_ = np.zeros((X.shape[1], X.shape[1]))

        for i, class_name in enumerate(self.classes_):
            class_count = np.count_nonzero(y == class_name)
            class_elements = X[y == class_name]
            self.pi_[i] = class_count / y.shape[0]
            self.mu_[i] = class_elements.mean(axis=0)
            temp = class_elements - self.mu_[i]
            self.cov_ += (temp.transpose() @ temp)

        self.cov_ /= (X.shape[0]-self.classes_[0])
        self._cov_inv = inv(self.cov_)


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
        return np.array([self.classes_[self._best_class_prediction(x)] for x in X])

    def _best_class_prediction(self, x):
        predictions_for_x = []
        for k in range(self.classes_.shape[0]):
            # check which class is the best fitting for the current sample (x)
            ak = self._cov_inv @ self.mu_[k].transpose()
            bk = np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ self._cov_inv @ self.mu_[k].transpose()
            pred_y = (ak.transpose() @ x + bk)
            predictions_for_x.append(pred_y)
        return np.argmax(predictions_for_x)

    def _calc_likelihood_of_sample(self, x):
        x_likelihood = []
        for i in range(self.classes_.shape[0]):
            sqrt_coef = 1 / np.sqrt((np.power(2 * np.pi, len(x))) * det(self.cov_))
            exp_factor = -0.5 * (x - self.mu_[i]) @ self._cov_inv @ np.transpose((x - self.mu_[i]))
            exp_res = np.exp(exp_factor)
            likelihood_x_by_class = self.pi_[i] * sqrt_coef * exp_res
            x_likelihood.append(likelihood_x_by_class)
        return x_likelihood

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

        return np.array([self._calc_likelihood_of_sample(x) for x in X])

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
