from __future__ import annotations
import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X)

        # mu_sum = 0
        # for x in X:
        #     mu_sum += x
        # self.mu_ = mu_sum / X.size

        var_sum = 0
        for x in X:
            var_sum += np.power( (x-self.mu_), 2)

        if self.biased_ is True:
            self.var_ = var_sum / X.size
        else:
            self.var_ = var_sum / (X.size -1)

        self.fitted_ = True
        return self


    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:  # forum
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        result = []
        scalar_coeff = 1 / (np.sqrt(2 * np.pi * self.var_))
        for x in X:
            exp_power = -(np.power((x - self.mu_), 2)) / (2 * self.var_)
            res = np.exp(exp_power)
            result.append(scalar_coeff * res)
        return result



    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        dist_sum = 0
        dist_var = np.power(sigma,2)
        for x in X:
            dist_sum += (x - mu) ** 2
        return -(X.size /2) * np.log(2 * np.pi * dist_var) - (1/(2*dist_var))*dist_sum


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")

        cov_inv = inv(self.cov_)
        exp_power = np.sum((X - self.mu_) @ cov_inv * (X - self.mu_), axis=1)
        exp_power *= (-0.5)
        cov_det = det(self.cov_)
        sqrt_res = (np.sqrt(np.power(2 * np.pi, len(self.cov_)) * cov_det))

        return (1 / sqrt_res) * np.exp(exp_power)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """

        dist_sum = 0
        cov_inv = inv(cov)
        for x in X:
            dist_sum += np.transpose(x-mu) @ cov_inv @ (x-mu)
        cov_det = det(cov)
        res_log = 1/(np.sqrt((np.power(2*np.pi, cov.shape[0])) * cov_det))
        return X.shape[0]*np.log(res_log)-0.5*dist_sum
