from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    f_x = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    epsilon = np.random.normal(0, noise, n_samples)
    # X = np.random.uniform(-1.2, 2, n_samples)
    X = np.linspace(-1.2,2,n_samples)
    y_noiseless = f_x(X)
    y = f_x(X) + epsilon

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.DataFrame(y), 2/3)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=X, y=y_noiseless, mode="markers", name="polynom"))
    fig1.add_trace(go.Scatter(x=train_X[0], y=train_y[0], mode="markers",
                             marker=dict(color="Red",colorscale=[custom[0], custom[-1]]), name="train"))
    fig1.add_trace(go.Scatter(x=test_X[0], y=test_y[0], mode="markers",
                             marker=dict(color="Purple", colorscale=[custom[0], custom[-1]]), name="test"))
    fig1.update_layout(title=f"Model of {n_samples} samples with noise {noise} - train and test sets")
    fig1.show()

    f_train_X = np.array(train_X).flatten()
    f_train_y = np.array(train_y).flatten()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    training_avg, validation_avg = [], []
    for k in range(11):
        pol_estimator = PolynomialFitting(k)
        train_score, validation_score = cross_validate(pol_estimator, f_train_X, f_train_y, mean_square_error)
        training_avg.append(train_score)
        validation_avg.append(validation_score)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=[i for i in range(11)], y=training_avg, name="average training error"))
    fig2.add_trace(go.Scatter(x=[i for i in range(11)], y=validation_avg, name="average validation error"))
    fig2.update_layout(title=f"Average training and validation score as a function of polynomial degree of model with "
                             f"{n_samples} samples and {noise} noise",
                       xaxis_title="degree")
    fig2.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k = np.argmin(validation_avg)
    full_train_estimator = PolynomialFitting(best_k).fit(np.array(train_X), np.array(train_y))
    prediction = full_train_estimator.predict(np.array(test_X))
    print("best k degree: ", best_k)
    print("test error of full train model with best k degree: ", np.round(mean_square_error(np.array(test_y), prediction), 2))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples], y[:n_samples]
    test_X, test_y = X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambda_values = np.linspace(0.01, 2, n_evaluations)
    train_error_ridge, validation_error_ridge, train_error_lasso, validation_error_lasso = [], [], [], []

    for lam in lambda_values:
        ridge_reg = RidgeRegression(lam)
        train_score_ridge, validation_score_ridge = cross_validate(ridge_reg, train_X, train_y, mean_square_error)
        train_error_ridge.append(train_score_ridge)
        validation_error_ridge.append(validation_score_ridge)

        lasso_reg = Lasso(alpha=lam)
        train_score_lasso, validation_score_lasso = cross_validate(lasso_reg, train_X, train_y, mean_square_error)
        train_error_lasso.append(train_score_lasso)
        validation_error_lasso.append(validation_score_lasso)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambda_values, y=train_error_ridge, mode="markers",
                             marker=dict(color="Red",colorscale=[custom[0], custom[-1]]),
                             name="average Ridge training error"))

    fig.add_trace(go.Scatter(x=lambda_values, y=validation_error_ridge, mode="markers",
                             marker=dict(color="Gray",colorscale=[custom[0], custom[-1]]),
                             name="average Ridge validation error"))

    fig.add_trace(go.Scatter(x=lambda_values, y=train_error_lasso, mode="markers",
                             marker=dict(color="Green",colorscale=[custom[0], custom[-1]]),
                             name="average Lasso training error"))

    fig.add_trace(go.Scatter(x=lambda_values, y=validation_error_lasso, mode="markers",
                             marker=dict(color="Purple",colorscale=[custom[0], custom[-1]]),
                             name="average Lasso validation error"))

    fig.update_layout(title="Average training and validation error as a function of regularization parameter value",
                       xaxis_title="regularization parameter value")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = lambda_values[np.argmin(validation_error_ridge)]
    best_lasso_lambda = lambda_values[np.argmin(validation_error_lasso)]
    print("best lambda for ridge: ", best_ridge_lambda)
    print("best lambda for lasso: ", best_lasso_lambda)

    full_train_ridge = RidgeRegression(best_ridge_lambda).fit(np.array(train_X), np.array(train_y))
    full_train_lasso = Lasso(alpha=best_lasso_lambda).fit(np.array(train_X), np.array(train_y))
    full_train_ls = LinearRegression().fit(np.array(train_X), np.array(train_y))

    print("test error of ridge estimator with best lambda: ",
          np.round(mean_square_error(np.array(test_y), full_train_ridge.predict(np.array(test_X))), 2))
    print("test error of lasso estimator with best lambda: ",
          np.round(mean_square_error(np.array(test_y), full_train_lasso.predict(np.array(test_X))), 2))
    print("test error of linear regression with best lambda: ",
          np.round(mean_square_error(np.array(test_y), full_train_ls.predict(np.array(test_X))), 2))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()