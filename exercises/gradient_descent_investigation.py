import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn.model_selection

from IMLearn.base import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    list_of_weights = []

    def callback(solver, weights, val, grad, t, eta, delta):
        values.append(val)
        list_of_weights.append(weights)
    return callback, values, list_of_weights


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    X = np.array([])
    y = np.array([])
    L1_min_losses = []
    L2_min_losses = []
    for eta in etas:
        L1_module = L1(init)
        L2_module = L2(init)
        lr = FixedLR(eta)
        L1_callback, L1_values, L1_weights = get_gd_state_recorder_callback()
        L2_callback, L2_values, L2_weights = get_gd_state_recorder_callback()
        l1_grad = GradientDescent(learning_rate=lr, callback=L1_callback).fit(f=L1_module, X=X, y=y)
        l2_grad = GradientDescent(learning_rate=lr, callback=L2_callback).fit(f=L2_module, X=X, y=y)
        L1_min_losses.append(L1(l1_grad).compute_output())
        L2_min_losses.append(L2(l2_grad).compute_output())

        L1_trajectory_fig = plot_descent_path(L1, np.array(L1_weights), title=f"L1 Module: eta={eta}")
        L1_trajectory_fig.show()
        L2_trajectory_fig = plot_descent_path(L2, np.array(L2_weights), title=f"L2 Module: eta={eta}")
        L2_trajectory_fig.show()

        L1_convergence_fig = go.Figure()
        L1_convergence_fig.add_trace(go.Scatter(x=[i for i in range(len(L1_values))], y=L1_values, mode="markers"))
        L1_convergence_fig.update_layout(title=f"L1 Convergence rate as a function of iteration number, eta={eta}",
                                         xaxis_title="Iteration Number", yaxis_title="L1 Norm")
        L1_convergence_fig.show()

        L2_convergence_fig = go.Figure()
        L2_convergence_fig.add_trace(go.Scatter(x=[i for i in range(len(L2_values))], y=L2_values, mode="markers"))
        L2_convergence_fig.update_layout(title=f"L2 Convergence rate as a function of iteration number, eta={eta}",
                                         xaxis_title="Iteration Number", yaxis_title="L2 Norm")
        L2_convergence_fig.show()

    L1_min_losses = np.array(L1_min_losses)
    L2_min_losses = np.array(L2_min_losses)
    print(f"Best eta for L1 model: {etas[np.argmin(L1_min_losses)]}, minimum loss: {np.min(L1_min_losses)}")
    print(f"Best eta for L2 model: {etas[np.argmin(L2_min_losses)]}, minimum loss: {np.min(L2_min_losses)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    X = np.array([])
    y = np.array([])
    L1_min_losses = []
    fig = make_subplots(rows=2, cols=2, subplot_titles=gammas,
                            horizontal_spacing=0.01, vertical_spacing=0.05)
    for i, gamma in enumerate(gammas):
        L1_module = L1(init)
        lr = ExponentialLR(eta, gamma)
        L1_callback, L1_values, L1_weights = get_gd_state_recorder_callback()
        l1_grad = GradientDescent(learning_rate=lr, callback=L1_callback).fit(f=L1_module, X=X, y=y)
        fig.add_trace(go.Scatter(x=[i for i in range(len(L1_values))], y=L1_values, mode="markers",
                                 name=f"Gamma = {gamma}"), row=(i // 2) + 1, col=(i % 2) + 1)
        L1_min_losses.append(L1(l1_grad).compute_output())

    fig.update_layout(title=f"L1 Convergence rate as a function of iteration number using different decay rates")
    fig.show()
    L1_min_losses = np.array(L1_min_losses)
    print(f"Best gamma for L1 model: {gammas[np.argmin(L1_min_losses)]}, minimum loss: {np.min(L1_min_losses)}")

    # Plot algorithm's convergence for the different values of gamma
    # Plot descent path for gamma=0.95
    gamma = 0.95
    L1_module = L1(init)
    L2_module = L2(init)
    lr = ExponentialLR(eta, gamma)
    L1_callback, L1_values, L1_weights = get_gd_state_recorder_callback()
    L2_callback, L2_values, L2_weights = get_gd_state_recorder_callback()
    GradientDescent(learning_rate=lr, callback=L1_callback).fit(f=L1_module, X=X, y=y)
    GradientDescent(learning_rate=lr, callback=L2_callback).fit(f=L2_module, X=X, y=y)

    L1_trajectory_fig = plot_descent_path(L1, np.array(L1_weights), title=f"L1 Module: eta={eta}, gamma={gamma}")
    L1_trajectory_fig.show()

    L2_trajectory_fig = plot_descent_path(L2, np.array(L2_weights), title=f"L2 Module: eta={eta}, gamma={gamma}")
    L2_trajectory_fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data
    alphas = np.arange(0, 1, 0.01)
    logistic_regression = LogisticRegression()
    logistic_regression.fit(X_train, y_train)
    y_pred_prob = logistic_regression.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob)
    fig = go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                           name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=alphas, name="", showlegend=False,
                           marker_size=5,
                           hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                           xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                           yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()
    best_alpha = np.round(thresholds[np.argmax(tpr - fpr)], 2)
    print(f"Best alpha: {best_alpha}")
    log_reg_best_alpha = LogisticRegression(alpha=best_alpha)

    pred = log_reg_best_alpha.fit(X_train, y_train).predict(X_test)
    print(f"Test error for alpha: {misclassification_error(y_test, pred)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    alpha = 0.5
    penalties = ["l1", "l2"]
    lambda_values = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for penalty in penalties:
        validation_scores = []
        for lam in lambda_values:
            log_reg_module = LogisticRegression(penalty=penalty, alpha=alpha, lam=lam)
            train_score, validation_score = cross_validate(log_reg_module, X_train,
                                                           y_train, misclassification_error)
            validation_scores.append(validation_score)
        validation_scores = np.array(validation_scores)
        best_lambda = lambda_values[np.argmin(validation_scores)]
        print(f"Best lambda for {penalty} module is: {best_lambda}")
        log_reg_module = LogisticRegression(penalty=penalty, alpha=0.5, lam=best_lambda)
        pred = log_reg_module.fit(X_train, y_train).predict(X_test)
        print(f"Test error for {penalty} module is: {misclassification_error(y_test, pred)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()