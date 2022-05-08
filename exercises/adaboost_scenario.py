import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def partial_decision_surface(predict, xrange, yrange, n_learners, density=120, dotted=False, colorscale=custom,
                             showscale=True):
    xrange, yrange = np.linspace(*xrange, density), np.linspace(*yrange, density)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = predict(np.c_[xx.ravel(), yy.ravel()], n_learners)

    if dotted:
        return go.Scatter(x=xx.ravel(), y=yy.ravel(), opacity=1, mode="markers",
                          marker=dict(color=pred, size=1, colorscale=colorscale, reversescale=False),
                          hoverinfo="skip", showlegend=False)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape),
                      colorscale=colorscale, reversescale=False, opacity=.7, connectgaps=True,
                      hoverinfo="skip", showlegend=False, showscale=showscale)


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaB_learner = AdaBoost(wl= lambda :DecisionStump(), iterations=n_learners)
    adaB_learner.fit(train_X, train_y)

    fig = go.Figure()
    learners = [i for i in range(1, n_learners+1)]
    train_loss_by_learners = [adaB_learner.partial_loss(train_X, train_y, i) for i in learners]
    test_loss_by_learners = [adaB_learner.partial_loss(test_X, test_y, i) for i in learners]

    fig.add_trace(go.Scatter(x=learners, y=train_loss_by_learners, name="train error"))
    fig.add_trace(go.Scatter(x=learners, y=test_loss_by_learners, name="test error"))

    fig.update_layout(title=f"Training and test errors as a function of the number of fitted learners, noise = {noise}",
                      title_x=0.5, xaxis_title="number of fitted learners")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X,test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    test_corrected_y = np.where(test_y < 0, 0, 1)

    plot_titles = [f"n = {num_learners}" for num_learners in T]
    fig2 = make_subplots(rows=2, cols=2, subplot_titles=plot_titles, horizontal_spacing=0.01, vertical_spacing=0.1)

    for i in range(len(T)):
        fig2.add_traces([partial_decision_surface(adaB_learner.partial_predict, lims[0], lims[1], T[i],
                                                  showscale=False),
                         go.Scatter(x=test_X[:,0], y=test_X[:,1], mode="markers",showlegend=False,
                                    marker=dict(color=test_y, colorscale=[custom[0],custom[-1]],
                                                line=dict(color="black", width=1)))],
                        rows=1 if 0 <= i <= 1 else 2, cols=i % 2 + 1)
    fig2.update_layout(title=f"Desicion boundary as a function of ensemble size, noise = {noise}", title_x=0.5)
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    best_test_ensemble_size = np.argmin(test_loss_by_learners) + 1
    best_test_ensemble_accuracy = 1 - test_loss_by_learners[best_test_ensemble_size-1]

    fig3 = go.Figure()
    fig3.add_traces([partial_decision_surface(adaB_learner.partial_predict, lims[0], lims[1], best_test_ensemble_size,
                                              showscale=False),
                     go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(color=test_y, colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))])
    fig3.update_layout(title=f"Ensemble size achieved the lowest test error is {best_test_ensemble_size} with "
                             f"accuracy of {best_test_ensemble_accuracy}, noise = {noise}", title_x=0.5,
                       xaxis_title="feature1 ", yaxis_title="feature 2")
    fig3.show()


    # Question 4: Decision surface with weighted samples
    D = adaB_learner.D_/np.max(adaB_learner.D_) * 10
    train_corrected_y = np.where(train_y < 0, 0, 1)

    fig4 = go.Figure()
    fig4.add_traces([decision_surface(adaB_learner.predict, lims[0], lims[1], showscale=False),
                     go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                marker=dict(size=D, color=train_y, colorscale=[custom[0], custom[-1]],
                                            line=dict(color="black", width=1)))])
    fig4.update_layout(title=f"Training set with a point size proportional to it's weight, noise = {noise}",
                       title_x=0.5, xaxis_title="feature1 ", yaxis_title="feature 2")
    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
