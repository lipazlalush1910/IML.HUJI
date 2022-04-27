from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X,y = load_dataset("C:/Users/lipaz/OneDrive/Desktop/iml/datasets/"+f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=lambda per, xi, yi: losses.append(per._loss(X,y)))
        perceptron.fit(X,y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(losses)+1)), y=losses, name="Graph"))
        fig.update_layout(title=f'Loss of {n} as a Function of Fitting Iteration', xaxis_title="iteration",
                          yaxis_title="Loss")
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X,y = load_dataset("C:/Users/lipaz/OneDrive/Desktop/iml/datasets/"+f)

        # Fit models and predict over training set
        models = [LDA(), GaussianNaiveBayes()]
        models[0].fit(X, y)
        models[1].fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        models_names = [f"Gaussian Naive Bayes, accuracy: {accuracy(y, models[1].predict(X))}",
                        f"Linear Discriminant Analysis, accuracy: {accuracy(y, models[0].predict(X))}"]
        fig = make_subplots(rows=1, cols=2, subplot_titles=models_names,
                            horizontal_spacing= 0.01, vertical_spacing=0.03)

        # Add traces for data-points setting symbols and colors
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

        prediction = [models[0].predict(X), models[1].predict(X)]
        for i, m in enumerate(models):
            fig.add_traces(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                       marker=go.scatter.Marker(color=prediction[i],symbol=np.array(y))),
                           rows=(i//3) + 1, cols=(i%3) +1)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=models[0].mu_[:, 0], y=models[0].mu_[:, 1], mode="markers",
                                 name="Center of fitted LDA",
                                 marker=dict(symbol='x-dot', color="Black")), row=1, col=2)
        fig.add_trace(go.Scatter(x=models[1].mu_[:,0], y=models[1].mu_[:,1], mode="markers",
                                 name="Center of fitted Gaussians",
                                 marker=dict(symbol='x-dot',color="Black")), row=1, col=1)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for k in range(models[0].classes_.shape[0]):
            fig.add_trace(get_ellipse(models[0].mu_[k], models[0].cov_), row=1, col=2)

        for k in range(models[1].classes_.shape[0]):
            fig.add_trace(get_ellipse(models[1].mu_[k], np.diag(models[1].vars_[k])), row=1, col=1)
        fig.update_layout(title_text=f"Predictions of Different Classifiers over {f}",
                          title_x=0.5)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()




