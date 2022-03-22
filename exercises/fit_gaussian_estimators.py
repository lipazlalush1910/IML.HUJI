from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    normal_distribution_obj = UnivariateGaussian()
    normal_distribution = np.random.normal(10, 1, 1000)
    normal_distribution_obj.fit(normal_distribution)
    print(f"({normal_distribution_obj.mu_},{normal_distribution_obj.var_})")

    normal_distribution_arr = np.array(normal_distribution)

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100)
    estimated_mean = []
    sample_obj = UnivariateGaussian()

    for i in range(10, 1001, 10):
        arr = normal_distribution_arr[0:i]
        sample_obj.fit(arr)
        estimated_mean.append(abs(10 - sample_obj.mu_))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ms, y=estimated_mean, name="Graph"))
    fig.update_layout(title='Expectation error of samples in increasing size', xaxis_title="sample size",
                      yaxis_title="Expectation error")
    fig.show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf_samples = normal_distribution_obj.pdf(normal_distribution)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=normal_distribution, y=pdf_samples, name="Graph", mode="markers"))
    fig2.update_layout(title="Empirical PDF function of samples", xaxis_title="sample", yaxis_title="PDF value")
    fig2.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])

    mult_normal_distribution = np.random.multivariate_normal(mean, cov, 1000)
    mult_normal_distribution_obj = MultivariateGaussian()
    mult_normal_distribution_obj.fit(mult_normal_distribution)

    print(mult_normal_distribution_obj.mu_)
    print(mult_normal_distribution_obj.cov_)

    # Question 5 - Likelihood evaluation
    samples = np.linspace(-10, 10, 200)
    estimated_log_likelihood = np.zeros((samples.size, samples.size))
    for f1 in range(samples.size):
        for f3 in range(samples.size):
            f1_val = samples[f1]
            f3_val = samples[f3]
            mu_temp = np.array([f1_val, 0, f3_val, 0]).transpose()
            estimated_log_likelihood[f1, f3] = mult_normal_distribution_obj.log_likelihood(mu_temp, cov,
                                                                                           mult_normal_distribution)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=samples, y=samples, z=estimated_log_likelihood))
    fig.update_layout(title="Heatmap log-likelihood of multivariate gaussian samples as a function of f1 and f3",
                      xaxis_title="f3", yaxis_title="f1")
    fig.show()

    # Question 6 - Maximum likelihood
    max_f1 = 0
    max_f3 = 0
    max_log_likelihood = -np.inf
    for f1 in range(samples.size):
        for f3 in range(samples.size):
            if estimated_log_likelihood[f1, f3] > max_log_likelihood:
                max_log_likelihood = estimated_log_likelihood[f1, f3]
                max_f1 = f1
                max_f3 = f3
    print(f"{samples[max_f1]},{samples[max_f3]},{estimated_log_likelihood[max_f1, max_f3]}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
