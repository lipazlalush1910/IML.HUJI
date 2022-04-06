from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

# import matplotlib.pyplot as plt  # just for correlation in Q1
# import seaborn as sns # just for correlation in Q1

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def calculate_house_dates(full_data):
    diff_built = [2022 - date for date in full_data["yr_built"]]
    full_data["house_age"] = diff_built
    full_data["is_renovated"] = np.where(full_data["yr_renovated"] == 0, 0, 1)
    full_data.drop(full_data[full_data.date == "0"].index, inplace=True)

    sell_year = []
    for date in full_data["date"]:
        sell_year.append(pd.to_datetime(date).year)
    full_data["sell_year"] = sell_year

    sell_year_dummies = pd.get_dummies(full_data.sell_year, prefix="sell_year_")
    return sell_year_dummies


def categorical_columns(full_data):
    # convert categorical features to binary columns
    full_data["has_view"] = np.where(full_data["view"] == 0, 0, 1)
    floor_dummies = pd.get_dummies(full_data.floors, prefix="floor_")
    return floor_dummies


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).dropna().drop_duplicates()
    # corr_data = pd.DataFrame(full_data.corr())
    # sns.heatmap(corr_data, annot=True)
    # plt.show()

    full_data = full_data[full_data["price"] > 0]

    sell_year_res = calculate_house_dates(full_data)
    categorical_res = categorical_columns(full_data)
    zipcode_dummies = zipcode_categories(full_data)

    features = full_data[["bedrooms",
                          "bathrooms",
                          "sqft_living",
                          "has_view",
                          "condition",
                          "grade",
                          "house_age",
                          # "zipcode",
                          "is_renovated"]]
    features = pd.concat([features, sell_year_res], axis=1)
    features = pd.concat([features, categorical_res], axis=1)
    features = pd.concat([features, zipcode_dummies], axis=1)

    price_response = full_data["price"]
    return features, price_response


def zipcode_categories(full_data):
    full_data["zipcode_prefix"] = full_data["zipcode"].apply(lambda x: str(np.round(x / 10)))
    zipcode_dummies = pd.get_dummies(full_data.zipcode_prefix, prefix="zipcode_")
    return zipcode_dummies


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson_correlation_res = []
    for col in X:
        cov_featureX_y = np.cov(X[col], y)[0, 1]
        std_featureX = np.sqrt(np.var(X[col]))
        std_y = np.sqrt(np.var(y))
        pearson = cov_featureX_y / (std_featureX * std_y)
        rounded_pearson = np.round(pearson, 3)
        pearson_correlation_res.append(pearson)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X[col], y=y, name=f"Feature {col} evaluation", mode="markers"))
        fig.update_layout(title=f'Evaluation of {col} And Price House with Pearson Correlation of {rounded_pearson}',
                          xaxis_title=col,yaxis_title=f"house price (respone)")
        fig.write_image(f"{output_path}\\{col}.png", format="png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    features, pricing_labels = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(features, pricing_labels, "C:\\Users\\lipaz\\OneDrive\\Desktop\\iml_res")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(features, pricing_labels, train_proportion=0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    samples_size = []
    mean_loss_predictions = []
    std_loss_predictions = []
    linear_estimator = LinearRegression()
    for p in range(10, 101):
        loss_p = []
        for i in range(0, 10):
            train_sample_X, train_sample_y, test_sample_X, test_sample_y = split_train_test(train_X, train_y,
                                                                                            train_proportion=p / 100)
            linear_estimator.fit(train_sample_X.to_numpy(), train_sample_y.to_numpy())
            loss_p.append(linear_estimator.loss(test_X.to_numpy(), test_y.to_numpy()))

        samples_size.append(p)
        mean_loss_predictions.append(np.mean(loss_p))
        std_loss_predictions.append(np.std(loss_p))

    mean_loss_predictions = np.array(mean_loss_predictions)
    std_loss_predictions = np.array(std_loss_predictions)

    fig2 = go.Figure([go.Scatter(x=samples_size, y=mean_loss_predictions,
                                 mode="markers+lines", name="average loss", line=dict(dash="dash"),
                                 marker=dict(color="green")),
                      go.Scatter(x=samples_size, y=(mean_loss_predictions + 2 * std_loss_predictions),
                                 fill="tonexty", mode="lines", line=dict(color="lightgrey"),
                                 showlegend=False),
                      go.Scatter(x=samples_size, y=(mean_loss_predictions - 2 * std_loss_predictions),
                                 fill="tonexty", mode="lines", line=dict(color="lightgrey"),
                                 showlegend=False)])
    fig2.update_layout(title="Mean Loss as a Function of Training Sample Size", xaxis_title="sample size",
                       yaxis_title="average loss")
    fig2.show()
