import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=['Date']).drop_duplicates()
    full_data = full_data[full_data["Temp"] > -72]

    dayOfYear = []
    for date in full_data["Date"]:
        dayOfYear.append(pd.Period(date, 'D').dayofyear)
    full_data["DayOfYear"] = dayOfYear

    features = full_data[["Country",
                          "City",
                          "DayOfYear",
                          "Year",
                          "Month",
                          "Day"]]

    # features = pd.concat([features, countries], axis=1)
    # features = pd.concat([features, cities], axis=1)

    temperature_label = full_data["Temp"]
    return features, temperature_label


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    features, temperature_label = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_features = features[features["Country"] == "Israel"]
    israel_temp = temperature_label.reindex_like(israel_features)

    fig2_1 = px.scatter(
        pd.DataFrame({'x': israel_features["DayOfYear"], 'y':israel_temp}),
        x='x', y='y', labels={"x": "day of year", "y": "temperature"},
        title="Average Daily Temperature in Israel As a Function of The Day Of Year",
        color=israel_features["Year"].astype(str))
    fig2_1.show()

    union_df = pd.concat([features, temperature_label], axis=1)
    israel_df = union_df[union_df["Country"] == "Israel"]
    israel_grouped_by_month = israel_df.groupby(["Month"])
    month_groups = israel_df["Month"].drop_duplicates().values
    std_temp_monthly = israel_grouped_by_month.agg({"Temp": "std"})

    fig2_2 = px.bar(std_temp_monthly, x=month_groups, y="Temp")
    fig2_2 = px.bar(pd.DataFrame({'x': month_groups, 'y': std_temp_monthly["Temp"]}), x='x', y='y',
                    labels={'x': "month", 'y':"standard deviation of the daily temperatures"},
                    title="Standard Deviation of The Daily Temperatures in Israel by Months")
    fig2_2.show()

    # Question 3 - Exploring differences between countries
    grouped_by_country_month = union_df.groupby(["Country", "Month"])
    mean_temp_grouped = grouped_by_country_month.mean().reset_index()
    std_temp_grouped = grouped_by_country_month.std().reset_index()

    fig3 = px.line(pd.DataFrame({'mean_temp': mean_temp_grouped['Temp'],
                                 'std_temp': std_temp_grouped['Temp'],
                                 'Country': mean_temp_grouped['Country'],
                                 'Month': mean_temp_grouped['Month']}),
                   x='Month', y='mean_temp', color='Country', error_y='std_temp',
                   labels={'Month': "month", 'mean_temp': "average monthly temperature"},
                   title="Average Monthly Temperature Colored by Country with Error Bars by Standard Deviation of "
                         "The Temperature")
    fig3.show()
    print()

    # Question 4 - Fitting model for different values of `k`
    train_israel_X, train_israel_y, test_israel_X, test_israel_y = split_train_test(israel_features,
                                                                                    israel_temp, train_proportion=0.75)
    loss_prediction = []
    for k in range(1, 11):
        polynomial_estimator = PolynomialFitting(k)
        polynomial_estimator.fit(train_israel_X["DayOfYear"].to_numpy(), train_israel_y.to_numpy())
        test_error = polynomial_estimator.loss(test_israel_X["DayOfYear"].to_numpy(), test_israel_y.to_numpy())
        loss_prediction.append(np.around(test_error, 2))

    print(loss_prediction)
    fig4 = px.bar(x=[range(1,11)], y=loss_prediction,
                  labels={'x': "k degree", 'y':"test error"},
                  title="Test Error as a Function of Polynomial Fitting Degree")
    fig4.show()

    # Question 5 - Evaluating fitted model on different countries
    k = np.argmin(loss_prediction) + 1
    print(k)
    israel_estimator = PolynomialFitting(k)
    israel_estimator.fit(israel_features["DayOfYear"].to_numpy(), israel_temp.to_numpy())

    loss_by_country = {}
    for country in set(union_df["Country"]):
        if country != "Israel":
            country_features = features[features["Country"] == country]
            country_temp = temperature_label.reindex_like(country_features)
            test_error = israel_estimator.loss(country_features["DayOfYear"].to_numpy(), country_temp.to_numpy())
            loss_by_country[country] = np.around(test_error, 2)

    print(loss_by_country)

    fig5 = px.bar(x=loss_by_country.keys(), y=loss_by_country.values(),
                  labels={'x': "country", 'y': "test error"},
                  color=loss_by_country.keys(),
                  title="Israel Based Model's error over other countries")
    fig5.show()