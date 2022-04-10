from pandas import DataFrame
from sklearn.base import BaseEstimator

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

from sklearn import svm
from sklearn import tree

import numpy as np
import pandas as pd


def calc_canceling_fund(estimated_vacation_time: int,
                        cancelling_policy_code: str,
                        original_selling_amount: float):
    policy_options = cancelling_policy_code.split("_")
    sum = 0
    for option in policy_options:
        if "D" in option:
            if "P" in option:
                charge = int(option[option.find("D") + 1:option.find("P")])
                charge /= 100
                sum += original_selling_amount * charge
            if "N" in option:
                charge = int(option[option.find("D") + 1:option.find("N")])
                charge /= estimated_vacation_time
                sum += original_selling_amount * charge
        elif "P" in option:
            charge = int(option[option.find("D") + 1:option.find("P")])
            charge /= 100
            sum += original_selling_amount * charge
    return sum / len(policy_options)


def data_conversion(full_data):
    countries_code = set(full_data["origin_country_code"]).union(
        set(full_data["hotel_country_code"]))
    countries_code_dict = {k: v for v, k in enumerate(countries_code)}
    full_data.replace({"origin_country_code": countries_code_dict},
                    inplace=True)
    full_data.replace({"hotel_country_code": countries_code_dict},
                    inplace=True)
    accommodation_type = set(full_data["accommadation_type_name"])
    accommodation_type_dict = {k: v for v, k in enumerate(accommodation_type)}
    full_data.replace({"accommadation_type_name": accommodation_type_dict},
                    inplace=True)
    payment_type = set(full_data["original_payment_type"])
    payment_type_dict = {k: v for v, k in enumerate(payment_type)}
    full_data.replace({"original_payment_type": payment_type_dict},
                     inplace=True)
    bool_dict = {True: 0, False: 1}
    full_data.replace({"is_first_booking": bool_dict}, inplace=True)


def insert_new_columns(full_data: DataFrame):
    charge_option = set(full_data["charge_option"])
    num_charge_option = {value: key for key, value in
                         enumerate(charge_option)}
    full_data.replace({"charge_option": num_charge_option}, inplace=True)
    num_of_days_from_booking = []
    estimated_stay_time = []
    cancelling_date_ind = 34
    policy_ind = 27
    cancelling_fund = []
    cancelling_days_from_booking = []
    for row in full_data.iterrows():
        booking_date = pd.to_datetime(row[1][1])
        check_in_date = pd.to_datetime(row[1][2])
        check_out_date = pd.to_datetime(row[1][3])
        estimated_vacation_time = (check_out_date - check_in_date).days
        if 'cancellation_datetime' in full_data.columns and row[1][cancelling_date_ind] is not None:
            cancelling_date = pd.to_datetime(row[1][cancelling_date_ind])
            delta_from_booking = (cancelling_date - booking_date).days
            cancelling_days_from_booking.append(delta_from_booking + 1)
        policy = row[1].cancellation_policy_code
        selling = row[1].original_selling_amount
        cancelling_fund.append((calc_canceling_fund(
            estimated_vacation_time, policy, selling) / selling) * 100)
        num_of_days_from_booking.append((check_in_date - booking_date).days)
        estimated_stay_time.append(estimated_vacation_time)
    full_data.insert(4, "time_from_booking_to_check_in",
                   num_of_days_from_booking)
    full_data.insert(5, "estimated_stay_time", estimated_stay_time)
    if "cancellation_datetime" in full_data.columns:
        full_data.insert(6, "cancelling_days_from_booking",
                         cancelling_days_from_booking)
    full_data.insert(policy_ind + 1, "avg_cancelling_fund_percent",
                   cancelling_fund)


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    insert_new_columns(full_data)
    data_conversion(full_data)

    features = full_data[["booking_datetime",
                          "time_from_booking_to_check_in",
                          "estimated_stay_time",
                          "guest_is_not_the_customer",
                          "original_payment_type",
                          "hotel_star_rating",
                          "is_first_booking",
                          "avg_cancelling_fund_percent"]]

    if "cancellation_datetime" in full_data.columns:
        labels = full_data["cancelling_days_from_booking"]
    else:
        labels = []
    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    prediction = estimator.predict(X)
    counter_estimated = 0
    res = []
    for i in range(len(X)):
        cancel_estimated = pd.to_datetime(X[i, 0])+pd.to_timedelta(prediction[i], unit="days")
        if pd.to_datetime("2018-12-07") <= cancel_estimated <= pd.to_datetime("2018-12-13"):
            res.append(1)
            counter_estimated += 1
        else:
            res.append(0)
    pd.DataFrame(res, columns=["predicted_values"]).to_csv(filename, index=False)
    # print(counter_estimated)
    return


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")
    # Fit model over data
    train_X, train_y, temp_X, temp_y = split_train_test(df, cancellation_labels, train_proportion=1) # for test-set
    # train_X, train_y, temp_X, temp_y = split_train_test(df, cancellation_labels) # for train-set

    train_X = train_X.fillna(0).to_numpy()
    train_y = train_y.fillna(0).to_numpy()

    temp_X = temp_X.fillna(0).to_numpy()  # for train set
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)

    # Store model predictions over test set
    df_test, cancellation_labels_test = load_data("../challenge/test_set_week_2.csv")
    # test_X, test_y, temp_X2, temp_y2 = split_train_test(df_test, cancellation_labels_test, train_proportion=1)
    test_X = df_test.fillna(0).to_numpy()

    # evaluate_and_export(estimator, temp_X, "id1_id2_id3.csv") # for train-set

    evaluate_and_export(estimator, test_X, "208385633_315997874_206948911.csv") # for test-set
