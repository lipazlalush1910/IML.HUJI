import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.base import BaseEstimator


def calc_canceling_fund(estimated_vacation_time,
                        cancelling_policy_code,
                        original_selling_amount,
                        normalize=True):

    policy_options = cancelling_policy_code.split("_")
    cost_sum = 0
    for option in policy_options:
        if "D" in option:
            if "P" in option:
                charge = int(option[option.find("D") + 1:option.find("P")])
                charge /= 100
                cost_sum += original_selling_amount * charge
            if "N" in option:
                charge = int(option[option.find("D") + 1:option.find("N")])
                charge /= estimated_vacation_time
                cost_sum += original_selling_amount * charge
        elif "P" in option:
            charge = int(option[option.find("D") + 1:option.find("P")])
            charge /= 100
            cost_sum += original_selling_amount * charge
    if normalize:
        return ((cost_sum / len(policy_options)) / original_selling_amount) * 100
    return cost_sum / len(policy_options)


def add_hour_of_day(full_data: DataFrame):
    full_data["hour"] = pd.to_datetime(full_data["booking_datetime"]).dt.hour


def add_month_of_booking(full_data: DataFrame):
    full_data["month_of_booking"] = pd.to_datetime(full_data["booking_datetime"]).dt.month


def add_month_of_cheking(full_data):
    full_data["month_of_checkin"] = pd.to_datetime(full_data["checkin_date"]).dt.month


# def payment_type_to_hashcode(full_data):
#     payment_type = set(full_data["original_payment_type"])
#     payment_type_dict = {k: v for v, k in enumerate(payment_type)}
#     full_data.replace({"original_payment_type": payment_type_dict},
#                       inplace=True)
#     bool_dict = {True: 0, False: 1}
#     full_data.replace({"is_first_booking": bool_dict}, inplace=True)


# def accommodation_to_hashcode(full_data):
#     accommodation_type = set(full_data["accommadation_type_name"])
#     accommodation_type_dict = {k: v for v, k in enumerate(accommodation_type)}
#     full_data.replace({"accommadation_type_name": accommodation_type_dict},
#                       inplace=True)


# def country_to_hashcode(full_data):
#     countries_code = set(full_data["origin_country_code"]).union(
#         set(full_data["hotel_country_code"]))
#
#     countries_code_dict = {k: v for v, k in enumerate(countries_code)}
#     full_data.replace({"origin_country_code": countries_code_dict},
#                       inplace=True)
#     full_data.replace({"hotel_country_code": countries_code_dict},
#                       inplace=True)


def calculate_canceling_fund_present(full_data):
    temp = full_data[["cancellation_policy_code", "original_selling_amount", "estimated_stay_time"]]
    res = temp.apply(lambda x: calc_canceling_fund(x['estimated_stay_time'], x['cancellation_policy_code'],
                                                   x['original_selling_amount']), axis=1)
    # res_ = temp.apply(lambda x: calc_canceling_fund(x['estimated_stay_time'], x['cancellation_policy_code'],
    #                                                 x['original_selling_amount'], False), axis=1)
    full_data["avg_cancelling_fund_percent"] = res
    # full_data["total_cancelling_fund"] = res_


def calc_booking_checking(full_data):
    checking_date = pd.to_datetime(full_data["checkin_date"])
    booking_date = pd.to_datetime(full_data["booking_datetime"])
    full_data["time_from_booking_to_check_in"] = (checking_date - booking_date).dt.days + 1


def calc_checking_checkout(full_data):
    checking_date = pd.to_datetime(full_data["checkin_date"])
    checkout_date = pd.to_datetime(full_data["checkout_date"])
    full_data["estimated_stay_time"] = (checkout_date - checking_date).dt.days + 1


def country_code(full_data: DataFrame):
    full_data["is_abroad"] = full_data["hotel_country_code"] == full_data["origin_country_code"]
    full_data["is_abroad"].astype(int)


def calc_cancel_checking(full_data):
    checkin_date = pd.to_datetime(full_data["checkin_date"])
    cancel_date = pd.to_datetime(full_data["cancellation_datetime"])
    full_data["cancelling_days_from_checkin"] = (checkin_date - cancel_date).dt.days + 1


def charge_option_to_hashcode(full_data):
    charge_option = set(full_data["charge_option"])
    num_charge_option = {value: key for key, value in
                         enumerate(charge_option)}
    full_data.replace({"charge_option": num_charge_option}, inplace=True)


def preprocess_data(full_data: pd.DataFrame,
                    hot_enc_: preprocessing.OneHotEncoder,
                    label_enc_: preprocessing.OrdinalEncoder):
    # country_to_hashcode(full_data)
    add_hour_of_day(full_data)
    add_month_of_booking(full_data)
    add_month_of_cheking(full_data)
    calc_booking_checking(full_data)
    calc_checking_checkout(full_data)
    country_code(full_data)
    if "cancellation_datetime" in full_data:
        add_if_cancel(full_data)
        calc_cancel_checking(full_data)
        full_data.drop('cancellation_datetime', axis=1, inplace=True)
    calculate_canceling_fund_present(full_data)

    list_to_drop = ['h_booking_id',
                    'booking_datetime',
                    'cancellation_policy_code',
                    'hotel_live_date',
                    'checkin_date',
                    'checkout_date']
    full_data.drop(list_to_drop, axis=1, inplace=True)
    dfOneHot = DataFrame(hot_enc_.transform(full_data[hot_enc_.feature_names_in_]))
    dfOneHot.columns = hot_enc_.get_feature_names_out()
    full_data = full_data.join(dfOneHot)
    full_data[list_to_label] = label_enc_.transform(full_data[list_to_label])
    full_data.drop(hot_enc_.feature_names_in_, axis=1, inplace=True)
    return full_data


def add_if_cancel(full_data):
    full_data['is_canceled'] = (~full_data['cancellation_datetime'].isna()).astype(int)


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
    return pd.read_csv(filename)


def evaluate_data(dt: pd.DataFrame):
    correlation = dt.corr()['is_canceled'].abs().sort_values(ascending=False)
    print(correlation)


def drop_data(full_data: pd.DataFrame):
    full_data = full_data.drop_duplicates(inplace=False)
    full_data.drop(['hotel_brand_code',
                    'hotel_chain_code',
                    'request_largebed',
                    'request_earlycheckin',
                    'request_airport',
                    'request_highfloor',
                    'request_latecheckin',
                    'request_nonesmoke',
                    'hotel_id',
                    'h_customer_id',
                    'request_twinbeds'],

                   axis=1,
                   inplace=True)
    # more than 40 present is missing (not including the canceling date which 73.91193167288907% nan)
    if 'cancellation_datetime' in full_data.columns:
        full_data = full_data.dropna(subset=full_data.columns.drop("cancellation_datetime")).reset_index(drop=True)

    return full_data


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
    res = []
    for i in range(len(X)):
        cancel_estimated = pd.to_datetime(X[i, 0]) + pd.to_timedelta(prediction[i], unit="days")
        if pd.to_datetime("2018-12-07") <= cancel_estimated <= pd.to_datetime("2018-12-13"):
            res.append(1)
        else:
            res.append(0)
    pd.DataFrame(res, columns=["predicted_values"]).to_csv(filename, index=False)
    return

if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df = load_data("../datasets/agoda_cancellation_train.csv")

    df = drop_data(df)

    hot_enc = preprocessing.OneHotEncoder(sparse=False,handle_unknown="ignore")
    label_enc = preprocessing.OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    list_to_hot_encode = ['accommadation_type_name', 'charge_option', 'original_payment_type',
                          'original_payment_method']
    list_to_label = ['customer_nationality', 'guest_nationality_country_name', 'language', 'original_payment_currency',
                     'is_user_logged_in', 'is_first_booking', 'hotel_country_code', 'origin_country_code']

    hot_enc.fit(df[list_to_hot_encode])
    label_enc.fit(df[list_to_label])

    df = preprocess_data(df, hot_enc, label_enc)

    label_bool = df.is_canceled
    label_time = df[label_bool.astype(bool)].cancelling_days_from_checkin
    df.drop('is_canceled', axis=1, inplace=True)
    df.drop('cancelling_days_from_checkin', axis=1, inplace=True)
    df_cancel = df[label_bool.astype(bool)]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsRegressor

    clf = RandomForestClassifier(warm_start=True)
    reg = KNeighborsRegressor()

    clf.fit(df, label_bool)
    reg.fit(df_cancel, label_time)
    full_test = load_data("week_6_test_data.csv")

    df_test = drop_data(full_test)
    df_test = preprocess_data(df_test, hot_enc, label_enc)
    cancelers = clf.predict(df_test)

    df_cancel = df_test[cancelers.astype(bool)]
    predication = reg.predict(df_cancel)
    df_cancel["prediction"] = predication
    df_test = df_test.merge(df_cancel, how='left', left_index=True, right_index=True)
    date_cancel = pd.to_datetime(full_test.checkin_date) - pd.to_timedelta(df_test.prediction, unit='D')
    res_1 = pd.to_datetime("2018-12-07") <= date_cancel
    res_2 = date_cancel <= pd.to_datetime("2018-12-13")
    res_all = res_1 & res_2
    pd.DataFrame(res_all.astype(int), columns=["predicted_values"]).to_csv('208385633_315997874_206948911_test_6.csv', index=False)

