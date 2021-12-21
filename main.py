# import torch
from crypto import *
from data import *
from plot import *
from lms import *
from ARIMA import *
from neuralnetwork import *
import pandas as pd
import numpy as np
from os.path import exists

# data collection settings

new_data = True

cryptocurrency = 'ETH'
target_currency = 'USD'
data_frequency = 'hour'
samples_per_request = 2000
exchange = 'CCCAGG'

# 1 hour = 3600 seconds
# 1 day = 86400 seconds
# 1 week = 604800 seconds
# 1 month = 2592000 seconds
# 1 year = 31536000 seconds

turn_back_time = 31536000 # num seconds in the past
percent_of_data = 0.5 # percent of data to use for testing (1 - testing = training)

window_size = 24

divisions = 1

tests = ["LSTM"]

def testing(X_train, y_train, X_test, df= None, title=None):

    y_train_preds, y_test_preds, result = None, None, None
    if title == "LMS":
        return get_least_means_squares(X_train, y_train, X_test)
    elif title == "ARIMA":
        return get_ARIMA(X_train, y_train, X_test)
    elif title == "LSTM":
        df = neuralnet(df, title, 1 - percent_of_data)
        return None, df['prediction'], None
    elif title == "RNN":
        df = neuralnet(df, title, 1 - percent_of_data)
        return None, df['prediction'], None
    else:
        return None


if __name__ == '__main__':

    # get data
    if os.path.exists('out.csv') and not new_data:
        data = pd.read_csv('out.csv', parse_dates=[0], infer_datetime_format=True)
        data.set_index('time', inplace=True)
    else:
        data = get_full_hist_data(cryptocurrency, target_currency, '', data_frequency, samples_per_request, 1, exchange, turn_back_time).sort_index()
        data = data.drop('conversionType', 1)
    
    # compression_opts = dict(method='zip',
    #                         archive_name='out.csv')  
    # data.to_csv('out.zip', index=True,
    #           compression=compression_opts)

    df_set = np.array_split(data, divisions)

    count = 1

    for df in df_set:
        for title in tests:
            nan_value = float("NaN")
            df_n = df.copy()
            df_n.replace("", nan_value, inplace=True)
            df_n.dropna(1, inplace=True)

            # serties1 = df_n[title]
            # serties1.index = df_n.index
            print ("===========================================================")
            print ("Testing: " + title + " on " + str(count))
            print ("===========================================================")
            X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df_n, 'close', percent_of_data)
            # X_train, y_train, X_test, y_test, train, test = get_test_train_sequence(series, window_size, percent_of_data)
            # X_train_LSTM, y_train_preds, X_test_LSTM, y_test_preds, train_LSTM, test_LSTM = get_test_train_sequence(serties1, window_size, percent_of_data)

            y_train_preds, y_test_preds, result = testing(X_train, y_train, X_test, df_n, title)
            
            mse = mean_squared_error(y_test, y_test_preds)
            print ("MSE: " + str(mse))
            mae = mean_absolute_error(y_test, y_test_preds)
            print ("MAE: " + str(mae))

            print ("===========================================================")

            # print(result.summary())
            print ("===========================================================")
            plot_data(y_train, y_test, y_train_preds, y_test_preds, (title + " " + str(count)))

        count += 1

