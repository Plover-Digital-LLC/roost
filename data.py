import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


def data_to_dataframe(data):
    #data from json is in array of dictionaries
    df = pd.DataFrame.from_dict(data)
    
    # time is stored as an epoch, we need normal dates
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df

def UniqueResults(dataframe):
    df = pd.DataFrame()
    for col in dataframe:
        S=pd.Series(dataframe[col].unique())
        df[col]=S.values
    return df

def split_sequence(sequence, window_size):
    X = []
    y = []
    # for all indexes
    for i in range(len(sequence)):
        end_idx = i + window_size
        # exit condition
        if end_idx > len(sequence) - 1:
            break
        # get X and Y values
        seq_x, seq_y = sequence[i:end_idx], sequence[end_idx]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def get_test_train_sequence(df, window_size, test_size):
    # split into train and test
    train, test = train_test_split(df, test_size=test_size)
    # generate lagged sequence
    X_train, y_train = split_sequence(train, window_size)
    X_test, y_test = split_sequence(test, window_size)
    return X_train, y_train, X_test, y_test, train, test

def generate_time_lags(df, n_lags):
    df_generated = df.copy()
    for i in range(n_lags):
        df_generated['close' + '_lag_' + str(i + 1)] = df['close'].shift(i + 1)
    return df_generated.copy()


def onehot_encode_pd(df, col_name):
    df_n = df.copy()
    dummies = pd.get_dummies(df_n, columns=col_name)
    return pd.concat([df, dummies], axis=1).drop(columns=col_name)

def generate_cyclical_features(df, col_name, period, start_num=0):
    series = df[col_name]
    print(series)
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2*np.pi*(df[col_name]-start_num)/period),
        f'cos_{col_name}' : lambda x: np.cos(2*np.pi*(df[col_name]-start_num)/period)    
             }
        
    return df.assign(**kwargs).drop(columns=[col_name])

def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

def train_val_test_split(df, target_col, test_ratio):
    val_ratio = test_ratio / (1 - test_ratio)
    X, y = feature_label_split(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test