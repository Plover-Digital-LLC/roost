import pmdarima as pm
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

def get_ARIMA(X_train, y_train, X_test):
    model = pm.auto_arima(y_train, 
                      start_p=1, # lag order starting value
                      start_q=1, # moving average order starting value
                      test='adf', #ADF test to decide the differencing order
                      max_p=3, # maximum lag order
                      max_q=3, # maximum moving average order
                      m=1, # seasonal frequency
                      d=None, # None so that the algorithm can chose the differencing order depending on the test
                      seasonal=False, 
                      start_P=0, 
                      D=0, # enforcing the seasonal frequencing with a positive seasonal difference value
                      trace=True,
                      suppress_warnings=True, 
                      stepwise=True)
    y_train_preds = model.predict(n_periods=len(X_train))
    y_test_preds = model.predict(n_periods=len(X_test))
    return y_train_preds, y_test_preds, model


    # # Fit regression model
    # model = sm.OLS(y_train, X_train)
    # result = model.fit()
    # # Make predictions
    # y_train_preds = result.predict(X_train)
    # y_test_preds = result.predict(X_test)
    # return y_train_preds, y_test_preds, result