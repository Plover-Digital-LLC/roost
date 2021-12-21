import statsmodels.api as sm

def get_least_means_squares(X_train, y_train, X_test):
    # Fit regression model
    model = sm.GLS(y_train, X_train)
    result = model.fit()
    # Make predictions
    y_train_preds = result.predict(X_train)
    y_test_preds = result.predict(X_test)
    return y_train_preds, y_test_preds, result





