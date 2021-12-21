import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('expand_frame_repr', False)

def plot_data(y_train, y_test, y_train_preds, y_test_preds, train_index, test_index, model_name):

    # fig, ax = plt.subplots()
    # ax.plot(train_index, y_train, label='Train values')
    # ax.plot(test_index, y_test, label='Test values')
    # ax.plot(train_index, y_train_preds, label='Train predictions')
    # ax.plot(test_index, y_test_preds, label='Test predictions')
    # ax.set_xlabel('Time')
    # ax.set_ylabel('Price')
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_xticks(range(0, len(train_index), 1000))
    # ax.legend()
    # ax.set_title(model_name)
    
    plt.plot(pd.Series(y_train, index=train_index), label='train values')
    plt.plot(pd.Series(y_test, index=test_index), label='test values')
    plt.plot(pd.Series(y_train_preds, index=train_index), label='train predictions')
    plt.plot(pd.Series(y_test_preds, index=test_index), label='test predictions')
    plt.title(model_name)
    plt.legend()
    plt.show()
    
    return None

def plot_data(y_train, y_test, y_train_preds, y_test_preds, model_name):
    
    plt.plot(y_train, label='train values')
    plt.plot(y_test, label='test values')
    if(y_train_preds != None):
        plt.plot(y_train_preds, label='train predictions')
    plt.plot(y_test_preds, label='test predictions')
    plt.title(model_name)
    plt.legend()
    plt.show()
    
    return None