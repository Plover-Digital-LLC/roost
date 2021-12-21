import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime

def data_to_dataframe(data):
    #data from json is in array of dictionaries
    df = pd.DataFrame.from_dict(data)
    
    # time is stored as an epoch, we need normal dates
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    return df

def get_current_data(from_sym='ETH', to_sym='USD', exchange='CCCAGG'):
    url = 'https://min-api.cryptocompare.com/data/price'    
    
    parameters = {'fsym': from_sym,
                  'tsyms': to_sym }
    
    if exchange:
        print('exchange: ', exchange)
        parameters['e'] = exchange
        
    # response comes as json
    response = requests.get(url, params=parameters)   
    data = response.json()
    
    return data  

def get_hist_data(from_sym='ETH', to_sym='USD', to_ts='', timeframe = 'hour', limit=2000, aggregation=1, exchange=''):
    
    url = 'https://min-api.cryptocompare.com/data/v2/histo'
    url += timeframe
    
    parameters = {'fsym': from_sym,
                  'tsym': to_sym,
                  'tots': to_ts,
                  'limit': limit,
                  'aggregate': aggregation,
                  'api_key': os.environ.get("KEY")}
    if exchange:
        print('exchange: ', exchange)
        parameters['e'] = exchange    
    
    print('baseurl: ', url) 
    print('timeframe: ', timeframe)
    print('parameters: ', parameters)
    
    # response comes as json
    response = requests.get(url, params=parameters)   
    
    data = response.json()['Data']['Data'] 
    
    return data

def get_full_hist_data(from_sym='ETH', to_sym='USD', to_ts='', timeframe = 'hour', limit=2000, aggregation=1, exchange='', endTime=0):
    url = 'https://min-api.cryptocompare.com/data/v2/histo'
    url += timeframe
    
    parameters = {'fsym': from_sym,
                  'tsym': to_sym,
                  'tots': to_ts,
                  'limit': limit,
                  'aggregate': aggregation,
                  'api_key': os.environ.get("KEY")}
    if exchange:
        print('exchange: ', exchange)
        parameters['e'] = exchange    
    
    print('baseurl: ', url) 
    print('timeframe: ', timeframe)
    print('parameters: ', parameters)
    
    # response comes as json
    response = requests.get(url, params=parameters)   
    
    data = response.json()['Data']['Data']

    endTime = response.json()['Data']['TimeTo'] - endTime;
    df = data_to_dataframe(data)

    while response.json()['Response'] == 'Success' and response.json()['Data']['TimeTo'] > endTime:
            
        parameters['toTs'] = response.json()['Data']['TimeFrom'] - 3600
        response = requests.get(url, params=parameters)
        temp = data_to_dataframe(response.json()['Data']['Data'])
        
        df = df.append(temp)

    return df