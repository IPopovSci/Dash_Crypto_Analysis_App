from pathlib import Path

import pandas as pd
import ta

'''Module that loads in and works with data'''

pd.options.mode.chained_assignment = None #Disable chained warning
Path = Path.cwd() / 'Data' #Grabs data path based on current working directory

ticker = 'btcusd' #Default Ticker
columns = ['time','open']
lags = []

'''Loads in data from Path location
accepts: ticker (string)
column names to load [list]
returns: Dataframe with data from ticker/columns'''
def get_data(ticker,columns):
    if ticker== None:
        ticker='btcusd'
    df = pd.read_csv(f'{Path}/{ticker}.csv',usecols=columns)
    df['time'] = pd.to_datetime(df['time'],format='%Y/%m/%d %H')
    df.set_index('time',inplace=True)

    return df

'''Adds technical analysis to the dataframe
Accepts: Dataframe w/ OHLCV information
Returns: Dataframe'''
def add_ta(df):
    df = ta.add_all_ta_features(df, open=f"open", high=f"high", low=f"low", close=f"close", volume=f"volume",
                                fillna=True, vectorized=True)  # Add all the ta!

    return df

'''Creates lagged returns
Accepts: Dataframe with OHLCV information
List of lags to create returns with
Returns: Dataframe w/ previous data + lagged returns'''
def lagged_returns(df, lags):
    for lag in lags:
        df[f'return_{lag}h'] = df['close'].pct_change(lag, axis=0)

    for t in lags:
        df[f'target_{t}h'] = df[f'return_{t}h'].shift(-t)
        df = df[[f'target_{t}h'] + [col for col in df.columns if
                                    col != f'target_{t}h']]  # Puts the return columns up front, for easier grabbing later

    return df

