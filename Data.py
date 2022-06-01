import pandas as pd
import os
import pathlib
from pathlib import Path
import ta

pd.options.mode.chained_assignment = None #Disable chained warning
Path = Path.cwd() / 'Data'

ticker = 'btcusd'
columns = ['time','open']
lags = []
def get_data(ticker,columns):
    if ticker== None:
        ticker='btcusd'
    df = pd.read_csv(f'{Path}/{ticker}.csv',usecols=columns)
    df['time'] = pd.to_datetime(df['time'],format='%Y/%m/%d %H')
    df.set_index('time',inplace=True)
    # if time:
    #     df = df[df['time'].dt.year == time]


    return df

def add_ta(df):
    df = ta.add_all_ta_features(df, open=f"open", high=f"high", low=f"low", close=f"close", volume=f"volume",
                                fillna=True, vectorized=True)  # Add all the ta!

    return df

#lags = [336,168,120,48,24]
def lagged_returns(df, lags):
    for lag in lags:
        df[f'return_{lag}h'] = df['close'].pct_change(lag, axis=0)

    for t in lags:
        df[f'target_{t}h'] = df[f'return_{t}h'].shift(-t)
        df = df[[f'target_{t}h'] + [col for col in df.columns if
                                    col != f'target_{t}h']]  # Puts the return columns up front, for easier grabbing later

    return df

