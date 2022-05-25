import pandas as pd
import os
import pathlib
from pathlib import Path

Path = Path.cwd() / 'Data'

ticker = 'btcusd'
columns = ['time','open']
def get_data(ticker,columns):
    if ticker== None:
        ticker='btcusd'
    df = pd.read_csv(f'{Path}/{ticker}.csv',usecols=columns)
    df['time'] = pd.to_datetime(df['time'],format='%Y/%m/%d %H')
    df.set_index('time',inplace=True)
    # if time:
    #     df = df[df['time'].dt.year == time]


    return df

# df = get_data(ticker,columns)
# for x in set(df['time'].dt.year):
#     print(x)


