import numpy as np


def max_drawdown(df_price):
    s = df_price
    idxmax = s.idxmax() #Find where the price is maximum
    s = s.loc[idxmax:] #Get rid of previous values, so that only next minimum is considered

    s.fillna(value=0, inplace=True)
    max_drawdown = np.ptp(s)/s.max() * 100

    return max_drawdown

def volatility(df):
    returns = np.log(df['close'] / df['close'].shift(1))
    returns.fillna(method='bfill', inplace=True)
    volatility = returns.rolling(window=24 * 7).std() * np.sqrt(24 * 7)
    return volatility