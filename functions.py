import numpy as np
import pandas as pd

def max_drawdown(df_price):
    s = df_price
    idxmax = s.idxmax() #Find where the price is maximum
    s = s.loc[idxmax:] #Get rid of previous values, so that only next minimum is considered

    s.fillna(value=0, inplace=True)
    max_drawdown = np.ptp(s)/s.max() * 100

    return max_drawdown