import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.arima.model import ARIMA

'''Module containing stand-alone functions to use in callbacks'''

'''Calculates Max Drawdown of returns
Accepts: Dataframe or Series with prices of the asset
Returns: % value of max drawdown'''
def max_drawdown(df_price):
    idxmax = df_price.idxmax() #Find where the price is maximum
    s = df_price.loc[idxmax:] #Get rid of previous values, so that only next minimum is considered

    s.fillna(value=0, inplace=True)
    max_drawdown = (s.min() - s.max())/s.max() * 100


    return max_drawdown

'Calculates volatility of the asset' \
'Uses 1 week interval rolling window' \
'Accepts: Dataframe with a closing column containing real prices' \
'Returns: Volatility values'
def volatility(df):
    returns = df['close'] / df['close'].shift(1)
    returns.fillna(method='bfill', inplace=True)
    volatility = returns.rolling(window=24 * 7).std() * np.sqrt(24 * 7)
    return volatility

'Calculates correlation features' \
'Redundant'
def correlation_array(df):
    df = df.corr()
    return df

'Left-over SARIMA code, in case it needs to be implemented later' \
'SARIMA takes too long to predict, and requires hyper-parameter search to work well' \
'Accepts: Dataframe with closing prices' \
'Returns: Prediction Array'
def linear_return_SARIMA(df):
    y_training = df['close'].iloc[:1000].shift(-1)
    y_training.fillna(method='bfill', inplace=True)
    y_training.fillna(method='ffill', inplace=True)

    x_training = df.iloc[:1000]
    x_test = df[1000:2000]
    x_training.fillna(method='bfill', inplace=True)
    x_training.fillna(method='ffill', inplace=True)
    print('data ok')
    model = ARIMA(y_training.asfreq('1h'),exog=x_training.asfreq('1h').fillna(method='bfill',inplace=True), order=(2, 1, 2),seasonal_order=(2,1,2,24))
    model_fit = model.fit()
    print('model fit')
    prediction = model_fit.predict(start=df.index[1000],end=df.index[2000],exog=x_test)
    print('prediction ok')

    prediction = prediction

    return prediction

'''A vector backtest function
This implementation assumes real values are unknown for lag amount of samples
This implementation is not a good real-life representation, designed only for basic backtest
Accepts: Y_true,Y_pred Pandas Series, starting balance, lag interval, bet amount
Returns: Pandas Series array of simulated balance'''
def advanced_vector_backtest(y_true, y_pred, balance, lag, bet_amount):
    assert type(y_pred) == type(y_true),'Backtesting Requires preds to be series'

    balance_tab = np.full([y_true.shape[0],], balance,dtype='float32')

    for tick in range(0, y_pred.shape[0]):
        if tick < y_pred.shape[0]:
            if tick < lag:  # position open only
                balance_tab[tick + 1] = balance_tab[tick] #Due to implementation, we don't know preds here, so no bet
            elif tick <= y_pred.shape[0]:  # trading
                profit = bet_amount * (1. + y_true[tick] * np.sign(y_pred[tick]))
                balance_tab[tick] = balance_tab[tick - 1] - bet_amount + profit
    return pd.Series(balance_tab)
'''A buy and hold balance simulation
Accepts: True returns as a Pandas Series, initial balance
Returns: Pandas Series array of balance tab simulation'''
def buy_hold(y_true, balance):
    lag = 1
    balance_tab = np.full([y_true.shape[0],], balance,dtype='float32')
    for tick in range(0, y_true.shape[0]):
        if tick + lag < y_true.shape[0]:  # Prevent open positions before the lag end time
            if tick < lag:  # position open only
                balance_tab[tick + 1] = balance_tab[tick]
            elif tick < y_true.shape[0] - lag:  # trading
                balance_tab[tick + 1] = balance_tab[tick] * (1. + y_true[tick])
        else:  # we don't know the future do we
            balance_tab[tick] = balance_tab[tick - 1]
    return pd.Series(balance_tab)

'''Computes information coefficient between real returns and predicted returns
Accepts: Pandas Series or numpy array for real and predicted returns
Returns: Correlation coefficient and rho value'''
def information_coefficient(y_true, y_pred):
    coef_r, p_r = spearmanr(y_true, y_pred)

    return coef_r, p_r

'''Generates a graph hover-over with information coefficient and max dradown values
Accepts: Pandas Series for true and predicted returns
Returns: String to use as a hover'''
def generate_hover(y_true,y_pred):
    coef_r, _ = spearmanr(y_true,
                          y_pred,nan_policy='omit')
    print('Type of y_pred in generate hover',type(y_pred))

    max_draw = max_drawdown(y_pred)


    hover_lin = f'Spearmann Correlation: {coef_r:.2f} <br>' + f'Max_dradown: {max_draw:.2f}'

    return hover_lin