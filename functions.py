import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import toeplitz
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
import numpy.polynomial.polynomial as poly
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

def max_drawdown(df_price):
    s = df_price
    idxmax = s.idxmax() #Find where the price is maximum
    s = s.loc[idxmax:] #Get rid of previous values, so that only next minimum is considered

    s.fillna(value=0, inplace=True)
    max_drawdown = np.ptp(s)/s.max() * 100

    return max_drawdown

def volatility(df):
    returns = df['close'] / df['close'].shift(1)
    returns.fillna(method='bfill', inplace=True)
    volatility = returns.rolling(window=24 * 7).std() * np.sqrt(24 * 7)
    return volatility

def correlation_array(df):
    df = df.corr()
    return df

# def buy_hold(df):
#
#     df['bnh_returns'] = df['close'].pct_change()
#
#     bnh_returns = (df['bnh_returns']+1).cumprod()
#
#     bnh_returns.fillna(method='bfill', inplace=True)
#     bnh_returns.fillna(method='ffill', inplace=True)
#
#     return bnh_returns


#How does the next close correlates to current data
#TODO: Add % dataset based training cutoff
def linear_return_OLS(df,percent_cutoff):
    y = df['close'].iloc[:int(len(df)*percent_cutoff)].pct_change().shift(-1)
    y.fillna(method='bfill', inplace=True)
    y.fillna(method='ffill', inplace=True)

    x_t = df.iloc[:int(len(df)*percent_cutoff)]
    #print(x_t)
    x_t.fillna(method='bfill', inplace=True)
    x_t.fillna(method='ffill', inplace=True)
    x_t = sm.add_constant(x_t)
    model = sm.OLS(y, x_t)

    model_fit = model.fit()

    df= sm.add_constant(df)
    prediction = model_fit.predict(df)

    prediction = prediction.shift(1) #This is line it up with real close


    return prediction

def polynomial_fit(df,percent_cutoff):
    y = df['close'].iloc[:int(len(df)*percent_cutoff)].pct_change().shift(-1)
    y_test = df['close'].pct_change().shift(-1)
    y.fillna(method='bfill', inplace=True)
    y.fillna(method='ffill', inplace=True)

    x_t = df.iloc[:int(len(df)*percent_cutoff)]
    x_test = df[int(len(df)*percent_cutoff):]
    #print(x_t)
    x_t.fillna(method='bfill', inplace=True)
    x_t.fillna(method='ffill', inplace=True)
    x_t = np.array(x_t)
    x_test = np.array(x_test)

    poly = PolynomialFeatures(degree=2,include_bias=True)
    X_ = poly.fit_transform(x_t)
    predict_ = poly.fit_transform(x_test)

    clf = linear_model.LinearRegression()
    # preform the actual regression
    clf.fit(X_,y)
    prediction = clf.predict(predict_)


    prediction = pd.Series(prediction,index=y_test.iloc[int(len(df)*percent_cutoff):].index).shift(1) #This is line it up with real close


    return prediction

def random_forest(df,percent_cutoff):
    y = df['close'].iloc[:int(len(df)*percent_cutoff)].pct_change().shift(-1)
    y_test = df['close'].pct_change().shift(-1)
    y.fillna(method='bfill', inplace=True)
    y.fillna(method='ffill', inplace=True)

    x_t = df.iloc[:int(len(df)*percent_cutoff)]
    x_test = df[int(len(df)*percent_cutoff):]
    #print(x_t)
    x_t.fillna(method='bfill', inplace=True)
    x_t.fillna(method='ffill', inplace=True)
    x_t = np.array(x_t)
    x_test = np.array(x_test)

    regr = RandomForestRegressor(n_estimators = 250,max_depth=15,max_features='sqrt',n_jobs=-1,min_samples_split = 2,  min_samples_leaf = 1,random_state=1337)
    # preform the actual regression
    regr.fit(x_t, y)
    prediction = regr.predict(x_test)


    prediction = pd.Series(prediction,index=y_test.iloc[int(len(df)*percent_cutoff):].index).shift(1) #This is line it up with real close

    return prediction

def gradient_boost(df,percent_cutoff):
    y = df['close'].iloc[:int(len(df)*percent_cutoff)].pct_change().shift(-1)
    y_test = df['close'].pct_change().shift(-1)
    y.fillna(method='bfill', inplace=True)
    y.fillna(method='ffill', inplace=True)

    x_t = df.iloc[:int(len(df)*percent_cutoff)]
    x_test = df[int(len(df)*percent_cutoff):]
    #print(x_t)
    x_t.fillna(method='bfill', inplace=True)
    x_t.fillna(method='ffill', inplace=True)
    x_t = np.array(x_t)
    x_test = np.array(x_test)

    reg = HistGradientBoostingRegressor(max_depth=10000,min_samples_leaf=1000,learning_rate=0.001,l2_regularization=0.001,loss='absolute_error',max_iter=2000,max_leaf_nodes=None)
    reg.fit(x_t, y)
    prediction = reg.predict(x_test)


    prediction = pd.Series(prediction,index=y_test.iloc[int(len(df)*percent_cutoff):].index).shift(1) #This is line it up with real close

    return prediction

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
    # print(int(len(df)-1001))
    # print(int(len(df)-1))
    prediction = model_fit.predict(start=df.index[1000],end=df.index[2000],exog=x_test)
    print('prediction ok')

    prediction = prediction
    # print(type(prediction))
    print(prediction)

    # index = prediction.index.strftime("%Y/%m/%d %H")
    # prediction.index = index
    return prediction

def advanced_vector_backtest(y_true, y_pred, balance, lag, bet_amount):
    assert type(y_pred) == type(y_true),'Backtesting Requires preds to be series'

    balance_tab = np.full([y_true.shape[0], 1], balance,dtype='float32')

    for tick in range(0, y_pred.shape[0]):
        #print('DEBUG - avb start loop')
        if tick < y_pred.shape[0]:
            #print('DEBUG - avb second loop')
            if tick < lag:  # position open only
                #print('DEBUG - avb position open loop')
                balance_tab[tick + 1] = balance_tab[tick] #Due to implementation, we don't know preds here, so no bet
            elif tick <= y_pred.shape[0]:  # trading
                #print('DEBUG - avb trading loop')
                #print(tick)
                profit = bet_amount * (1. +  y_true[tick] * np.sign(y_pred[tick]))
                balance_tab[tick] = balance_tab[tick - 1] - bet_amount + profit
        # else:  # we don't know the future returns
        #     balance_tab[tick] = balance_tab[tick - 1] #Due to implementation, we always know final outcome
    return pd.DataFrame(balance_tab)

def buy_hold(y_true, balance):
    lag = 1
    balance_tab = np.full([y_true.shape[0], 1], balance,dtype='float32')
    for tick in range(0, y_true.shape[0]):
        if tick + lag < y_true.shape[0]:  # Prevent open positions before the lag end time
            if tick < lag:  # position open only
                balance_tab[tick + 1] = balance_tab[tick]
            elif tick < y_true.shape[0] - lag:  # trading
                balance_tab[tick + 1] = balance_tab[tick] * (1. + y_true[tick])
        else:  # we don't know the future do we
            balance_tab[tick] = balance_tab[tick - 1]
    return pd.DataFrame(balance_tab)
