from functools import wraps

import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

from functions import advanced_vector_backtest, generate_hover

'''Module containing various classes and decorates that are used in callbacks.py'''


'''Look I know we technically could just done this with a function, but I wanted a decorator, ok?
This decorator is used to generate graphs
Use it on prediction functions
Performs vector backtest, and generates a hover text'''
def generate_return_plot(func):
    @wraps(func)
    def wrapper(self,*args,**kwargs):
                df = self.df
                training_percent_cutoff = self.percent_cutoff
                init_balance = self.init_balance
                bet_amount = self.bet_amount
                fig = self.fig

                pred = func(self,*args,**kwargs)

                returns = advanced_vector_backtest(
                    df['close'].pct_change().iloc[int(len(df) * training_percent_cutoff):], pred, init_balance, lag=1,
                    bet_amount=bet_amount)

                hover_lin = generate_hover(df['close'].pct_change().iloc[int(len(df) * training_percent_cutoff):],
                                           pred)

                trace_one_hour = go.Figure(go.Scattergl(
                    x=df.iloc[int(len(df) * training_percent_cutoff):].index, connectgaps=True,
                    y=returns, name=f'1H {func.__name__} Strategy Return', hovertemplate=hover_lin))

                fig.add_trace(trace_one_hour.data[0])


    return wrapper

'''Regression class for generating predictions
Prepares the data on init and uses methods to output predictions
On instantiation:
Accepts: Dataframe with OHLCV information
Data percent cutoff for training (Float)
Initial balance (Float)
bet size (float or int)
Figure object (Graph)
lag size (By default = 1)'''

class Regression:

    def __init__(self,df,percent_cutoff,init_balance,bet_amount,fig,lag):
        self.df = df
        self.percent_cutoff = percent_cutoff
        self.y = df['close'].iloc[:int(len(df)*percent_cutoff)].pct_change().shift(-1).fillna(method='bfill').fillna(method='ffill')
        self.x_t = df.iloc[:int(len(df)*percent_cutoff)].fillna(method='bfill').fillna(method='ffill')
        self.x_test = df[int(len(df) * percent_cutoff):]
        self.y_test = df['close'].pct_change().shift(-1)
        self.init_balance = init_balance
        self.bet_amount = bet_amount
        self.fig = fig
        self.lag = lag

    @generate_return_plot
    def OLS(self):
        x_t = sm.add_constant(self.x_t)
        model = sm.OLS(self.y, x_t)

        model_fit = model.fit()


        x_test = sm.add_constant(self.x_test)
        prediction = model_fit.predict(x_test)

        prediction = prediction.shift(1)

        return prediction

    @generate_return_plot
    def polynomial(self):
        poly = PolynomialFeatures(degree=2, include_bias=True)
        X_ = poly.fit_transform(self.x_t)
        predict_ = poly.fit_transform(self.x_test)

        clf = linear_model.LinearRegression()
        # preform the actual regression
        clf.fit(X_, self.y)
        prediction = clf.predict(predict_)

        prediction = pd.Series(prediction,index=self.y_test.iloc[int(len(self.df)*self.percent_cutoff):].index).shift(1)

        return prediction

    @generate_return_plot
    def random_forest(self):
        regr = RandomForestRegressor(n_estimators=250, max_depth=15, max_features='sqrt', n_jobs=-1,
                                     min_samples_split=2, min_samples_leaf=1, random_state=1337)
        # preform the actual regression
        regr.fit(self.x_t, self.y)
        prediction = regr.predict(self.x_test)

        prediction = pd.Series(prediction, index=self.y_test.iloc[int(len(self.df) * self.percent_cutoff):].index).shift(
            1)  # This is line it up with real close

        return prediction

    @generate_return_plot
    def gradient_boost(self):
        reg = HistGradientBoostingRegressor(max_depth=10000, min_samples_leaf=1000, learning_rate=0.001,
                                            l2_regularization=0.001, loss='absolute_error', max_iter=2000,
                                            max_leaf_nodes=None)
        reg.fit(self.x_t, self.y)
        prediction = reg.predict(self.x_test)

        prediction = pd.Series(prediction, index=self.y_test.iloc[int(len(self.df) * self.percent_cutoff):].index).shift(
            1)  # This is line it up with real close

        return prediction





