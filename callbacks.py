from pathlib import Path

import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import callback
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
from scipy.stats import linregress

from Classes_Decorators import Regression
from Data import get_data
from Data import lagged_returns as create_lagged_returns, add_ta
from functions import buy_hold, generate_hover
from functions import max_drawdown as draw
from functions import volatility as vol
from layouts import get_page_1_layout, get_page_2_layout, get_page_3_layout

'''This module is for outlying the DASH callbacks'''

"""Get initial data and load page layouts"""
df = get_data('btcusd', ['time', 'open', 'high', 'low', 'close'])
data_path = Path.cwd() / 'Data'

page_1_layout = get_page_1_layout(df, data_path)
page_2_layout = get_page_2_layout(df, data_path)
page_3_layout = get_page_3_layout(df, data_path)

'''Get_callbacks function is used to load callback into the app
It houses other functions that act as a callback
Requires a Dash app as input'''


def get_callbacks(app):
    # Callback to load correct pages
    @callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/':
            return page_1_layout
        elif pathname == '/tam':
            return page_2_layout
        elif pathname == '/bcktst':
            return page_3_layout

    # Callbacks for the first page
    '''This callback controls the graph on the first page
    Uses year and ticker dropdowns, as well as check boxes for missing data and volatility switch
    Returns graph and 2 textboxes for stock performance analysis and missing data'''

    @callback(
        Output('close-graph', 'figure'),
        Output('textarea_missingno', 'value'),
        Output('textarea_performance', 'value'),
        Input('year_dropdown', 'value'),
        Input('ticker_dropdown', 'value'),
        Input('check-box', 'value'),
        Input('vol-check', 'value'))
    def update_page1_figure(year_value, ticker_value, check_box, vol_check):
        df = get_data(ticker_value, ['time', 'open', 'high', 'low', 'close', 'volume'])

        df = df.loc[str(year_value)]

        text_gaps = 'Enable show gaps to see statistics about missing data'

        if check_box:
            x = df.index
            full_ind = pd.date_range(start=x[0], end=x[-1],
                                     freq='h')  # Constructs a full date range, with df index as starting and ending points
            miss = full_ind.difference(x)  # Compares existing dataframe with the full index built above
            text_gaps = f'Number of gaps in the data for ticker {ticker_value} and year {year_value} is {len(miss)}.    {round(len(miss) / len(full_ind) * 100, 1)} % of data is missing!'
        else:
            x = df.index.strftime(
                "%Y/%m/%d %H")  # If checkbox is not enabled, we hard convert pandas index to strftime for Dash to not plot missing dates

        tick_index = df.index.strftime("%Y/%m/%d").values[1::730]  # Designate tick interval every 730 hours

        fig = make_subplots(rows=2, cols=1, print_grid=False, shared_xaxes=True, vertical_spacing=0.009,
                            horizontal_spacing=0.009, specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                            row_heights=[0.5, 0.1])  # Make 2 graphs in a row, with a secondary y axis

        # Create candlestick graph w/ volume bars subplot
        fig.add_trace(go.Candlestick(
            x=x,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            yaxis='y1', xaxis='x1'
            , name=f'{ticker_value} OHLCV'), 1, 1)
        fig.update_xaxes(gridcolor='Gray')
        fig.update_yaxes(gridcolor='Gray')

        fig.add_trace(go.Bar(x=x, y=df['volume'], yaxis='y3', name='Volume', xaxis='x1'), 2, 1)

        # Calculating and displaying the volatility of the ticker
        if vol_check:
            volatility = vol(df)
            fig.add_trace(
                go.Scatter(line=dict(color='royalblue', width=2), x=x, y=volatility, type='scatter', name='volatility',
                           xaxis='x1'), 1,
                1,
                secondary_y=True)

        #Making the graphs look pretty
        fig.update_layout(paper_bgcolor="rgb(34,34,34)", plot_bgcolor='rgb(48,48,48)',
                          font=dict(color="#F39A1B"),
                          xaxis_rangeslider_visible=False, yaxis_showgrid=False, yaxis2_showgrid=False, xaxis=dict(
                tickangle=0, ticks='outside', tickformat="%Y/%m/%d", showgrid=False, rangeslider_visible=False,
                showticklabels=False, automargin=False), xaxis2=dict(
                tickangle=0, type='category', ticks='outside', tickformat="%Y/%m/%d", ticktext=tick_index,
                showgrid=False, rangeslider_visible=False, showticklabels=True, automargin=False, tickmode='array',
                tickvals=[x for x in range(1, len(df), 720)]), margin=dict(l=35, r=0, b=30, t=0), legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0
            ))

        # Calculating Drawdown
        max_drawdown = draw(df['close'])

        text_perf = f'Average return in {year_value} for {ticker_value} is {(df.loc[x[-1], "close"] - df.loc[x[0], "close"]) * 100 / df.loc[x[0], "close"]} % \n' \
                    f'Max drawdown is {max_drawdown} %'

        return fig, text_gaps, text_perf

    '''Callback to update year dropdown to match avaliable years of selected ticker'''
    @callback(
        Output('year_dropdown', 'options'),
        Input('ticker_dropdown', 'value'))
    def update_page1_year_dropdown(ticker_value):
        df = get_data(ticker_value, ['time', 'open', 'high', 'low', 'close'])
        years = list(sorted(set(df.index.year.astype(int))))
        return years

    # Callbacks for second page
    '''This callback creates 2 graphs that display correlation matrix and correlation of 2 selected assets'''
    @callback(
        Output('corr-mat', 'figure'),
        Output('corr-element', 'figure'),
        Input('ticker_dropdown_page2', 'value'),
        Input('corr-mat', 'clickData'),
        Input('poly-slider', 'value'))
    def update_page2(ticker_value, click_data, poly_slider):
        lags = [336, 168, 120, 48, 24]

        #Create our data, along with lagged returns and technical analysis
        df = get_data(ticker_value, ['time', 'open', 'high', 'low', 'close', 'volume'])
        df = add_ta(df)
        df = create_lagged_returns(df, lags)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        #Generate initial asset correlation graph
        fig2 = px.scatter(data_frame=df, x='volume', y=f'target_{lags[-1]}h', trendline="ols",
                          trendline_color_override="orange")

        #On click, generates polynomial fit for graph #2
        if click_data:

            x_click = click_data['points'][0]['x']
            y_click = click_data['points'][0]['y']
            x = df[str(x_click)]
            y = df[str(y_click)]

            # Polynomial fit
            z = poly.polyfit(df[str(x_click)], df[str(y_click)], poly_slider)

            # calculate new x's and y's
            x_new = np.linspace(x.min(), x.max(), len(x))
            y_new = poly.polyval(x_new, z)
            slope, intercept, r_value, p_value, std_err = linregress(x_new, y_new)

            #This can be used to generate the equation of the polynomial fit, it's too long to be displayed on hover
            def poly_formula(z):
                # z = z[::-1]
                for degree, value in enumerate(z):
                    if value > 0:
                        formula = f'+{value}*x^{degree}'
                        yield formula
                    elif value < 0:
                        formula = f'{value}*x^{degree}'
                        yield formula
                    elif value == 0:
                        formula = ''
                        yield formula

            #Polynomial fit hover text
            poly_hover = f'<b>Polynomial fit</b>' + f'<br>R^2 = {1 - r_value ** 2}'

            # Draw polynomial fit first
            fig2 = go.Figure(go.Scattergl(
                x=x_new,
                y=y_new, hovertemplate=poly_hover, line=dict(color='teal'), line_color='teal', showlegend=False
            ))
            fig2.data[0].line.color = "teal"

            fig2 = px.scatter(data_frame=df, x=x_click, y=y_click, trendline="ols", trendline_color_override="orange")

            # Draw the scatter plot/ linear trendline

            trace_graph = go.Figure(go.Scattergl(
                x=x_new,
                y=y_new, hovertemplate=poly_hover, line=dict(color='teal'), line_color='teal', showlegend=False
            ))

            fig2.add_trace(trace_graph.data[0])
            fig2.data = (
                fig2.data[0], fig2.data[1], fig2.data[2])  # Set the display order, in lieue of having proper tools

        #Calculate the correlation matrix
        df_corr = df.corr()
        df_corr = df_corr.iloc[:5, :]

        #Create correlation matrix visual graph
        fig = (px.imshow(df_corr, color_continuous_scale='Viridis', aspect="auto"))

        fig.update_layout(paper_bgcolor="rgb(34,34,34)", plot_bgcolor='rgb(48,48,48)',
                          font=dict(color="#F39A1B"), margin=dict(l=0, r=0, b=0, t=0))
        fig2.update_layout(paper_bgcolor="rgb(34,34,34)", plot_bgcolor='rgb(48,48,48)',
                           font=dict(color="#F39A1B"), margin=dict(l=30, r=0, b=100, t=0))

        return fig, fig2

    # Page 3 callbacks
    '''Callback for actions on page 3
    Allows to generate backtesting predictions and plot them
    Generates hover information about the backtest results on hover'''
    @callback(
        Output('backtest_graph', 'figure'),
        Input('ticker_dropdown3', 'value'),
        Input('init_balance', 'value'),
        Input('bet_amount', 'value'),
        Input('training_percent3', 'value'),
        Input('strategydrp3', 'value')
    )
    def update_page3(ticker_value, init_balance, bet_amount, training_percent_cutoff, strategy):
        #Prepare data
        df = get_data(ticker_value, ['time', 'open', 'high', 'low', 'close', 'volume'])
        df = df.loc[(df.index >= '2018-01-01')]
        df = add_ta(df)
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

        #Generate empty plot, since we don't have a strategy selected
        fig = go.Figure(go.Scattergl(
            x=df.index,
            y=[], name='Strategy Returns', connectgaps=False))

        #Initialize regression object that will perform predictions and generate graph information
        Regressor = Regression(df, training_percent_cutoff, init_balance, bet_amount, fig, 1)

        #Buy and hold is an exception, due to not using regression class
        if 'Buy And Hold' in strategy:
            bnh_returns = buy_hold(df['close'].pct_change(), init_balance)

            hover_poly_bns = generate_hover(df['close'].pct_change().iloc[int(len(df) * training_percent_cutoff):],
                                            df['close'].pct_change().iloc[int(len(df) * training_percent_cutoff):])

            bnh_trace = go.Figure(go.Scattergl(
                x=df.iloc[int(len(df) * training_percent_cutoff):].index,
                y=bnh_returns, name='Buy and Hold Returns', connectgaps=True, hovertemplate=hover_poly_bns))

            fig.add_trace(bnh_trace.data[0])

        if 'Linear' in strategy:
            Regressor.OLS()

        if 'Polynomial' in strategy:
            Regressor.polynomial()

        if 'Random Forest' in strategy:
            Regressor.random_forest()

        if 'Gradient Boost' in strategy:
            Regressor.gradient_boost()

        # TODO: Add spearmann corr coeff hover on
        # TODO: SARIMA (Auto-tuning), Random Forest, Polynomial Fit, Unobserved Components Model,MANOVA,VARMAX ; CUSTOM LOSS ON GRADIENT BOOST;Neighrest neighbour

        #Adds a vertical line indicating training cutoff, and adds an annotation to it
        fig.add_vline(
            x=df.index[int(len(df) * training_percent_cutoff)], line_width=3, line_dash="dash",
            line_color="green")
        fig.add_annotation(x=df.index[int(len(df) * training_percent_cutoff)], yref="y domain", y=1,
                           text="Training Cutoff")  # Can't add annotation above, coz Dash is great ~_~


        return fig
