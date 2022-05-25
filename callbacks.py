from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import callback
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

from Data import get_data
from functions import max_drawdown as draw
from functions import volatility as vol
from layouts import get_page_1_layout, get_page_2_layout

df = get_data('btcusd', ['time', 'open', 'high', 'low', 'close'])
data_path = Path.cwd() / 'Data'

page_1_layout = get_page_1_layout(df,data_path)
page_2_layout = get_page_2_layout()

def get_callbacks(app):
    @callback(
        Output('close-graph', 'figure'),
        Output('textarea_missingno', 'value'),
        Output('textarea_performance', 'value'),
        Input('year_dropdown', 'value'),
        Input('ticker_dropdown', 'value'),
        Input('check-box', 'value'),
        Input('vol-check', 'value'))
    def update_figure(year_value, ticker_value, check_box, vol_check):
        df = get_data(ticker_value, ['time', 'open', 'high', 'low', 'close', 'volume'])

        df = df.loc[str(year_value)]

        text_gaps = 'No gaps, no problems'
        text_perf = ''

        if check_box:
            x = df.index
            full_ind = pd.date_range(start=x[0], end=x[-1], freq='h')
            miss = full_ind.difference(x)
            text_gaps = f'Number of gaps in the data for ticker {ticker_value} and year {year_value} is {len(miss)}.    {round(len(miss) / len(full_ind) * 100, 1)} % of data is missing!'
        else:
            x = df.index.strftime("%Y/%m/%d %H")
            # x = df.index.to_datetime(df,infer_datetime_format=True)

        tick_index = df.index.strftime("%Y/%m/%d").values[1::730]

        fig = make_subplots(rows=2, cols=1, print_grid=False, shared_xaxes=True, vertical_spacing=0.009,
                            horizontal_spacing=0.009, specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
                            row_heights=[0.5, 0.1])

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
        # Calculating the volatility of the ticker

        # print(volatility)
        volatility = vol(df)

        if vol_check:
            fig.add_trace(
                go.Scatter(line=dict(color='royalblue', width=2), x=x, y=volatility, type='scatter', name='volatility',
                           xaxis='x1'), 1,
                1,
                secondary_y=True)

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

    @callback(
        Output('year_dropdown', 'options'),
        Input('ticker_dropdown', 'value'))
    def update_figure(ticker_value):
        df = get_data(ticker_value, ['time', 'open', 'high', 'low', 'close'])
        # print(df.head)
        years = list(sorted(set(df.index.year.astype(int))))
        return years

    @callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
    def display_page(pathname):
        if pathname == '/':
            return page_1_layout
        elif pathname == '/tam':
            return page_2_layout
