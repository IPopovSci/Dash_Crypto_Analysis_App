import os

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html


def get_page_1_layout(df, data_path):
    page_1_layout = dbc.Container([
        dbc.Row([dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Analysis Dashboard", active=True, href="/")),
                dbc.NavItem(dbc.NavLink("TA Correlation Matrix", href="/tam")),
                dbc.NavItem(dbc.NavLink("Fitted Backtest", href="/bcktst")),
            ]
            , justified=False)]),
        dbc.Row([
            dbc.Col(html.H1("Crypto Dashboard Analysis",
                            className='text-center bg-dark text-warning mb-4'),
                    width=8)
        ], justify='center'),
        dbc.Row([
            dbc.Col([dcc.Dropdown(id='year_dropdown', multi=False, value=2022,
                                  className='bg-dark text-dark',
                                  options=[{'label': x, 'value': x}
                                           for x in sorted(set(df.index.year))])], width={'size': 1, 'order': 2}
                    ),
            dbc.Col([dcc.Dropdown(id='ticker_dropdown', multi=False, value='btcusd',
                                  className='bg-dark text-dark',
                                  options=[{'label': os.path.splitext(x)[0], 'value': os.path.splitext(x)[0]}
                                           for x in sorted(os.listdir(data_path))])], width={'size': 1, 'order': 1})

        ], justify='center'),
        dbc.Row([

            dbc.Col([dcc.Checklist(id='check-box', options={True: 'Show Gaps'})], width={'size': 1, 'order': 1},
                    align='center'),

            dbc.Col([dcc.Checklist(id='vol-check', options={True: 'Enable Volatility'})], width={'size': 1, 'order': 2},
                    align='center'),

        ], justify='center'),
        dbc.Row([dbc.Col([
            dcc.Graph(id='close-graph', figure={'layout': {'plot_bgcolor': '#d3d3d3', 'paper_bgcolor': '#111111'}},
                      style={'height': 675}), ], width={'size': 12, 'order': 1})

        ], justify='center'),

        dbc.Row([
            dbc.Col([dcc.Textarea(
                id='textarea_missingno', disabled=True,
                value='',
                style={'width': '100%', 'height': 80, 'color': 'orange'})], width={'size': 3, 'order': 1}),
            dbc.Col([dcc.Textarea(
                id='textarea_performance', disabled=True,
                value='',
                style={'width': '100%', 'height': 80, 'color': 'orange'})], width={'size': 4, 'order': 2}),
        ], justify='evenly')
    ], fluid=True)
    return page_1_layout


def get_page_2_layout(df, data_path):
    page_2_layout = dbc.Container([

        dbc.Row([dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Analysis Dashboard", active=True, href="/")),
                dbc.NavItem(dbc.NavLink("TA Correlation Matrix", href="/tam")),
                dbc.NavItem(dbc.NavLink("Fitted Backtest", href="/bcktst")),
            ]
        )
        ]),
        dbc.Row([
            dbc.Col(html.H1("TA Feature Analysis",
                            className='text-center bg-dark text-warning mb-4'),
                    width=8)
        ], justify='center'),
        dbc.Row([
            dbc.Col([dcc.Dropdown(id='ticker_dropdown_page2', multi=False, value='btcusd',
                                  className='bg-dark text-dark',
                                  options=[{'label': os.path.splitext(x)[0], 'value': os.path.splitext(x)[0]}
                                           for x in sorted(os.listdir(data_path))])], width={'size': 1, 'order': 1})

        ], justify='center'),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='corr-mat', figure={'layout': {'plot_bgcolor': '#d3d3d3', 'paper_bgcolor': '#111111'}},
                          style={'height': 675})], width={'size': 6, 'order': 1}),

            dbc.Col([
                dcc.Graph(id='corr-element', figure={'layout': {'plot_bgcolor': '#d3d3d3', 'paper_bgcolor': '#111111'}},
                          style={'height': 675}),

                dcc.Slider(
                    id='poly-slider',
                    min=2, max=15, step=1,
                    value=2,
                )], width={'size': 6, 'order': 2}),

        ], justify='center'),

        # dbc.Row([
        #     dbc.Col([
        #         dcc.Graph(id='corr-element', figure={'layout': {'plot_bgcolor': '#d3d3d3', 'paper_bgcolor': '#111111'}},
        #                   style={'height': 675})], width={'size': 12, 'order': 1}),
        #
        #     # dbc.Col([
        #     #     dcc.Graph(id='corr-element', figure={'layout': {'plot_bgcolor': '#d3d3d3', 'paper_bgcolor': '#111111'}},
        #     #               style={'height': 675})], width={'size': 8, 'order': 2}),
        #
        # ], justify='center')

    ], fluid=True)
    return page_2_layout


def get_page_3_layout(df, data_path):
    page_3_layout = dbc.Container([
        dbc.Row([dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Analysis Dashboard", active=True, href="/")),
                dbc.NavItem(dbc.NavLink("TA Correlation Matrix", href="/tam")),
                dbc.NavItem(dbc.NavLink("Fitted Backtest", href="/bcktst")),
            ]
            , justified=False)]),
        dbc.Row([
            dbc.Col(html.H1("Backtest Fit Analysis",
                            className='text-center bg-dark text-warning mb-4'),
                    width=8)
        ], justify='center'),
        dbc.Row([
            dbc.Col(['% Training Data',dcc.Dropdown(id='training_percent3', multi=False, value=0.7,
                                  className='bg-dark text-dark',
                                  options=[{'label': '50%', 'value': 0.5},
                                           {'label': '60%', 'value': 0.6},
                                           {'label': '70%', 'value': 0.7}, {'label': '80%', 'value': 0.8}])],
                    width={'size': 1, 'order': 2}
                    ),
            dbc.Col(['Select Ticker',dcc.Dropdown(id='ticker_dropdown3', multi=False, value='btcusd',
                                  className='bg-dark text-dark',
                                  options=[{'label': os.path.splitext(x)[0], 'value': os.path.splitext(x)[0]}
                                           for x in sorted(os.listdir(data_path))])], width={'size': 1, 'order': 1}),
            dbc.Col(['Select Regression Type',dcc.Dropdown(
                                  ['Buy And Hold', 'Linear', 'Polynomial','Random Forest','Gradient Boost'],
                                  ['Buy And Hold'],
                                  multi=True,id='strategydrp3'
                                  )], width={'size': 6, 'order': 3}),
            dbc.Col(['Initial Balance',dcc.Input(
                id="init_balance",
                type='number',
                value=10000,
                placeholder="Initial Balance")], width={'size': 1, 'order': 4},style={"margin-right": "15px"}),

            dbc.Col(['Bet Size',dcc.Input(
                id="bet_amount",
                type='number',
                value=100,
                placeholder="Bet Amount")], width={'size': 1, 'order': 5}),

        ], justify='center'),

        dbc.Row([dbc.Col([
            dcc.Graph(id='backtest_graph', figure={'layout': {'plot_bgcolor': '#d3d3d3', 'paper_bgcolor': '#111111'}},
                      style={'height': 675}), ], width={'size': 12, 'order': 1})

        ], justify='center'),

        dbc.Row([
            dbc.Col([dcc.Textarea(
                id='textarea_missingno', disabled=True,
                value='',
                style={'width': '100%', 'height': 80, 'color': 'orange'})], width={'size': 3, 'order': 1}),
            dbc.Col([dcc.Textarea(
                id='textarea_performance', disabled=True,
                value='',
                style={'width': '100%', 'height': 80, 'color': 'orange'})], width={'size': 4, 'order': 2}),
        ], justify='evenly')
    ], fluid=True)
    return page_3_layout
