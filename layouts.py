import os

import dash_bootstrap_components as dbc
from dash import dcc
from dash import html


def get_page_1_layout(df,data_path):
    page_1_layout = dbc.Container([
        dbc.Row([dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Analysis Dashboard", active=True, href="/")),
                dbc.NavItem(dbc.NavLink("TA Correlation Matrix", href="/tam")),
            ]
            , justified=True)]),
        dbc.Row([
            dbc.Col(html.H1("Crypto Dashboard Analysis",
                            className='text-center bg-dark text-warning mb-4'),
                    width=8)
        ], justify='center'),
        dbc.Row([
            dbc.Col([dcc.Dropdown(id='year_dropdown', multi=False, value=2022,
                                  className='bg-dark text-dark',
                                  options=[{'label': x, 'value': x}
                                           for x in sorted(set(df.index.year))])], width={'size': 3, 'order': 2}
                    ),
            dbc.Col([dcc.Dropdown(id='ticker_dropdown', multi=False, value='btcusd',
                                  className='bg-dark text-dark',
                                  options=[{'label': os.path.splitext(x)[0], 'value': os.path.splitext(x)[0]}
                                           for x in sorted(os.listdir(data_path))])], width={'size': 3, 'order': 1})

        ], justify='center', className='g-0'),
        dbc.Row([

            dbc.Col([dcc.Checklist(id='check-box', options={True: 'Show Gaps'})], width={'size': 3, 'order': 1},
                    align='left'),

            dbc.Col([dcc.Checklist(id='vol-check', options={True: 'Enable Volatility'})], width={'size': 3, 'order': 1},
                    align='left'),

        ], justify='center', className='g-0'),
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

def get_page_2_layout():
    page_2_layout = dbc.Container([
        dbc.Row([dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Analysis Dashboard", active=True, href="/")),
                dbc.NavItem(dbc.NavLink("TA Correlation Matrix", href="/tam")),
            ]
        )])])
    return page_2_layout