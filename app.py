import dash_bootstrap_components as dbc
from dash import Dash
from dash import dcc
from dash import html

from callbacks import get_callbacks

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
           meta_tags=[{'name': 'viewport',
                       'content': 'width=device-width, initial-scale=1.0'}],
           suppress_callback_exceptions=True)

# Layout section: Bootstrap
# -------------------
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

get_callbacks(app) #Get the callbacks from callbacks.py

if __name__ == '__main__':
    app.run_server(debug=True,dev_tools_prune_errors=False)
