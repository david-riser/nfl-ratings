""" 

David Riser 
October 20, 2019
app.py 

Dash application to serve predictions for NFL 
games. 

"""

# Python Standard Libs
import os

# Third Party Libs 
import dash_core_components as dcc
import dash_html_components as html
import flask 
import pandas as pd 
import plotly_express as px

from dash import Dash 
from dash.dependencies import (Input, Output)

data_dir = os.path.normpath(
    os.path.dirname(os.path.abspath(__file__)) + '/../data'
)

data = pd.read_csv(data_dir + '/weekly_preds.csv')
print('Loading data from: {0}'.format(data_dir))
print('Loaded {} games.'.format(len(data)))

app = Dash('app', server=flask.Flask('app'))
app.layout = html.Div([
    html.H1('Hello World!'),
    dcc.Graph(figure=px.scatter(
        data, x='elo_prob1', y='team1'
    ))
])


if __name__ == '__main__':
    app.run_server(port=5678, debug=True)