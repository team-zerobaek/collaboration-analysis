from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from subjective.overall import initialize_overall_app
import pandas as pd

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/subjective/')

# Load dataset
dataset = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')

# Define the layout
dash_app.layout = html.Div([
    html.H1("Collaboration Analysis", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Monitoring', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective Scoring', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('A/B Test', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('ML', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.H2("Subjective Scoring", style={'text-align': 'center'}),
    html.Div([
        dcc.Dropdown(
            id='meeting-dropdown',
            placeholder="Select a meeting",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='speaker-dropdown',
            placeholder="Select speakers",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.RadioItems(
            id='view-type-radio',
            options=[
                {'label': 'Total', 'value': 'total'},
                {'label': 'By Speakers', 'value': 'by_speakers'}
            ],
            value='total',
            labelStyle={'display': 'inline-block'}
        ),
        html.Button('Reset', id='reset-button', n_clicks=0)
    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
    dcc.Graph(id='collaboration-score-graph'),
])

# Initialize the individual app
initialize_overall_app(dash_app, dataset)

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
