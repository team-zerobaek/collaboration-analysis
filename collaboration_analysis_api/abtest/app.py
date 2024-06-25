from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from abtest.on_off import initialize_abtest_app
from abtest.casual import initialize_casual_app
from abtest.text_voice import initialize_text_voice_app
import pandas as pd

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/abtest/')

# Load dataset
dataset_voice = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')
dataset_text = pd.read_csv('/app/data/kakao_data.csv')

# Define the layout
dash_app.layout = html.Div([
    html.H1("Collaboration Analysis", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Monitoring', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective Scoring', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('A/B Test', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('ML', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.H2("A/B Test", style={'text-align': 'center'}),
    html.H3("A/B Test: Online vs. Offline", style={'text-align': 'center'}),
    html.Div([
        dcc.RadioItems(
            id='abtest-view-type',
            options=[
                {'label': 'Total', 'value': 'total'},
                {'label': 'By Speakers', 'value': 'by_speakers'}
            ],
            value='total',
            labelStyle={'display': 'inline-block'}
        ),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='abtest-graph-speech'),
            html.Div(id='abtest-table-speech')
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='abtest-graph-interaction'),
            html.Div(id='abtest-table-interaction')
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'})
])

# Initialize the individual apps
initialize_abtest_app(dash_app, dataset_voice)
initialize_casual_app(dash_app, dataset_voice)
initialize_text_voice_app(dash_app, dataset_voice, dataset_text)

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
