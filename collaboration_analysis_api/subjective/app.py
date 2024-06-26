from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from subjective.overall import initialize_overall_app
from subjective.individual_others import initialize_individual_app
from subjective.individual_self import initialize_self_score_app
from subjective.gap import initialize_gap_app  # New import
import pandas as pd

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/subjective/')

# Load dataset
dataset = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')
dataset = dataset[(dataset['overall_collaboration_score'] != -1) & (dataset['individual_collaboration_score'] != -1)]

# Define the layout
dash_app.layout = html.Div([
    html.H1("Collaboration Analysis", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Monitoring', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective Scoring', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('A/B Test', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('ML', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
])

# Initialize the individual apps
initialize_gap_app(dash_app, dataset)  # Initialize the new gap app
initialize_overall_app(dash_app, dataset)
initialize_individual_app(dash_app, dataset)
initialize_self_score_app(dash_app, dataset)

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
