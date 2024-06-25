from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from behavioral.degree_centrality import initialize_degree_centrality_app
from behavioral.frequency import initialize_frequency_app
from behavioral.interaction import initialize_interaction_app
from behavioral.gini import initialize_gini_app
from behavioral.sna import initialize_sna_app
import pandas as pd

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/dash/')

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
    html.H2("Monitoring", style={'text-align': 'center'}),
    html.Div([
        dcc.Dropdown(
            id='project-dropdown',
            options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
            placeholder="Select a project",
            multi=True,
            style={'width': '200px'}
        ),
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
        html.Button('Reset', id='reset-button', n_clicks=0)
    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
    dcc.Graph(id='network-graph'),
])

# Initialize the individual apps
initialize_sna_app(dash_app, dataset)
initialize_frequency_app(dash_app, dataset)
initialize_interaction_app(dash_app, dataset)
initialize_gini_app(dash_app, dataset)
initialize_degree_centrality_app(dash_app, dataset)

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
