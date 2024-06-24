from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
from asgiref.wsgi import WsgiToAsgi
from dashboard.sna import initialize_sna_app
from dashboard.frequency import initialize_frequency_app
from dashboard.interaction import initialize_interaction_app
from dashboard.gini import initialize_gini_app
from dashboard.degree_centrality import initialize_degree_centrality_app

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Redirect root to /dash
@fastapi_app.get("/")
async def main():
    return RedirectResponse(url="/dash")

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/dash/')

# Load dataset
dataset = pd.read_csv('/app/data/dataset_collaboration.csv')

# Define the layout
dash_app.layout = html.Div([
    html.H1("Interactive Network Graph Dashboard"),
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
    html.H1("Normalized Speech Frequency"),
    html.Div([
        dcc.Dropdown(
            id='speech-project-dropdown',
            options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
            placeholder="Select projects",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='speech-meeting-dropdown',
            placeholder="Select meetings",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='speech-speaker-dropdown',
            placeholder="Select speakers",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.RadioItems(
            id='speech-type-radio',
            options=[
                {'label': 'Total', 'value': 'total'},
                {'label': 'By Speakers', 'value': 'by_speakers'}
            ],
            value='total',
            labelStyle={'display': 'inline-block'}
        ),
        html.Button('Reset', id='reset-speech-button', n_clicks=0)
    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
    dcc.Graph(id='speech-frequency-graph'),
    html.H1("Normalized Interaction Frequency"),
    html.Div([
        dcc.Dropdown(
            id='interaction-project-dropdown',
            options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
            placeholder="Select projects",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='interaction-meeting-dropdown',
            placeholder="Select meetings",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='interaction-speaker-dropdown',
            placeholder="Select speakers",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.RadioItems(
            id='interaction-type-radio',
            options=[
                {'label': 'Total', 'value': 'total'},
                {'label': 'By Speakers', 'value': 'by_speakers'}
            ],
            value='total',
            labelStyle={'display': 'inline-block'}
        ),
        html.Button('Reset', id='reset-interaction-button', n_clicks=0)
    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
    dcc.Graph(id='interaction-frequency-graph'),
])

# Initialize the individual apps
initialize_sna_app(dash_app, dataset)
initialize_frequency_app(dash_app, dataset)
initialize_interaction_app(dash_app, dataset)
initialize_gini_app(dash_app, dataset)
initialize_degree_centrality_app(dash_app, dataset)

# Convert the Dash app to an ASGI app
asgi_dash_app = WsgiToAsgi(dash_app.server)

# Mount the Dash app to FastAPI
fastapi_app.mount("/dash", asgi_dash_app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8080)
