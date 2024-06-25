from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/abtest/')

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
    # Add your layout and components here for A/B Test page
])

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
