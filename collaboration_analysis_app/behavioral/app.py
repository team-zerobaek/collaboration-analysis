from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from behavioral.degree_centrality import initialize_degree_centrality_app
from behavioral.frequency import initialize_frequency_app
from behavioral.interaction import initialize_interaction_app
from behavioral.gini import initialize_gini_app
from behavioral.sna import initialize_sna_app
import pandas as pd
import os
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/dash/')

# Data Load and Configuration
data_directory = '/app/data'
os.makedirs(data_directory, exist_ok=True)

# Load default dataset
default_dataset = pd.read_csv(os.path.join(data_directory, 'dataset_collaboration_with_survey_scores.csv'))
dataset = default_dataset.copy()

def check_uploaded_data_exists():
    return os.path.exists(os.path.join(data_directory, 'dataset_collaboration_manual.csv'))

# Check if the uploaded dataset exists initially
uploaded_data_exists = check_uploaded_data_exists()

# Define the layout for the main page
main_layout = html.Div([
    html.H1("How We Collaborate", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Upload', style={'margin-right': '10px'}), href='/upload'),
        html.A(html.Button('Behavioral', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('Effective Channels', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('Predict', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

    html.Div([
        dcc.RadioItems(
            id='dataset-selection-radio',
            options=[
                {'label': 'Default Data', 'value': 'default'},
                {'label': 'Uploaded Data', 'value': 'uploaded', 'disabled': not uploaded_data_exists}
            ],
            value='default',
            style={'text-align': 'center'}
        )
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

    html.H2("Behavioral Data Analysis", style={'text-align': 'center'}),
    html.Div([
        html.A("Interaction Network Graph", href='#sna', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("How Evenly Interacted?", href='#gini', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("How Much Contributed?", href='#degree', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("How Much Interacted?", href='#turn', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("How Many Words Were Spoken?", href='#word', className='scroll-link')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Div(id='graphs-container'),
    html.Script('''
        document.addEventListener('DOMContentLoaded', function() {
            var links = document.querySelectorAll('.scroll-link');
            links.forEach(function(link) {
                link.addEventListener('click', function(event) {
                    event.preventDefault();
                    var targetId = this.getAttribute('href').substring(1);
                    var targetElement = document.getElementById(targetId);
                    window.scrollTo({
                        top: targetElement.offsetTop,
                        behavior: 'smooth'
                    });
                });
            });
        });
    ''')
])

dash_app.layout = main_layout

dash_app.layout.children.append(
    html.Button('Top', id='top-button', style={
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'padding': '10px 20px',
        'font-size': '16px',
        'z-index': '1000',
        'background-color': '#007bff',
        'color': 'white',
        'border': 'none',
        'border-radius': '5px',
        'cursor': 'pointer'
    })
)

dash_app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0, n_clicks === 0) {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        }
        return '';
    }
    """,
    Output('top-button', 'n_clicks'),
    [Input('top-button', 'n_clicks')]
)

# Callback to update the radio button options based on file existence
@dash_app.callback(
    [Output('dataset-selection-radio', 'options')],
    [Input('dataset-selection-radio', 'value')]
)
def update_radio_options(dataset_selection):
    uploaded_data_exists = check_uploaded_data_exists()
    options = [
        {'label': 'Default Data', 'value': 'default'},
        {'label': 'Uploaded Data', 'value': 'uploaded', 'disabled': not uploaded_data_exists}
    ]
    return [options]

# Initialize the individual apps
initialize_sna_app(dash_app, dataset)
initialize_gini_app(dash_app, dataset)
initialize_degree_centrality_app(dash_app, dataset)
initialize_interaction_app(dash_app, dataset)
initialize_frequency_app(dash_app, dataset)

# Log initialization
print("Initialized individual apps with dataset")

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
