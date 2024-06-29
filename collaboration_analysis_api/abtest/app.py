from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from abtest.on_off import initialize_abtest_app
from abtest.casual import initialize_casual_app
from abtest.text_voice import initialize_text_voice_app
import pandas as pd
from dash.dependencies import Input, Output


# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/abtest/')

# Load dataset
dataset_voice = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')
dataset_text = pd.read_csv('/app/data/kakao_data.csv')

# Define the layout
dash_app.layout = html.Div([
    html.H1("How We Collaborate", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Upload', style={'margin-right': '10px'}), href='/upload'),
        html.A(html.Button('Monitoring', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective Scoring', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('A/B Test', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('ML', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.H2("A/B Test", style={'text-align': 'center'}),
    html.Div([
        html.A("Online vs. Offline", href='#onoff', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("Casual Language Used", href='#casual', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("Text-based vs. Voice-based", href='#voice', style={'margin-right': '20px'}, className='scroll-link'),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

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
        if (n_clicks > 0) {
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

# Initialize the individual apps
initialize_abtest_app(dash_app, dataset_voice)
initialize_casual_app(dash_app, dataset_voice)
initialize_text_voice_app(dash_app, dataset_voice, dataset_text)

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
