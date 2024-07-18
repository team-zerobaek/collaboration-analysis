from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from subjective.overall import initialize_overall_app
from subjective.individual_others import initialize_individual_app
from subjective.individual_self import initialize_self_score_app
from subjective.gap import initialize_gap_app
import pandas as pd
from dash.dependencies import Input, Output

fastapi_app = FastAPI()

dash_app = Dash(__name__, requests_pathname_prefix='/subjective/')

dataset = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')
dataset = dataset[(dataset['overall_collaboration_score'] != -1) & (dataset['individual_collaboration_score'] != -1)]

dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.H1("How We Collaborate", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Upload', style={'margin-right': '10px'}), href='/upload'),
        html.A(html.Button('Behavioral', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('Effective Channels', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('Predict', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.H2("Subjective Data Analysis", style={'text-align': 'center'}),
    html.Div([
        html.A("Members' Perception of Collaboration", href='#overall', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("Gap of Peer Evaluation Score", href='#gap', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("Peer Evaluation Scores for Collaboration", href='#individual-others', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("Self Evaluation Scores for Collaboration", href='#individual-self', className='scroll-link')
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


initialize_overall_app(dash_app, dataset)
initialize_gap_app(dash_app, dataset)
initialize_individual_app(dash_app, dataset)
initialize_self_score_app(dash_app, dataset)

fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
