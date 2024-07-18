from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from ml.ml_overall import initialize_overall_ml_app
from ml.ml_individual_others import initialize_individual_others_ml_app
from ml.ml_individual_self import initialize_individual_self_ml_app
import pandas as pd
from dash.dependencies import Input, Output

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/ml/')

# Load dataset
dataset = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')

# Define the layout
dash_app.layout = html.Div([
    html.H1("How We Collaborate", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Upload', style={'margin-right': '10px'}), href='/upload'),
        html.A(html.Button('Behavioral', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('Effective Channels', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('Predict', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.H2("Prediction by Machine Learning", style={'text-align': 'center'}),
    html.Div([
        html.A("Predict Members' Perception of Collaboration", href='#overall', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("Predict Peer Evaluation Scores", href='#other', style={'margin-right': '20px'}, className='scroll-link'),
        html.A("Predict Self-Evaluation Scores", href='#self', style={'margin-right': '20px'}, className='scroll-link'),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),
    html.Details([
        html.Summary('Machine Learning Main Description', style={'margin-bottom': '11px','text-align': 'center'}),
        html.Div([
            dcc.Markdown('''
        ### Machine Learning (ML)
        Machine Learning (ML) is a field of artificial intelligence that uses statistical techniques to give computer systems the ability to "learn" from data, without being explicitly programmed. In this dashboard, we use ML to predict collaboration scores based on various input features derived from meeting data.
    ''', style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'width': '100%', 'margin-bottom': '5px'}),

            html.Div([
                dcc.Markdown('''
            #### Model Selection and Performance Evaluation
            Grid search was used to tune the hyperparameters of the model. The performance of the model was evaluated using the following metrics to ensure generalizability and accuracy:
            - **R² score**: Measures the proportion of the variance in the dependent variable (self-evaluation scores) that is predictable from the independent variables. A higher R² indicates better model performance.
            - **MSE (Mean Squared Error)**: Represents the average of the squares of the errors—that is, the average squared difference between the predicted values and the actual self-evaluation scores. Lower MSE indicates better model performance.
            - **Cross-Validation**: Ensures that the model's performance is consistent across different subsets of the data, indicating its ability to generalize well to unseen data.
        ''', style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'width': '50%', 'display': 'inline-block'}),

                dcc.Markdown('''
            #### Features considered in the model include(Feature Importance):
            - **Meeting number**: This feature might capture specific contextual or temporal elements of the meetings that significantly influence self-evaluation. Different meetings could have varying dynamics, expectations, or participation levels, which in turn affect how individuals rate their own performance.
            - **Gini coefficient**: This measure of inequality in contribution among participants can highlight how balanced or skewed the participation was. A higher Gini coefficient could indicate a few dominant speakers, potentially leading individuals to rate their own performance lower if they felt overshadowed.
            - **Degree centrality**: This network analysis measure indicates how central a person is within the interaction network. A higher degree centrality suggests greater involvement in the discussion, likely leading individuals to evaluate their own performance more positively.
            - **Normalized speech frequency**: This feature represents the frequency of speech adjusted for the length of the meeting or number of participants. Higher speech frequency generally reflects more active participation, which could positively influence self-evaluation as individuals perceive themselves as more engaged and contributing.
        ''', style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'width': '50%', 'display': 'inline-block'})
            ], style={'display': 'flex', 'justify-content': 'space-between', 'width': '100%'})
        ], style={'backgroundColor': '#f0f0f0', 'borderRadius': '5px', 'width': '100%', 'padding': '20px'})
    ], style={'margin-top': '10px', 'margin-bottom': '10px'}),

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

# Ensure unique ID for the 'Top' button
top_button_id = 'top-button-ml'
if not any(hasattr(comp, 'id') and comp.id == top_button_id for comp in dash_app.layout.children):
    dash_app.layout.children.append(
        html.Button('Top', id=top_button_id, style={
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
    Output(top_button_id, 'n_clicks'),
    [Input(top_button_id, 'n_clicks')]
)

# Initialize the individual ML apps with unique IDs to avoid duplication
initialize_overall_ml_app(dash_app, dataset)
initialize_individual_others_ml_app(dash_app, dataset)
initialize_individual_self_ml_app(dash_app, dataset)

# Mount the Dash app to FastAPI using WSGIMiddleware
fastapi_app.mount("/", WSGIMiddleware(dash_app.server))
