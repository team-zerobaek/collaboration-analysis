from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from dash.dependencies import Input, Output, State
import datetime
import pandas as pd
import os

# Import processing functions
from upload.preprocessing_behavioral import process_transcripts, extract_speaker_turns, create_dataset, process_files_in_directory, process_and_save
from upload.preview import initialize_summary_app

# Initialize the FastAPI app
upload_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/upload/')

# Load default dataset
default_dataset = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')
dataset = default_dataset.copy()

# Define the layout for the upload page
dash_app.layout = html.Div([
    html.H1("How We Collaborate", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Upload', style={'margin-right': '10px'}), href='/upload'),
        html.A(html.Button('Monitoring', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective Scoring', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('A/B Test', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('ML', style={'margin-right': '10px'}), href='/ml')
    ], style={'text-align': 'center', 'margin-bottom': '20px'}),

    # Upload section
    html.Div(id='upload-section', children=[
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=True
        ),
        html.Div(style={'text-align': 'center'}, children=[
            html.Button('Use Default Data', id='use-default-data', n_clicks=0, style={
                'font-size': '20px',
                'padding': '10px 20px',
                'margin-top': '10px'
            })
        ]),
        html.Div(id='output-data-upload'),
    ]),

    # Preview section
    html.Div(id='preview-section', children=[
        html.H1("Preview", style={'text-align': 'center'}),
        html.Div(id='preview-content', style={'display': 'none'})  # Placeholder for preview content
    ])
])

# Initialize the summary charts
initialize_summary_app(dash_app, dataset)

# Callback to handle file uploads and process the transcripts
@dash_app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(contents, filenames, last_modified):
    global dataset
    if contents is not None:
        transcriptions = process_transcripts(contents, filenames)
        dfs = [extract_speaker_turns(data) for data in transcriptions]
        durations = process_files_in_directory(transcriptions)
        dataset = create_dataset(dfs, project_number=99)
        dataset['duration'] = durations * len(dataset) // len(durations)
        dataset['duration'] = dataset['duration'] / 60  # Convert duration from minutes to hours
        dataset['normalized_speech_frequency'] = dataset['speech_frequency'] / dataset['duration']

        # Save to CSV
        process_and_save(dataset, 'data/dataset_collaboration_manual.csv')

        return html.Div([
            html.H5(', '.join(filenames)),
            html.H6(datetime.datetime.fromtimestamp(last_modified[0])),
            html.Div('Data uploaded and processed successfully!'),
            dcc.Location(pathname='/upload', id='redirect')
        ])

# Callback to handle the use of default data and show the preview section
@dash_app.callback(
    Output('preview-content', 'style'),
    Input('use-default-data', 'n_clicks')
)
def use_default_data(n_clicks):
    if n_clicks:
        global dataset
        dataset = default_dataset.copy()
        return {'display': 'block'}
    return {'display': 'none'}

# Log initialization
print("Initialized upload page")

# Mount the Dash app to FastAPI using WSGIMiddleware
upload_app.mount("/", WSGIMiddleware(dash_app.server))
