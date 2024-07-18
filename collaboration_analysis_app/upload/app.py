# upload/app.py
from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from dash.dependencies import Input, Output, State
import pandas as pd
import os
import logging
from dash.exceptions import PreventUpdate

# Import processing functions
from upload.preprocessing_behavioral import process_uploaded_files
from upload.preview import initialize_summary_app

# Initialize the FastAPI app
upload_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/upload/')

# Data Load and Configuration
data_directory = '/app/data'
os.makedirs(data_directory, exist_ok=True)

# Load default datasets
default_dataset_voice = pd.read_csv(os.path.join(data_directory, 'dataset_collaboration_with_survey_scores.csv'))
default_dataset_text = pd.read_csv(os.path.join(data_directory, 'kakao_data.csv'))

# Configure logging
logging.basicConfig(level=logging.INFO)

def check_uploaded_data_exists():
    return os.path.exists(os.path.join(data_directory, 'dataset_collaboration_manual.csv'))

# Check if the uploaded dataset exists initially
uploaded_data_exists = check_uploaded_data_exists()

# Define the layout for the upload page
dash_app.layout = html.Div([
    html.H1("How We Collaborate", style={'text-align': 'center'}),
    html.Div([
        html.A(html.Button('Upload', style={'margin-right': '10px'}), href='/upload'),
        html.A(html.Button('Behavioral', style={'margin-right': '10px'}), href='/dash'),
        html.A(html.Button('Subjective', style={'margin-right': '10px'}), href='/subjective'),
        html.A(html.Button('Effective Channels', style={'margin-right': '10px'}), href='/abtest'),
        html.A(html.Button('Predict', style={'margin-right': '10px'}), href='/ml')
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
        html.Div(style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'justifyContent': 'center'}, children=[
            dcc.RadioItems(
                id='dataset-selection-radio',
                options=[
                    {'label': 'Default Data', 'value': 'default'},
                    {'label': 'Uploaded Data', 'value': 'uploaded', 'disabled': not uploaded_data_exists}
                ],
                value='default',
                style={'text-align': 'center'}
            ),
        ]),
        dcc.Store(id='dataset-selection-store'),  # Store to hold dataset selection
        dcc.Store(id='upload-info-store'),
        html.Div(id='output-data-upload', style={'font-size': '20px', 'text-align': 'center', 'margin-top': '20px'}),
    ]),

    # Preview section
    html.Div(id='preview-section', children=[
        html.H1("Preview", style={'text-align': 'center'}),
        html.Div(id='preview-content')  # Placeholder for preview content
    ])
])

# Initialize the summary charts with default data
initial_preview_content = initialize_summary_app(dash_app, default_dataset_voice, default_dataset_text)

# Set the initial preview content
dash_app.layout.children[2].children[1].children.append(initial_preview_content)

# Add the "Top" button
dash_app.layout.children.append(
    html.Button('Top', id='top-button-upload', style={
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

# Client-side callback for the "Top" button
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
    Output('top-button-upload', 'n_clicks'),
    [Input('top-button-upload', 'n_clicks')]
)

# Callback to handle file uploads and process the transcripts
@dash_app.callback(
    [Output('upload-info-store', 'data'),
     Output('preview-content', 'children'),
     Output('dataset-selection-radio', 'options'),
     Output('dataset-selection-radio', 'value'),
     Output('dataset-selection-store', 'data')],
    [Input('upload-data', 'contents'),
     Input('dataset-selection-radio', 'value')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_output(contents, dataset_selection, filenames, last_modified):
    global default_dataset_voice, default_dataset_text, uploaded_data_exists

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    logging.info(f"Triggered by: {trigger_id}")

    preview_content = initial_preview_content
    upload_info = {'status': 'default', 'filenames': []}

    if trigger_id == 'dataset-selection-radio' and dataset_selection == 'default':
        default_dataset_voice = pd.read_csv(os.path.join(data_directory, 'dataset_collaboration_with_survey_scores.csv'))
        preview_content = initialize_summary_app(dash_app, default_dataset_voice, default_dataset_text)
        logging.info("Using default dataset for preview.")
        return upload_info, preview_content, dash.no_update, 'default', 'default'

    if trigger_id == 'dataset-selection-radio' and dataset_selection == 'uploaded':
        try:
            default_dataset_voice = pd.read_csv(os.path.join(data_directory, 'dataset_collaboration_manual.csv'))
        except FileNotFoundError as e:
            logging.error(f"Error loading uploaded data: {e}")
            upload_info = {'status': 'error', 'filenames': [], 'error': "Uploaded data not found."}
            return upload_info, html.Div(""), dash.no_update, 'default', 'default'

        preview_content = initialize_summary_app(dash_app, default_dataset_voice, default_dataset_text)
        logging.info("Using uploaded dataset for preview.")
        return upload_info, preview_content, dash.no_update, 'uploaded', 'uploaded'

    if trigger_id == 'upload-data' and contents:
        try:
            logging.info("Contents received for processing.")
            if filenames is None:
                filenames = []

            logging.info(f"Filenames: {filenames}")

            # Process the uploaded files
            process_uploaded_files(contents, filenames)

            # Load the newly created dataset
            default_dataset_voice = pd.read_csv(os.path.join(data_directory, 'dataset_collaboration_manual.csv'))

            # Initialize summary charts with the new dataset
            preview_content = initialize_summary_app(dash_app, default_dataset_voice, default_dataset_text)
            logging.info("Preview content updated with uploaded dataset.")

            # Check if the uploaded data now exists
            uploaded_data_exists = check_uploaded_data_exists()

            # Update the dataset-selection-radio options
            updated_options = [
                {'label': 'Default Data', 'value': 'default'},
                {'label': 'Uploaded Data', 'value': 'uploaded', 'disabled': not uploaded_data_exists}
            ]

            upload_info = {'status': 'uploaded', 'filenames': filenames}

            return upload_info, preview_content, updated_options, 'uploaded', 'uploaded'
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            upload_info = {'status': 'error', 'filenames': filenames, 'error': str(e)}
            return upload_info, html.Div(""), dash.no_update, 'default', 'default'

    # Update the dataset-selection-radio options if nothing triggered it
    uploaded_data_exists = check_uploaded_data_exists()
    updated_options = [
        {'label': 'Default Data', 'value': 'default'},
        {'label': 'Uploaded Data', 'value': 'uploaded', 'disabled': not uploaded_data_exists}
    ]

    return upload_info, preview_content, updated_options, dash.no_update, dash.no_update

# Callback to update the output-data-upload based on the stored upload info
@dash_app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-info-store', 'data')
)
def display_upload_info(upload_info):
    if upload_info['status'] == 'default':
        return html.Div([
            html.H5("Used data: Default data"),
            html.H6("No upload attempted")
        ])
    elif upload_info['status'] == 'uploaded':
        filenames = upload_info['filenames'] if isinstance(upload_info['filenames'], list) else []
        return html.Div([
            html.H5(f"Used data: {', '.join(filenames)}"),
            html.H6("Upload successful")
        ])
    elif upload_info['status'] == 'error':
        filenames = upload_info['filenames'] if isinstance(upload_info['filenames'], list) else []
        return html.Div([
            html.H5(f"Used data: {', '.join(filenames)}"),
            html.H6(f"Upload failed: {upload_info['error']}")
        ])

# Log initialization
print("Initialized upload page")

# Mount the Dash app to FastAPI using WSGIMiddleware
upload_app.mount("/", WSGIMiddleware(dash_app.server))

# Log initialization
logging.info("Initialized upload page")
