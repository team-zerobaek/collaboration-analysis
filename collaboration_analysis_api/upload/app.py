from fastapi import FastAPI
from dash import Dash, dcc, html
from fastapi.middleware.wsgi import WSGIMiddleware
from dash.dependencies import Input, Output, State
import networkx as nx
import datetime
import pandas as pd
import os
import logging

# Import processing functions
from upload.preprocessing_behavioral import (
    process_transcripts, extract_speaker_turns, create_dataset, process_files_in_directory,
    process_and_save, compute_interaction_frequency, generate_all_pairs, compute_centralities,
    compute_density, weighted_density, compute_gini, compute_equality_index, save_dataframe_to_csv
)
from upload.preview import initialize_summary_app

# Initialize the FastAPI app
upload_app = FastAPI()

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/upload/')

# Load default datasets
default_dataset_voice = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')
default_dataset_text = pd.read_csv('/app/data/kakao_data.csv')

# Configure logging
logging.basicConfig(level=logging.INFO)

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
initialize_summary_app(dash_app, default_dataset_voice, default_dataset_text)

# Ensure 'data' directory exists
data_directory = '/app/data'
os.makedirs(data_directory, exist_ok=True)

# Callback to handle file uploads and process the transcripts
@dash_app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(contents, filenames, last_modified):
    if contents is not None:
        logging.info("Contents received for processing.")
        try:
            transcriptions = process_transcripts(contents, filenames)
            logging.info(f"Processed {len(transcriptions)} transcriptions.")

            dfs = []
            for i, data in enumerate(transcriptions):
                df = extract_speaker_turns(data)
                dfs.append(df)
                df.to_csv(f'data/extracted_speaker_turns_{i+1}.csv', index=False)

            logging.info(f"Extracted speaker turns for {len(dfs)} datasets.")

            durations = process_files_in_directory(transcriptions)
            logging.info(f"Computed durations: {durations}")

            new_dataset_voice = create_dataset(dfs, project_number=0)  # Set default project number
            new_dataset_voice.to_csv('data/created_dataset_voice.csv', index=False)

            duration_mapping = []
            for i, duration in enumerate(durations):
                num_rows = len(new_dataset_voice[new_dataset_voice['meeting_number'] == i + 1])
                duration_mapping.extend([duration / 60] * num_rows)  # Convert to hours and replicate

            if len(duration_mapping) != len(new_dataset_voice):
                raise ValueError("Duration mapping length does not match the new_dataset_voice length.")

            new_dataset_voice['duration'] = duration_mapping
            new_dataset_voice['normalized_speech_frequency'] = new_dataset_voice['speech_frequency'] / new_dataset_voice['duration']

            logging.info("New dataset created and durations computed.")

            # Compute interaction frequencies
            interaction_records = pd.concat([compute_interaction_frequency(df, 0) for df in dfs], ignore_index=True)
            all_pairs = generate_all_pairs(interaction_records, new_dataset_voice)
            interaction_records = pd.concat([interaction_records, all_pairs], ignore_index=True)
            interaction_records = interaction_records.sort_values(by=['project', 'meeting_number', 'speaker_id', 'next_speaker_id']).reset_index(drop=True)

            logging.info("Interaction frequencies computed.")
            # Debugging: Save interaction records to CSV for verification
            interaction_records.to_csv('data/interaction_records_debug.csv', index=False)

            # Save intermediate dataset to CSV for debugging
            intermediate_filename = os.path.join(data_directory, 'intermediate_combined_dataset.csv')
            process_and_save(interaction_records, intermediate_filename)
            logging.info(f"Intermediate dataset saved to {intermediate_filename}")

            # Compute centralities and densities
            for df in dfs:
                centralities = compute_centralities(df, interaction_records)
                G = nx.DiGraph()
                for i in range(len(df)):
                    prev_speaker = df.iloc[i]['Speaker']
                    if i < len(df) - 1:
                        next_speaker = df.iloc[i+1]['Speaker']
                    else:
                        next_speaker = df.iloc[i]['Speaker']  # Self-interaction if last speaker
                    if prev_speaker != next_speaker and df.iloc[i]['Text'].strip() != '':
                        try:
                            combined_row = interaction_records[(interaction_records['project'] == df['project'].iloc[0]) & (interaction_records['meeting_number'] == df['meeting_number'].iloc[0]) & (interaction_records['speaker_id'] == int(prev_speaker.split('_')[1])) & (interaction_records['next_speaker_id'] == int(next_speaker.split('_')[1]))]
                            if not combined_row.empty:
                                weight = combined_row['normalized_weight'].values[0]
                                if G.has_edge(prev_speaker, next_speaker):
                                    G[prev_speaker][next_speaker]['weight'] += weight
                                else:
                                    G.add_edge(prev_speaker, next_speaker, weight=weight)
                        except IndexError as e:
                            logging.error(f"IndexError: {e}. Problem with speaker: {prev_speaker}, next_speaker: {next_speaker}, meeting_number: {df['meeting_number'].iloc[0]}")
                            continue
                for centrality_measure, centrality_values in centralities.items():
                    for node, value in centrality_values.items():
                        interaction_records.loc[(interaction_records['project'] == df['project'].iloc[0]) & (interaction_records['meeting_number'] == df['meeting_number'].iloc[0]) & (interaction_records['speaker_id'] == int(node.split('_')[1])), centrality_measure] = value
                interaction_records.loc[(interaction_records['project'] == df['project'].iloc[0]) & (interaction_records['meeting_number'] == df['meeting_number'].iloc[0]), 'network_density'] = compute_density(G)
                interaction_records.loc[(interaction_records['project'] == df['project'].iloc[0]) & (interaction_records['meeting_number'] == df['meeting_number'].iloc[0]), 'weighted_network_density'] = weighted_density(G)

            logging.info("Centralities and densities computed.")

            # Compute Gini coefficient and Interaction Equality Index
            interaction_records['gini_coefficient'] = 0
            interaction_records['interaction_equality_index'] = 0
            for project in interaction_records['project'].unique():
                project_data = interaction_records[interaction_records['project'] == project]
                gini_values = compute_gini(project_data)
                equality_index_values = compute_equality_index(project_data)
                for i, meeting_number in enumerate(project_data['meeting_number'].unique()):
                    interaction_records.loc[(interaction_records['project'] == project) & (interaction_records['meeting_number'] == meeting_number), 'gini_coefficient'] = gini_values[i]
                    interaction_records.loc[(interaction_records['project'] == project) & (interaction_records['meeting_number'] == meeting_number), 'interaction_equality_index'] = equality_index_values[i]

            # Save final dataset to CSV
            final_filename = os.path.join(data_directory, 'dataset_collaboration_manual.csv')
            process_and_save(interaction_records, final_filename)
            logging.info(f"Final dataset saved to {final_filename}")

            return html.Div([
                html.H5(', '.join(filenames)),
                html.H6(datetime.datetime.fromtimestamp(last_modified[0])),
                html.Div('Data uploaded and processed successfully!'),
                dcc.Location(pathname='/upload', id='redirect')
            ])
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            return html.Div([
                html.H5(', '.join(filenames)),
                html.H6(datetime.datetime.fromtimestamp(last_modified[0])),
                html.Div(f"Error processing data: {e}")
            ])


# Callback to handle the use of default data and show the preview section
@dash_app.callback(
    Output('preview-content', 'style'),
    Input('use-default-data', 'n_clicks')
)
def use_default_data(n_clicks):
    if n_clicks:
        global default_dataset_voice, default_dataset_text
        default_dataset_voice = pd.read_csv('/app/data/dataset_collaboration_with_survey_scores.csv')
        default_dataset_text = pd.read_csv('/app/data/kakao_data.csv')
        return {'display': 'block'}
    return {'display': 'none'}

# Log initialization
print("Initialized upload page")

# Mount the Dash app to FastAPI using WSGIMiddleware
upload_app.mount("/", WSGIMiddleware(dash_app.server))
