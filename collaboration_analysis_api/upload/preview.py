# upload/preview.py
from dash.dependencies import Input, Output
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors
from abtest.on_off import get_meetings_by_condition  # Import function to get meeting conditions

dash_app = None
dataset = None

# Predefined base colors
base_colors = ['blue', 'red', 'green', 'purple', 'orange']

def get_color_map(num_speakers):
    # Use base colors for the first few speakers
    color_map = {i: base_colors[i] for i in range(min(num_speakers, len(base_colors)))}
    # Generate additional distinct colors if there are more speakers
    if num_speakers > len(base_colors):
        additional_colors = list(mcolors.TABLEAU_COLORS) + list(mcolors.CSS4_COLORS)
        additional_colors = [c for c in additional_colors if c not in base_colors]
        for i in range(len(base_colors), num_speakers):
            color_map[i] = additional_colors[i - len(base_colors)]
    return color_map

def initialize_summary_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    # Ensure the normalized_interaction_frequency column exists and is calculated correctly
    if 'normalized_interaction_frequency' not in dataset.columns:
        dataset['normalized_interaction_frequency'] = dataset['count'] / dataset['duration']

    most_recent_project = dataset['project'].max()

    def get_valid_projects_for_all_meetings():
        valid_projects = []
        for project in dataset['project'].unique():
            project_data = dataset[dataset['project'] == project]
            online_meetings, offline_meetings = get_meetings_by_condition(project_data)
            if any(project_data['meeting_number'].isin(online_meetings)) and any(project_data['meeting_number'].isin(offline_meetings)):
                valid_projects.append(project)
        return valid_projects

    valid_projects_for_all_meetings = get_valid_projects_for_all_meetings()

    dash_app.layout.children.append(
        html.Div(id='preview-section-content', children=[
            html.Div([
                html.Div([
                    html.H3("Who Spoke the Most"),
                    dcc.Dropdown(
                        id='project-dropdown-speech',
                        options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                        value=most_recent_project,
                        style={'width': '80%', 'margin': '0 auto'}
                    ),
                    dcc.Graph(id='pie-chart-speech')
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H3("Who Interacted the Most"),
                    dcc.Dropdown(
                        id='project-dropdown-interaction',
                        options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                        value=most_recent_project,
                        style={'width': '80%', 'margin': '0 auto'}
                    ),
                    dcc.Graph(id='pie-chart-interaction')
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H3("Who Was the Center of Meetings the Most"),
                    dcc.Dropdown(
                        id='project-dropdown-degree-centrality',
                        options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                        value=most_recent_project,
                        style={'width': '80%', 'margin': '0 auto'}
                    ),
                    dcc.Graph(id='pie-chart-degree-centrality')
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ], style={'text-align': 'center', 'margin-bottom': '20px'}),

            # New pie charts for Individual Collaboration Score (Others) and (Self)
            html.Div([
                html.Div([
                    html.H3("Mean Individual Collaboration Score (Others)"),
                    dcc.Dropdown(
                        id='project-dropdown-individual-others',
                        options=[{'label': f'Project {i}', 'value': i} for i in valid_projects_for_all_meetings],
                        value=most_recent_project if most_recent_project in valid_projects_for_all_meetings else None,
                        style={'width': '80%', 'margin': '0 auto'}
                    ),
                    dcc.Graph(id='pie-chart-individual-others')
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H3("Mean Individual Collaboration Score (Self)"),
                    dcc.Dropdown(
                        id='project-dropdown-individual-self',
                        options=[{'label': f'Project {i}', 'value': i} for i in valid_projects_for_all_meetings],
                        value=most_recent_project if most_recent_project in valid_projects_for_all_meetings else None,
                        style={'width': '80%', 'margin': '0 auto'}
                    ),
                    dcc.Graph(id='pie-chart-individual-self')
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
            ], style={'text-align': 'center', 'margin-bottom': '20px'}),

            # New bar chart for interaction differences
            html.Div([
                html.H3("Interaction Difference Between Online and Offline Conditions"),
                dcc.Dropdown(
                    id='project-dropdown-interaction-diff',
                    options=[{'label': f'Project {i}', 'value': i} for i in valid_projects_for_all_meetings],
                    value=most_recent_project if most_recent_project in valid_projects_for_all_meetings else None,
                    style={'width': '80%', 'margin': '0 auto'}
                ),
                dcc.Graph(id='bar-chart-interaction-diff')
            ], style={'text-align': 'center', 'margin-bottom': '20px'})
        ])
    )

    @dash_app.callback(
        Output('pie-chart-speech', 'figure'),
        [Input('project-dropdown-speech', 'value')]
    )
    def update_speech_pie_chart(selected_project):
        if selected_project is None:
            selected_project = most_recent_project
        filtered_df = dataset[dataset['project'] == selected_project]
        speech_freq = filtered_df.groupby('speaker_number')['speech_frequency'].sum().reset_index()
        color_map = get_color_map(len(speech_freq))
        fig = go.Figure(data=[go.Pie(labels=speech_freq['speaker_number'], values=speech_freq['speech_frequency'], hole=.3)])
        fig.update_traces(marker=dict(colors=[color_map[int(speaker)] for speaker in speech_freq['speaker_number']]))
        fig.update_layout(title=f'Speech Frequency Distribution for Project {selected_project}')
        return fig

    @dash_app.callback(
        Output('pie-chart-interaction', 'figure'),
        [Input('project-dropdown-interaction', 'value')]
    )
    def update_interaction_pie_chart(selected_project):
        if selected_project is None:
            selected_project = most_recent_project
        filtered_df = dataset[(dataset['project'] == selected_project) & (dataset['speaker_id'] != dataset['next_speaker_id'])]
        interaction_freq = filtered_df.groupby('speaker_number')['normalized_interaction_frequency'].sum().reset_index()
        color_map = get_color_map(len(interaction_freq))
        fig = go.Figure(data=[go.Pie(labels=interaction_freq['speaker_number'], values=interaction_freq['normalized_interaction_frequency'], hole=.3)])
        fig.update_traces(marker=dict(colors=[color_map[int(speaker)] for speaker in interaction_freq['speaker_number']]))
        fig.update_layout(title=f'Interaction Frequency Distribution for Project {selected_project}')
        return fig

    @dash_app.callback(
        Output('pie-chart-degree-centrality', 'figure'),
        [Input('project-dropdown-degree-centrality', 'value')]
    )
    def update_degree_centrality_pie_chart(selected_project):
        if selected_project is None:
            selected_project = most_recent_project
        filtered_df = dataset[dataset['project'] == selected_project]
        degree_centrality_freq = filtered_df.groupby('speaker_number')['degree_centrality'].sum().reset_index()
        color_map = get_color_map(len(degree_centrality_freq))
        fig = go.Figure(data=[go.Pie(labels=degree_centrality_freq['speaker_number'], values=degree_centrality_freq['degree_centrality'], hole=.3)])
        fig.update_traces(marker=dict(colors=[color_map[int(speaker)] for speaker in degree_centrality_freq['speaker_number']]))
        fig.update_layout(title=f'Degree Centrality Distribution for Project {selected_project}')
        return fig

    @dash_app.callback(
        Output('pie-chart-individual-others', 'figure'),
        [Input('project-dropdown-individual-others', 'value')]
    )
    def update_individual_others_pie_chart(selected_project):
        if selected_project is None:
            selected_project = most_recent_project
        filtered_df = dataset[dataset['project'] == selected_project]
        others_scores = filtered_df.groupby('speaker_number')['individual_collaboration_score'].mean().reset_index()
        color_map = get_color_map(len(others_scores))
        fig = go.Figure(data=[go.Pie(labels=others_scores['speaker_number'], values=others_scores['individual_collaboration_score'], hole=.3)])
        fig.update_traces(marker=dict(colors=[color_map[int(speaker)] for speaker in others_scores['speaker_number']]))
        fig.update_layout(title=f'Individual Collaboration Score (Others) for Project {selected_project}')
        return fig

    @dash_app.callback(
        Output('pie-chart-individual-self', 'figure'),
        [Input('project-dropdown-individual-self', 'value')]
    )
    def update_individual_self_pie_chart(selected_project):
        if selected_project is None:
            selected_project = most_recent_project
        filtered_df = dataset[dataset['project'] == selected_project]
        self_scores = filtered_df[filtered_df['speaker_id'] == filtered_df['next_speaker_id']].groupby('speaker_number')['individual_collaboration_score'].mean().reset_index()
        color_map = get_color_map(len(self_scores))
        fig = go.Figure(data=[go.Pie(labels=self_scores['speaker_number'], values=self_scores['individual_collaboration_score'], hole=.3)])
        fig.update_traces(marker=dict(colors=[color_map[int(speaker)] for speaker in self_scores['speaker_number']]))
        fig.update_layout(title=f'Individual Collaboration Score (Self) for Project {selected_project}')
        return fig

    @dash_app.callback(
        Output('bar-chart-interaction-diff', 'figure'),
        [Input('project-dropdown-interaction-diff', 'value')]
    )
    def update_bar_chart_difference(selected_project):
        if selected_project is None:
            selected_project = most_recent_project
        online_meetings, offline_meetings = get_meetings_by_condition(dataset)
        online_meetings = dataset[(dataset['project'] == selected_project) & (dataset['meeting_number'].isin(online_meetings)) & (dataset['speaker_id'] != dataset['next_speaker_id'])]
        offline_meetings = dataset[(dataset['project'] == selected_project) & (dataset['meeting_number'].isin(offline_meetings)) & (dataset['speaker_id'] != dataset['next_speaker_id'])]
        online_interactions = online_meetings.groupby('speaker_number')['normalized_interaction_frequency'].mean().reset_index()
        offline_interactions = offline_meetings.groupby('speaker_number')['normalized_interaction_frequency'].mean().reset_index()
        interaction_diff = pd.merge(online_interactions, offline_interactions, on='speaker_number', suffixes=('_online', '_offline'))
        interaction_diff['diff'] = interaction_diff['normalized_interaction_frequency_offline'] - interaction_diff['normalized_interaction_frequency_online']
        interaction_diff = interaction_diff.sort_values(by='diff', ascending=False)
        color_map = get_color_map(len(interaction_diff))
        fig = go.Figure(data=[go.Bar(
            x=interaction_diff['speaker_number'].astype(str),
            y=interaction_diff['diff'],
            marker=dict(color=[color_map[int(speaker)] for speaker in interaction_diff['speaker_number']])
        )])
        fig.update_layout(title=f'Interaction Difference Between Online and Offline for Project {selected_project}',
                          xaxis_title='Speaker Number',
                          yaxis_title='Interaction Difference (Offline - Online)',
                          xaxis={'categoryorder':'total descending'})  # Ensure bars are sorted by y-values
        return fig
