# upload/preview.py
from dash.dependencies import Input, Output
from dash import dcc, html
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors

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

    most_recent_project = dataset['project'].max()

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
                        options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                        value=most_recent_project,
                        style={'width': '80%', 'margin': '0 auto'}
                    ),
                    dcc.Graph(id='pie-chart-individual-others')
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
                html.Div([
                    html.H3("Mean Individual Collaboration Score (Self)"),
                    dcc.Dropdown(
                        id='project-dropdown-individual-self',
                        options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                        value=most_recent_project,
                        style={'width': '80%', 'margin': '0 auto'}
                    ),
                    dcc.Graph(id='pie-chart-individual-self')
                ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
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
        interaction_freq = filtered_df.groupby('speaker_number')['count'].sum().reset_index()
        color_map = get_color_map(len(interaction_freq))
        fig = go.Figure(data=[go.Pie(labels=interaction_freq['speaker_number'], values=interaction_freq['count'], hole=.3)])
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
