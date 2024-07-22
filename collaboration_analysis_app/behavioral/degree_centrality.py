from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors
import dash

dash_app = None
dataset = None

def initialize_degree_centrality_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    most_recent_project = dataset['project'].max()

    # Define a consistent color map for the projects
    color_map = {
        0: 'grey',
        1: 'purple',
        2: 'green',
        3: 'blue',
        4: 'red',
        5: 'orange'
        # Add more colors as needed
    }

    dash_app.layout.children.append(html.Div(id='degree', children=[
        html.H1("How Much Contributed?"),
        html.Div([
            dcc.Dropdown(
                id='degree-centrality-project-dropdown',
                options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                value=most_recent_project,  # Initially select the most recent project
                placeholder="Select a project",
                multi=False,  # Allow only one option to be selected
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='degree-centrality-meeting-dropdown',
                placeholder="Select meetings",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='degree-centrality-speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
            ),
            html.Button('Reset', id='reset-degree-centrality-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='degree-centrality-graph'),

        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Degree Centrality Graph Explanation
                This graph shows the degree centrality of each participant in the meeting, highlighting how many connections each participant has with others.
                - **Degree Centrality = 0**: Participants on that line did not interact with other participants in the meeting
                - **Degree Centrality = 1**: Participants on that line interacted with all other participants in the meeting.

                ### Axis Labels
                - **X-axis**: Indicates the session number of the meeting.
                - **Y-axis**: Indicates the Degree Centrality, with higher values representing greater interaction with other participants.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px', 'margin-bottom': '80px'})
    ]))

    def display_empty_degree_centrality_graph():
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Degree Centrality',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            showlegend=False
        )
        return fig

    @dash_app.callback(
        Output('degree-centrality-meeting-dropdown', 'options'),
        [Input('degree-centrality-project-dropdown', 'value')]
    )
    def set_meeting_options(selected_project):
        if not selected_project:
            return []
        filtered_df = dataset[dataset['project'] == selected_project]
        meetings = filtered_df['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('degree-centrality-speaker-dropdown', 'options'),
        [Input('degree-centrality-project-dropdown', 'value'),
         Input('degree-centrality-meeting-dropdown', 'value')]
    )
    def set_speaker_options(selected_project, selected_meetings):
        if not selected_project:
            return []
        filtered_df = dataset[dataset['project'] == selected_project]
        if selected_meetings:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meetings)]
        speakers = filtered_df['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    def create_degree_centrality_figure(filtered_df, color_map):
        fig = go.Figure()
        for speaker in filtered_df['speaker_number'].unique():
            speaker_df = filtered_df[filtered_df['speaker_number'] == speaker]
            fig.add_trace(go.Scatter(x=speaker_df['meeting_number'],
                                     y=speaker_df['degree_centrality'],
                                     mode='lines+markers',
                                     name=f'Speaker {speaker}'))

        # Calculate the average degree centrality for all speakers
        avg_degree_centrality = dataset[dataset['project'] == filtered_df['project'].iloc[0]].groupby('meeting_number')['degree_centrality'].mean().reset_index()

        fig.add_trace(go.Scatter(
            x=avg_degree_centrality['meeting_number'],
            y=avg_degree_centrality['degree_centrality'],
            mode='lines',
            name='Average Degree Centrality',
            line=dict(color='black', dash='dash')
        ))

        xticks = filtered_df['meeting_number'].unique()

        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Degree Centrality',
            xaxis=dict(tickmode='array', tickvals=xticks),
            showlegend=True
        )

        return fig

    @dash_app.callback(
        [Output('degree-centrality-graph', 'figure'),
         Output('degree-centrality-project-dropdown', 'value'),
         Output('degree-centrality-meeting-dropdown', 'value'),
         Output('degree-centrality-speaker-dropdown', 'value')],
        [Input('degree-centrality-project-dropdown', 'value'),
         Input('degree-centrality-meeting-dropdown', 'value'),
         Input('degree-centrality-speaker-dropdown', 'value'),
         Input('reset-degree-centrality-button', 'n_clicks')]
    )
    def update_degree_centrality_graph(selected_project, selected_meetings, selected_speakers, reset_clicks):
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'reset-degree-centrality-button':
            return display_empty_degree_centrality_graph(), None, [], []

        if not selected_project:
            selected_project = dataset['project'].max()

        filtered_df = dataset[dataset['project'] == selected_project]

        if selected_meetings:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meetings)]
        if selected_speakers:
            filtered_df = filtered_df[filtered_df['speaker_number'].isin(selected_speakers)]

        if filtered_df.empty:
            return display_empty_degree_centrality_graph(), selected_project, selected_meetings, selected_speakers

        fig = create_degree_centrality_figure(filtered_df, color_map)

        if selected_meetings:
            bar_data = filtered_df.groupby(['meeting_number', 'speaker_number'])['degree_centrality'].sum().reset_index()
            colorscale = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
            bar_colors = bar_data['degree_centrality'].apply(lambda x: mcolors.rgb2hex(colorscale(x / bar_data['degree_centrality'].max())))

            fig = go.Figure(data=[go.Bar(
                x=bar_data['speaker_number'],
                y=bar_data['degree_centrality'],
                marker_color=bar_colors
            )])
            fig.update_layout(
                xaxis_title='Speaker Number',
                yaxis_title='Degree Centrality',
                showlegend=False
            )

            color_bar = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    colorscale=[[0, 'blue'], [1, 'red']],
                    cmin=0,
                    cmax=bar_data['degree_centrality'].max(),
                    colorbar=dict(
                        title="Degree Centrality",
                        titleside="right"
                    )
                ),
                hoverinfo='none'
            )
            fig.add_trace(color_bar)

        return fig, selected_project, selected_meetings, selected_speakers
