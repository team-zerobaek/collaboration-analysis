from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors
from dash import dcc, html
import dash

dash_app = None
dataset = None
interaction_summary = None

def initialize_interaction_app(dash_app_instance, dataset_instance):
    global dash_app, dataset, interaction_summary
    dash_app = dash_app_instance
    dataset = dataset_instance

    # Filter out self-interactions
    dataset = dataset[dataset['speaker_number'] != dataset['next_speaker_id']]

    interaction_summary = dataset.groupby(['project', 'meeting_number', 'speaker_number', 'next_speaker_id'])['count'].sum().reset_index()
    interaction_summary = pd.merge(interaction_summary, dataset[['project', 'meeting_number', 'duration']].drop_duplicates(),
                                   on=['project', 'meeting_number'])
    interaction_summary['normalized_interaction_count'] = interaction_summary['count'] / interaction_summary['duration']

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

    dash_app.layout.children.append(html.Div(id='turn', children=[
        html.H1("How Much Interacted?"),
        html.Div([
            dcc.Dropdown(
                id='interaction-project-dropdown',
                options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                value=[i for i in dataset['project'].unique()],  # Initially select all projects
                placeholder="Select projects",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='interaction-meeting-dropdown',
                placeholder="Select meetings",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='interaction-speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.RadioItems(
                id='interaction-type-radio',
                options=[
                    {'label': 'Total', 'value': 'total'},
                    {'label': 'By Speakers', 'value': 'by_speakers'}
                ],
                value='total',  # Initially select the 'total' option
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('Reset', id='reset-interaction-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='interaction-frequency-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Normalized Interaction Frequencies Explanation
                - This line plot compares the normalized interaction frequencies across meetings for two different projects.

                    - X-axis: Represents the sequential number of meetings analyzed. Each point on the x-axis corresponds to a specific meeting.

                    - Y-axis: Indicates the average frequency of interactions normalized. This metric helps in understanding the relative intensity of interactions per meeting.

                    - Lines and Markers:

                        - Blue Line: Represents the normalized interaction frequency for Project 3 across meetings.

                        - Orange Line: Represents the normalized interaction frequency for Project 4 across meetings.

                - This visualization helps in comparing how interaction frequencies evolve over meetings. It highlights trends such as changes in interaction dynamics or potential differences in communication patterns over time.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px'})
    ]))

    @dash_app.callback(
        Output('interaction-meeting-dropdown', 'options'),
        [Input('interaction-project-dropdown', 'value')]
    )
    def set_interaction_meeting_options(selected_project):
        if not selected_project:
            return []
        meetings = dataset[dataset['project'].isin(selected_project)]['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('interaction-speaker-dropdown', 'options'),
        [Input('interaction-project-dropdown', 'value'),
         Input('interaction-meeting-dropdown', 'value')]
    )
    def set_interaction_speaker_options(selected_project, selected_meeting):
        if not selected_project:
            return []
        filtered_df = dataset[dataset['project'].isin(selected_project)]
        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        speakers = filtered_df['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    def create_interaction_figure(filtered_df, interaction_type, color_map):
        fig = go.Figure()

        if interaction_type == 'total':
            interaction_comparison_df = filtered_df.groupby(['project', 'meeting_number'])['normalized_interaction_count'].sum().reset_index()
            interaction_comparison_df_pivot = interaction_comparison_df.pivot(index='meeting_number', columns='project', values='normalized_interaction_count')
            interaction_comparison_df_pivot.columns = [f'Normalized_Interaction_Frequency_Project{i}' for i in interaction_comparison_df_pivot.columns]
            interaction_comparison_df_pivot.reset_index(inplace=True)

            meeting_numbers_with_data = interaction_comparison_df_pivot[(interaction_comparison_df_pivot > 0).any(axis=1)]['meeting_number']

            for column in interaction_comparison_df_pivot.columns[1:]:
                project_number = int(column.split('Project')[-1])
                fig.add_trace(go.Scatter(x=interaction_comparison_df_pivot['meeting_number'],
                                         y=interaction_comparison_df_pivot[column],
                                         mode='lines+markers',
                                         name=column,
                                         marker=dict(symbol='circle'),
                                         line=dict(color=color_map.get(project_number, 'black'))  # Use predefined color or default to black
                                         ))

            fig.update_layout(
                xaxis_title='Meeting Number',
                yaxis_title='Normalized Interaction Frequency',
                xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
                showlegend=True
            )
        else:
            speaker_interactions = filtered_df.groupby(['meeting_number', 'speaker_number'])['normalized_interaction_count'].sum().reset_index()
            for speaker in speaker_interactions['speaker_number'].unique():
                speaker_df = speaker_interactions[speaker_interactions['speaker_number'] == speaker]
                fig.add_trace(go.Scatter(x=speaker_df['meeting_number'],
                                         y=speaker_df['normalized_interaction_count'],
                                         mode='lines+markers',
                                         name=f'Speaker {speaker}'))

            meeting_numbers_with_data = speaker_interactions['meeting_number'].unique()

            fig.update_layout(
                xaxis_title='Meeting Number',
                yaxis_title='Normalized Interaction Frequency',
                xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
                showlegend=True
            )

        return fig

    def display_empty_interaction_graph():
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Normalized Interaction Frequency',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            showlegend=False
        )
        return fig

    @dash_app.callback(
        [Output('interaction-frequency-graph', 'figure'),
         Output('interaction-project-dropdown', 'value'),
         Output('interaction-meeting-dropdown', 'value'),
         Output('interaction-speaker-dropdown', 'value')],
        [Input('interaction-project-dropdown', 'value'),
         Input('interaction-meeting-dropdown', 'value'),
         Input('interaction-speaker-dropdown', 'value'),
         Input('interaction-type-radio', 'value'),
         Input('reset-interaction-button', 'n_clicks')]
    )
    def update_interaction_graph(selected_project, selected_meeting, selected_speakers, interaction_type, reset_clicks):
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'reset-interaction-button':
            return display_empty_interaction_graph(), None, None, None

        if not selected_project:
            return display_empty_interaction_graph(), selected_project, selected_meeting, selected_speakers

        filtered_df = interaction_summary

        if selected_project:
            filtered_df = filtered_df[filtered_df['project'].isin(selected_project)]
        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        if selected_speakers:
            filtered_df = filtered_df[filtered_df['speaker_number'].isin(selected_speakers)]

        fig = create_interaction_figure(filtered_df, interaction_type, color_map)

        if selected_meeting:
            bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)].groupby(['speaker_number'])['normalized_interaction_count'].sum().reset_index()
            colorscale = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
            bar_colors = bar_data['normalized_interaction_count'].apply(lambda x: mcolors.rgb2hex(colorscale(x / bar_data['normalized_interaction_count'].max())))
            fig = go.Figure(data=[go.Bar(
                x=bar_data['speaker_number'],
                y=bar_data['normalized_interaction_count'],
                marker_color=bar_colors
            )])
            fig.update_layout(
                xaxis_title='Speaker Number',
                yaxis_title='Normalized Interaction Frequency',
                xaxis=dict(tickmode='array', tickvals=bar_data['speaker_number'].unique()),  # Ensure x-ticks are displayed
                showlegend=False
            )

            color_bar = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(
                    colorscale=[[0, 'blue'], [1, 'red']],
                    cmin=0,
                    cmax=bar_data['normalized_interaction_count'].max(),
                    colorbar=dict(
                        title="Interaction Frequency",
                        titleside="right"
                    )
                ),
                hoverinfo='none'
            )
            fig.add_trace(color_bar)

        return fig, selected_project, selected_meeting, selected_speakers
