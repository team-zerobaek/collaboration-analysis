from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors
from dash import dcc, html
import dash

dash_app = None
dataset = None
speech_summary = None

def initialize_frequency_app(dash_app_instance, dataset_instance):
    global dash_app, dataset, speech_summary
    dash_app = dash_app_instance
    dataset = dataset_instance

    speech_summary = dataset.groupby(['project', 'meeting_number', 'speaker_number'])['speech_frequency'].sum().reset_index()
    num_speakers_per_meeting = dataset.groupby(['project', 'meeting_number'])['speaker_number'].nunique().reset_index()
    num_speakers_per_meeting.columns = ['project', 'meeting_number', 'num_speakers']
    speech_summary = pd.merge(speech_summary, num_speakers_per_meeting, on=['project', 'meeting_number'])
    speech_summary['adjusted_speech_frequency'] = speech_summary['speech_frequency'] / speech_summary['num_speakers']
    total_words_summary = dataset.groupby(['project', 'meeting_number'])['total_words'].first().reset_index()
    speech_summary = pd.merge(speech_summary, total_words_summary, on=['project', 'meeting_number'])
    meeting_durations = dataset.groupby(['project', 'meeting_number'])['duration'].first().reset_index()
    speech_summary = pd.merge(speech_summary, meeting_durations, on=['project', 'meeting_number'])
    speech_summary['normalized_speech_frequency'] = speech_summary['adjusted_speech_frequency'] / speech_summary['duration']

    color_map = {
        0: 'grey',
        1: 'purple',
        2: 'green',
        3: 'blue',
        4: 'red',
        5: 'orange'
    }

    dropdown_style = {
        'width': '200px'
    }

    dash_app.layout.children.append(html.Div(id='word', children=[
        html.H1("How Many Words Were Spoken?"),
        html.Div([
            dcc.Dropdown(
                id='speech-project-dropdown',
                options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                value=[i for i in dataset['project'].unique()],  # Initially select all projects
                placeholder="Select projects",
                multi=True,
                style=dropdown_style
            ),
            dcc.Dropdown(
                id='speech-meeting-dropdown',
                placeholder="Select meetings",
                multi=True,
                style=dropdown_style
            ),
            dcc.Dropdown(
                id='speech-speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style=dropdown_style
            ),
            dcc.RadioItems(
                id='speech-type-radio',
                options=[
                    {'label': 'Total', 'value': 'total'},
                    {'label': 'By Speakers', 'value': 'by_speakers'}
                ],
                value='total',
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('Select All Meetings', id='speech-select-all-meetings-button', n_clicks=0),
            html.Button('Select All Speakers', id='speech-select-all-speakers-button', n_clicks=0),
            html.Button('Reset', id='reset-speech-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='speech-frequency-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Normalized Speech Frequencies Explanation
                - This line plot explains the normalized speech frequencies.

                    - X-axis: Represents the sequential number of meetings analyzed. Each point on the x-axis corresponds to a specific meeting.

                    - Y-axis: Indicates the normalized speech frequency, which is calculated by dividing the total speech amount by the meeting duration.

                    - Lines and Markers:

                        - Blue Line: Represents the normalized interaction frequency for Project 3 across meetings.

                        - Orange Line: Represents the normalized interaction frequency for Project 4 across meetings.

                - This visualization helps in comparing how speech frequencies change over time during meetings. It highlights trends such as variations in speech activity or potential differences in communication patterns between the projects.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px'})
    ]))

    @dash_app.callback(
        Output('speech-meeting-dropdown', 'options'),
        [Input('speech-project-dropdown', 'value')]
    )
    def set_speech_meeting_options(selected_project):
        if not selected_project:
            return []
        meetings = dataset[dataset['project'].isin(selected_project)]['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('speech-speaker-dropdown', 'options'),
        [Input('speech-project-dropdown', 'value'),
         Input('speech-meeting-dropdown', 'value')]
    )
    def set_speech_speaker_options(selected_project, selected_meeting):
        if not selected_project:
            return []
        filtered_df = dataset[dataset['project'].isin(selected_project)]
        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        speakers = filtered_df['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    def create_speech_figure(filtered_df, speech_type, color_map):
        fig = go.Figure()

        if speech_type == 'total':
            speech_comparison_df = filtered_df.groupby(['project', 'meeting_number'])['normalized_speech_frequency'].sum().reset_index()
            speech_comparison_df_pivot = speech_comparison_df.pivot(index='meeting_number', columns='project', values='normalized_speech_frequency')
            speech_comparison_df_pivot.columns = [f'Normalized_Speech_Frequency_Project{i}' for i in speech_comparison_df_pivot.columns]
            speech_comparison_df_pivot.reset_index(inplace=True)

            meeting_numbers_with_data = speech_comparison_df_pivot[(speech_comparison_df_pivot > 0).any(axis=1)]['meeting_number']

            for column in speech_comparison_df_pivot.columns[1:]:
                project_number = int(column.split('Project')[-1])
                fig.add_trace(go.Scatter(x=speech_comparison_df_pivot['meeting_number'],
                                         y=speech_comparison_df_pivot[column],
                                         mode='lines+markers',
                                         name=column,
                                         marker=dict(symbol='circle'),
                                         line=dict(color=color_map.get(project_number, 'black'))  # Use predefined color or default to black
                                         ))

            fig.update_layout(
                xaxis_title='Meeting Number',
                yaxis_title='Normalized Speech Frequency',
                xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
                showlegend=True
            )
        else:
            speaker_interactions = filtered_df.groupby(['meeting_number', 'speaker_number'])['normalized_speech_frequency'].sum().reset_index()
            for speaker in speaker_interactions['speaker_number'].unique():
                speaker_df = speaker_interactions[speaker_interactions['speaker_number'] == speaker]
                fig.add_trace(go.Scatter(x=speaker_df['meeting_number'],
                                         y=speaker_df['normalized_speech_frequency'],
                                         mode='lines+markers',
                                         name=f'Speaker {speaker}'))

            meeting_numbers_with_data = speaker_interactions['meeting_number'].unique()

            fig.update_layout(
                xaxis_title='Meeting Number',
                yaxis_title='Normalized Speech Frequency',
                xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
                showlegend=True
            )

        return fig

    def display_empty_speech_graph():
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Normalized Speech Frequency',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            showlegend=False
        )
        return fig

    @dash_app.callback(
        [Output('speech-frequency-graph', 'figure'),
         Output('speech-project-dropdown', 'value'),
         Output('speech-meeting-dropdown', 'value'),
         Output('speech-speaker-dropdown', 'value')],
        [Input('speech-project-dropdown', 'value'),
         Input('speech-meeting-dropdown', 'value'),
         Input('speech-speaker-dropdown', 'value'),
         Input('speech-type-radio', 'value'),
         Input('reset-speech-button', 'n_clicks'),
         Input('speech-select-all-meetings-button', 'n_clicks'),
         Input('speech-select-all-speakers-button', 'n_clicks')],
        [State('speech-project-dropdown', 'value')]
    )
    def update_speech_graph(selected_project, selected_meeting, selected_speakers, speech_type, reset_clicks, select_all_meetings_clicks, select_all_speakers_clicks, state_project):
        ctx = dash.callback_context
        if ctx.triggered:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'reset-speech-button':
                return display_empty_speech_graph(), None, [], []
            if button_id == 'speech-select-all-meetings-button':
                meetings = dataset[dataset['project'].isin(selected_project)]['meeting_number'].unique()
                return dash.no_update, selected_project, list(meetings), dash.no_update
            if button_id == 'speech-select-all-speakers-button':
                speakers = dataset[dataset['project'].isin(selected_project)]['speaker_number'].unique()
                return dash.no_update, selected_project, dash.no_update, list(speakers)

        if not selected_project:
            return display_empty_speech_graph(), selected_project, selected_meeting, selected_speakers

        filtered_df = speech_summary

        if selected_project:
            filtered_df = filtered_df[filtered_df['project'].isin(selected_project)]
        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        if selected_speakers:
            filtered_df = filtered_df[filtered_df['speaker_number'].isin(selected_speakers)]

        fig = create_speech_figure(filtered_df, speech_type, color_map)

        if selected_meeting:
            bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)].groupby(['speaker_number'])['normalized_speech_frequency'].sum().reset_index()
            colorscale = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
            bar_colors = bar_data['normalized_speech_frequency'].apply(lambda x: mcolors.rgb2hex(colorscale(x / bar_data['normalized_speech_frequency'].max())))
            fig = go.Figure(data=[go.Bar(
                x=bar_data['speaker_number'],
                y=bar_data['normalized_speech_frequency'],
                marker_color=bar_colors
            )])
            fig.update_layout(
                xaxis_title='Speaker Number',
                yaxis_title='Normalized Speech Frequency',
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
                    cmax=bar_data['normalized_speech_frequency'].max(),
                    colorbar=dict(
                        title="Speech Frequency",
                        titleside="right"
                    )
                ),
                hoverinfo='none'
            )
            fig.add_trace(color_bar)

        return fig, selected_project, selected_meeting, selected_speakers