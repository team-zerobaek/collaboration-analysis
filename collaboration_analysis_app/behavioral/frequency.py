from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import matplotlib.colors as mcolors
from dash import dcc, html

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

    dash_app.layout.children.append(html.Div(id ='word', children=[
        html.H1("How Many Words Were Spoken?"),
        html.Div([
            dcc.Dropdown(
                id='speech-project-dropdown',
                options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                placeholder="Select projects",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='speech-meeting-dropdown',
                placeholder="Select meetings",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='speech-speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
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
            html.Button('Reset', id='reset-speech-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='speech-frequency-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Middle Title
                - Content 1
                - Content 2
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

    @dash_app.callback(
        [Output('speech-project-dropdown', 'value'),
         Output('speech-meeting-dropdown', 'value'),
         Output('speech-speaker-dropdown', 'value')],
        [Input('reset-speech-button', 'n_clicks')]
    )
    def reset_speech_filters(n_clicks):
        return None, None, None

    @dash_app.callback(
        Output('speech-frequency-graph', 'figure'),
        [Input('speech-project-dropdown', 'value'),
         Input('speech-meeting-dropdown', 'value'),
         Input('speech-speaker-dropdown', 'value'),
         Input('speech-type-radio', 'value')]
    )
    def update_speech_graph(selected_project, selected_meeting, selected_speakers, speech_type):
        if speech_type == 'total':
            filtered_df = speech_summary
            if not selected_project:
                # Show all projects when "Total" is selected
                selected_project = dataset['project'].unique()
            filtered_df = filtered_df[filtered_df['project'].isin(selected_project)]
        else:
            if not selected_project:
                # Show most recent project (project 4) when "By Speakers" is selected
                selected_project = [4]
            filtered_df = speech_summary[speech_summary['project'].isin(selected_project)]

        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        if selected_speakers:
            filtered_df = filtered_df[filtered_df['speaker_number'].isin(selected_speakers)]

        fig = go.Figure()

        if speech_type == 'total':
            speech_comparison_df = filtered_df.groupby(['project', 'meeting_number'])['normalized_speech_frequency'].sum().reset_index()
            speech_comparison_df_pivot = speech_comparison_df.pivot(index='meeting_number', columns='project', values='normalized_speech_frequency')
            speech_comparison_df_pivot.columns = [f'Normalized_Speech_Frequency_Project{i}' for i in speech_comparison_df_pivot.columns]
            speech_comparison_df_pivot.reset_index(inplace=True)

            meeting_numbers_with_data = speech_comparison_df_pivot[(speech_comparison_df_pivot > 0).any(axis=1)]['meeting_number']

            for column in speech_comparison_df_pivot.columns[1:]:
                fig.add_trace(go.Scatter(x=speech_comparison_df_pivot['meeting_number'],
                                         y=speech_comparison_df_pivot[column],
                                         mode='lines+markers',
                                         name=column))

            fig.update_layout(
                xaxis_title='Meeting Number',
                yaxis_title='Normalized Speech Frequency',
                xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
                showlegend=True
            )
        else:
            for speaker in filtered_df['speaker_number'].unique():
                speaker_df = filtered_df[filtered_df['speaker_number'] == speaker]
                fig.add_trace(go.Scatter(x=speaker_df['meeting_number'],
                                         y=speaker_df['normalized_speech_frequency'],
                                         mode='lines+markers',
                                         name=f'Speaker {speaker}'))

            meeting_numbers_with_data = filtered_df['meeting_number'].unique()

            fig.update_layout(
                xaxis_title='Meeting Number',
                yaxis_title='Normalized Speech Frequency',
                xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
                showlegend=True
            )

        if selected_meeting:
            bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
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

        return fig
