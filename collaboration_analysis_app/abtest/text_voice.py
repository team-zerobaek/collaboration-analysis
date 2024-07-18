from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import ttest_ind
from dash import html, dcc

def initialize_text_voice_app(dash_app, dataset_voice, dataset_text):
    def calculate_team_meeting_metrics(meetings):
        unique_speech_frequencies = meetings.groupby(['meeting_number', 'speaker_id'])['speech_frequency'].mean().reset_index()
        meeting_metrics = unique_speech_frequencies.groupby('meeting_number').agg({'speech_frequency': 'sum'}).reset_index()
        interaction_metrics = meetings.groupby('meeting_number').agg({'count': 'sum'}).reset_index()
        self_interactions = meetings[meetings['speaker_id'] == meetings['next_speaker_id']]
        total_self_interactions = self_interactions.groupby('meeting_number')['count'].sum().reset_index()
        interaction_metrics = interaction_metrics.merge(total_self_interactions, on='meeting_number', how='left', suffixes=('', '_self'))
        interaction_metrics['count'] = interaction_metrics['count'] - interaction_metrics['count_self'].fillna(0)
        interaction_metrics.drop(columns=['count_self'], inplace=True)
        combined_metrics = meeting_metrics.merge(interaction_metrics, on='meeting_number')
        return combined_metrics

    def calculate_individual_metrics(meetings, meeting_count):
        unique_speech_frequencies = meetings.groupby(['meeting_number', 'speaker_id'])['speech_frequency'].mean().reset_index()
        individual_metrics = unique_speech_frequencies.groupby('speaker_id').agg({'speech_frequency': 'sum'}).reset_index()
        individual_metrics['speech_frequency'] /= meeting_count
        interaction_metrics = meetings.groupby('speaker_id').agg({'count': 'sum'}).reset_index()
        self_interactions = meetings[meetings['speaker_id'] == meetings['next_speaker_id']]
        total_self_interactions = self_interactions.groupby('speaker_id')['count'].sum().reset_index()
        interaction_metrics = interaction_metrics.merge(total_self_interactions, on='speaker_id', how='left', suffixes=('', '_self'))
        interaction_metrics['count'] = interaction_metrics['count'] - interaction_metrics['count_self'].fillna(0)
        interaction_metrics.drop(columns=['count_self'], inplace=True)
        interaction_metrics['count'] /= meeting_count
        combined_metrics = individual_metrics.merge(interaction_metrics, on='speaker_id')
        return combined_metrics

    def calculate_individual_meeting_metrics(meetings):
        grouped = meetings.groupby(['meeting_number', 'speaker_id']).agg({'speech_frequency': 'mean', 'count': 'sum'}).reset_index()
        return grouped

    def perform_ttest(group1, group2):
        ttest_results = {}
        ttest_results['speech_frequency'] = ttest_ind(group1['speech_frequency'], group2['speech_frequency'], equal_var=False)
        ttest_results['count'] = ttest_ind(group1['count'], group2['count'], equal_var=False)
        return ttest_results

    def dataframe_generator(ttest_results, group1, group2, view_type):
        rows_speech = []
        rows_interaction = []
        if view_type == 'total':
            variables = ['speech_frequency', 'count']
            for var in variables:
                if var == 'speech_frequency':
                    row_voice = {
                        'Group': 'Voice',
                        'Mean': round(group1[var].mean(), 2),
                        'Std': round(group1[var].std(), 2),
                        'df': len(group1[var]) - 1,
                        't-statistic': round(ttest_results[var].statistic, 2),
                        'p-value': round(ttest_results[var].pvalue, 2)
                    }
                    row_text = {
                        'Group': 'Text',
                        'Mean': round(group2[var].mean(), 2),
                        'Std': round(group2[var].std(), 2),
                        'df': len(group2[var]) - 1,
                        't-statistic': '',
                        'p-value': ''
                    }
                    rows_speech.append(row_voice)
                    rows_speech.append(row_text)
                else:
                    row_voice = {
                        'Group': 'Voice',
                        'Mean': round(group1[var].mean(), 2),
                        'Std': round(group1[var].std(), 2),
                        'df': len(group1[var]) - 1,
                        't-statistic': round(ttest_results[var].statistic, 2),
                        'p-value': round(ttest_results[var].pvalue, 2)
                    }
                    row_text = {
                        'Group': 'Text',
                        'Mean': round(group2[var].mean(), 2),
                        'Std': round(group2[var].std(), 2),
                        'df': len(group2[var]) - 1,
                        't-statistic': '',
                        'p-value': ''
                    }
                    rows_interaction.append(row_voice)
                    rows_interaction.append(row_text)
        else:
            speakers = group1['speaker_id'].unique()
            for speaker in speakers:
                voice_speaker = group1[group1['speaker_id'] == speaker]
                text_speaker = group2[group2['speaker_id'] == speaker]

                if len(voice_speaker) > 1 and len(text_speaker) > 1:
                    ttest_speech = ttest_ind(voice_speaker['speech_frequency'], text_speaker['speech_frequency'], equal_var=False)
                    ttest_count = ttest_ind(voice_speaker['count'], text_speaker['count'], equal_var=False)
                else:
                    ttest_speech = ttest_count = None

                row_voice = {
                    'Group': f'Voice (Speaker {speaker})',
                    'Mean': round(voice_speaker['speech_frequency'].mean(), 2),
                    'Std': round(voice_speaker['speech_frequency'].std(), 2),
                    'df': len(voice_speaker['speech_frequency']) - 1,
                    't-statistic': round(ttest_speech.statistic, 2) if ttest_speech else None,
                    'p-value': round(ttest_speech.pvalue, 2) if ttest_speech else None
                }
                row_text = {
                    'Group': f'Text (Speaker {speaker})',
                    'Mean': round(text_speaker['speech_frequency'].mean(), 2),
                    'Std': round(text_speaker['speech_frequency'].std(), 2),
                    'df': len(text_speaker['speech_frequency']) - 1,
                    't-statistic': '',
                    'p-value': ''
                }
                rows_speech.append(row_voice)
                rows_speech.append(row_text)

                row_voice = {
                    'Group': f'Voice (Speaker {speaker})',
                    'Mean': round(voice_speaker['count'].mean(), 2),
                    'Std': round(voice_speaker['count'].std(), 2),
                    'df': len(voice_speaker['count']) - 1,
                    't-statistic': round(ttest_count.statistic, 2) if ttest_count else None,
                    'p-value': round(ttest_count.pvalue, 2) if ttest_count else None
                }
                row_text = {
                    'Group': f'Text (Speaker {speaker})',
                    'Mean': round(text_speaker['count'].mean(), 2),
                    'Std': round(text_speaker['count'].std(), 2),
                    'df': len(text_speaker['count']) - 1,
                    't-statistic': '',
                    'p-value': ''
                }
                rows_interaction.append(row_voice)
                rows_interaction.append(row_text)

        detailed_df_speech = pd.DataFrame(rows_speech)
        detailed_df_interaction = pd.DataFrame(rows_interaction)
        return detailed_df_speech, detailed_df_interaction

    @dash_app.callback(
        [Output('text-voice-graph-speech', 'figure'),
         Output('text-voice-graph-interaction', 'figure'),
         Output('text-voice-table-speech', 'children'),
         Output('text-voice-table-interaction', 'children')],
        [Input('text-voice-view-type', 'value')]
    )
    def update_text_voice_graph_table(view_type):
        voice_meetings = dataset_voice[dataset_voice['meeting_number'] >= 10]
        text_meetings = dataset_text[dataset_text['meeting_number'] >= 10]

        if view_type == 'total':
            voice_metrics = calculate_team_meeting_metrics(voice_meetings)
            text_metrics = calculate_team_meeting_metrics(text_meetings)
        else:
            voice_metrics = calculate_individual_meeting_metrics(voice_meetings)
            text_metrics = calculate_individual_meeting_metrics(text_meetings)

        ttest_results = perform_ttest(voice_metrics, text_metrics)

        fig_speech = go.Figure()
        fig_interaction = go.Figure()

        if view_type == 'total':
            fig_speech.add_trace(go.Bar(
                x=['Voice', 'Text'],
                y=[voice_metrics['speech_frequency'].mean(), text_metrics['speech_frequency'].mean()],
                error_y=dict(type='data', array=[voice_metrics['speech_frequency'].std(), text_metrics['speech_frequency'].std()]),
                marker_color=['#1f77b4', '#ff7f0e']
            ))

            fig_interaction.add_trace(go.Bar(
                x=['Voice', 'Text'],
                y=[voice_metrics['count'].mean(), text_metrics['count'].mean()],
                error_y=dict(type='data', array=[voice_metrics['count'].std(), text_metrics['count'].std()]),
                marker_color=['#1f77b4', '#ff7f0e']
            ))

            speech_y_max = max(voice_metrics['speech_frequency'].mean() + 1.1 * voice_metrics['speech_frequency'].std(),
                               text_metrics['speech_frequency'].mean() + 1.1 * text_metrics['speech_frequency'].std())
            interaction_y_max = max(voice_metrics['count'].mean() + 1.1 * voice_metrics['count'].std(),
                                    text_metrics['count'].mean() + 1.1 * text_metrics['count'].std())

            fig_speech.update_layout(
                xaxis_title='Condition',
                yaxis_title=' Speech Frequency',
                yaxis=dict(range=[0, speech_y_max]),
                showlegend=False
            )

            fig_interaction.update_layout(
                xaxis_title='Condition',
                yaxis_title=' Interaction Count',
                yaxis=dict(range=[0, interaction_y_max]),
                showlegend=False
            )
        else:
            for speaker in voice_metrics['speaker_id'].unique():
                fig_speech.add_trace(go.Bar(
                    x=['Voice', 'Text'],
                    y=[voice_metrics[voice_metrics['speaker_id'] == speaker]['speech_frequency'].mean(),
                       text_metrics[text_metrics['speaker_id'] == speaker]['speech_frequency'].mean()],
                    error_y=dict(type='data', array=[voice_metrics[voice_metrics['speaker_id'] == speaker]['speech_frequency'].std(),
                                                     text_metrics[text_metrics['speaker_id'] == speaker]['speech_frequency'].std()]),
                    name=f'Speaker {speaker}'
                ))

                fig_interaction.add_trace(go.Bar(
                    x=['Voice', 'Text'],
                    y=[voice_metrics[voice_metrics['speaker_id'] == speaker]['count'].mean(),
                       text_metrics[text_metrics['speaker_id'] == speaker]['count'].mean()],
                    error_y=dict(type='data', array=[voice_metrics[voice_metrics['speaker_id'] == speaker]['count'].std(),
                                                     text_metrics[text_metrics['speaker_id'] == speaker]['count'].std()]),
                    name=f'Speaker {speaker}'
                ))

            fig_speech.update_layout(
                xaxis_title='Condition',
                yaxis_title=' Speech Frequency',
                showlegend=True
            )

            fig_interaction.update_layout(
                xaxis_title='Condition',
                yaxis_title=' Interaction Count',
                showlegend=True
            )

        detailed_df_speech, detailed_df_interaction = dataframe_generator(ttest_results, voice_metrics, text_metrics, view_type)

        speech_table = html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in ['Group', 'Mean', 'Std', 'df', 't-statistic', 'p-value']])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(detailed_df_speech.iloc[i][col]) for col in ['Group', 'Mean', 'Std', 'df', 't-statistic', 'p-value']
                ]) for i in range(len(detailed_df_speech))
            ])
        ], style={'margin': 'auto', 'text-align': 'center', 'width': '100%', 'white-space': 'nowrap'})

        interaction_table = html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in ['Group', 'Mean', 'Std', 'df', 't-statistic', 'p-value']])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(detailed_df_interaction.iloc[i][col]) for col in ['Group', 'Mean', 'Std', 'df', 't-statistic', 'p-value']
                ]) for i in range(len(detailed_df_interaction))
            ])
        ], style={'margin': 'auto', 'text-align': 'center', 'width': '100%', 'white-space': 'nowrap'})

        return fig_speech, fig_interaction, speech_table, interaction_table

    dash_app.layout.children.append(html.Div(id ='voice', children=[html.H2("A/B Test: Text-based vs. Voice-based", style={'text-align': 'center'})]))
    dash_app.layout.children.append(html.Div([
        dcc.RadioItems(
            id='text-voice-view-type',
            options=[
                {'label': 'Total', 'value': 'total'},
                {'label': 'By Speakers', 'value': 'by_speakers'}
            ],
            value='total',
            labelStyle={'display': 'inline-block'}
        ),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}))
    dash_app.layout.children.append(html.Div([
        html.Div([
            dcc.Graph(id='text-voice-graph-speech'),
            html.Div(id='text-voice-table-speech'),
            html.Details([
                html.Summary('Text-based vs. Voice-based - Speech Frequency',style={'margin-top': '20px','margin-bottom': '10px'}),
                dcc.Markdown("""
                    #### Speech Frequency - Total  
                    - By observing the changes in the Speech Frequency values in the Total graph, we can determine how conducting a meeting based on text or voice affects the overall speech frequency in the meeting.

                    #### Speech Frequency - by Speaker
                    - The by Speaker graph allows us to see the differences in speech frequency for each speaker. This helps determine in which meeting format a specific speaker is more actively speaking.
                """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
            ], style={'margin-top': '5px', 'margin-bottom': '40px'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='text-voice-graph-interaction'),
            html.Div(id='text-voice-table-interaction'),
            html.Details([
                html.Summary('Text-based vs. Voice-based - Interaction Count',style={'margin-top': '10px','margin-bottom': '10px'}),
                dcc.Markdown("""
                    #### Interaction Count - Total
                    - By observing the changes in the Interaction Count values in the Total graph, we can determine how conducting a meeting based on text or voice affects the overall interaction count in the meeting.

                    #### Interaction Count - by Speaker
                    - The by Speaker graph allows us to see the differences in interaction count for each speaker. This helps determine in which meeting format a specific speaker is more actively communicating with other speakers.
                """, style={'backgroundColor': '#f0f0f0', 'padding': '20px', 'borderRadius': '5px'})
            ], style={'margin-top': '5px', 'margin-bottom': '40px'})
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}))
