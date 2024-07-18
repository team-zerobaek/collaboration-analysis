from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import ttest_ind
from dash import html, dcc

def get_meetings_by_condition(dataset):
    online_meetings = [1, 2, 3, 4, 5, 6, 7, 13, 14, 17]
    offline_meetings = [8, 9, 11, 12, 15, 16]
    return online_meetings, offline_meetings

def initialize_abtest_app(dash_app, dataset):
    # Ensure the normalized_interaction_frequency column exists and is calculated correctly
    if 'normalized_interaction_frequency' not in dataset.columns:
        dataset['normalized_interaction_frequency'] = dataset['count'] / dataset['duration']

    # Exclude self-interactions
    dataset = dataset[dataset['speaker_id'] != dataset['next_speaker_id']]

    def calculate_team_meeting_metrics(meetings):
        unique_speech_frequencies = meetings.groupby(['meeting_number', 'speaker_id'])['normalized_speech_frequency'].mean().reset_index()
        meeting_metrics = unique_speech_frequencies.groupby('meeting_number').agg({'normalized_speech_frequency': 'sum'}).reset_index()
        interaction_metrics = meetings.groupby('meeting_number').agg({'normalized_interaction_frequency': 'sum'}).reset_index()
        self_interactions = meetings[meetings['speaker_id'] == meetings['next_speaker_id']]
        total_self_interactions = self_interactions.groupby('meeting_number')['normalized_interaction_frequency'].sum().reset_index()
        interaction_metrics = interaction_metrics.merge(total_self_interactions, on='meeting_number', how='left', suffixes=('', '_self'))
        interaction_metrics['normalized_interaction_frequency'] = interaction_metrics['normalized_interaction_frequency'] - interaction_metrics['normalized_interaction_frequency_self'].fillna(0)
        interaction_metrics.drop(columns=['normalized_interaction_frequency_self'], inplace=True)
        combined_metrics = meeting_metrics.merge(interaction_metrics, on='meeting_number')
        return combined_metrics

    def calculate_individual_meeting_metrics(meetings):
        grouped = meetings.groupby(['meeting_number', 'speaker_id']).agg({'normalized_speech_frequency': 'mean', 'normalized_interaction_frequency': 'sum'}).reset_index()
        return grouped

    def perform_ttest(group1, group2):
        ttest_results = {}
        ttest_results['normalized_speech_frequency'] = ttest_ind(group1['normalized_speech_frequency'], group2['normalized_speech_frequency'], equal_var=False)
        ttest_results['normalized_interaction_frequency'] = ttest_ind(group1['normalized_interaction_frequency'], group2['normalized_interaction_frequency'], equal_var=False)
        return ttest_results

    def dataframe_generator(ttest_results, group1, group2, view_type):
        rows_speech = []
        rows_interaction = []
        if view_type == 'total':
            variables = ['normalized_speech_frequency', 'normalized_interaction_frequency']
            for var in variables:
                if var == 'normalized_speech_frequency':
                    row_online = {
                        'Group': 'Online',
                        'Mean': round(group1[var].mean(), 2),
                        'Std': round(group1[var].std(), 2),
                        'df': len(group1[var]) - 1,
                        't-statistic': round(ttest_results[var].statistic, 2),
                        'p-value': round(ttest_results[var].pvalue, 2)
                    }
                    row_offline = {
                        'Group': 'Offline',
                        'Mean': round(group2[var].mean(), 2),
                        'Std': round(group2[var].std(), 2),
                        'df': len(group2[var]) - 1,
                        't-statistic': '',
                        'p-value': ''
                    }
                    rows_speech.append(row_online)
                    rows_speech.append(row_offline)
                else:
                    row_online = {
                        'Group': 'Online',
                        'Mean': round(group1[var].mean(), 2),
                        'Std': round(group1[var].std(), 2),
                        'df': len(group1[var]) - 1,
                        't-statistic': round(ttest_results[var].statistic, 2),
                        'p-value': round(ttest_results[var].pvalue, 2)
                    }
                    row_offline = {
                        'Group': 'Offline',
                        'Mean': round(group2[var].mean(), 2),
                        'Std': round(group2[var].std(), 2),
                        'df': len(group2[var]) - 1,
                        't-statistic': '',
                        'p-value': ''
                    }
                    rows_interaction.append(row_online)
                    rows_interaction.append(row_offline)
        else:
            speakers = group1['speaker_id'].unique()
            for speaker in speakers:
                online_speaker = group1[group1['speaker_id'] == speaker]
                offline_speaker = group2[group2['speaker_id'] == speaker]

                if len(online_speaker) > 1 and len(offline_speaker) > 1:
                    ttest_speech = ttest_ind(online_speaker['normalized_speech_frequency'], offline_speaker['normalized_speech_frequency'], equal_var=False)
                    ttest_count = ttest_ind(online_speaker['normalized_interaction_frequency'], offline_speaker['normalized_interaction_frequency'], equal_var=False)
                else:
                    ttest_speech = ttest_count = None

                row_online = {
                    'Group': f'Online (Speaker {speaker})',
                    'Mean': round(online_speaker['normalized_speech_frequency'].mean(), 2),
                    'Std': round(online_speaker['normalized_speech_frequency'].std(), 2),
                    'df': len(online_speaker['normalized_speech_frequency']) - 1,
                    't-statistic': round(ttest_speech.statistic, 2) if ttest_speech else None,
                    'p-value': round(ttest_speech.pvalue, 2) if ttest_speech else None
                }
                row_offline = {
                    'Group': f'Offline (Speaker {speaker})',
                    'Mean': round(offline_speaker['normalized_speech_frequency'].mean(), 2),
                    'Std': round(offline_speaker['normalized_speech_frequency'].std(), 2),
                    'df': len(offline_speaker['normalized_speech_frequency']) - 1,
                    't-statistic': '',
                    'p-value': ''
                }
                rows_speech.append(row_online)
                rows_speech.append(row_offline)

                row_online = {
                    'Group': f'Online (Speaker {speaker})',
                    'Mean': round(online_speaker['normalized_interaction_frequency'].mean(), 2),
                    'Std': round(online_speaker['normalized_interaction_frequency'].std(), 2),
                    'df': len(online_speaker['normalized_interaction_frequency']) - 1,
                    't-statistic': round(ttest_count.statistic, 2) if ttest_count else None,
                    'p-value': round(ttest_count.pvalue, 2) if ttest_count else None
                }
                row_offline = {
                    'Group': f'Offline (Speaker {speaker})',
                    'Mean': round(offline_speaker['normalized_interaction_frequency'].mean(), 2),
                    'Std': round(offline_speaker['normalized_interaction_frequency'].std(), 2),
                    'df': len(offline_speaker['normalized_interaction_frequency']) - 1,
                    't-statistic': '',
                    'p-value': ''
                }
                rows_interaction.append(row_online)
                rows_interaction.append(row_offline)

        detailed_df_speech = pd.DataFrame(rows_speech)
        detailed_df_interaction = pd.DataFrame(rows_interaction)
        return detailed_df_speech, detailed_df_interaction

    @dash_app.callback(
        [Output('abtest-graph-speech', 'figure'),
         Output('abtest-graph-interaction', 'figure'),
         Output('abtest-table-speech', 'children'),
         Output('abtest-table-interaction', 'children')],
        [Input('abtest-view-type', 'value')]
    )
    def update_abtest_graph_table(view_type):
        online_meetings, offline_meetings = get_meetings_by_condition(dataset)
        online_meetings_df = dataset[dataset['meeting_number'].isin(online_meetings)]
        offline_meetings_df = dataset[dataset['meeting_number'].isin(offline_meetings)]

        if view_type == 'total':
            online_metrics = calculate_team_meeting_metrics(online_meetings_df)
            offline_metrics = calculate_team_meeting_metrics(offline_meetings_df)
        else:
            online_metrics = calculate_individual_meeting_metrics(online_meetings_df)
            offline_metrics = calculate_individual_meeting_metrics(offline_meetings_df)

        ttest_results = perform_ttest(online_metrics, offline_metrics)

        fig_speech = go.Figure()
        fig_interaction = go.Figure()

        if view_type == 'total':
            fig_speech.add_trace(go.Bar(
                x=['Online', 'Offline'],
                y=[online_metrics['normalized_speech_frequency'].mean(), offline_metrics['normalized_speech_frequency'].mean()],
                error_y=dict(type='data', array=[online_metrics['normalized_speech_frequency'].std(), offline_metrics['normalized_speech_frequency'].std()]),
                marker_color=['#2ca02c', '#d62728']
            ))

            fig_interaction.add_trace(go.Bar(
                x=['Online', 'Offline'],
                y=[online_metrics['normalized_interaction_frequency'].mean(), offline_metrics['normalized_interaction_frequency'].mean()],
                error_y=dict(type='data', array=[online_metrics['normalized_interaction_frequency'].std(), offline_metrics['normalized_interaction_frequency'].std()]),
                marker_color=['#2ca02c', '#d62728']
            ))

            fig_speech.update_layout(
                xaxis_title='Condition',
                yaxis_title='Normalized Speech Frequency',
                showlegend=False
            )

            fig_interaction.update_layout(
                xaxis_title='Condition',
                yaxis_title='Normalized Interaction Frequency',
                showlegend=False
            )
        else:
            for speaker in online_metrics['speaker_id'].unique():
                fig_speech.add_trace(go.Bar(
                    x=['Online', 'Offline'],
                    y=[online_metrics[online_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].mean(),
                       offline_metrics[offline_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].mean()],
                    error_y=dict(type='data', array=[online_metrics[online_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].std(),
                                                     offline_metrics[offline_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].std()]),
                    name=f'Speaker {speaker}'
                ))

                fig_interaction.add_trace(go.Bar(
                    x=['Online', 'Offline'],
                    y=[online_metrics[online_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].mean(),
                       offline_metrics[offline_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].mean()],
                    error_y=dict(type='data', array=[online_metrics[online_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].std(),
                                                     offline_metrics[offline_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].std()]),
                    name=f'Speaker {speaker}'
                ))

            fig_speech.update_layout(
                xaxis_title='Condition',
                yaxis_title='Normalized Speech Frequency',
                showlegend=True
            )

            fig_interaction.update_layout(
                xaxis_title='Condition',
                yaxis_title='Normalized Interaction Frequency',
                showlegend=True
            )

        detailed_df_speech, detailed_df_interaction = dataframe_generator(ttest_results, online_metrics, offline_metrics, view_type)

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

    dash_app.layout.children.append(html.Div(id='onoff', children=[
        html.H2("A/B Test: Online vs. Offline", style={'text-align': 'center'})
    ]))

    dash_app.layout.children.append(html.Div([
        dcc.RadioItems(
            id='abtest-view-type',
            options=[
                {'label': 'Total', 'value': 'total'},
                {'label': 'By Speakers', 'value': 'by_speakers'}
            ],
            value='total',
            labelStyle={'display': 'inline-block'}
        ),
    ], style={'text-align': 'center', 'margin-bottom': '20px'}))

    dash_app.layout.children.append(html.Div([
        html.Div([
            dcc.Graph(id='abtest-graph-speech'),
            html.Div(id='abtest-table-speech'),
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
            dcc.Graph(id='abtest-graph-interaction'),
            html.Div(id='abtest-table-interaction'),
            html.Details([
                html.Summary('Text-based vs. Voice-based - Speech Frequency',style={'margin-top': '20px','margin-bottom': '10px'}),
                dcc.Markdown("""
                    #### Speech Frequency - Total
                    - By observing the changes in the Speech Frequency values in the Total graph, we can determine how conducting a meeting based on text or voice affects the overall speech frequency in the meeting.

                    #### Speech Frequency - by Speaker
                    - The by Speaker graph allows us to see the differences in speech frequency for each speaker. This helps determine in which meeting format a specific speaker is more actively speaking.
                """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
            ], style={'margin-top': '5px', 'margin-bottom': '40px'})
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}))

    return dash_app
