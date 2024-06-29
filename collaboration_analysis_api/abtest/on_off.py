from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import ttest_ind
from dash import html, dcc

def initialize_abtest_app(dash_app, dataset):
    def calculate_team_meeting_metrics(meetings):
        unique_speech_frequencies = meetings.groupby(['meeting_number', 'speaker_id'])['normalized_speech_frequency'].mean().reset_index()
        meeting_metrics = unique_speech_frequencies.groupby('meeting_number').agg({'normalized_speech_frequency': 'sum'}).reset_index()
        interaction_metrics = meetings.groupby('meeting_number').agg({'count': 'sum'}).reset_index()
        self_interactions = meetings[meetings['speaker_id'] == meetings['next_speaker_id']]
        total_self_interactions = self_interactions.groupby('meeting_number')['count'].sum().reset_index()
        interaction_metrics = interaction_metrics.merge(total_self_interactions, on='meeting_number', how='left', suffixes=('', '_self'))
        interaction_metrics['count'] = interaction_metrics['count'] - interaction_metrics['count_self'].fillna(0)
        interaction_metrics.drop(columns=['count_self'], inplace=True)
        combined_metrics = meeting_metrics.merge(interaction_metrics, on='meeting_number')
        return combined_metrics

    def calculate_individual_metrics(meetings, meeting_count):
        unique_speech_frequencies = meetings.groupby(['meeting_number', 'speaker_id'])['normalized_speech_frequency'].mean().reset_index()
        individual_metrics = unique_speech_frequencies.groupby('speaker_id').agg({'normalized_speech_frequency': 'sum'}).reset_index()
        individual_metrics['normalized_speech_frequency'] /= meeting_count
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
        grouped = meetings.groupby(['meeting_number', 'speaker_id']).agg({'normalized_speech_frequency': 'mean', 'count': 'sum'}).reset_index()
        return grouped

    def perform_ttest(group1, group2):
        ttest_results = {}
        ttest_results['normalized_speech_frequency'] = ttest_ind(group1['normalized_speech_frequency'], group2['normalized_speech_frequency'], equal_var=False)
        ttest_results['count'] = ttest_ind(group1['count'], group2['count'], equal_var=False)
        return ttest_results

    def dataframe_generator(ttest_results, group1, group2, view_type):
        rows_speech = []
        rows_interaction = []
        if view_type == 'total':
            variables = ['normalized_speech_frequency', 'count']
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
                    ttest_count = ttest_ind(online_speaker['count'], offline_speaker['count'], equal_var=False)
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
                    'Mean': round(online_speaker['count'].mean(), 2),
                    'Std': round(online_speaker['count'].std(), 2),
                    'df': len(online_speaker['count']) - 1,
                    't-statistic': round(ttest_count.statistic, 2) if ttest_count else None,
                    'p-value': round(ttest_count.pvalue, 2) if ttest_count else None
                }
                row_offline = {
                    'Group': f'Offline (Speaker {speaker})',
                    'Mean': round(offline_speaker['count'].mean(), 2),
                    'Std': round(offline_speaker['count'].std(), 2),
                    'df': len(offline_speaker['count']) - 1,
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
        online_meetings = dataset[dataset['meeting_number'].isin([1, 2, 3, 4, 5, 6, 7, 13, 14])]
        offline_meetings = dataset[dataset['meeting_number'].isin([8, 9, 11, 12])]

        if view_type == 'total':
            online_metrics = calculate_team_meeting_metrics(online_meetings)
            offline_metrics = calculate_team_meeting_metrics(offline_meetings)
        else:
            online_metrics = calculate_individual_meeting_metrics(online_meetings)
            offline_metrics = calculate_individual_meeting_metrics(offline_meetings)

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
                y=[online_metrics['count'].mean(), offline_metrics['count'].mean()],
                error_y=dict(type='data', array=[online_metrics['count'].std(), offline_metrics['count'].std()]),
                marker_color=['#2ca02c', '#d62728']
            ))

            speech_y_max = max(online_metrics['normalized_speech_frequency'].mean() + 1.1 * online_metrics['normalized_speech_frequency'].std(),
                               offline_metrics['normalized_speech_frequency'].mean() + 1.1 * offline_metrics['normalized_speech_frequency'].std())
            interaction_y_max = max(online_metrics['count'].mean() + 1.1 * online_metrics['count'].std(),
                                    offline_metrics['count'].mean() + 1.1 * offline_metrics['count'].std())

            fig_speech.update_layout(
                title='A/B Test: Normalized Speech Frequency',
                xaxis_title='Condition',
                yaxis_title='Normalized Speech Frequency',
                yaxis=dict(range=[0, speech_y_max]),
                showlegend=False
            )

            fig_interaction.update_layout(
                title='A/B Test: Normalized Interaction Count',
                xaxis_title='Condition',
                yaxis_title='Normalized Interaction Count',
                yaxis=dict(range=[0, interaction_y_max]),
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
                    y=[online_metrics[online_metrics['speaker_id'] == speaker]['count'].mean(),
                       offline_metrics[offline_metrics['speaker_id'] == speaker]['count'].mean()],
                    error_y=dict(type='data', array=[online_metrics[online_metrics['speaker_id'] == speaker]['count'].std(),
                                                     offline_metrics[offline_metrics['speaker_id'] == speaker]['count'].std()]),
                    name=f'Speaker {speaker}'
                ))

            fig_speech.update_layout(
                title='A/B Test: Normalized Speech Frequency by Speaker',
                xaxis_title='Condition',
                yaxis_title='Normalized Speech Frequency',
                showlegend=True
            )

            fig_interaction.update_layout(
                title='A/B Test: Normalized Interaction Count by Speaker',
                xaxis_title='Condition',
                yaxis_title='Normalized Interaction Count',
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

    dash_app.layout.children.append(html.Div(id ='onoff', children=[html.H2("A/B Test: Online vs. Offline", style={'text-align': 'center'})]))
    
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
                html.Summary('Online vs. Offline - Normalized Speech Frequency',style={'margin-top': '20px','margin-bottom': '10px'}),
                dcc.Markdown("""
                    **Normalized Speech Frequency - Total**  
                    Normalized Speech Frequency refers to the speech amount during a meeting normalized by dividing it by the duration. By observing the changes in the Normalized Speech Frequency values in the Total graph, we can determine how the difference between online and offline environments affected the overall speech frequency in the meeting.  
                      
                    **Normalized Speech Frequency - by Speaker**  
                    The by Speaker graph allows us to see the differences in speech frequency for each speaker. This helps determine if the change had a significant effect on specific individuals.  
                """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
            ], style={'margin-top': '5px', 'margin-bottom': '10px'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='abtest-graph-interaction'),
            html.Div(id='abtest-table-interaction'),
            html.Details([
                html.Summary('Online vs. Offline - Normalized Interaction Count',style={'margin-top': '20px','margin-bottom': '10px'}),
                dcc.Markdown("""
                    **Normalized Interaction Count - Total**  
                    Normalized Interaction Count refers to the number of interactions by the speakers. This allows us to see how much interaction occurred between speakers during the meeting. By observing the changes in the Normalized Interaction Count values in the Total graph, we can determine how the difference between online and offline environments affected the overall interaction count in the meeting.  
                      
                    **Normalized Interaction Count - by Speaker**  
                    The by Speaker graph allows us to see the interaction count for each speaker. This helps determine if the change had a significant effect on specific individuals.  
                """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
            ], style={'margin-top': '5px', 'margin-bottom': '10px'})
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}))

    