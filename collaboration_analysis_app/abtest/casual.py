# abtest/casual.py
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
from scipy.stats import ttest_ind
from dash import html, dcc

def initialize_casual_app(dash_app, dataset):
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
                    row_not_used = {
                        'Group': 'Not Used',
                        'Mean': round(group1[var].mean(), 2),
                        'Std': round(group1[var].std(), 2),
                        'df': len(group1[var]) - 1,
                        't-statistic': round(ttest_results[var].statistic, 2),
                        'p-value': round(ttest_results[var].pvalue, 2)
                    }
                    row_used = {
                        'Group': 'Used',
                        'Mean': round(group2[var].mean(), 2),
                        'Std': round(group2[var].std(), 2),
                        'df': len(group2[var]) - 1,
                        't-statistic': '',
                        'p-value': ''
                    }
                    rows_speech.append(row_not_used)
                    rows_speech.append(row_used)
                else:
                    row_not_used = {
                        'Group': 'Not Used',
                        'Mean': round(group1[var].mean(), 2),
                        'Std': round(group1[var].std(), 2),
                        'df': len(group1[var]) - 1,
                        't-statistic': round(ttest_results[var].statistic, 2),
                        'p-value': round(ttest_results[var].pvalue, 2)
                    }
                    row_used = {
                        'Group': 'Used',
                        'Mean': round(group2[var].mean(), 2),
                        'Std': round(group2[var].std(), 2),
                        'df': len(group2[var]) - 1,
                        't-statistic': '',
                        'p-value': ''
                    }
                    rows_interaction.append(row_not_used)
                    rows_interaction.append(row_used)
        else:
            speakers = group1['speaker_id'].unique()
            for speaker in speakers:
                not_used_speaker = group1[group1['speaker_id'] == speaker]
                used_speaker = group2[group2['speaker_id'] == speaker]

                if len(not_used_speaker) > 1 and len(used_speaker) > 1:
                    ttest_speech = ttest_ind(not_used_speaker['normalized_speech_frequency'], used_speaker['normalized_speech_frequency'], equal_var=False)
                    ttest_count = ttest_ind(not_used_speaker['normalized_interaction_frequency'], used_speaker['normalized_interaction_frequency'], equal_var=False)
                else:
                    ttest_speech = ttest_count = None

                row_not_used = {
                    'Group': f'Not Used (Speaker {speaker})',
                    'Mean': round(not_used_speaker['normalized_speech_frequency'].mean(), 2),
                    'Std': round(not_used_speaker['normalized_speech_frequency'].std(), 2),
                    'df': len(not_used_speaker['normalized_speech_frequency']) - 1,
                    't-statistic': round(ttest_speech.statistic, 2) if ttest_speech else None,
                    'p-value': round(ttest_speech.pvalue, 2) if ttest_speech else None
                }
                row_used = {
                    'Group': f'Used (Speaker {speaker})',
                    'Mean': round(used_speaker['normalized_speech_frequency'].mean(), 2),
                    'Std': round(used_speaker['normalized_speech_frequency'].std(), 2),
                    'df': len(used_speaker['normalized_speech_frequency']) - 1,
                    't-statistic': '',
                    'p-value': ''
                }
                rows_speech.append(row_not_used)
                rows_speech.append(row_used)

                row_not_used = {
                    'Group': f'Not Used (Speaker {speaker})',
                    'Mean': round(not_used_speaker['normalized_interaction_frequency'].mean(), 2),
                    'Std': round(not_used_speaker['normalized_interaction_frequency'].std(), 2),
                    'df': len(not_used_speaker['normalized_interaction_frequency']) - 1,
                    't-statistic': round(ttest_count.statistic, 2) if ttest_count else None,
                    'p-value': round(ttest_count.pvalue, 2) if ttest_count else None
                }
                row_used = {
                    'Group': f'Used (Speaker {speaker})',
                    'Mean': round(used_speaker['normalized_interaction_frequency'].mean(), 2),
                    'Std': round(used_speaker['normalized_interaction_frequency'].std(), 2),
                    'df': len(used_speaker['normalized_interaction_frequency']) - 1,
                    't-statistic': '',
                    'p-value': ''
                }
                rows_interaction.append(row_not_used)
                rows_interaction.append(row_used)

        detailed_df_speech = pd.DataFrame(rows_speech)
        detailed_df_interaction = pd.DataFrame(rows_interaction)
        return detailed_df_speech, detailed_df_interaction

    @dash_app.callback(
        [Output('casual-graph-speech', 'figure'),
         Output('casual-graph-interaction', 'figure'),
         Output('casual-table-speech', 'children'),
         Output('casual-table-interaction', 'children')],
        [Input('casual-view-type', 'value')]
    )
    def update_casual_graph_table(view_type):
        not_used_meetings = dataset[dataset['meeting_number'].isin([1, 2, 3, 4, 5, 6, 7, 8])]
        used_meetings = dataset[dataset['meeting_number'].isin([9, 10, 11, 12, 13, 14, 15, 16, 17])]

        if view_type == 'total':
            not_used_metrics = calculate_team_meeting_metrics(not_used_meetings)
            used_metrics = calculate_team_meeting_metrics(used_meetings)
        else:
            not_used_metrics = calculate_individual_meeting_metrics(not_used_meetings)
            used_metrics = calculate_individual_meeting_metrics(used_meetings)

        ttest_results = perform_ttest(not_used_metrics, used_metrics)

        fig_speech = go.Figure()
        fig_interaction = go.Figure()

        if view_type == 'total':
            fig_speech.add_trace(go.Bar(
                x=['Not Used', 'Used'],
                y=[not_used_metrics['normalized_speech_frequency'].mean(), used_metrics['normalized_speech_frequency'].mean()],
                error_y=dict(type='data', array=[not_used_metrics['normalized_speech_frequency'].std(), used_metrics['normalized_speech_frequency'].std()]),
                marker_color=['#1f77b4', '#ff7f0e']
            ))

            fig_interaction.add_trace(go.Bar(
                x=['Not Used', 'Used'],
                y=[not_used_metrics['normalized_interaction_frequency'].mean(), used_metrics['normalized_interaction_frequency'].mean()],
                error_y=dict(type='data', array=[not_used_metrics['normalized_interaction_frequency'].std(), used_metrics['normalized_interaction_frequency'].std()]),
                marker_color=['#1f77b4', '#ff7f0e']
            ))

            speech_y_max = max(not_used_metrics['normalized_speech_frequency'].mean() + 1.1 * not_used_metrics['normalized_speech_frequency'].std(),
                               used_metrics['normalized_speech_frequency'].mean() + 1.1 * used_metrics['normalized_speech_frequency'].std())
            interaction_y_max = max(not_used_metrics['normalized_interaction_frequency'].mean() + 1.1 * not_used_metrics['normalized_interaction_frequency'].std(),
                                    used_metrics['normalized_interaction_frequency'].mean() + 1.1 * used_metrics['normalized_interaction_frequency'].std())

            fig_speech.update_layout(
                xaxis_title='Condition',
                yaxis_title='Normalized Speech Frequency',
                yaxis=dict(range=[0, speech_y_max]),
                showlegend=False
            )

            fig_interaction.update_layout(
                xaxis_title='Condition',
                yaxis_title='Normalized Interaction Frequency',
                yaxis=dict(range=[0, interaction_y_max]),
                showlegend=False
            )
        else:
            for speaker in not_used_metrics['speaker_id'].unique():
                fig_speech.add_trace(go.Bar(
                    x=['Not Used', 'Used'],
                    y=[not_used_metrics[not_used_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].mean(),
                       used_metrics[used_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].mean()],
                    error_y=dict(type='data', array=[not_used_metrics[not_used_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].std(),
                                                     used_metrics[used_metrics['speaker_id'] == speaker]['normalized_speech_frequency'].std()]),
                    name=f'Speaker {speaker}'
                ))

                fig_interaction.add_trace(go.Bar(
                    x=['Not Used', 'Used'],
                    y=[not_used_metrics[not_used_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].mean(),
                       used_metrics[used_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].mean()],
                    error_y=dict(type='data', array=[not_used_metrics[not_used_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].std(),
                                                     used_metrics[used_metrics['speaker_id'] == speaker]['normalized_interaction_frequency'].std()]),
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

        detailed_df_speech, detailed_df_interaction = dataframe_generator(ttest_results, not_used_metrics, used_metrics, view_type)

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

    dash_app.layout.children.append(html.Div(id ='casual', children=[html.H2("A/B Test: Casual Language Used", style={'text-align': 'center'})]))
    dash_app.layout.children.append(html.Div([
        dcc.RadioItems(
            id='casual-view-type',
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
            dcc.Graph(id='casual-graph-speech'),
            html.Div(id='casual-table-speech'),
            html.Details([
                html.Summary('Casual Language Used - Normalized Speech Frequency', style={'margin-top': '20px','margin-bottom': '10px'}),
                dcc.Markdown("""
                    #### Normalized Speech Frequency - Total
                    - By observing the changes in the Normalized Speech Frequency values in the Total graph, we can determine how the use of casual language between participants affected the overall speech frequency in the meeting.

                    #### Normalized Speech Frequency - by Speaker
                    - The by Speaker graph allows us to see the differences in speech frequency for each speaker. This helps determine if using casual language had a significant effect on speech frequency changes for specific individuals.
                """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
            ], style={'margin-top': '5px', 'margin-bottom': '20px'})
        ], style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='casual-graph-interaction'),
            html.Div(id='casual-table-interaction'),
            html.Details([
                html.Summary('Casual Language Used - Normalized Interaction Frequency', style={'margin-top': '20px','margin-bottom': '10px'}),
                dcc.Markdown("""
                    #### Normalized Interaction Frequency - Total
                    - By observing the changes in the Normalized Interaction Frequency values in the Total graph, we can determine how the use of casual language between participants affected the overall interaction frequency in the meeting.

                    #### Normalized Interaction Frequency - by Speaker
                    - The by Speaker graph allows us to see the differences in interaction frequency for each speaker. This helps determine if using casual language had a significant effect on interaction frequency changes for specific individuals.
                """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
            ], style={'margin-top': '5px', 'margin-bottom': '20px'})
        ], style={'width': '48%', 'display': 'inline-block'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}))

    return dash_app
