from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors

def initialize_individual_app(dash_app, dataset):
    @dash_app.callback(
        Output('individual-meeting-dropdown', 'options'),
        Input('individual-view-type-radio', 'value')
    )
    def set_individual_meeting_options(view_type):
        meetings = dataset['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('individual-speaker-dropdown', 'options'),
        [Input('individual-meeting-dropdown', 'value'),
         Input('individual-view-type-radio', 'value')]
    )
    def set_individual_speaker_options(selected_meeting, view_type):
        if not selected_meeting:
            speakers = dataset['next_speaker_id'].unique()
        else:
            filtered_df = dataset[dataset['meeting_number'].isin(selected_meeting)]
            speakers = filtered_df['next_speaker_id'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    @dash_app.callback(
        [Output('individual-meeting-dropdown', 'value'),
         Output('individual-speaker-dropdown', 'value')],
        Input('individual-reset-button', 'n_clicks')
    )
    def reset_individual_filters(n_clicks):
        return None, None

    @dash_app.callback(
        Output('individual-score-graph', 'figure'),
        [Input('individual-meeting-dropdown', 'value'),
         Input('individual-speaker-dropdown', 'value'),
         Input('individual-view-type-radio', 'value')]
    )
    def update_individual_score_graph(selected_meeting, selected_speakers, view_type):
        filtered_df = dataset[dataset['individual_collaboration_score'] != -1]
        filtered_df = filtered_df[filtered_df['overall_collaboration_score'] != -1]

        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        if selected_speakers:
            filtered_df = filtered_df[filtered_df['next_speaker_id'].isin(selected_speakers)]

        fig = go.Figure()

        if view_type == 'total':
            others_scores_mean = filtered_df.groupby('meeting_number')['individual_collaboration_score'].agg(['mean', 'sem']).reset_index()
            fig.add_trace(go.Scatter(
                x=others_scores_mean['meeting_number'],
                y=others_scores_mean['mean'],
                mode='lines+markers',
                name='Mean Individual Collaboration Score (Others)',
                line=dict(color='red'),
                marker=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=others_scores_mean['meeting_number'],
                y=others_scores_mean['mean'],
                mode='markers',
                marker=dict(size=8, color='red'),
                error_y=dict(
                    type='data',
                    array=others_scores_mean['sem'],
                    visible=True
                ),
                showlegend=False
            ))

        else:  # view_type == 'by_speakers'
            others_scores_by_speaker = filtered_df.groupby(['meeting_number', 'next_speaker_id'])['individual_collaboration_score'].agg(['mean', 'sem']).reset_index()
            palette = sns.color_palette('tab10', n_colors=others_scores_by_speaker['next_speaker_id'].nunique())
            color_map = {speaker: mcolors.rgb2hex(palette[i % len(palette)]) for i, speaker in enumerate(others_scores_by_speaker['next_speaker_id'].unique())}

            for speaker in others_scores_by_speaker['next_speaker_id'].unique():
                speaker_data = others_scores_by_speaker[others_scores_by_speaker['next_speaker_id'] == speaker]
                color = color_map[speaker]
                fig.add_trace(go.Scatter(
                    x=speaker_data['meeting_number'],
                    y=speaker_data['mean'],
                    mode='lines+markers',
                    name=f'Next Speaker {speaker}',
                    line=dict(color=color),
                    marker=dict(color=color),
                    error_y=dict(
                        type='data',
                        array=speaker_data['sem'],
                        visible=True
                    )
                ))

        fig.update_layout(
            title='Mean Individual Collaboration Score (Others) by Meeting',
            xaxis_title='Meeting Number',
            yaxis_title='Mean Individual Collaboration Score (Others)',
            xaxis=dict(tickmode='array', tickvals=filtered_df['meeting_number'].unique()),
            showlegend=True
        )

        # Bar plot for selected meetings
        if selected_meeting:
            bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
            if not bar_data.empty:
                bar_data_agg = bar_data.groupby('next_speaker_id')['individual_collaboration_score'].mean().reset_index()
                fig = go.Figure(data=[go.Bar(
                    x=bar_data_agg['next_speaker_id'],
                    y=bar_data_agg['individual_collaboration_score'],
                    marker_color=[color_map[speaker] for speaker in bar_data_agg['next_speaker_id']]
                )])
                fig.update_layout(
                    title='Mean Individual Collaboration Score (Others) by Next Speaker for Selected Meetings',
                    xaxis_title='Next Speaker ID',
                    yaxis_title='Individual Collaboration Score (Others)',
                    showlegend=False
                )

        return fig
