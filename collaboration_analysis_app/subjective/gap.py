from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.colors as mcolors
from dash import dcc, html

def initialize_gap_app(dash_app, dataset):
    gap_layout = html.Div(id ='gap', children=[
        html.H1("Gap of Peer Evaluation Score", style={'text-align': 'left'}),
        html.Div([
            dcc.Dropdown(
                id='gap-meeting-dropdown',
                placeholder="Select a meeting",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='gap-speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.RadioItems(
                id='gap-view-type-radio',
                options=[
                    {'label': 'Total', 'value': 'total'},
                    {'label': 'By Meeting', 'value': 'by_meeting'}
                ],
                value='total',
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('Reset', id='gap-reset-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='gap-score-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Graph Explanation
                This graph visualizes the individual collaboration score differences between others' average and self for specific meetings and speakers.

                - **X-axis**: Speaker number
                - **Y-axis**: Collaboration score gap (Others - Self)
                - **Graph type**: Bar chart, allowing visual comparison of score differences for each speaker

                The graph can be dynamically updated based on the selected meetings and speakers.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px','margin-bottom': '20px'})
    ])
    dash_app.layout.children.append(gap_layout)

    @dash_app.callback(
        Output('gap-meeting-dropdown', 'options'),
        Input('gap-view-type-radio', 'value')
    )
    def set_gap_meeting_options(view_type):
        meetings = dataset['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('gap-speaker-dropdown', 'options'),
        [Input('gap-meeting-dropdown', 'value'),
         Input('gap-view-type-radio', 'value')]
    )
    def set_gap_speaker_options(selected_meeting, view_type):
        if not selected_meeting:
            speakers = dataset['speaker_number'].unique()
        else:
            filtered_df = dataset[dataset['meeting_number'].isin(selected_meeting)]
            speakers = filtered_df['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    @dash_app.callback(
        [Output('gap-meeting-dropdown', 'value'),
         Output('gap-speaker-dropdown', 'value')],
        Input('gap-reset-button', 'n_clicks')
    )
    def reset_gap_filters(n_clicks):
        return None, None

    @dash_app.callback(
        Output('gap-meeting-dropdown', 'disabled'),
        Input('gap-view-type-radio', 'value')
    )
    def disable_meeting_dropdown(view_type):
        return view_type == 'by_meeting'

    @dash_app.callback(
        Output('gap-score-graph', 'figure'),
        [Input('gap-meeting-dropdown', 'value'),
         Input('gap-speaker-dropdown', 'value'),
         Input('gap-view-type-radio', 'value')]
    )
    def update_gap_score_graph(selected_meeting, selected_speakers, view_type):
        filtered_df = dataset[(dataset['individual_collaboration_score'] != -1) &
                              (dataset['overall_collaboration_score'] != -1)]

        # Always calculate others_scores based on all speakers
        others_scores = dataset[dataset['speaker_id'] != dataset['next_speaker_id']].groupby(
            ['meeting_number', 'next_speaker_id'])['individual_collaboration_score'].mean().reset_index()

        self_scores = filtered_df[filtered_df['speaker_id'] == filtered_df['next_speaker_id']].groupby(
            ['meeting_number', 'speaker_number'])['individual_collaboration_score'].mean().reset_index()

        combined_scores = pd.merge(others_scores, self_scores, left_on=['meeting_number', 'next_speaker_id'],
                                   right_on=['meeting_number', 'speaker_number'],
                                   suffixes=('_others', '_self'))
        combined_scores['gap'] = combined_scores['individual_collaboration_score_others'] - combined_scores['individual_collaboration_score_self']

        fig = go.Figure()

        color_map = {
            0: 'blue',
            1: 'red',
            2: 'green',
            3: 'purple',
            4: 'orange'
        }

        if view_type == 'total':
            if selected_meeting:
                combined_scores = combined_scores[combined_scores['meeting_number'].isin(selected_meeting)]
            gap_scores_total = combined_scores.groupby('next_speaker_id')['gap'].agg(['mean', 'sem']).reset_index()
            fig.add_trace(go.Bar(
                x=gap_scores_total['next_speaker_id'],
                y=gap_scores_total['mean'],
                error_y=dict(
                    type='data',
                    array=gap_scores_total['sem'],
                    visible=True
                ),
                marker_color=[color_map[speaker] for speaker in gap_scores_total['next_speaker_id']]
            ))
            fig.update_layout(
                xaxis_title='Speaker Number',
                yaxis_title='Gap (Others - Self)',
                showlegend=False
            )
        else:
            if selected_speakers:
                for speaker in selected_speakers:
                    speaker_data = combined_scores[combined_scores['next_speaker_id'] == speaker]
                    color = color_map[speaker]
                    fig.add_trace(go.Scatter(
                        x=speaker_data['meeting_number'],
                        y=speaker_data['gap'],
                        mode='lines+markers',
                        name=f'Speaker {speaker}',
                        line=dict(color=color),
                        marker=dict(color=color)
                    ))
            else:
                for speaker in dataset['speaker_number'].unique():
                    speaker_data = combined_scores[combined_scores['next_speaker_id'] == speaker]
                    fig.add_trace(go.Scatter(
                        x=speaker_data['meeting_number'],
                        y=speaker_data['gap'],
                        mode='lines+markers',
                        name=f'Speaker {speaker}',
                        line=dict(color=color_map[speaker]),
                        marker=dict(color=color_map[speaker])
                    ))

            fig.add_shape(
                type="line",
                x0=combined_scores['meeting_number'].min(),
                x1=combined_scores['meeting_number'].max(),
                y0=0,
                y1=0,
                line=dict(color="black", dash="dash")
            )

            # Set x-ticks to show when at least one datapoint exists
            x_ticks = combined_scores['meeting_number'].unique()

            fig.update_layout(
                xaxis_title='Meeting Number',
                yaxis_title='Gap (Others - Self)',
                xaxis=dict(tickmode='array', tickvals=x_ticks),
                showlegend=True
            )

        return fig
