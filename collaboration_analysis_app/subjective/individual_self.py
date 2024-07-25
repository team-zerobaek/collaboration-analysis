from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.colors as pcolors
import pandas as pd
from dash import dcc, html
import dash

def initialize_self_score_app(dash_app, dataset):
    # Filter for project 4 data only
    dataset = dataset[dataset['project'] == 4]

    # Define a consistent color map using D3 palette
    color_palette = pcolors.qualitative.D3
    color_map = {i: color_palette[i % len(color_palette)] for i in range(len(color_palette))}

    self_score_layout = html.Div(id='individual-self', children=[
        html.H1("Self Evaluation Scores for Collaboration", style={'text-align': 'left'}),
        html.Div([
            dcc.Dropdown(
                id='self-meeting-dropdown',
                placeholder="Select a meeting",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='self-speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.RadioItems(
                id='self-view-type-radio',
                options=[
                    {'label': 'Total', 'value': 'total'},
                    {'label': 'By Speakers', 'value': 'by_speakers'}
                ],
                value='total',
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('Reset', id='self-reset-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='self-score-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Individual Collaboration Score (Self) Explanation
                This graph shows the average self-assessed collaboration scores for each meeting, as well as the individual scores of participants.

                ### Axis Labels
                - **X-axis**: Indicates the session number of the meeting.
                - **Y-axis**: Indicates self-assessed Collaboration Score
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px','margin-bottom': '20px'})
    ])
    dash_app.layout.children.append(self_score_layout)

    @dash_app.callback(
        Output('self-meeting-dropdown', 'options'),
        Input('self-view-type-radio', 'value')
    )
    def set_self_meeting_options(view_type):
        meetings = dataset['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('self-speaker-dropdown', 'options'),
        [Input('self-meeting-dropdown', 'value'),
         Input('self-view-type-radio', 'value')]
    )
    def set_self_speaker_options(selected_meeting, view_type):
        if not selected_meeting:
            speakers = dataset['speaker_number'].unique()
        else:
            filtered_df = dataset[dataset['meeting_number'].isin(selected_meeting)]
            speakers = filtered_df['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    def display_empty_graph():
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Mean Individual Collaboration Score (Self)',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            showlegend=False
        )
        return fig

    @dash_app.callback(
        [Output('self-meeting-dropdown', 'disabled'),
         Output('self-speaker-dropdown', 'disabled')],
        Input('self-view-type-radio', 'value')
    )
    def disable_dropdowns(view_type):
        if view_type == 'total':
            return True, True
        else:
            return False, False

    @dash_app.callback(
        [Output('self-score-graph', 'figure'),
         Output('self-meeting-dropdown', 'value'),
         Output('self-speaker-dropdown', 'value')],
        [Input('self-meeting-dropdown', 'value'),
         Input('self-speaker-dropdown', 'value'),
         Input('self-view-type-radio', 'value'),
         Input('self-reset-button', 'n_clicks')]
    )
    def update_self_score_graph(selected_meeting, selected_speakers, view_type, reset_clicks):
        ctx = dash.callback_context

        if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'self-reset-button':
            return display_empty_graph(), None, None

        filtered_df = dataset[dataset['individual_collaboration_score'] != -1]
        filtered_df = filtered_df[filtered_df['overall_collaboration_score'] != -1]

        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]

        fig = go.Figure()

        if view_type == 'total':
            self_scores = filtered_df[filtered_df['speaker_id'] == filtered_df['next_speaker_id']].groupby('meeting_number')['individual_collaboration_score'].agg(['mean', 'sem']).reset_index()
            fig.add_trace(go.Scatter(
                x=self_scores['meeting_number'],
                y=self_scores['mean'],
                mode='lines+markers',
                name='Mean Individual Collaboration Score (Self)',
                line=dict(color='green'),
                marker=dict(color='green')
            ))
            fig.add_trace(go.Scatter(
                x=self_scores['meeting_number'],
                y=self_scores['mean'],
                mode='markers',
                marker=dict(size=8, color='green'),
                error_y=dict(
                    type='data',
                    array=self_scores['sem'],
                    visible=True
                ),
                showlegend=False
            ))

        else:  # view_type == 'by_speakers'
            self_scores_by_speaker = filtered_df[filtered_df['speaker_id'] == filtered_df['next_speaker_id']].groupby(['meeting_number', 'speaker_number'])['individual_collaboration_score'].agg(['mean', 'sem']).reset_index()

            if selected_speakers:
                for speaker in selected_speakers:
                    speaker_data = self_scores_by_speaker[self_scores_by_speaker['speaker_number'] == speaker]
                    color = color_map[speaker % len(color_map)]
                    fig.add_trace(go.Scatter(
                        x=speaker_data['meeting_number'],
                        y=speaker_data['mean'],
                        mode='lines+markers',
                        name=f'Speaker {speaker}',
                        line=dict(color=color),
                        marker=dict(color=color),
                        error_y=dict(
                            type='data',
                            array=speaker_data['sem'],
                            visible=True
                        )
                    ))
            else:
                for speaker in self_scores_by_speaker['speaker_number'].unique():
                    speaker_data = self_scores_by_speaker[self_scores_by_speaker['speaker_number'] == speaker]
                    color = color_map[speaker % len(color_map)]
                    fig.add_trace(go.Scatter(
                        x=speaker_data['meeting_number'],
                        y=speaker_data['mean'],
                        mode='lines+markers',
                        name=f'Speaker {speaker}',
                        line=dict(color=color),
                        marker=dict(color=color),
                        error_y=dict(
                            type='data',
                            array=speaker_data['sem'],
                            visible=True
                        )
                    ))

        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Mean Individual Collaboration Score (Self)',
            xaxis=dict(tickmode='array', tickvals=filtered_df['meeting_number'].unique()),
            showlegend=True
        )

        # Bar plot for selected meetings
        if selected_meeting:
            bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
            if not bar_data.empty:
                bar_data_agg = bar_data[bar_data['speaker_id'] == bar_data['next_speaker_id']].groupby('speaker_number')['individual_collaboration_score'].mean().reset_index()
                fig = go.Figure(data=[go.Bar(
                    x=bar_data_agg['speaker_number'],
                    y=bar_data_agg['individual_collaboration_score'],
                    marker_color=[color_map[speaker % len(color_map)] for speaker in bar_data_agg['speaker_number']]
                )])
                fig.update_layout(
                    xaxis_title='Speaker Number',
                    yaxis_title='Individual Collaboration Score (Self)',
                    showlegend=False
                )

        return fig, None, None
