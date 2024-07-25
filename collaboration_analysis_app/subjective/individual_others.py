from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.colors as pcolors
import pandas as pd
from dash import dcc, html
import dash

def initialize_individual_app(dash_app, dataset):
    # Filter for project 4 data only
    dataset = dataset[dataset['project'] == 4]

    # Define a consistent color map using D3 palette
    color_palette = pcolors.qualitative.D3
    color_map = {i: color_palette[i % len(color_palette)] for i in range(len(color_palette))}

    individual_others_layout = html.Div(id='individual-others', children=[
        html.H1("Peer Evaluation Scores for Collaboration", style={'text-align': 'left'}),
        html.Div([
            dcc.Dropdown(
                id='individual-meeting-dropdown',
                placeholder="Select a meeting",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='individual-speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.RadioItems(
                id='individual-view-type-radio',
                options=[
                    {'label': 'Total', 'value': 'total'},
                    {'label': 'By Speakers', 'value': 'by_speakers'}
                ],
                value='total',
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('Reset', id='individual-reset-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='individual-score-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Individual Collaboration Score (Others) Explanation
                - This graph represents the collaboration score each member received from other members.

                    - X-axis: Indicates the sequential number of analyzed meetings. Each dot on the x-axis represents a specific meeting.

                    - Y-axis: The "total" tab shows average collaboration score each individual received as a member, and the "by speakers" tab shows individual's collaboration score as evaluated by others.

                - This visualization helps you understand how each person's collaboration score changes over time and how they perform compared to other people's evaluation scores. It can be used to evaluate individual collaboration performance and analyze collaboration dynamics within a team.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px','margin-bottom': '20px'})
    ])
    dash_app.layout.children.append(individual_others_layout)

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
            speakers = dataset['speaker_number'].unique()
        else:
            filtered_df = dataset[dataset['meeting_number'].isin(selected_meeting)]
            speakers = filtered_df['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    def display_empty_graph():
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Mean Individual Collaboration Score (Others)',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            showlegend=False
        )
        return fig

    @dash_app.callback(
        [Output('individual-meeting-dropdown', 'disabled'),
         Output('individual-speaker-dropdown', 'disabled')],
        Input('individual-view-type-radio', 'value')
    )
    def disable_dropdowns(view_type):
        if view_type == 'total':
            return True, True
        else:
            return False, False

    @dash_app.callback(
        [Output('individual-score-graph', 'figure'),
         Output('individual-meeting-dropdown', 'value'),
         Output('individual-speaker-dropdown', 'value')],
        [Input('individual-meeting-dropdown', 'value'),
         Input('individual-speaker-dropdown', 'value'),
         Input('individual-view-type-radio', 'value'),
         Input('individual-reset-button', 'n_clicks')]
    )
    def update_individual_score_graph(selected_meeting, selected_speakers, view_type, reset_clicks):
        ctx = dash.callback_context

        if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'individual-reset-button':
            return display_empty_graph(), None, None

        filtered_df = dataset[dataset['individual_collaboration_score'] != -1]
        filtered_df = filtered_df[filtered_df['overall_collaboration_score'] != -1]

        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]

        fig = go.Figure()

        if view_type == 'total':
            others_scores = filtered_df.groupby('meeting_number')['individual_collaboration_score'].agg(['mean', 'sem']).reset_index()
            fig.add_trace(go.Scatter(
                x=others_scores['meeting_number'],
                y=others_scores['mean'],
                mode='lines+markers',
                name='Mean Individual Collaboration Score (Others)',
                line=dict(color='red'),
                marker=dict(color='red')
            ))
            fig.add_trace(go.Scatter(
                x=others_scores['meeting_number'],
                y=others_scores['mean'],
                mode='markers',
                marker=dict(size=8, color='red'),
                error_y=dict(
                    type='data',
                    array=others_scores['sem'],
                    visible=True
                ),
                showlegend=False
            ))

        else:  # view_type == 'by_speakers'
            others_scores_by_speaker = filtered_df.groupby(['meeting_number', 'speaker_number'])['individual_collaboration_score'].agg(['mean', 'sem']).reset_index()

            if selected_speakers:
                for speaker in selected_speakers:
                    speaker_data = others_scores_by_speaker[others_scores_by_speaker['speaker_number'] == speaker]
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
                for speaker in others_scores_by_speaker['speaker_number'].unique():
                    speaker_data = others_scores_by_speaker[others_scores_by_speaker['speaker_number'] == speaker]
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
            yaxis_title='Mean Individual Collaboration Score (Others)',
            xaxis=dict(tickmode='array', tickvals=filtered_df['meeting_number'].unique()),
            showlegend=True
        )

        # Bar plot for selected meetings
        if selected_meeting:
            bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
            if not bar_data.empty:
                bar_data_agg = bar_data.groupby('speaker_number')['individual_collaboration_score'].mean().reset_index()
                fig = go.Figure(data=[go.Bar(
                    x=bar_data_agg['speaker_number'],
                    y=bar_data_agg['individual_collaboration_score'],
                    marker_color=[color_map[speaker % len(color_map)] for speaker in bar_data_agg['speaker_number']]
                )])
                fig.update_layout(
                    xaxis_title='Speaker Number',
                    yaxis_title='Individual Collaboration Score (Others)',
                    showlegend=False
                )

        return fig, None, None
