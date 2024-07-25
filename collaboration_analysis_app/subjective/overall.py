from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.colors as pcolors
import pandas as pd
from dash import dcc, html
import dash

def initialize_overall_app(dash_app, dataset):
    # Define a consistent color map using D3 palette
    color_palette = pcolors.qualitative.D3
    color_map = {i: color_palette[i % len(color_palette)] for i in range(len(color_palette))}

    overall_layout = html.Div(id='overall', children=[
        html.H1("Members' Perception of Collaboration", style={'text-align': 'left'}),
        html.Div([
            dcc.Dropdown(
                id='meeting-dropdown',
                placeholder="Select a meeting",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='speaker-dropdown',
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.RadioItems(
                id='view-type-radio',
                options=[
                    {'label': 'Total', 'value': 'total'},
                    {'label': 'By Speakers', 'value': 'by_speakers'}
                ],
                value='total',
                labelStyle={'display': 'inline-block'}
            ),
            html.Button('Reset', id='reset-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='collaboration-score-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Members' Perception of Collaboration
                - This graph shows the overall collaboration scores as perceived by the members in each meeting.

                    - **X-axis**: Indicates the session number of the meeting.
                    - **Y-axis**: Indicates the overall collaboration score.

                - The "total" view shows the mean collaboration score across all members, while the "by speakers" view shows the scores for individual speakers.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px','margin-bottom': '20px'})
    ])
    dash_app.layout.children.append(overall_layout)

    @dash_app.callback(
        Output('meeting-dropdown', 'options'),
        Input('view-type-radio', 'value')
    )
    def set_meeting_options(view_type):
        meetings = dataset['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('speaker-dropdown', 'options'),
        Input('view-type-radio', 'value')
    )
    def set_speaker_options(view_type):
        speakers = dataset['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    def display_empty_graph():
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Mean Overall Collaboration Score',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            showlegend=False
        )
        return fig

    @dash_app.callback(
        [Output('meeting-dropdown', 'disabled'),
         Output('speaker-dropdown', 'disabled')],
        Input('view-type-radio', 'value')
    )
    def disable_dropdowns(view_type):
        if view_type == 'total':
            return True, True
        else:
            return False, False

    @dash_app.callback(
        [Output('collaboration-score-graph', 'figure'),
         Output('meeting-dropdown', 'value'),
         Output('speaker-dropdown', 'value')],
        [Input('meeting-dropdown', 'value'),
         Input('speaker-dropdown', 'value'),
         Input('view-type-radio', 'value'),
         Input('reset-button', 'n_clicks')]
    )
    def update_collaboration_score_graph(selected_meeting, selected_speakers, view_type, reset_clicks):
        ctx = dash.callback_context

        if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'reset-button':
            return display_empty_graph(), None, None

        filtered_df = dataset

        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        if selected_speakers:
            filtered_df = filtered_df[filtered_df['speaker_number'].isin(selected_speakers)]

        fig = go.Figure()

        if view_type == 'total':
            overall_score_by_meeting = filtered_df.groupby('meeting_number')['overall_collaboration_score'].agg(['mean', 'std']).reset_index()
            fig.add_trace(go.Scatter(
                x=overall_score_by_meeting['meeting_number'],
                y=overall_score_by_meeting['mean'],
                mode='lines+markers',
                name='Mean Collaboration Score',
                line=dict(color='blue'),
                marker=dict(color='blue')
            ))
            fig.add_trace(go.Scatter(
                x=overall_score_by_meeting['meeting_number'],
                y=overall_score_by_meeting['mean'],
                mode='markers',
                marker=dict(size=8, color='blue'),
                error_y=dict(
                    type='data',
                    array=overall_score_by_meeting['std'],
                    visible=True
                ),
                showlegend=False
            ))

        else:  # view_type == 'by_speakers'
            overall_score_by_meeting_speaker = filtered_df.groupby(['meeting_number', 'speaker_number'])['overall_collaboration_score'].agg(['mean']).reset_index()

            for speaker in overall_score_by_meeting_speaker['speaker_number'].unique():
                speaker_data = overall_score_by_meeting_speaker[overall_score_by_meeting_speaker['speaker_number'] == speaker]
                color = color_map[speaker % len(color_map)]
                fig.add_trace(go.Scatter(
                    x=speaker_data['meeting_number'],
                    y=speaker_data['mean'],
                    mode='lines+markers',
                    name=f'Speaker {speaker}',
                    line=dict(color=color),
                    marker=dict(color=color)
                ))

        fig.update_layout(
            xaxis_title='Meeting Number',
            yaxis_title='Mean Overall Collaboration Score',
            xaxis=dict(tickmode='array', tickvals=filtered_df['meeting_number'].unique()),
            showlegend=True
        )

        # Bar plot for selected meetings
        if selected_meeting:
            bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
            if not bar_data.empty:
                bar_data_agg = bar_data.groupby('speaker_number')['overall_collaboration_score'].mean().reset_index()
                fig = go.Figure(data=[go.Bar(
                    x=bar_data_agg['speaker_number'],
                    y=bar_data_agg['overall_collaboration_score'],
                    marker_color=[color_map[speaker % len(color_map)] for speaker in bar_data_agg['speaker_number']]
                )])
                fig.update_layout(
                    xaxis_title='Speaker Number',
                    yaxis_title='Overall Collaboration Score',
                    showlegend=False
                )

        return fig, dash.no_update, dash.no_update
