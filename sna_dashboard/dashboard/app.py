from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from asgiref.wsgi import WsgiToAsgi
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Initialize the FastAPI app
fastapi_app = FastAPI()

# Redirect root to /dash
@fastapi_app.get("/")
async def main():
    return RedirectResponse(url="/dash")

# Initialize the Dash app
dash_app = Dash(__name__, requests_pathname_prefix='/dash/')

# Load dataset
dataset = pd.read_csv('/app/data/dataset_collaboration.csv')

# Summarize speech frequencies by speaker within each meeting
speech_summary = dataset.groupby(['project', 'meeting_number', 'speaker_number'])['speech_frequency'].sum().reset_index()

# Calculate adjusted speech frequencies by dividing by the number of speakers in each meeting
num_speakers_per_meeting = dataset.groupby(['project', 'meeting_number'])['speaker_number'].nunique().reset_index()
num_speakers_per_meeting.columns = ['project', 'meeting_number', 'num_speakers']
speech_summary = pd.merge(speech_summary, num_speakers_per_meeting, on=['project', 'meeting_number'])
speech_summary['adjusted_speech_frequency'] = speech_summary['speech_frequency'] / speech_summary['num_speakers']

# Ensure the sum of each speaker's adjusted speech frequency is equal to total_words within one meeting
total_words_summary = dataset.groupby(['project', 'meeting_number'])['total_words'].first().reset_index()
speech_summary = pd.merge(speech_summary, total_words_summary, on=['project', 'meeting_number'])

# Normalize speech frequency by duration
meeting_durations = dataset.groupby(['project', 'meeting_number'])['duration'].first().reset_index()
speech_summary = pd.merge(speech_summary, meeting_durations, on=['project', 'meeting_number'])
speech_summary['normalized_speech_frequency'] = speech_summary['adjusted_speech_frequency'] / speech_summary['duration']

# Compute Interaction Frequencies Including Self-Interactions
interaction_summary = dataset.groupby(['project', 'meeting_number', 'speaker_number', 'next_speaker_id'])['count'].sum().reset_index()

# Normalize interaction frequencies by duration
interaction_summary = pd.merge(interaction_summary, dataset[['project', 'meeting_number', 'duration']].drop_duplicates(),
                               on=['project', 'meeting_number'])
interaction_summary['normalized_interaction_count'] = interaction_summary['count'] / interaction_summary['duration']


def create_interaction_graph(df):
    G = nx.DiGraph()
    for i in range(len(df)):
        prev_speaker = f"SPEAKER_{df.iloc[i]['speaker_number']:02d}"
        next_speaker = f"SPEAKER_{df.iloc[i]['next_speaker_id']:02d}"
        count = df.iloc[i]['count']
        if count > 0:
            if G.has_edge(prev_speaker, next_speaker):
                G[prev_speaker][next_speaker]['weight'] += count
            else:
                G.add_edge(prev_speaker, next_speaker, weight=count)
    return G


def get_pos_and_labels(G):
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    return pos, edge_labels


def create_blue_red_colormap():
    return mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])


def plot_interaction_network(G):
    pos, edge_labels = get_pos_and_labels(G)
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    node_sizes = []
    node_trace_text = []
    self_interaction_labels = []
    hover_texts = []
    edge_traces = []
    annotations = []

    min_size = 30
    max_size = 70
    min_edge_width = 1
    max_edge_width = 10

    # Find the range of edge weights for scaling excluding self-interactions
    weights = [G[u][v]['weight'] for u, v in G.edges() if u != v]
    if weights:
        min_weight = min(weights)
        max_weight = max(weights)
    else:
        min_weight = max_weight = 1

    # Create a colormap
    norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
    cmap = create_blue_red_colormap()

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        self_weight = G[node][node]['weight'] if G.has_edge(node, node) else 0
        node_size = min_size + 2 * self_weight  # Adjust the size based on self-interaction weight
        node_size = max(min_size, min(max_size, node_size))  # Ensure the size is within limits
        node_sizes.append(node_size)
        node_trace_text.append(str(node))
        if self_weight > 0:
            self_interaction_labels.append((x, y, f"({self_weight})"))

        # Calculate the sum of direct interactions excluding self-interactions
        total_interactions = sum([G[node][nbr]['weight'] for nbr in G.neighbors(node) if nbr != node])

        # Find the node with the biggest interaction and its number
        if len(G[node]) > 0:
            max_interaction_node, max_interaction_weight = max(G[node].items(), key=lambda item: item[1]['weight'] if item[0] != node else -1)
            if max_interaction_node == node:
                max_interaction_node = 'None'
                max_interaction_weight = 0
        else:
            max_interaction_node = 'None'
            max_interaction_weight = 0

        hover_texts.append(f"Node: {node}<br>Total Interactions (excluding self): {total_interactions}<br>Self-Interactions: {self_weight}<br>Max Interaction: {max_interaction_node} ({max_interaction_weight})")

    for edge in G.edges(data=True):
        if edge[0] != edge[1]:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            # Scale edge width based on weight
            edge_width = ((edge[2]['weight'] - min_weight) / (max_weight - min_weight)) * (max_edge_width - min_edge_width) + min_edge_width
            edge_color = mcolors.rgb2hex(cmap(norm(edge[2]['weight'])))  # Get the color from the colormap
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=edge_width, color=edge_color),
                    hoverinfo='none',
                    mode='lines'
                )
            )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(
            color='skyblue',
            size=node_sizes,
            line=dict(width=2, color='black')
        ),
        hoverinfo='text',
        hovertext=hover_texts
    )

    for edge in G.edges(data=True):
        if edge[0] != edge[1]:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            annotations.append(
                dict(
                    x=(x0 + x1) / 2,
                    y=(y0 + y1) / 2,
                    text=str(edge[2]['weight']),
                    showarrow=False,
                    font=dict(color='red', size=14, weight='bold'),  # Increased size and boldness
                    bgcolor='#f7f7f7'
                )
            )

    for (x, y, weight) in self_interaction_labels:
        annotations.append(
            dict(
                x=x,
                y=y + 0.1,  # Adjust the position slightly above the node
                text=weight,
                showarrow=False,
                font=dict(color='red', size=14, weight='bold'),  # Increased size and boldness
                bgcolor='#f7f7f7'
            )
        )

    for i, text in enumerate(node_trace_text):
        annotations.append(
            dict(
                x=node_x[i],
                y=node_y[i],
                text=text,
                showarrow=False,
                font=dict(size=14, color='black', weight='bold'),  # Increased size and boldness
                align='center',
                bgcolor='#f7f7f7'
            )
        )

    # Add color bar for the edge weights excluding self-interactions
    color_bar = go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            colorscale=[[0, 'blue'], [1, 'red']],
            cmin=min_weight,
            cmax=max_weight,
            colorbar=dict(
                title="Edge Weight (excluding self-interactions)",
                titleside="right"
            )
        ),
        hoverinfo='none'
    )

    # Adjust the x coordinate to move the example nodes to the right
    example_x = 1.5

    # Add example nodes for min and max self-interaction sizes
    example_min_node = go.Scatter(
        x=[example_x],
        y=[1.0],
        mode='markers',
        marker=dict(
            color='skyblue',
            size=min_size,
            line=dict(width=2, color='black')
        ),
        showlegend=False,
        hoverinfo='none'
    )

    example_max_node = go.Scatter(
        x=[example_x],
        y=[0.6],
        mode='markers',
        marker=dict(
            color='skyblue',
            size=max_size,
            line=dict(width=2, color='black')
        ),
        showlegend=False,
        hoverinfo='none'
    )

    example_annotations = [
        dict(
            x=example_x,
            y=1.2,
            xref="x",
            yref="y",
            text="Node Scales by Self-Interactions",
            showarrow=False,
            font=dict(size=15, color="black", weight="bold"),
            align="center",
            bgcolor='#f7f7f7'
        ),
        dict(
            x=example_x,
            y=1.0,
            xref="x",
            yref="y",
            text="(0)",
            showarrow=False,
            font=dict(size=12, color="red", weight="bold"),
            align="center",
            bgcolor='#f7f7f7'
        ),
        dict(
            x=example_x,
            y=0.6,
            xref="x",
            yref="y",
            text="(>=20)",
            showarrow=False,
            font=dict(size=12, color="red", weight="bold"),
            align="center",
            bgcolor='#f7f7f7'
        )
    ]

    fig = go.Figure(data=edge_traces + [node_trace] + [color_bar, example_min_node, example_max_node],
                    layout=go.Layout(
                        annotations=annotations + example_annotations,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        paper_bgcolor='#f7f7f7',  # Background color
                        plot_bgcolor='#f7f7f7'  # Plot area background color
                    ))

    return fig


# Define the layout
dash_app.layout = html.Div([
    html.H1("Interactive Network Graph Dashboard"),
    html.Div([
        dcc.Dropdown(
            id='project-dropdown',
            options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
            placeholder="Select a project",
            multi=True,
            style={'width': '200px'}
        ),
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
        html.Button('Reset', id='reset-button', n_clicks=0)
    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
    dcc.Graph(id='network-graph'),
    html.H1("Normalized Speech Frequency"),
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
    html.H1("Normalized Interaction Frequency"),
    html.Div([
        dcc.Dropdown(
            id='interaction-project-dropdown',
            options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
            placeholder="Select projects",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='interaction-meeting-dropdown',
            placeholder="Select meetings",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.Dropdown(
            id='interaction-speaker-dropdown',
            placeholder="Select speakers",
            multi=True,
            style={'width': '200px'}
        ),
        dcc.RadioItems(
            id='interaction-type-radio',
            options=[
                {'label': 'Total', 'value': 'total'},
                {'label': 'By Speakers', 'value': 'by_speakers'}
            ],
            value='total',
            labelStyle={'display': 'inline-block'}
        ),
        html.Button('Reset', id='reset-interaction-button', n_clicks=0)
    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
    dcc.Graph(id='interaction-frequency-graph')
])


@dash_app.callback(
    Output('meeting-dropdown', 'options'),
    [Input('project-dropdown', 'value')]
)
def set_meeting_options(selected_project):
    if not selected_project:
        return []
    meetings = dataset[dataset['project'].isin(selected_project)]['meeting_number'].unique()
    return [{'label': f'Meeting {i}', 'value': i} for i in meetings]


@dash_app.callback(
    Output('speaker-dropdown', 'options'),
    [Input('project-dropdown', 'value'),
     Input('meeting-dropdown', 'value')]
)
def set_speaker_options(selected_project, selected_meeting):
    if not selected_project:
        return []
    filtered_df = dataset[dataset['project'].isin(selected_project)]
    if selected_meeting:
        filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
    speakers = filtered_df['speaker_number'].unique()
    return [{'label': f'Speaker {i}', 'value': i} for i in speakers]


@dash_app.callback(
    [Output('project-dropdown', 'value'),
     Output('meeting-dropdown', 'value'),
     Output('speaker-dropdown', 'value')],
    [Input('reset-button', 'n_clicks')]
)
def reset_filters(n_clicks):
    return None, None, None


@dash_app.callback(
    Output('network-graph', 'figure'),
    [Input('project-dropdown', 'value'),
     Input('meeting-dropdown', 'value'),
     Input('speaker-dropdown', 'value'),
     Input('reset-button', 'n_clicks')]
)
def update_graph(selected_project, selected_meeting, selected_speakers, reset_clicks):
    if not selected_project:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='#f7f7f7',
            plot_bgcolor='#f7f7f7',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    df_filtered = dataset[dataset['project'].isin(selected_project)]

    if selected_meeting:
        df_filtered = df_filtered[df_filtered['meeting_number'].isin(selected_meeting)]
    if selected_speakers:
        df_filtered = df_filtered[df_filtered['speaker_number'].isin(selected_speakers) |
                                  df_filtered['next_speaker_id'].isin(selected_speakers)]

    G = create_interaction_graph(df_filtered)

    return plot_interaction_network(G)


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
    filtered_df = speech_summary

    if selected_project:
        filtered_df = filtered_df[filtered_df['project'].isin(selected_project)]
    if selected_meeting:
        filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
    if selected_speakers:
        filtered_df = filtered_df[filtered_df['speaker_number'].isin(selected_speakers)]

    if speech_type == 'total':
        speech_comparison_df = filtered_df.groupby(['project', 'meeting_number'])['normalized_speech_frequency'].sum().reset_index()
        speech_comparison_df_pivot = speech_comparison_df.pivot(index='meeting_number', columns='project', values='normalized_speech_frequency')
        speech_comparison_df_pivot.columns = [f'Normalized_Speech_Frequency_Project{i}' for i in speech_comparison_df_pivot.columns]
        speech_comparison_df_pivot.reset_index(inplace=True)

        meeting_numbers_with_data = speech_comparison_df_pivot[(speech_comparison_df_pivot > 0).any(axis=1)]['meeting_number']

        fig = go.Figure()
        for column in speech_comparison_df_pivot.columns[1:]:
            fig.add_trace(go.Scatter(x=speech_comparison_df_pivot['meeting_number'],
                                     y=speech_comparison_df_pivot[column],
                                     mode='lines+markers',
                                     name=column))

        fig.update_layout(
            title='Comparison of Normalized Speech Frequencies by Meeting',
            xaxis_title='Meeting Number',
            yaxis_title='Normalized Speech Frequency',
            xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
            showlegend=True
        )
    else:
        fig = go.Figure()
        for speaker in filtered_df['speaker_number'].unique():
            speaker_df = filtered_df[filtered_df['speaker_number'] == speaker]
            fig.add_trace(go.Scatter(x=speaker_df['meeting_number'],
                                     y=speaker_df['normalized_speech_frequency'],
                                     mode='lines+markers',
                                     name=f'Speaker {speaker}'))

        fig.update_layout(
            title='Normalized Speech Frequencies by Meeting and Speaker',
            xaxis_title='Meeting Number',
            yaxis_title='Normalized Speech Frequency',
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
            title=f'Normalized Speech Frequency for Selected Meetings',
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


@dash_app.callback(
    Output('interaction-meeting-dropdown', 'options'),
    [Input('interaction-project-dropdown', 'value')]
)
def set_interaction_meeting_options(selected_project):
    if not selected_project:
        return []
    meetings = dataset[dataset['project'].isin(selected_project)]['meeting_number'].unique()
    return [{'label': f'Meeting {i}', 'value': i} for i in meetings]


@dash_app.callback(
    Output('interaction-speaker-dropdown', 'options'),
    [Input('interaction-project-dropdown', 'value'),
     Input('interaction-meeting-dropdown', 'value')]
)
def set_interaction_speaker_options(selected_project, selected_meeting):
    if not selected_project:
        return []
    filtered_df = dataset[dataset['project'].isin(selected_project)]
    if selected_meeting:
        filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
    speakers = filtered_df['speaker_number'].unique()
    return [{'label': f'Speaker {i}', 'value': i} for i in speakers]


@dash_app.callback(
    [Output('interaction-project-dropdown', 'value'),
     Output('interaction-meeting-dropdown', 'value'),
     Output('interaction-speaker-dropdown', 'value')],
    [Input('reset-interaction-button', 'n_clicks')]
)
def reset_interaction_filters(n_clicks):
    return None, None, None


@dash_app.callback(
    Output('interaction-frequency-graph', 'figure'),
    [Input('interaction-project-dropdown', 'value'),
     Input('interaction-meeting-dropdown', 'value'),
     Input('interaction-speaker-dropdown', 'value'),
     Input('interaction-type-radio', 'value')]
)
def update_interaction_graph(selected_project, selected_meeting, selected_speakers, interaction_type):
    filtered_df = interaction_summary

    if selected_project:
        filtered_df = filtered_df[filtered_df['project'].isin(selected_project)]
    if selected_meeting:
        filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
    if selected_speakers:
        filtered_df = filtered_df[filtered_df['speaker_number'].isin(selected_speakers)]

    if interaction_type == 'total':
        interaction_comparison_df = filtered_df.groupby(['project', 'meeting_number'])['normalized_interaction_count'].sum().reset_index()
        interaction_comparison_df_pivot = interaction_comparison_df.pivot(index='meeting_number', columns='project', values='normalized_interaction_count')
        interaction_comparison_df_pivot.columns = [f'Normalized_Interaction_Frequency_Project{i}' for i in interaction_comparison_df_pivot.columns]
        interaction_comparison_df_pivot.reset_index(inplace=True)

        meeting_numbers_with_data = interaction_comparison_df_pivot[(interaction_comparison_df_pivot > 0).any(axis=1)]['meeting_number']

        fig = go.Figure()
        for column in interaction_comparison_df_pivot.columns[1:]:
            fig.add_trace(go.Scatter(x=interaction_comparison_df_pivot['meeting_number'],
                                     y=interaction_comparison_df_pivot[column],
                                     mode='lines+markers',
                                     name=column))

        fig.update_layout(
            title='Comparison of Normalized Interaction Frequencies by Meeting',
            xaxis_title='Meeting Number',
            yaxis_title='Normalized Interaction Frequency',
            xaxis=dict(tickmode='array', tickvals=meeting_numbers_with_data),
            showlegend=True
        )
    else:
        fig = go.Figure()
        for speaker in filtered_df['speaker_number'].unique():
            speaker_df = filtered_df[filtered_df['speaker_number'] == speaker]
            fig.add_trace(go.Scatter(x=speaker_df['meeting_number'],
                                     y=speaker_df['normalized_interaction_count'],
                                     mode='lines+markers',
                                     name=f'Speaker {speaker}'))

        fig.update_layout(
            title='Normalized Interaction Frequencies by Meeting and Speaker',
            xaxis_title='Meeting Number',
            yaxis_title='Normalized Interaction Frequency',
            showlegend=True
        )

    if selected_meeting:
        bar_data = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        colorscale = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])
        bar_colors = bar_data['normalized_interaction_count'].apply(lambda x: mcolors.rgb2hex(colorscale(x / bar_data['normalized_interaction_count'].max())))
        fig = go.Figure(data=[go.Bar(
            x=bar_data['speaker_number'],
            y=bar_data['normalized_interaction_count'],
            marker_color=bar_colors
        )])
        fig.update_layout(
            title=f'Normalized Interaction Frequency for Selected Meetings',
            xaxis_title='Speaker Number',
            yaxis_title='Normalized Interaction Frequency',
            showlegend=False
        )

        color_bar = go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale=[[0, 'blue'], [1, 'red']],
                cmin=0,
                cmax=bar_data['normalized_interaction_count'].max(),
                colorbar=dict(
                    title="Interaction Frequency",
                    titleside="right"
                )
            ),
            hoverinfo='none'
        )
        fig.add_trace(color_bar)

    return fig


# Convert the Dash app to an ASGI app
asgi_dash_app = WsgiToAsgi(dash_app.server)

# Mount the Dash app to FastAPI
fastapi_app.mount("/dash", asgi_dash_app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8080)
