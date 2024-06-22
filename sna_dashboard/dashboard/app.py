from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from asgiref.wsgi import WsgiToAsgi

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

# Compute Interaction Frequencies Excluding Self-Interactions
interaction_summary = dataset[dataset['speaker_number'] != dataset['next_speaker_id']].groupby(
    ['project', 'meeting_number', 'speaker_number', 'next_speaker_id'])['count'].sum().reset_index()

# Ensure not to count the same speaker multiple times within a meeting
interaction_summary = interaction_summary.groupby(
    ['project', 'meeting_number', 'speaker_number'])['count'].sum().reset_index()

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

            if G.has_edge(next_speaker, prev_speaker) and prev_speaker != next_speaker:
                G[next_speaker][prev_speaker]['weight'] += count
            else:
                G.add_edge(next_speaker, prev_speaker, weight=count)
    return G


def get_pos_and_labels(G):
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    return pos, edge_labels

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

    min_size = 30
    max_size = 70

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

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='gray'),
        hoverinfo='none',
        mode='lines'
    )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_trace_text,
        marker=dict(
            color='skyblue',
            size=node_sizes,
            line=dict(width=2, color='black')
        ),
        textposition='middle center',
        textfont=dict(
            size=12,
            color='black'
        ),
        hoverinfo='text',
        hovertext=hover_texts
    )

    edge_label_trace = []
    for edge in G.edges(data=True):
        if edge[0] != edge[1]:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_label_trace.append(
                go.Scatter(
                    x=[(x0 + x1) / 2],
                    y=[(y0 + y1) / 2],
                    text=[str(edge[2]['weight'])],
                    mode='text',
                    textfont=dict(color='red', size=10),
                    hoverinfo='none'
                )
            )

    self_interaction_trace = []
    for (x, y, weight) in self_interaction_labels:
        self_interaction_trace.append(
            go.Scatter(
                x=[x],
                y=[y + 0.1],  # Adjust the position slightly above the node
                text=[weight],
                mode='text',
                textfont=dict(color='red', size=10),
                hoverinfo='none'
            )
        )

    fig = go.Figure(data=[edge_trace, node_trace] + edge_label_trace + self_interaction_trace,
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False)
                    ))

    return fig


# Define the layout
dash_app.layout = html.Div([
    html.H1("Interactive Network Graph Dashboard"),
    dcc.Dropdown(
        id='project-dropdown',
        options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
        placeholder="Select a project",
    ),
    dcc.Dropdown(
        id='meeting-dropdown',
        placeholder="Select a meeting",
    ),
    dcc.Dropdown(
        id='speaker-dropdown',
        multi=True,
        placeholder="Select speakers",
    ),
    html.Button('Reset', id='reset-button', n_clicks=0),
    dcc.Graph(id='network-graph')
])


@dash_app.callback(
    Output('meeting-dropdown', 'options'),
    [Input('project-dropdown', 'value')]
)
def set_meeting_options(selected_project):
    if selected_project is None:
        return []
    meetings = dataset[dataset['project'] == selected_project]['meeting_number'].unique()
    return [{'label': f'Meeting {i}', 'value': i} for i in meetings]


@dash_app.callback(
    Output('speaker-dropdown', 'options'),
    [Input('project-dropdown', 'value'),
     Input('meeting-dropdown', 'value')]
)
def set_speaker_options(selected_project, selected_meeting):
    if selected_project is None or selected_meeting is None:
        return []
    speakers = dataset[(dataset['project'] == selected_project) &
                       (dataset['meeting_number'] == selected_meeting)]['speaker_number'].unique()
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
    # Clear the graph if no project or meeting is selected
    if selected_project is None or selected_meeting is None:
        return go.Figure()

    # Filter the dataset based on the selected filters
    df_filtered = dataset[(dataset['project'] == selected_project) &
                          (dataset['meeting_number'] == selected_meeting)]

    if selected_speakers:
        df_filtered = df_filtered[df_filtered['speaker_number'].isin(selected_speakers) |
                                  df_filtered['next_speaker_id'].isin(selected_speakers)]

    # Create the interaction graph
    G = create_interaction_graph(df_filtered)

    # Plot the interaction network
    return plot_interaction_network(G)


# Convert the Dash app to an ASGI app
asgi_dash_app = WsgiToAsgi(dash_app.server)

# Mount the Dash app to FastAPI
fastapi_app.mount("/dash", asgi_dash_app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8080)
