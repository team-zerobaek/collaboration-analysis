from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import networkx as nx
import matplotlib.colors as mcolors
from dash import dcc, html
import dash

# Initialize Dash app and dataset within the SNA context
dash_app = None
dataset = None

def initialize_sna_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    def create_interaction_graph(df, selected_speakers=None):
        G = nx.DiGraph()
        for i in range(len(df)):
            prev_speaker = f"SPEAKER_{df.iloc[i]['speaker_number']:02d}"
            next_speaker = f"SPEAKER_{df.iloc[i]['next_speaker_id']:02d}"
            count = df.iloc[i]['count']
            if count > 0:
                if selected_speakers and (prev_speaker not in selected_speakers or next_speaker not in selected_speakers):
                    continue  # Skip edges not between selected speakers
                if G.has_edge(prev_speaker, next_speaker):
                    G[prev_speaker][next_speaker]['weight'] += count
                else:
                    G.add_edge(prev_speaker, next_speaker, weight=count)
        # Add nodes that are selected but have no edges
        if selected_speakers:
            for speaker in selected_speakers:
                if speaker not in G:
                    G.add_node(speaker)
        return G

    def plot_interaction_network(G):
        pos = nx.spring_layout(G)
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
        cmap = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])

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
                            plot_bgcolor='#f7f7f7',  # Plot area background color
                            height=800
                        ))

        return fig

    def display_empty_graph():
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='#f7f7f7',
            plot_bgcolor='#f7f7f7',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig

    @dash_app.callback(
        [Output('network-graph', 'figure'),
         Output('project-dropdown', 'value'),
         Output('meeting-dropdown', 'value'),
         Output('speaker-dropdown', 'value')],
        [Input('project-dropdown', 'value'),
         Input('meeting-dropdown', 'value'),
         Input('speaker-dropdown', 'value'),
         Input('reset-button', 'n_clicks')],
        [State('project-dropdown', 'value')]
    )
    def update_graph(selected_project, selected_meeting, selected_speakers, reset_clicks, state_project):
        ctx = dash.callback_context
        if not ctx.triggered:
            if state_project:
                selected_project = state_project
            else:
                return display_empty_graph(), dash.no_update, dash.no_update, dash.no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'reset-button' or not selected_project:
            # Return an empty graph and reset all dropdowns
            return display_empty_graph(), None, [], []

        df_filtered = dataset[dataset['project'] == selected_project]

        if selected_meeting:
            df_filtered = df_filtered[df_filtered['meeting_number'].isin(selected_meeting)]

        if selected_speakers:
            selected_speakers = [f"SPEAKER_{speaker:02d}" for speaker in selected_speakers]
            df_filtered = df_filtered[df_filtered['speaker_number'].isin([int(s.split('_')[1]) for s in selected_speakers]) |
                                      df_filtered['next_speaker_id'].isin([int(s.split('_')[1]) for s in selected_speakers])]
            G = create_interaction_graph(df_filtered, selected_speakers)
        else:
            G = create_interaction_graph(df_filtered)

        return plot_interaction_network(G), dash.no_update, dash.no_update, dash.no_update

    @dash_app.callback(
        Output('meeting-dropdown', 'options'),
        [Input('project-dropdown', 'value')]
    )
    def set_meeting_options(selected_project):
        if not selected_project:
            return []
        meetings = dataset[dataset['project'] == selected_project]['meeting_number'].unique()
        return [{'label': f'Meeting {i}', 'value': i} for i in meetings]

    @dash_app.callback(
        Output('speaker-dropdown', 'options'),
        [Input('project-dropdown', 'value'),
         Input('meeting-dropdown', 'value')]
    )
    def set_speaker_options(selected_project, selected_meeting):
        if not selected_project:
            return []
        filtered_df = dataset[dataset['project'] == selected_project]
        if selected_meeting:
            filtered_df = filtered_df[filtered_df['meeting_number'].isin(selected_meeting)]
        speakers = filtered_df['speaker_number'].unique()
        return [{'label': f'Speaker {i}', 'value': i} for i in speakers]

    @dash_app.callback(
        Output('dataset-selection-radio', 'value'),
        [Input('upload-info-store', 'data')]
    )
    def update_radio_value(upload_info):
        if upload_info and 'status' in upload_info and upload_info['status'] == 'uploaded':
            return 'uploaded'
        return 'default'

    # Define the layout for SNA-specific elements
    initial_project = dataset['project'].max()
    initial_meetings = dataset[dataset['project'] == initial_project]['meeting_number'].unique()
    initial_speakers = dataset[dataset['project'] == initial_project]['speaker_number'].unique()

    sna_layout = html.Div([
        html.Div(id='sna', children=[
            html.H1("Interaction Network Graph")
        ]),
        html.Div([
            dcc.Dropdown(
                id='project-dropdown',
                options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                placeholder="Select a project",
                multi=False,
                style={'width': '200px'},
                value=initial_project  # Set the initial value to the most recent project
            ),
            dcc.Dropdown(
                id='meeting-dropdown',
                options=[{'label': f'Meeting {i}', 'value': i} for i in initial_meetings],
                placeholder="Select a meeting",
                multi=True,
                style={'width': '200px'}
            ),
            dcc.Dropdown(
                id='speaker-dropdown',
                options=[{'label': f'Speaker {i}', 'value': i} for i in initial_speakers],
                placeholder="Select speakers",
                multi=True,
                style={'width': '200px'}
            ),
            html.Button('Reset', id='reset-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='network-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Network Interaction Graph Explanation
                This graph represents a directed network of communication interactions recorded during a meeting.

                #### Components
                - **Nodes**: Each node represents a meeting participant.
                - **Edges**: Each edge signifies the number of interactions between participants. The thickness and color of the edges denote the frequency of interactions, ranging from blue (fewer interactions) to red (more interactions).
                - **Self-Interactions**: Numbers in parentheses above the nodes indicate self-interactions, which occur when a participant speaks at length or explains something extensively.

                #### Insights
                - **Interaction Frequency**: The edge thickness and color give a clear visual representation of how frequently participants interact with each other.
                - **Meeting Dynamics**: The dashboard provides detailed information about participant interactions in a specific meeting.
                - **Cumulative Data**: By selecting multiple meetings, you can view the cumulative number of interactions across those meetings.
                - **Focused Analysis**: The dashboard allows for selecting specific participants to analyze their interactions and relationships within the group.

                This tool is useful for analyzing communication patterns, understanding meeting dynamics, and identifying key contributors in discussions.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px'})
    ])

    dash_app.layout.children.append(sna_layout)
