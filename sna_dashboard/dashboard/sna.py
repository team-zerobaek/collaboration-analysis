from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
import matplotlib.colors as mcolors

dash_app = None
dataset = None

def initialize_sna_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

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

        weights = [G[u][v]['weight'] for u, v in G.edges() if u != v]
        if weights:
            min_weight = min(weights)
            max_weight = max(weights)
        else:
            min_weight = max_weight = 1

        norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
        cmap = mcolors.LinearSegmentedColormap.from_list('blue_red', ['blue', 'red'])

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            self_weight = G[node][node]['weight'] if G.has_edge(node, node) else 0
            node_size = min_size + 2 * self_weight
            node_size = max(min_size, min(max_size, node_size))
            node_sizes.append(node_size)
            node_trace_text.append(str(node))
            if self_weight > 0:
                self_interaction_labels.append((x, y, f"({self_weight})"))

            total_interactions = sum([G[node][nbr]['weight'] for nbr in G.neighbors(node) if nbr != node])

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
                edge_width = ((edge[2]['weight'] - min_weight) / (max_weight - min_weight)) * (max_edge_width - min_edge_width) + min_edge_width
                edge_color = mcolors.rgb2hex(cmap(norm(edge[2]['weight'])))
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
                        font=dict(color='red', size=14, weight='bold'),
                        bgcolor='#f7f7f7'
                    )
                )

        for (x, y, weight) in self_interaction_labels:
            annotations.append(
                dict(
                    x=x,
                    y=y + 0.1,
                    text=weight,
                    showarrow=False,
                    font=dict(color='red', size=14, weight='bold'),
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
                    font=dict(size=14, color='black', weight='bold'),
                    align='center',
                    bgcolor='#f7f7f7'
                )
            )

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

        example_x = 1.5
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
                            paper_bgcolor='#f7f7f7',
                            plot_bgcolor='#f7f7f7'
                        ))

        return fig

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
