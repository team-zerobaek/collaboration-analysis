from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import dash

dash_app = None
dataset = None

def initialize_gini_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    # Define a consistent color map for the projects
    color_map = {
        0: 'grey',
        1: 'purple',
        2: 'green',
        3: 'blue',
        4: 'red',
        5: 'orange'
        # Add more colors as needed
    }

    dash_app.layout.children.append(html.Div(id='gini', children=[
        html.H1("How Evenly Interacted?"),
        html.Div([
            dcc.Dropdown(
                id='gini-project-dropdown',
                options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                value=[i for i in dataset['project'].unique()],  # Initially select all projects
                placeholder="Select projects",
                multi=True,
                style={'width': '200px'}
            ),
            html.Button('Reset', id='gini-reset-button', n_clicks=0)
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='gini-graph'),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Gini Coefficient Graph Explanation
                This graph shows the equality of the meeting.
                - **Gini Coefficient = 0**: All participants spoke at the same rate in the meeting
                - **Gini Coefficient = 1**: Only one participant spoke, the rest did not speak at all

                ### Axis Labels
                - **X-axis**: Indicates the session number of the meeting.
                - **Y-axis**: Indicates the Gini coefficient, with higher values representing greater inequality in participation.
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px'})
    ]))

    @dash_app.callback(
        [Output('gini-graph', 'figure'),
         Output('gini-project-dropdown', 'value')],
        [Input('gini-project-dropdown', 'value'),
         Input('gini-reset-button', 'n_clicks')]
    )
    def update_gini_graph(selected_projects, reset_clicks):
        fig = go.Figure()

        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'].split('.')[0] == 'gini-reset-button':
            selected_projects = []  # Reset to empty selection
            return fig.update_layout(
                xaxis_title='Meeting',
                yaxis_title='Gini Coefficient',
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                showlegend=False
            ), None

        if selected_projects:
            for project in selected_projects:
                gini_data = dataset[dataset['project'] == project].groupby('meeting_number')['gini_coefficient'].unique().apply(lambda x: x[0])
                gini_df = pd.DataFrame({'Meeting': range(1, len(gini_data) + 1), 'Gini Coefficient': gini_data})

                fig.add_trace(go.Scatter(
                    x=gini_df['Meeting'],
                    y=gini_df['Gini Coefficient'],
                    mode='lines+markers',
                    name=f'Gini Coefficient Project {project}',
                    marker=dict(symbol='circle'),
                    line=dict(color=color_map.get(project, 'black'))  # Use predefined color or default to black
                ))

            # Determine x-ticks only if there's at least one data point
            xticks = gini_df['Meeting'] if not gini_df.empty else []
            fig.update_layout(
                xaxis_title='Meeting',
                yaxis_title='Gini Coefficient',
                yaxis=dict(autorange='reversed'),  # Reverse y-axis
                xaxis=dict(tickmode='array', tickvals=xticks),
                showlegend=True
            )
        else:
            fig.update_layout(
                xaxis_title='Meeting',
                yaxis_title='Gini Coefficient',
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
                showlegend=False
            )

        return fig, selected_projects

