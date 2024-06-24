from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from dash import dcc, html

dash_app = None
dataset = None

def initialize_gini_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    # Check if the Gini components already exist
    existing_gini_components = [comp.id for comp in dash_app.layout.children if hasattr(comp, 'id') and comp.id in ['gini-project-dropdown-unique', 'gini-graph']]

    if 'gini-project-dropdown-unique' not in existing_gini_components and 'gini-graph' not in existing_gini_components:
        dash_app.layout.children.append(
            html.Div([
                html.H1("Gini Coefficient"),
                html.Div([
                    dcc.Dropdown(
                        id='gini-project-dropdown-unique',
                        options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                        placeholder="Select projects",
                        multi=True,
                        style={'width': '200px'}
                    ),
                    html.Button('Reset', id='reset-gini-button', n_clicks=0)
                ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
                dcc.Graph(id='gini-graph')
            ], style={'margin-bottom': '20px'})
        )

    @dash_app.callback(
        Output('gini-graph', 'figure'),
        [Input('gini-project-dropdown-unique', 'value')],
        [State('gini-project-dropdown-unique', 'options')]
    )
    def update_gini_graph(selected_project, options):
        if not selected_project:
            selected_project = [opt['value'] for opt in options]

        fig = go.Figure()
        for project in selected_project:
            gini_data = dataset[dataset['project'] == project].groupby('meeting_number')['gini_coefficient'].unique().apply(lambda x: x[0])
            gini_df = pd.DataFrame({'Meeting': range(1, len(gini_data) + 1), 'Gini Coefficient': gini_data})

            fig.add_trace(go.Scatter(x=gini_df['Meeting'],
                                     y=gini_df['Gini Coefficient'],
                                     mode='lines+markers',
                                     name=f'Gini Coefficient Project {project}',
                                     marker=dict(symbol='circle')))

        fig.update_layout(
            title='Gini Coefficient by Meeting',
            xaxis_title='Meeting',
            yaxis_title='Gini Coefficient',
            showlegend=True
        )

        return fig

    @dash_app.callback(
        Output('gini-project-dropdown-unique', 'value'),
        [Input('reset-gini-button', 'n_clicks')]
    )
    def reset_gini_filters(n_clicks):
        return []
