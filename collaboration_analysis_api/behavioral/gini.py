from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

dash_app = None
dataset = None

def initialize_gini_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    dash_app.layout.children.append(html.Div([
        html.H1("How evenly interacted? (Gini Coefficient)"),
        html.Div([
            dcc.Dropdown(
                id='gini-project-dropdown',
                options=[{'label': f'Project {i}', 'value': i} for i in dataset['project'].unique()],
                placeholder="Select projects",
                multi=True,
                style={'width': '200px'}
            )
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
        dcc.Graph(id='gini-graph')
    ]))

    @dash_app.callback(
        Output('gini-graph', 'figure'),
        [Input('gini-project-dropdown', 'value')]
    )
    def update_gini_graph(selected_projects):
        fig = go.Figure()

        if not selected_projects:
            selected_projects = dataset['project'].unique()

        for project in selected_projects:
            gini_data = dataset[dataset['project'] == project].groupby('meeting_number')['gini_coefficient'].unique().apply(lambda x: x[0])
            gini_df = pd.DataFrame({'Meeting': range(1, len(gini_data) + 1), 'Gini Coefficient': gini_data})

            fig.add_trace(go.Scatter(
                x=gini_df['Meeting'],
                y=gini_df['Gini Coefficient'],
                mode='lines+markers',
                name=f'Gini Coefficient Project {project}',
                marker=dict(symbol='circle')
            ))

        # Determine x-ticks only if there's at least one data point
        xticks = gini_df['Meeting'] if not gini_df.empty else []

        fig.update_layout(
            title='Gini Coefficient by Meeting',
            xaxis_title='Meeting',
            yaxis_title='Gini Coefficient',
            yaxis=dict(autorange='reversed'),  # Reverse y-axis
            xaxis=dict(tickmode='array', tickvals=xticks),
            showlegend=True
        )

        return fig
