from dash import Dash, dcc, html, Input, Output
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64
from statsmodels.stats.outliers_influence import variance_inflation_factor

dash_app = None
dataset = None

def initialize_individual_others_ml_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    if dataset is None or dataset.empty:
        raise ValueError("Dataset is not properly initialized or is empty")

    # Exclude self-interactions
    dataset = dataset[dataset['speaker_id'] != dataset['next_speaker_id']].copy()

    # Create necessary columns
    dataset['num_speakers'] = dataset.groupby(['project', 'meeting_number'])['speaker_id'].transform('nunique')

    # Ensure interaction_count column exists
    dataset['interaction_count'] = dataset['count']
    if 'interaction_count' not in dataset.columns:
        dataset['interaction_count'] = 0  # Or some appropriate default value

    dataset['normalized_interaction_frequency'] = dataset['interaction_count'] / dataset['duration']

    dash_app.layout.children.append(html.Div(id ='other', children=[
        html.H1("ML Models for Individual Collaboration Score"),
        html.Div(id='ml-output-individual'),
        html.Button('Run Dummy Model', id='run-dummy-individual', n_clicks=0),
        html.Button('Run Actual Model', id='run-actual-individual', n_clicks=0),
        dcc.Loading(id="loading", type="default", children=html.Div(id="loading-output-individual")),
        html.Details([
            html.Summary('Description', style={'margin-bottom': '10px'}),
            dcc.Markdown("""
                ### Middle Title
                - Content 1
                - Content 2
            """, style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px'})
        ], style={'margin-top': '10px','margin-bottom': '20px'})
    ]))

    @dash_app.callback(
        Output('loading-output-individual', 'children'),
        [Input('run-dummy-individual', 'n_clicks'), Input('run-actual-individual', 'n_clicks')]
    )
    def build_model(dummy_clicks, actual_clicks):
        ctx = dash.callback_context

        if not ctx.triggered:
            return ""
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'run-dummy-individual':
            return build_dummy_model()
        elif button_id == 'run-actual-individual':
            return build_actual_model()
        return ""

def build_actual_model():
    global dataset

    # Filter dataset for individual_collaboration_score between 1 and 10
    dataset_filtered = dataset[dataset['individual_collaboration_score'].between(1, 10)]

    # Cleaning the dataset
    dataset_filtered = dataset_filtered.replace([np.inf, -np.inf], np.nan)
    dataset_filtered = dataset_filtered.dropna()

    # Features and target
    features = dataset_filtered[['meeting_number', 'normalized_speech_frequency', 'gini_coefficient',
                                 'degree_centrality', 'num_speakers', 'normalized_interaction_frequency', 'speaker_id', 'next_speaker_id']]
    target = dataset_filtered['individual_collaboration_score']

    column_transformer = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), ['meeting_number', 'normalized_speech_frequency', 'gini_coefficient', 'degree_centrality', 'num_speakers', 'normalized_interaction_frequency']),
            ('onehot', OneHotEncoder(), ['speaker_id', 'next_speaker_id'])
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    regression_models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(random_state=42),
        'XGBRegressor': XGBRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
        'LightGBM Regressor': LGBMRegressor(random_state=42, verbose=-1),
        'CatBoost Regressor': CatBoostRegressor(random_state=42, verbose=0),
        'SVM Regressor': SVR()
    }

    param_grids_regression = {
        'Linear Regression': {},
        'Decision Tree': {'model__max_depth': [3, 5, 7]},
        'Random Forest Regressor': {'model__n_estimators': [50, 100, 150], 'model__max_depth': [5, 10, 15]},
        'XGBRegressor': {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5, 7], 'model__learning_rate': [0.01, 0.1, 0.2]},
        'Gradient Boosting Regressor': {'model__n_estimators': [50, 100], 'model__max_depth': [3, 5], 'model__learning_rate': [0.01, 0.1, 0.2, 0.5]},
        'K-Nearest Neighbors Regressor': {'model__n_neighbors': [3, 5, 7]},
        'LightGBM Regressor': {'model__n_estimators': [50, 100, 200], 'model__num_leaves': [31, 62], 'model__learning_rate': [0.01, 0.1, 0.3]},
        'CatBoost Regressor': {'model__iterations': [100, 200, 400], 'model__depth': [4, 6, 10]},
        'SVM Regressor': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
    }

    def find_best_hyperparameters_regression(X_train, y_train, X_test, y_test):
        alpha = 0.35  # Weight for the performance metric (adjust as necessary)
        beta = 0.33
        gamma = 0.3

        best_performance = -float('inf')
        best_model_info = {}
        model_performance = []

        for model_name, model in regression_models.items():
            start_time = time.time()
            pipeline = Pipeline([
                ('preprocessor', column_transformer),
                ('model', model)
            ])
            grid = GridSearchCV(pipeline, param_grids_regression[model_name], cv=3, scoring='r2')
            grid.fit(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time

            y_pred = grid.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=kf, scoring='r2')
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)

            performance = alpha * r2 + gamma * mean_cv_score + beta * (1/mse) + (1 - alpha - beta - gamma)

            model_performance.append({
                'Model': model_name,
                'Performance': round(performance, 2),
                'R2': round(r2, 2),
                'MSE': round(mse, 2),
                'CV Mean': round(mean_cv_score, 2),
                'CV Std': round(std_cv_score, 2),
                'Training Time': round(training_time, 2)
            })

            if performance > best_performance:
                best_performance = performance
                best_model_info = {
                    'model': model_name,
                    'r2': round(r2, 2),
                    'mse': round(mse, 2),
                    'cv_mean_r2': round(mean_cv_score, 2),
                    'cv_std_r2': round(std_cv_score, 2),
                    'params': grid.best_params_,
                    'training_time': round(training_time, 2),
                    'model_object': grid.best_estimator_
                }

        return best_model_info, model_performance

    best_reg_model_info, model_performance = find_best_hyperparameters_regression(X_train, y_train, X_test, y_test)

    def predict_and_evaluate(best_reg_model_info, X_train, y_train, X_test, y_test):
        best_model = best_reg_model_info['model_object']
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        results_with_test_df = pd.concat([X_test.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Actual', y='Predicted', data=results_df)
        plt.plot([results_df.min().min(), results_df.max().max()], [results_df.min().min(), results_df.max().max()], color='red', linewidth=2)
        plt.title('Actual vs Predicted Values')
        plt.savefig('actual_vs_predicted.png')
        plt.show()

        return results_with_test_df

    results_df = predict_and_evaluate(best_reg_model_info, X_train, y_train, X_test, y_test)

    # Multicollinearity Check (VIF)
    X_vif = pd.DataFrame(column_transformer.fit_transform(features), columns=column_transformer.get_feature_names_out())
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X_vif.columns
    vif_data['VIF'] = [round(variance_inflation_factor(X_vif.values, i), 2) for i in range(X_vif.shape[1])]

    # Feature Importance
    if hasattr(best_reg_model_info['model_object'].named_steps['model'], 'feature_importances_'):
        best_model = best_reg_model_info['model_object'].named_steps['model']
        feature_importances_reg = best_model.feature_importances_

        importance_df_reg = pd.DataFrame({'Feature': column_transformer.get_feature_names_out(), 'Importance': feature_importances_reg})
        importance_df_reg = importance_df_reg.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df_reg)
        plt.title('Feature Importance Analysis (Regression)')
        plt.savefig('feature_importance.png')
        plt.show()

    # Model Performance Table
    model_performance_df = pd.DataFrame(model_performance)

    # Hyperparameter details
    hyperparameter_details = [
        {'Model': 'Decision Tree', 'Parameters': 'max_depth: [3, 5, 7]'},
        {'Model': 'Random Forest Regressor', 'Parameters': 'n_estimators: [50, 100, 150], max_depth: [5, 10, 15]'},
        {'Model': 'XGBRegressor', 'Parameters': 'n_estimators: [50, 100], max_depth: [3, 5, 7], learning_rate: [0.01, 0.1, 0.2]'},
        {'Model': 'Gradient Boosting Regressor', 'Parameters': 'n_estimators: [50, 100], max_depth: [3, 5], learning_rate: [0.01, 0.1, 0.2, 0.5]'},
        {'Model': 'K-Nearest Neighbors Regressor', 'Parameters': 'n_neighbors: [3, 5, 7]'},
        {'Model': 'LightGBM Regressor', 'Parameters': 'n_estimators: [50, 100, 200], num_leaves: [31, 62], learning_rate: [0.01, 0.1, 0.3]'},
        {'Model': 'CatBoost Regressor', 'Parameters': 'iterations: [100, 200, 400], depth: [4, 6, 10]'},
        {'Model': 'SVM Regressor', 'Parameters': 'C: [0.1, 1, 10], kernel: [linear, rbf]'}
    ]

    hyperparameter_df = pd.DataFrame(hyperparameter_details)

    return html.Div([
        html.H3('Model Building Completed!'),
        html.H4('Input Summary'),
        html.P('Features and Label used:'),
        html.P('Features: meeting_number, normalized_speech_frequency, gini_coefficient, degree_centrality, num_speakers, normalized_interaction_frequency, speaker_id, next_speaker_id'),
        html.P('Label: individual_collaboration_score'),
        html.P('Encodings used: StandardScaler for all features, OneHotEncoder for speaker_id and next_speaker_id'),
        html.P('Models compared: Linear Regression, Decision Tree, Random Forest Regressor, XGBRegressor, Gradient Boosting Regressor, K-Nearest Neighbors Regressor, LightGBM Regressor, CatBoost Regressor, SVM Regressor'),
        html.H4('Hyperparameter Tuning'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(hyperparameter_df.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[hyperparameter_df[col] for col in hyperparameter_df.columns],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
        html.H4('Performance Summary'),
        html.H5('Model Performance Comparison:'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(model_performance_df.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[model_performance_df[col] for col in model_performance_df.columns],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
        html.H5('Selected Model:'),
        html.P(f"Model: {best_reg_model_info['model']}"),
        html.P(f"R2: {best_reg_model_info['r2']}"),
        html.P(f"MSE: {best_reg_model_info['mse']}"),
        html.P(f"CV Mean R2: {best_reg_model_info['cv_mean_r2']}"),
        html.P(f"CV Std R2: {best_reg_model_info['cv_std_r2']}"),
        html.P(f"Training Time: {best_reg_model_info['training_time']} seconds"),
        html.P(f"Best Params: {best_reg_model_info['params']}"),
        html.H5('Feature Importance:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('feature_importance.png', 'rb').read()).decode()),
        html.H5('Actual vs Predicted:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('actual_vs_predicted.png', 'rb').read()).decode()),
        html.H4('Model Validity Check'),
        html.H5('Overfitting Check (Cross-Validation Results):'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=['Model', 'CV Mean R2', 'CV Std R2'],
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[[model['Model'] for model in model_performance],
                                           [model['CV Mean'] for model in model_performance],
                                           [model['CV Std'] for model in model_performance]],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
        html.H5('Multicollinearity Check (VIF):'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(vif_data.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[vif_data[col] for col in vif_data.columns],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
    ])

def build_dummy_model():
    # Dummy values for demonstration purposes
    model_performance = [
        {'Model': 'Linear Regression', 'Performance': 0.75, 'R2': 0.76, 'MSE': 1.25, 'CV Mean': 0.72, 'CV Std': 0.03, 'Training Time': 0.1},
        {'Model': 'Decision Tree', 'Performance': 0.82, 'R2': 0.85, 'MSE': 1.15, 'CV Mean': 0.80, 'CV Std': 0.04, 'Training Time': 0.2},
        {'Model': 'Random Forest Regressor', 'Performance': 0.90, 'R2': 0.92, 'MSE': 0.85, 'CV Mean': 0.88, 'CV Std': 0.02, 'Training Time': 0.5},
        {'Model': 'XGBRegressor', 'Performance': 0.88, 'R2': 0.90, 'MSE': 0.95, 'CV Mean': 0.87, 'CV Std': 0.03, 'Training Time': 0.7},
        {'Model': 'Gradient Boosting Regressor', 'Performance': 0.89, 'R2': 0.91, 'MSE': 0.90, 'CV Mean': 0.88, 'CV Std': 0.03, 'Training Time': 0.6},
        {'Model': 'K-Nearest Neighbors Regressor', 'Performance': 0.80, 'R2': 0.83, 'MSE': 1.10, 'CV Mean': 0.79, 'CV Std': 0.04, 'Training Time': 0.3},
        {'Model': 'LightGBM Regressor', 'Performance': 0.91, 'R2': 0.93, 'MSE': 0.80, 'CV Mean': 0.89, 'CV Std': 0.02, 'Training Time': 0.4},
        {'Model': 'CatBoost Regressor', 'Performance': 0.92, 'R2': 0.94, 'MSE': 0.75, 'CV Mean': 0.91, 'CV Std': 0.02, 'Training Time': 0.5},
        {'Model': 'SVM Regressor', 'Performance': 0.78, 'R2': 0.80, 'MSE': 1.30, 'CV Mean': 0.76, 'CV Std': 0.03, 'Training Time': 0.4}
    ]

    best_reg_model_info = {
        'model': 'CatBoost Regressor',
        'r2': 0.94,
        'mse': 0.75,
        'cv_mean_r2': 0.91,
        'cv_std_r2': 0.02,
        'params': {'model__iterations': 200, 'model__depth': 6},
        'training_time': 0.5
    }

    # Generate dummy importance plot
    importance_df_reg = pd.DataFrame({
        'Feature': ['meeting_number', 'normalized_speech_frequency', 'gini_coefficient', 'degree_centrality', 'num_speakers', 'normalized_interaction_frequency', 'speaker_id', 'next_speaker_id'],
        'Importance': [0.15, 0.25, 0.10, 0.20, 0.18, 0.12, 0.13, 0.14]
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df_reg)
    plt.title('Feature Importance Analysis (Dummy)')
    plt.savefig('feature_importance.png')
    plt.show()

    # Generate dummy actual vs predicted plot
    dummy_results_df = pd.DataFrame({'Actual': np.random.randint(1, 10, 100), 'Predicted': np.random.randint(1, 10, 100)})
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Actual', y='Predicted', data=dummy_results_df)
    plt.plot([dummy_results_df.min().min(), dummy_results_df.max().max()], [dummy_results_df.min().min(), dummy_results_df.max().max()], color='red', linewidth=2)
    plt.title('Actual vs Predicted Values (Dummy)')
    plt.savefig('actual_vs_predicted.png')
    plt.show()

    # Dummy VIF data
    vif_data = pd.DataFrame({
        'Feature': ['meeting_number', 'normalized_speech_frequency', 'gini_coefficient', 'degree_centrality', 'num_speakers', 'normalized_interaction_frequency', 'speaker_id', 'next_speaker_id'],
        'VIF': [1.5, 2.0, 1.2, 1.8, 1.4, 1.6, 1.7, 1.3]
    })

    model_performance_df = pd.DataFrame(model_performance)

    # Dummy hyperparameter details
    hyperparameter_details = [
        {'Model': 'Decision Tree', 'Parameters': 'max_depth: [3, 5, 7]'},
        {'Model': 'Random Forest Regressor', 'Parameters': 'n_estimators: [50, 100, 150], max_depth: [5, 10, 15]'},
        {'Model': 'XGBRegressor', 'Parameters': 'n_estimators: [50, 100], max_depth: [3, 5, 7], learning_rate: [0.01, 0.1, 0.2]'},
        {'Model': 'Gradient Boosting Regressor', 'Parameters': 'n_estimators: [50, 100], max_depth: [3, 5], learning_rate: [0.01, 0.1, 0.2, 0.5]'},
        {'Model': 'K-Nearest Neighbors Regressor', 'Parameters': 'n_neighbors: [3, 5, 7]'},
        {'Model': 'LightGBM Regressor', 'Parameters': 'n_estimators: [50, 100, 200], num_leaves: [31, 62], learning_rate: [0.01, 0.1, 0.3]'},
        {'Model': 'CatBoost Regressor', 'Parameters': 'iterations: [100, 200, 400], depth: [4, 6, 10]'},
        {'Model': 'SVM Regressor', 'Parameters': 'C: [0.1, 1, 10], kernel: [linear, rbf]'}
    ]

    hyperparameter_df = pd.DataFrame(hyperparameter_details)

    return html.Div([
        html.H3('Model Building Completed! (Dummy)'),
        html.H4('Input Summary'),
        html.P('Features and Label used:'),
        html.P('Features: meeting_number, normalized_speech_frequency, gini_coefficient, degree_centrality, num_speakers, normalized_interaction_frequency, speaker_id, next_speaker_id'),
        html.P('Label: individual_collaboration_score'),
        html.P('Encodings used: StandardScaler for all features, OneHotEncoder for speaker_id and next_speaker_id'),
        html.P('Models compared: Linear Regression, Decision Tree, Random Forest Regressor, XGBRegressor, Gradient Boosting Regressor, K-Nearest Neighbors Regressor, LightGBM Regressor, CatBoost Regressor, SVM Regressor'),
        html.H4('Hyperparameter Tuning'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(hyperparameter_df.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[hyperparameter_df[col] for col in hyperparameter_df.columns],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
        html.H4('Performance Summary'),
        html.H5('Model Performance Comparison:'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(model_performance_df.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[model_performance_df[col] for col in model_performance_df.columns],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
        html.H5('Selected Model:'),
        html.P(f"Model: {best_reg_model_info['model']}"),
        html.P(f"R2: {best_reg_model_info['r2']}"),
        html.P(f"MSE: {best_reg_model_info['mse']}"),
        html.P(f"CV Mean R2: {best_reg_model_info['cv_mean_r2']}"),
        html.P(f"CV Std R2: {best_reg_model_info['cv_std_r2']}"),
        html.P(f"Training Time: {best_reg_model_info['training_time']} seconds"),
        html.P(f"Best Params: {best_reg_model_info['params']}"),
        html.H5('Feature Importance:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('feature_importance.png', 'rb').read()).decode()),
        html.H5('Actual vs Predicted:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('actual_vs_predicted.png', 'rb').read()).decode()),
        html.H4('Model Validity Check'),
        html.H5('Overfitting Check (Cross-Validation Results):'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=['Model', 'CV Mean R2', 'CV Std R2'],
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[[model['Model'] for model in model_performance],
                                           [model['CV Mean'] for model in model_performance],
                                           [model['CV Std'] for model in model_performance]],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
        html.H5('Multicollinearity Check (VIF):'),
        dcc.Graph(
            figure=go.Figure(
                data=[
                    go.Table(
                        header=dict(values=list(vif_data.columns),
                                    fill_color='paleturquoise',
                                    align='left'),
                        cells=dict(values=[vif_data[col] for col in vif_data.columns],
                                   fill_color='lavender',
                                   align='left'))
                ]
            )
        ),
    ])

# Example call to initialize the app
# initialize_individual_others_ml_app(dash_app_instance, dataset_instance)
