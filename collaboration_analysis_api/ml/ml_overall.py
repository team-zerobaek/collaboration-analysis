from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

def initialize_overall_ml_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    if dataset is None or dataset.empty:
        raise ValueError("Dataset is not properly initialized or is empty")

    # Create necessary columns
    dataset['num_speakers'] = dataset.groupby(['project', 'meeting_number'])['speaker_number'].transform('nunique')

    # Ensure interaction_count column exists
    if 'interaction_count' not in dataset.columns:
        dataset['interaction_count'] = 0  # Or some appropriate default value

    dataset['normalized_interaction_frequency'] = dataset['interaction_count'] / dataset['duration']

    dash_app.layout.children.append(html.Div([
        html.H1("ML Models for Overall Collaboration Score"),
        html.Div(id='ml-output'),
        html.Button('Build Model for Overall Collaboration Score', id='build-ml-overall', n_clicks=0),
        dcc.Loading(id="loading", type="default", children=html.Div(id="loading-output"))
    ]))

    @dash_app.callback(
        Output('loading-output', 'children'),
        [Input('build-ml-overall', 'n_clicks')]
    )
    def build_model(n_clicks):
        if n_clicks > 0:
            return build_overall_model()
        return ""

def build_overall_model():
    global dataset

    # Filter dataset for project 4 and overall_collaboration_score between 1 and 10
    dataset_filtered = dataset[(dataset['project'] == 4) & (dataset['overall_collaboration_score'].between(1, 10))]

    # Cleaning the dataset
    dataset_filtered = dataset_filtered.replace([np.inf, -np.inf], np.nan)
    dataset_filtered = dataset_filtered.dropna()

    # Features and target
    features = dataset_filtered[['meeting_number', 'normalized_speech_frequency', 'gini_coefficient',
                                 'degree_centrality', 'num_speakers', 'normalized_interaction_frequency']]
    target = dataset_filtered['overall_collaboration_score']

    column_transformer = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), ['meeting_number', 'normalized_speech_frequency', 'gini_coefficient', 'degree_centrality', 'num_speakers', 'normalized_interaction_frequency'])
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
                'Performance': performance,
                'R2': r2,
                'MSE': mse,
                'CV Mean': mean_cv_score,
                'CV Std': std_cv_score,
                'Training Time': training_time,
                'Params': grid.best_params_
            })

            if performance > best_performance:
                best_performance = performance
                best_model_info = {
                    'model': model_name,
                    'r2': r2,
                    'mse': mse,
                    'cv_mean_r2': mean_cv_score,
                    'cv_std_r2': std_cv_score,
                    'params': grid.best_params_,
                    'model_object': grid.best_estimator_
                }

            # Update intermediate results
            dash_app.layout.children.append(html.Div([
                html.H4(f'Current Model: {model_name}'),
                html.P(f'R2: {r2}'),
                html.P(f'MSE: {mse}'),
                html.P(f'CV Mean R2: {mean_cv_score}'),
                html.P(f'CV Std R2: {std_cv_score}'),
                html.P(f'Best Params: {grid.best_params_}')
            ]))

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
    vif_data['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

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

    return html.Div([
        html.H3('Model Building Completed!'),
        html.H4('Model Performance Comparison:'),
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
        html.H4('Multicollinearity Check (VIF):'),
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
        html.H4('Feature Importance:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('feature_importance.png', 'rb').read()).decode()),
        html.H4('Actual vs Predicted:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('actual_vs_predicted.png', 'rb').read()).decode())
    ])

# Example call to initialize the app
# initialize_overall_ml_app(dash_app_instance, dataset_instance)
