from dash import dcc, html, Input, Output
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

dash_app = None
dataset = None

def initialize_individual_self_ml_app(dash_app_instance, dataset_instance):
    global dash_app, dataset
    dash_app = dash_app_instance
    dataset = dataset_instance

    # Create necessary columns
    dataset['num_speakers'] = dataset.groupby(['project', 'meeting_number'])['speaker_number'].transform('nunique')

    # Ensure interaction_count column exists
    if 'interaction_count' not in dataset.columns:
        dataset['interaction_count'] = 0  # Or some appropriate default value

    dataset['normalized_interaction_frequency'] = dataset['interaction_count'] / dataset['duration']

    dash_app.layout.children.append(html.Div([
        html.H1("ML Models for Individual Self Collaboration Score"),
        html.Div(id='ml-individual-self-output'),
        html.Button('Build Model for Individual Self Collaboration Score', id='build-ml-individual-self', n_clicks=0),
        dcc.Loading(id="loading-individual-self", type="default", children=html.Div(id="loading-individual-self-output"))
    ]))

    @dash_app.callback(
        Output('loading-individual-self-output', 'children'),
        [Input('build-ml-individual-self', 'n_clicks')]
    )
    def build_model(n_clicks):
        if n_clicks > 0:
            return build_individual_self_model()
        return ""

def build_individual_self_model():
    global dataset

    # Cleaning the dataset
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna()

    features = dataset.drop(columns=['overall_collaboration_score', 'individual_collaboration_score', 'id', 'project', 'speaker_number', 'indegree_centrality', 'outdegree_centrality', 'betweenness_centrality', 'closeness_centrality', 'eigenvector_centrality', 'pagerank'])
    target = dataset['individual_self_collaboration_score']

    # One-hot encoding for num_speakers
    column_transformer = ColumnTransformer([
        ('onehot', OneHotEncoder(), ['num_speakers']),
        ('scale', StandardScaler(), features.drop(columns=['num_speakers']).columns)
    ], remainder='passthrough')

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
        'LightGBM Regressor': {'model__n_estimators': [50, 100, 200], 'model__num_leaves': [31, 62,], 'model__learning_rate': [0.01, 0.1, 0.3]},
        'CatBoost Regressor': {'model__iterations': [100, 200, 400], 'model__depth': [4, 6, 10]},
        'SVM Regressor': {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}
    }

    def find_best_hyperparameters_regression(X_train, y_train, X_test, y_test):
        alpha = 0.35  # Weight for the performance metric (adjust as necessary)
        beta = 0.33
        gamma = 0.3

        best_performance = -float('inf')
        best_model_info = {}
        model_performance = {}

        for model_name, model in regression_models.items():
            start_time = time.time()
            pipeline = Pipeline([
                ('transformer', column_transformer),
                ('model', model)
            ])
            grid = GridSearchCV(
                pipeline, param_grids_regression[model_name], cv=3, scoring='r2')
            grid.fit(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time

            y_pred = grid.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                grid.best_estimator_, X_train, y_train, cv=kf, scoring='r2')
            mean_cv_score = np.mean(cv_scores)
            std_cv_score = np.std(cv_scores)

            performance = alpha * (r2) + gamma * \
                (mean_cv_score) + beta * (1/mse) + (1 - alpha-beta-gamma)

            model_performance[model_name] = {
                'performance': performance, 'r2': r2, 'mse': mse, 'cv_mean': mean_cv_score, 'cv_std': std_cv_score, 'training_time': training_time, 'params': grid.best_params_}

            if performance > best_performance:
                best_performance = performance
                best_model_info = {'model': model_name, 'r2': r2, 'mse': mse, 'cv_mean_r2': mean_cv_score, 'cv_std_r2': std_cv_score,
                                   'params': grid.best_params_, 'model_object': grid.best_estimator_}

        return best_model_info, model_performance

    best_reg_model_info, model_performance = find_best_hyperparameters_regression(
        X_train, y_train, X_test, y_test)

    def predict_and_evaluate(best_reg_model_info, X_train, y_train, X_test, y_test):
        best_model = best_reg_model_info['model_object']
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results_df = pd.DataFrame(
            {'Actual': y_test, 'Predicted': y_pred})
        results_with_test_df = pd.concat(
            [X_test.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Actual', y='Predicted', data=results_df)
        plt.plot([results_df.min().min(), results_df.max().max()], [
                 results_df.min().min(), results_df.max().max()], color='red', linewidth=2)
        plt.title('Actual vs Predicted Values')
        plt.savefig('actual_vs_predicted_individual_self.png')
        plt.show()

        return results_with_test_df

    results_df = predict_and_evaluate(
        best_reg_model_info, X_train, y_train, X_test, y_test)

    if hasattr(best_reg_model_info['model_object'].named_steps['model'], 'feature_importances_'):
        best_model = best_reg_model_info['model_object'].named_steps['model']
        feature_importances_reg = best_model.feature_importances_

        column_transformer.fit(X_train)  # Ensure transformer is fitted
        feature_names = np.array(X_train.columns)
        if isinstance(column_transformer.transformers_[0][1], OneHotEncoder):
            onehot_columns = column_transformer.transformers_[0][1].get_feature_names(['num_speakers'])
            feature_names = np.concatenate([onehot_columns, feature_names])

        importance_df_reg = pd.DataFrame(
            {'Feature': feature_names, 'Importance': feature_importances_reg})
        importance_df_reg = importance_df_reg.sort_values(
            by='Importance', ascending=False)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df_reg)
        plt.title('Feature Importance Analysis (Regression)')
        plt.savefig('feature_importance_individual_self.png')
        plt.show()

    model_comparison_df = pd.DataFrame(model_performance).T
    model_comparison_df.reset_index(inplace=True)
    model_comparison_df.rename(columns={'index': 'Model'}, inplace=True)

    fig = px.bar(model_comparison_df, x='Model', y='r2', title='Model Comparison (R2 Scores)')
    fig.show()

    return html.Div([
        html.H3('Model Building Completed!'),
        html.P(f"Model: {best_reg_model_info['model']}"),
        html.P(f"R2: {best_reg_model_info['r2']}"),
        html.P(f"MSE: {best_reg_model_info['mse']}"),
        html.P(f"CV Mean R2: {best_reg_model_info['cv_mean_r2']}"),
        html.P(f"CV Std R2: {best_reg_model_info['cv_std_r2']}"),
        html.P(f"Best Params: {best_reg_model_info['params']}"),
        html.H4('Feature Importance:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('feature_importance_individual_self.png', 'rb').read()).decode()),
        html.H4('Actual vs Predicted:'),
        html.Img(src='data:image/png;base64,' + base64.b64encode(open('actual_vs_predicted_individual_self.png', 'rb').read()).decode()),
        html.H4('Model Comparison:'),
        dcc.Graph(figure=fig)
    ])
