import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from data_processing import process_data
import joblib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_input_example(df):
    """Create an input example for MLflow model signature."""
    numeric_cols = ['Amount', 'Value', 'PricingStrategy', 'TransactionHour', 'TransactionDay',
                    'TransactionMonth', 'TransactionYear', 'TotalAmount', 'AvgAmount',
                    'TransactionCount', 'StdAmount']
    example = df[numeric_cols].iloc[0:1]
    return example.to_dict(orient='records')[0]

def train_model(model, X_train, y_train, X_test, y_test, model_name, input_example=None):
    """Train a model and log to MLflow."""
    if mlflow.active_run():
        mlflow.end_run()
    
    with mlflow.start_run(run_name=model_name) as run:
        logger.info(f"Training {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, zero_division=0))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred))
        
        mlflow.sklearn.log_model(model, artifact_path=model_name, input_example=input_example)
        return model, run.info.run_id

def tune_model(X_train, y_train):
    """Perform hyperparameter tuning for RandomForest."""
    if mlflow.active_run():
        mlflow.end_run()
    
    with mlflow.start_run(run_name="RandomForest_Tuning"):
        logger.info("Performing hyperparameter tuning for RandomForest")
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 20]
        }
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        return grid_search.best_estimator_

def main():
    logger.info("Loading and processing data")
    X, df, preprocessor = process_data('data/raw/data.csv')
    y = df['is_high_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save fitted preprocessor
    joblib.dump(preprocessor, 'data/processed/preprocessor.pkl')
    mlflow.log_artifact('data/processed/preprocessor.pkl')
    
    input_example = create_input_example(df)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": tune_model(X_train, y_train)
    }
    
    best_model = None
    best_run_id = None
    best_roc_auc = 0
    for name, model in models.items():
        trained_model, run_id = train_model(model, X_train, y_train, X_test, y_test, name, input_example)
        y_pred = trained_model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = name
            best_run_id = run_id
    
    if best_run_id:
        logger.info(f"Registering model {best_model} with run ID {best_run_id}")
        mlflow.register_model(f"runs:/{best_run_id}/{best_model}", "BestCreditRiskModel")
        print(f"Registered model {best_model} with run ID {best_run_id}")

if __name__ == "__main__":
    main()