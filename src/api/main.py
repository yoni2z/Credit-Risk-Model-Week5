from fastapi import FastAPI
import mlflow.sklearn
import pandas as pd
import joblib
from src.data_processing import build_preprocessing_pipeline
from .pydantic_models import CustomerData, PredictionResponse

app = FastAPI()

# Load model and preprocessor
model_path = "mlruns/0/4de556fc8c254c75aba3a34d72e1a749/artifacts/model"
model = mlflow.sklearn.load_model(model_path)
preprocessor = joblib.load('data/processed/preprocessor.pkl')

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    processed_data = preprocessor.transform(df)
    prob = model.predict_proba(processed_data)[:, 1][0]
    return {"CustomerId": data.CustomerId, "RiskProbability": float(prob)}