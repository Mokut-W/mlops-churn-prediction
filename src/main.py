"""
Customer Churn Prediction - FastAPI Serving Layer
Phase 3 of the MLOps pipeline: wraps the trained model in a REST API.
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Customer Churn Prediction API")

# Load model artifacts ONCE at startup - not per-request
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")


class CustomerInput(BaseModel):
    gender: str  # "Male" or "Female"
    SeniorCitizen: int  # 0 or 1
    Partner: str  # "Yes" or "No"
    Dependents: str  # "Yes" or "No"
    tenure: int
    PhoneService: str  # "Yes" or "No"
    MultipleLines: str  # "Yes", "No", "No phone service"
    InternetService: str  # "DSL", "Fiber optic", "No"
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str  # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str  # "Yes" or "No"
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class PredictionOutput(BaseModel):
    churn_prediction: str  # "Yes" or "No"
    churn_probability: float


def preprocess_input(data: CustomerInput) -> pd.DataFrame:
    """Apply the SAME encoding used during training."""
    df = pd.DataFrame([data.dict()])

    # Binary encoding - matches train.py exactly
    binary_map = {"Yes": 1, "No": 0}
    df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Partner"] = df["Partner"].map(binary_map)
    df["Dependents"] = df["Dependents"].map(binary_map)
    df["PhoneService"] = df["PhoneService"].map(binary_map)
    df["PaperlessBilling"] = df["PaperlessBilling"].map(binary_map)

    # One-hot encode the same multi-category columns as training
    multi_cat_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # Align columns to match training exactly - fills missing dummy columns with 0
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict", response_model=PredictionOutput)
def predict(data: CustomerInput):
    try:
        processed = preprocess_input(data)
        scaled = scaler.transform(processed)

        prediction = model.predict(scaled)[0]
        probability = model.predict_proba(scaled)[0][
            1
        ]  # probability of churn (class 1)

        return PredictionOutput(
            churn_prediction="Yes" if prediction == 1 else "No",
            churn_probability=round(float(probability), 4),
        )
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(e))
