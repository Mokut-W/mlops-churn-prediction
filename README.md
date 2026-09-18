# Customer Churn Prediction — MLOps Pipeline

An end-to-end machine learning pipeline that predicts customer churn using the Telco Customer Churn dataset. Goes from raw data exploration through a trained model to a live FastAPI inference endpoint.

## What it does

Given a customer's account details (contract type, tenure, services subscribed, billing info, etc.), the API predicts whether they're likely to churn and returns a probability score.

## Project structure

```
mlops/
├── data/                       # Raw and processed datasets (gitignored)
├── models/                     # Saved model artifacts (gitignored — regenerate via train.py)
├── src/
│   ├── train.py                # Trains the model, saves model/scaler/feature columns to models/
│   └── main.py                 # FastAPI app exposing the /predict endpoint
├── churn_exploration.ipynb     # Initial EDA and model experimentation
├── .gitignore
└── README.md
```

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone https://github.com/Mokut-W/mlops-churn-prediction.git
   cd mlops-churn-prediction
   python -m venv venv
   venv\Scripts\activate      # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the Telco Customer Churn CSV in `data/` (dataset not included in this repo — see note below).

## Usage

**Train the model:**
```bash
cd src
python train.py
```
This trains a Logistic Regression model and saves `churn_model.pkl`, `scaler.pkl`, and `feature_columns.pkl` to `models/`.

**Run the API (from the project root):**
```bash
uvicorn src.main:app --reload
```
The API will be live at `http://127.0.0.1:8000`. Interactive docs (Swagger UI) are available at `http://127.0.0.1:8000/docs`.

> Run this from the project root, not from inside `src/` — `main.py` loads model artifacts using relative paths (e.g. `models/churn_model.pkl`), which resolve against your current working directory.

## API reference

### `GET /`
Health check. Returns:
```json
{"message": "Customer Churn Prediction API is running"}
```

### `POST /predict`

**Request body:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}
```

**Response:**
```json
{
  "churn_prediction": "No",
  "churn_probability": 0.1883
}
```

## Model

- **Algorithm:** Logistic Regression
- **Dataset:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset)
- Model, scaler, and feature column order are persisted via `joblib` after training and loaded at API startup.

## Notes / next steps

- Dataset and trained model artifacts are gitignored — run `train.py` to regenerate them locally.
- Git setup was deferred during initial development; this repo was checkpointed once the pipeline was verified end-to-end.
- Potential next steps: Dockerize the API, add input validation edge cases, add a `requirements.txt` if not already present, write basic tests for `/predict`.
