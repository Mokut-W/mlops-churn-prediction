"""
Customer Churn Prediction - Training Script
Phase 2 of the MLOps pipeline: notebook logic ported into a clean, reproducible script.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """Load the pre-processed, encoded dataset."""
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    """Split into features/target, then train/test sets."""
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale numeric features - important for Logistic Regression."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """Print key metrics - accuracy alone is misleading on imbalanced churn data."""
    preds = model.predict(X_test)

    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {accuracy_score(y_test, preds):.4f}")
    print(f"Precision: {precision_score(y_test, preds):.4f}")
    print(f"Recall:    {recall_score(y_test, preds):.4f}")
    print(f"F1 Score:  {f1_score(y_test, preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    return f1_score(y_test, preds)


def main():
    df = load_data("data/churn_processed.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    feature_columns = X_train.columns.tolist()  # <-- add this line

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    log_reg = train_logistic_regression(X_train_scaled, y_train)
    rf = train_random_forest(X_train, y_train)

    log_reg_f1 = evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")
    rf_f1 = evaluate_model(rf, X_test, y_test, "Random Forest")

    if log_reg_f1 >= rf_f1:
        best_model, best_name = log_reg, "Logistic Regression"
    else:
        best_model, best_name = rf, "Random Forest"

    print(f"\nBest model: {best_name} (F1: {max(log_reg_f1, rf_f1):.4f})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/churn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")  # <-- add this line
    print("Model, scaler, and feature columns saved to models/")


if __name__ == "__main__":
    main()
