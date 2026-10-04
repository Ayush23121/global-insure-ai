"""
check.py
--------
Trains a Linear Regression model on the standard Kaggle 'insurance.csv'
dataset (https://www.kaggle.com/datasets/mirichoi0218/insurance) and
persists it to model.pkl for the FastAPI backend to consume.

Features used (deliberately kept to 4, mirroring the original frontend
prototype's hardcoded coefficients):
    - age
    - bmi
    - smoker            (0 = no, 1 = yes)
    - age_smoker         (engineered interaction term = age * smoker)

Run:
    python check.py

Expects insurance.csv to be in the same directory as this script.
Download it from Kaggle if you don't already have it.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os

DATA_PATH = "insurance.csv"
MODEL_OUTPUT_PATH = "model.pkl"


def load_data(path: str) -> pd.DataFrame:
    """Load the raw Kaggle insurance dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Download the Kaggle 'insurance.csv' "
            "dataset and place it next to check.py before running this script."
        )
    df = pd.read_csv(path)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features:
      - Map 'smoker' yes/no -> 1/0
      - Create the age * smoker interaction term
    """
    df = df.copy()

    # Map categorical smoker column to binary
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    # Feature engineering: interaction term captures the fact that the
    # health-cost penalty for smoking compounds with age.
    df["age_smoker"] = df["age"] * df["smoker"]

    return df


def train_model(df: pd.DataFrame):
    """Train a Linear Regression model on the engineered feature set."""
    feature_cols = ["age", "bmi", "smoker", "age_smoker"]
    target_col = "charges"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate on held-out test data
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, feature_cols, r2, mae


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("Engineering features (smoker encoding + age*smoker interaction)...")
    df = engineer_features(df)

    print("Training Linear Regression model...")
    model, feature_cols, r2, mae = train_model(df)

    print("\n========== MODEL EVALUATION ==========")
    print(f"R^2 Score        : {r2:.4f}")
    print(f"Mean Abs. Error  : {mae:.2f}")

    print("\n========== EXTRACTED COEFFICIENTS ==========")
    print(f"INTERCEPT : {model.intercept_}")
    for name, coef in zip(feature_cols, model.coef_):
        print(f"{name.upper():<12}: {coef}")

    # Persist the trained model (and the feature order it expects)
    joblib.dump({"model": model, "features": feature_cols}, MODEL_OUTPUT_PATH)
    print(f"\nModel successfully saved to '{MODEL_OUTPUT_PATH}'")


if __name__ == "__main__":
    main()
