from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_loader import load_clean_data
from src.feature_engineering import build_modeling_dataset


# -----------------------------
# Define Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "Models"

MODEL_PATH.mkdir(exist_ok=True)


def train_model():

    print("Loading cleaned dataset...")
    df = load_clean_data()

    print("Building RFM + CLV dataset...")
    clv_data = build_modeling_dataset(df)

    feature_columns = ["Recency", "Frequency", "Monetary", "TotalQuantity"]

    X = clv_data[feature_columns]
    y = clv_data["CLV"]

    # Log transform target
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )

    print("Training Gradient Boosting with hyperparameter tuning...")

    param_dist = {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
    }

    model = GradientBoostingRegressor(random_state=42)

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=15,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=42,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred_log = best_model.predict(X_test)

    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print("\nModel Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # -----------------------------
    # Save Model
    # -----------------------------
    joblib.dump(best_model, MODEL_PATH / "clv_model.pkl")
    joblib.dump(feature_columns, MODEL_PATH / "clv_features.pkl")

    print("\nModel saved successfully to Models folder.")


if __name__ == "__main__":
    train_model()
