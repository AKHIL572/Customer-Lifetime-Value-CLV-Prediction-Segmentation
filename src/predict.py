from pathlib import Path
import numpy as np
import pandas as pd
import joblib


# -----------------------------
# Define Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "Models"


# -----------------------------
# Load Model Assets
# -----------------------------
def load_model():
    """
    Load trained model and feature names.
    """

    model_file = MODEL_PATH / "clv_model.pkl"
    feature_file = MODEL_PATH / "clv_features.pkl"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_file}. "
            "Run train.py first."
        )

    model = joblib.load(model_file)
    features = joblib.load(feature_file)

    return model, features


# -----------------------------
# Predict CLV
# -----------------------------
def predict_clv(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict 6-month CLV for given input data.

    Parameters
    ----------
    input_data : pd.DataFrame
        Must contain columns:
        ['Recency', 'Frequency', 'Monetary', 'TotalQuantity']

    Returns
    -------
    pd.DataFrame
        Original data + Predicted_CLV column
    """

    model, feature_names = load_model()

    # Ensure correct column order
    input_data = input_data[feature_names]

    # Model predicts log scale
    log_predictions = model.predict(input_data)

    # Convert back to original scale
    predictions = np.expm1(log_predictions)

    result = input_data.copy()
    result["Predicted_CLV"] = predictions

    return result
