"""
Prediction utilities for the CLV model.

Loads the trained model, feature list, and the £ segment thresholds
saved by train.py, so that any caller (the Streamlit app, a batch
scoring job, this module's own CLI) gets a CLV prediction AND a
Low/Medium/High label that is guaranteed consistent with how the
model was evaluated at training time.
"""

import json
import logging

import numpy as np
import pandas as pd
import joblib

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class ModelNotTrainedError(Exception):
    """Raised when model artifacts are missing — train.py has not been run yet."""


def load_model():
    """
    Load the trained model and its expected feature column order.

    Returns
    -------
    tuple(model, feature_names: list[str])

    Raises
    ------
    ModelNotTrainedError
        If the model has not been trained yet.
    """
    model_file = config.MODEL_DIR / config.MODEL_FILE
    feature_file = config.MODEL_DIR / config.FEATURES_FILE

    if not model_file.exists() or not feature_file.exists():
        raise ModelNotTrainedError(
            f"Model artifacts not found in {config.MODEL_DIR}. Run `python -m src.train` first."
        )

    model = joblib.load(model_file)
    features = joblib.load(feature_file)
    logger.info("Loaded model and %s features from %s", len(features), config.MODEL_DIR)
    return model, features


def load_segment_thresholds() -> dict:
    """
    Load the £ segment thresholds computed at training time.

    Returns
    -------
    dict with keys: quantiles, labels, value_thresholds_gbp

    Raises
    ------
    ModelNotTrainedError
        If thresholds have not been computed yet (i.e., model not trained).
    """
    thresholds_file = config.MODEL_DIR / config.SEGMENT_THRESHOLDS_FILE
    if not thresholds_file.exists():
        raise ModelNotTrainedError(
            f"Segment thresholds not found at {thresholds_file}. Run `python -m src.train` first."
        )
    with open(thresholds_file) as f:
        return json.load(f)


def _validate_input(input_data: pd.DataFrame, feature_names: list) -> None:
    """Raise a clear, actionable error if input_data doesn't match the model's contract."""
    if not isinstance(input_data, pd.DataFrame):
        raise TypeError(f"input_data must be a pandas DataFrame, got {type(input_data)}")

    missing_cols = set(feature_names) - set(input_data.columns)
    if missing_cols:
        raise ValueError(
            f"input_data is missing required columns: {sorted(missing_cols)}. "
            f"Expected columns: {feature_names}"
        )

    for col in feature_names:
        if not pd.api.types.is_numeric_dtype(input_data[col]):
            raise ValueError(f"Column '{col}' must be numeric, got dtype {input_data[col].dtype}")
        if input_data[col].isnull().any():
            raise ValueError(f"Column '{col}' contains missing values, which is not supported")
        if (input_data[col] < 0).any():
            raise ValueError(f"Column '{col}' contains negative values, which is not valid for CLV features")


def assign_segment(predicted_clv: float, thresholds: dict) -> str:
    """
    Assign a Low/Medium/High Value label to a single predicted CLV value,
    using the £ thresholds saved at training time (NOT a re-derived quantile,
    since a single new prediction can't be quantile-ranked against itself).
    """
    value_thresholds = thresholds["value_thresholds_gbp"]
    labels = thresholds["labels"]

    for i, label in enumerate(labels):
        if predicted_clv <= value_thresholds[i + 1]:
            return label
    return labels[-1]


def predict_clv(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Predict 6-month CLV (and assign a business segment) for the given input.

    Parameters
    ----------
    input_data : pd.DataFrame
        Must contain all columns in config.FEATURE_COLUMNS, numeric,
        non-null, non-negative. Extra columns are ignored.

    Returns
    -------
    pd.DataFrame
        Input data (reordered to feature_names) plus:
        - Predicted_CLV : float, clipped at 0 (CLV cannot be negative)
        - CLV_Segment   : str, one of the configured segment labels

    Raises
    ------
    ModelNotTrainedError
        If the model hasn't been trained yet.
    ValueError, TypeError
        If input_data doesn't match the expected schema.
    """
    model, feature_names = load_model()
    thresholds = load_segment_thresholds()

    _validate_input(input_data, feature_names)

    ordered_input = input_data[feature_names].copy()

    log_predictions = model.predict(ordered_input)
    predictions = np.clip(np.expm1(log_predictions), a_min=0, a_max=None)

    result = ordered_input.copy()
    result["Predicted_CLV"] = predictions
    result["CLV_Segment"] = [assign_segment(p, thresholds) for p in predictions]

    logger.info("Generated %s predictions", len(result))
    return result
