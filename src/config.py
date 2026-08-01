"""
Central configuration for the CLV Prediction project.

Every path, constant, and hyperparameter search space used across
data_loader.py, feature_engineering.py, train.py, and predict.py is
defined here ONCE. Nothing downstream should hardcode a filename,
feature list, or threshold — import it from here instead.

This is what makes app.py, train.py, and the notebooks agree with
each other on things like segmentation thresholds and feature order.
"""

from pathlib import Path

# -----------------------------
# Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
MODEL_DIR = PROJECT_ROOT / "Models"

RAW_DATA_FILE = "OnlineRetail.csv"
CLEANED_DATA_FILE = "cleaned_online_retail_transactions.csv"
CANCELLED_ORDERS_FILE = "cancelled_orders.csv"
PREDICTIONS_FILE = "clv_predictions.csv"
AUDIT_LOG_FILE = "data_cleaning_audit_log.csv"
EDA_SUMMARY_FILE = "customer_level_eda_summary.csv"

MODEL_FILE = "clv_model.pkl"
FEATURES_FILE = "clv_features.pkl"
MODEL_CARD_FILE = "model_card.json"
SEGMENT_THRESHOLDS_FILE = "segment_thresholds.json"

# -----------------------------
# Modeling Constants
# -----------------------------
PREDICTION_WINDOW_MONTHS = 6
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature set — MUST match the columns produced by feature_engineering.build_modeling_dataset().
# Order matters: this is the exact column order the model was trained on.
FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "TotalQuantity",
    "AvgOrderValue",
    "AvgBasketSize",
    "AvgPurchaseIntervalDays",
    "TenureDays",
]

TARGET_COLUMN = "CLV_6M"

# -----------------------------
# Segmentation
# -----------------------------
# Quantile cutpoints used at training time to derive segment_thresholds.json.
# The actual £ thresholds are computed once during training and reused everywhere
# else (predict.py, app.py) so segmentation is consistent across the whole project.
SEGMENT_QUANTILES = [0.0, 0.7, 0.9, 1.0]
SEGMENT_LABELS = ["Low Value", "Medium Value", "High Value"]

# -----------------------------
# Candidate Models & Hyperparameter Search Spaces
# -----------------------------
HYPERPARAMETER_GRIDS = {
    "GradientBoosting": {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
    },
    "XGBoost": {
        "n_estimators": [200, 300, 400],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "subsample": [0.8, 1.0],
    },
    "RandomForest": {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 8, 12],
        "min_samples_leaf": [1, 2, 4],
    },
    "DecisionTree": {
        "max_depth": [None, 5, 8, 12],
        "min_samples_leaf": [1, 2, 4, 8],
    },
    "LinearRegression": {},
}

RANDOM_SEARCH_N_ITER = 15
