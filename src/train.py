"""
Train the CLV prediction model end-to-end.

Usage
-----
    python -m src.train
    python -m src.train --prediction-window-months 6

This script:
1. Loads the cleaned transaction dataset
2. Builds the leakage-safe feature/target dataset
3. Computes two baselines (mean prediction, naive "past = future")
4. Cross-validates multiple candidate models and picks a champion
   based on evidence, not a hardcoded assumption
5. Tunes the champion model's hyperparameters
6. Evaluates on a held-out test set and reports metrics against
   both baselines
7. Computes and saves £ segment thresholds (Low/Medium/High) so
   predict.py and app.py use the exact same segmentation logic
8. Saves the model, feature list, and a full model card (metadata)
"""

import argparse
import json
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import joblib

from src import config
from src.data_loader import load_clean_data
from src.feature_engineering import build_modeling_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

CANDIDATE_MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=config.RANDOM_STATE),
    "RandomForest": RandomForestRegressor(n_estimators=300, random_state=config.RANDOM_STATE, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
    "XGBoost": XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0),
}

BASE_MODEL_LOOKUP = {
    "GradientBoosting": GradientBoostingRegressor(random_state=config.RANDOM_STATE),
    "XGBoost": XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0),
    "RandomForest": RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
    "DecisionTree": DecisionTreeRegressor(random_state=config.RANDOM_STATE),
    "LinearRegression": LinearRegression(),
}


def select_champion_model(X_train: pd.DataFrame, y_train: pd.Series) -> str:
    """Cross-validate all candidate models and return the name of the best one."""
    cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    results = []

    for name, model in CANDIDATE_MODELS.items():
        scores = -cross_val_score(
            model, X_train, y_train, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
        )
        results.append({"Model": name, "CV_MAE_log_mean": scores.mean(), "CV_MAE_log_std": scores.std()})
        logger.info("CV result — %-18s MAE(log): %.4f (+/- %.4f)", name, scores.mean(), scores.std())

    results_df = pd.DataFrame(results).sort_values("CV_MAE_log_mean")
    champion = results_df.iloc[0]["Model"]
    logger.info("Champion model selected via cross-validation: %s", champion)
    return champion


def tune_champion_model(champion_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Run RandomizedSearchCV for the selected champion model."""
    grid = config.HYPERPARAMETER_GRIDS[champion_name]
    base_model = BASE_MODEL_LOOKUP[champion_name]

    if not grid:
        base_model.fit(X_train, y_train)
        return base_model, {}

    search = RandomizedSearchCV(
        base_model,
        param_distributions=grid,
        n_iter=config.RANDOM_SEARCH_N_ITER,
        scoring="neg_mean_absolute_error",
        cv=3,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    logger.info("Best hyperparameters for %s: %s", champion_name, search.best_params_)
    return search.best_estimator_, search.best_params_


def train_model(prediction_window_months: int = config.PREDICTION_WINDOW_MONTHS) -> None:
    config.MODEL_DIR.mkdir(exist_ok=True)

    logger.info("Loading cleaned dataset...")
    df = load_clean_data()

    logger.info("Building leakage-safe feature/target dataset...")
    clv_data = build_modeling_dataset(df, prediction_window_months=prediction_window_months)

    X = clv_data[config.FEATURE_COLUMNS]
    y = clv_data[config.TARGET_COLUMN]
    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
        X, y_log, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    logger.info("Train size: %s | Test size: %s", len(X_train), len(X_test))

    # --- Baselines ---
    mean_pred = np.full_like(y_test_raw, y_train_raw.mean(), dtype=float)
    mean_baseline_mae = mean_absolute_error(y_test_raw, mean_pred)

    naive_pred = X_test["Monetary"].values
    naive_baseline_mae = mean_absolute_error(y_test_raw, naive_pred)

    logger.info("Baseline — mean prediction MAE:  £%.2f", mean_baseline_mae)
    logger.info("Baseline — naive past=future MAE: £%.2f", naive_baseline_mae)

    # --- Model selection & tuning ---
    champion_name = select_champion_model(X_train, y_train)
    best_model, best_params = tune_champion_model(champion_name, X_train, y_train)

    # --- Evaluation ---
    y_pred_log = best_model.predict(X_test)
    y_pred = np.clip(np.expm1(y_pred_log), a_min=0, a_max=None)

    final_mae = mean_absolute_error(y_test_raw, y_pred)
    final_rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
    final_r2 = r2_score(y_test_raw, y_pred)
    mae_pct_of_mean = final_mae / y_train_raw.mean() * 100
    improvement_vs_naive = (naive_baseline_mae - final_mae) / naive_baseline_mae * 100

    logger.info("Final Test MAE:  £%.2f (%.1f%% of mean CLV)", final_mae, mae_pct_of_mean)
    logger.info("Final Test RMSE: £%.2f", final_rmse)
    logger.info("Final Test R2:   %.4f", final_r2)
    logger.info("Improvement over naive baseline: %.1f%%", improvement_vs_naive)

    if final_mae > naive_baseline_mae:
        logger.warning(
            "Model MAE (£%.2f) is WORSE than the naive 'past=future' baseline (£%.2f). "
            "This should be disclosed honestly in any reporting of this model, not hidden.",
            final_mae, naive_baseline_mae,
        )

    # --- Refit on full data for deployment, compute segment thresholds ---
    best_model.fit(X, y_log)
    full_predictions = np.clip(np.expm1(best_model.predict(X)), a_min=0, a_max=None)

    thresholds = np.quantile(full_predictions, config.SEGMENT_QUANTILES)
    segment_thresholds = {
        "quantiles": config.SEGMENT_QUANTILES,
        "labels": config.SEGMENT_LABELS,
        "value_thresholds_gbp": [round(float(t), 2) for t in thresholds],
    }

    # --- Save artifacts ---
    joblib.dump(best_model, config.MODEL_DIR / config.MODEL_FILE)
    joblib.dump(config.FEATURE_COLUMNS, config.MODEL_DIR / config.FEATURES_FILE)

    with open(config.MODEL_DIR / config.SEGMENT_THRESHOLDS_FILE, "w") as f:
        json.dump(segment_thresholds, f, indent=2)

    model_card = {
        "model_name": champion_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_definition": (
            f"Total revenue in the {prediction_window_months}-month window "
            f"following the cutoff date"
        ),
        "features": config.FEATURE_COLUMNS,
        "best_hyperparameters": best_params,
        "n_training_customers": int(len(X)),
        "test_set_metrics": {
            "MAE_GBP": round(float(final_mae), 2),
            "RMSE_GBP": round(float(final_rmse), 2),
            "R2": round(float(final_r2), 4),
            "MAE_as_pct_of_mean_CLV": round(float(mae_pct_of_mean), 2),
        },
        "baseline_metrics": {
            "mean_baseline_MAE_GBP": round(float(mean_baseline_mae), 2),
            "naive_past_equals_future_MAE_GBP": round(float(naive_baseline_mae), 2),
            "improvement_over_naive_baseline_pct": round(float(improvement_vs_naive), 2),
            "beats_naive_baseline": bool(final_mae <= naive_baseline_mae),
        },
        "pct_customers_with_zero_target": round(float((y == 0).mean() * 100), 2),
        "segment_thresholds_gbp": segment_thresholds,
        "known_limitations": [
            "No multi-period walk-forward validation due to dataset length",
            "Cannot score customers with zero purchase history (cold-start)",
            "Wider prediction error for extreme high-value customers",
        ],
    }

    with open(config.MODEL_DIR / config.MODEL_CARD_FILE, "w") as f:
        json.dump(model_card, f, indent=2)

    logger.info("Model saved to:      %s", config.MODEL_DIR / config.MODEL_FILE)
    logger.info("Features saved to:   %s", config.MODEL_DIR / config.FEATURES_FILE)
    logger.info("Thresholds saved to: %s", config.MODEL_DIR / config.SEGMENT_THRESHOLDS_FILE)
    logger.info("Model card saved to: %s", config.MODEL_DIR / config.MODEL_CARD_FILE)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the CLV prediction model.")
    parser.add_argument(
        "--prediction-window-months",
        type=int,
        default=config.PREDICTION_WINDOW_MONTHS,
        help="Length of the forward-looking CLV target window, in months.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_model(prediction_window_months=args.prediction_window_months)
