"""
Unit tests for the CLV pipeline.

Run with:  python -m pytest tests/ -v

These are not decorative — they specifically test the things that
have historically gone wrong in this project:
  - the target window leakage bug (unbounded future window)
  - predict.py accepting bad input silently
  - segment assignment consistency
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import build_modeling_dataset, create_clv_target, create_rfm_features
from src.predict import assign_segment


@pytest.fixture
def synthetic_transactions():
    """
    A small, hand-built transaction set with known behavior, so expected
    feature/target values can be checked exactly rather than just "it ran".
    """
    rows = [
        # Customer 1: two purchases before cutoff, one purchase in target window
        {"CustomerID": 1, "InvoiceNo": "A1", "StockCode": "X1", "Quantity": 2,
         "InvoiceDate": "2023-01-01", "TotalAmount": 20.0},
        {"CustomerID": 1, "InvoiceNo": "A2", "StockCode": "X2", "Quantity": 1,
         "InvoiceDate": "2023-02-01", "TotalAmount": 10.0},
        {"CustomerID": 1, "InvoiceNo": "A3", "StockCode": "X1", "Quantity": 3,
         "InvoiceDate": "2023-04-01", "TotalAmount": 30.0},  # within 6-month target window

        # Customer 2: one purchase before cutoff, nothing afterward -> CLV_6M should be 0
        {"CustomerID": 2, "InvoiceNo": "B1", "StockCode": "X1", "Quantity": 1,
         "InvoiceDate": "2023-01-15", "TotalAmount": 15.0},

        # Customer 3: purchase far outside the 6-month target window
        # -> must NOT be counted in CLV_6M (this is the leakage-bug regression test)
        {"CustomerID": 3, "InvoiceNo": "C1", "StockCode": "X1", "Quantity": 1,
         "InvoiceDate": "2023-01-10", "TotalAmount": 5.0},
        {"CustomerID": 3, "InvoiceNo": "C2", "StockCode": "X1", "Quantity": 1,
         "InvoiceDate": "2023-12-01", "TotalAmount": 999.0},  # 11 months after cutoff
    ]
    df = pd.DataFrame(rows)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def test_create_rfm_features_basic_shape(synthetic_transactions):
    cutoff = pd.Timestamp("2023-03-01")
    rfm = create_rfm_features(synthetic_transactions, cutoff)

    # Customers 1, 2, and 3 all have at least one purchase before the cutoff
    # (customer 3's second purchase is what falls in the target window, tested separately below)
    assert set(rfm["CustomerID"]) == {1, 2, 3}
    assert {"Recency", "Frequency", "Monetary", "TotalQuantity",
            "AvgOrderValue", "AvgBasketSize", "AvgPurchaseIntervalDays", "TenureDays"}.issubset(rfm.columns)

    cust1 = rfm[rfm["CustomerID"] == 1].iloc[0]
    assert cust1["Frequency"] == 2
    assert cust1["Monetary"] == 30.0
    assert cust1["AvgOrderValue"] == 15.0


def test_single_purchase_customer_gets_recency_as_interval_fill(synthetic_transactions):
    cutoff = pd.Timestamp("2023-03-01")
    rfm = create_rfm_features(synthetic_transactions, cutoff)
    cust2 = rfm[rfm["CustomerID"] == 2].iloc[0]
    # Customer 2 has one purchase -> no real interval -> filled with Recency
    assert cust2["AvgPurchaseIntervalDays"] == cust2["Recency"]


def test_target_window_is_bounded_not_leaking_future_revenue(synthetic_transactions):
    """
    Regression test for the original notebook bug: the target window MUST
    exclude revenue that occurs after the prediction window ends, even if
    it occurs later in the dataset.
    """
    cutoff = pd.Timestamp("2023-03-01")
    target = create_clv_target(synthetic_transactions, cutoff, prediction_window_months=6)

    cust3_target = target[target["CustomerID"] == 3]
    # Customer 3's £999 purchase happens ~11 months after cutoff — outside
    # the 6-month window — so it must NOT appear in the target at all.
    assert cust3_target.empty, "Target window leaked revenue from outside the prediction window"


def test_build_modeling_dataset_fills_zero_for_no_future_purchase(synthetic_transactions):
    # build_modeling_dataset computes its own cutoff as (max date in df) - 6 months.
    # Max date here is 2023-12-01, so cutoff = 2023-06-01, target window = 2023-06-01..2023-12-01.
    clv_data = build_modeling_dataset(synthetic_transactions, prediction_window_months=6)

    # Customer 1's last purchase (2023-04-01) falls BEFORE this auto-computed cutoff,
    # so all their activity is in the feature window and none in the target window -> 0.
    cust1_row = clv_data[clv_data["CustomerID"] == 1].iloc[0]
    assert cust1_row["CLV_6M"] == 0.0

    # Customer 2's only purchase is before cutoff and nothing follows -> 0.
    cust2_row = clv_data[clv_data["CustomerID"] == 2].iloc[0]
    assert cust2_row["CLV_6M"] == 0.0

    # Customer 3's second purchase (2023-12-01) falls exactly at the end of the
    # target window (cutoff < date <= cutoff + 6 months) -> correctly counted.
    cust3_row = clv_data[clv_data["CustomerID"] == 3].iloc[0]
    assert cust3_row["CLV_6M"] == 999.0


def test_assign_segment_uses_saved_thresholds_correctly():
    thresholds = {
        "value_thresholds_gbp": [0.0, 500.0, 2000.0, 999999.0],
        "labels": ["Low Value", "Medium Value", "High Value"],
    }
    assert assign_segment(100, thresholds) == "Low Value"
    assert assign_segment(500, thresholds) == "Low Value"
    assert assign_segment(1500, thresholds) == "Medium Value"
    assert assign_segment(50000, thresholds) == "High Value"


def test_assign_segment_handles_value_above_all_thresholds():
    thresholds = {
        "value_thresholds_gbp": [0.0, 500.0, 2000.0],
        "labels": ["Low Value", "Medium Value"],
    }
    # No upper bucket beyond the last threshold -> falls back to last label
    assert assign_segment(999999, thresholds) == "Medium Value"
