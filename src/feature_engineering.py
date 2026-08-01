"""
Feature engineering for CLV modeling.

Builds the same leakage-safe, cutoff-date-based feature and target
tables used in Notebook 3. This module is the single source of truth
for feature construction — the notebook should eventually be refactored
to call these functions directly instead of duplicating the logic.

IMPORTANT: the target window is explicitly bounded
(cutoff_date < InvoiceDate <= cutoff_date + prediction_window_months).
An earlier version of this project's notebook left the target window
unbounded, which happened to work only because the dataset ended
exactly N months after the chosen cutoff. That bug does not exist here.
"""

import logging

import pandas as pd
from pandas.tseries.offsets import DateOffset

from src import config

logger = logging.getLogger(__name__)


def create_rfm_features(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create the expanded RFM + behavioral feature set from transaction
    data occurring on or before the cutoff date.

    Features produced
    ------------------
    Recency                  Days between cutoff_date and the customer's last purchase
    Frequency                Number of distinct invoices
    Monetary                 Total historical spend
    TotalQuantity             Total units purchased
    AvgOrderValue             Monetary / Frequency
    AvgBasketSize             Average distinct items (StockCodes) per order
    AvgPurchaseIntervalDays   Median days between consecutive purchases
                              (single-purchase customers are filled with
                              their Recency value as a neutral placeholder)
    TenureDays                Days between first and last purchase before cutoff

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataset (must include InvoiceDate, CustomerID,
        InvoiceNo, StockCode, Quantity, TotalAmount).
    cutoff_date : pd.Timestamp
        Date separating the feature window from the prediction window.

    Returns
    -------
    pd.DataFrame
        One row per CustomerID with the features above.
    """
    feature_df = df[df["InvoiceDate"] <= cutoff_date]

    rfm = feature_df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (cutoff_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
        TotalQuantity=("Quantity", "sum"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
    ).reset_index()

    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    rfm["TenureDays"] = (rfm["LastPurchase"] - rfm["FirstPurchase"]).dt.days

    basket_size = feature_df.groupby("InvoiceNo")["StockCode"].nunique().rename("ItemsInBasket")
    basket_by_customer = (
        feature_df[["CustomerID", "InvoiceNo"]]
        .drop_duplicates()
        .merge(basket_size, on="InvoiceNo")
        .groupby("CustomerID")["ItemsInBasket"]
        .mean()
        .rename("AvgBasketSize")
    )
    rfm = rfm.merge(basket_by_customer, on="CustomerID", how="left")

    purchase_dates = feature_df.groupby(["CustomerID", "InvoiceNo"])["InvoiceDate"].min().reset_index()
    purchase_dates = purchase_dates.sort_values(["CustomerID", "InvoiceDate"])
    purchase_dates["DaysSincePrev"] = purchase_dates.groupby("CustomerID")["InvoiceDate"].diff().dt.days
    avg_interval = purchase_dates.groupby("CustomerID")["DaysSincePrev"].median().rename("AvgPurchaseIntervalDays")
    rfm = rfm.merge(avg_interval, on="CustomerID", how="left")

    # Single-purchase customers have no interval — fill with Recency as a
    # neutral placeholder (documented assumption, matches Notebook 3).
    rfm["AvgPurchaseIntervalDays"] = rfm["AvgPurchaseIntervalDays"].fillna(rfm["Recency"])

    rfm = rfm.drop(columns=["FirstPurchase", "LastPurchase"])

    logger.info("Built feature table for %s customers as of cutoff %s", len(rfm), cutoff_date.date())
    return rfm


def create_clv_target(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    prediction_window_months: int = config.PREDICTION_WINDOW_MONTHS,
) -> pd.DataFrame:
    """
    Create the future CLV target variable, strictly bounded to the
    prediction window (not open-ended).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataset.
    cutoff_date : pd.Timestamp
        Feature window cutoff date.
    prediction_window_months : int
        Length of the future window to sum revenue over.

    Returns
    -------
    pd.DataFrame
        Columns: CustomerID, CLV_6M
    """
    future_end_date = cutoff_date + DateOffset(months=prediction_window_months)

    target_df = df[(df["InvoiceDate"] > cutoff_date) & (df["InvoiceDate"] <= future_end_date)]

    clv_target = target_df.groupby("CustomerID")["TotalAmount"].sum().reset_index()
    clv_target.columns = ["CustomerID", config.TARGET_COLUMN]

    logger.info(
        "Built target table for %s customers over window %s to %s",
        len(clv_target), cutoff_date.date(), future_end_date.date(),
    )
    return clv_target


def build_modeling_dataset(
    df: pd.DataFrame,
    prediction_window_months: int = config.PREDICTION_WINDOW_MONTHS,
) -> pd.DataFrame:
    """
    Build the final leakage-safe dataset for CLV modeling.

    Steps
    -----
    1. Define cutoff date as (max date in df) - prediction_window_months
    2. Build RFM + behavioral features from data before the cutoff
    3. Build the CLV target from data strictly within the bounded future window
    4. Merge; customers with no purchases in the target window get CLV_6M = 0
       (a genuine behavioral outcome, not missing data)

    Returns
    -------
    pd.DataFrame
        One row per customer: FEATURE_COLUMNS + CLV_6M
    """
    cutoff_date = df["InvoiceDate"].max() - DateOffset(months=prediction_window_months)

    rfm = create_rfm_features(df, cutoff_date)
    clv_target = create_clv_target(df, cutoff_date, prediction_window_months)

    clv_data = rfm.merge(clv_target, on="CustomerID", how="left")
    clv_data[config.TARGET_COLUMN] = clv_data[config.TARGET_COLUMN].fillna(0)

    zero_pct = (clv_data[config.TARGET_COLUMN] == 0).mean() * 100
    logger.info(
        "Modeling dataset built: %s customers, %.1f%% with CLV_6M = 0",
        len(clv_data), zero_pct,
    )

    return clv_data
