import pandas as pd
from pandas.tseries.offsets import DateOffset


# -----------------------------
# Create RFM Features
# -----------------------------
def create_rfm_features(df: pd.DataFrame,
                        cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """
    Create RFM features from transaction data.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataset
    cutoff_date : pd.Timestamp
        Date separating feature window and prediction window

    Returns
    -------
    pd.DataFrame
        RFM feature dataset
    """

    feature_df = df[df["InvoiceDate"] <= cutoff_date]

    rfm = feature_df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (cutoff_date - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalAmount": "sum",
        "Quantity": "sum"
    }).reset_index()

    rfm.columns = [
        "CustomerID",
        "Recency",
        "Frequency",
        "Monetary",
        "TotalQuantity"
    ]

    return rfm


# -----------------------------
# Create CLV Target
# -----------------------------
def create_clv_target(df: pd.DataFrame,
                      cutoff_date: pd.Timestamp,
                      prediction_window_months: int = 6) -> pd.DataFrame:
    """
    Create future CLV target variable.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned transaction dataset
    cutoff_date : pd.Timestamp
        Feature window cutoff date
    prediction_window_months : int
        Future prediction window

    Returns
    -------
    pd.DataFrame
        CLV target dataset
    """

    future_end_date = cutoff_date + DateOffset(months=prediction_window_months)

    target_df = df[
        (df["InvoiceDate"] > cutoff_date) &
        (df["InvoiceDate"] <= future_end_date)
    ]

    clv_target = (
        target_df.groupby("CustomerID")["TotalAmount"]
        .sum()
        .reset_index()
    )

    clv_target.columns = ["CustomerID", "CLV"]

    return clv_target


# -----------------------------
# Merge Features & Target
# -----------------------------
def build_modeling_dataset(df: pd.DataFrame,
                           prediction_window_months: int = 6) -> pd.DataFrame:
    """
    Build final dataset for CLV modeling.

    Steps:
    1. Define cutoff date
    2. Create RFM features
    3. Create CLV target
    4. Merge features and target

    Returns
    -------
    pd.DataFrame
    """

    cutoff_date = df["InvoiceDate"].max() - DateOffset(
        months=prediction_window_months
    )

    rfm = create_rfm_features(df, cutoff_date)
    clv_target = create_clv_target(
        df,
        cutoff_date,
        prediction_window_months
    )

    clv_data = rfm.merge(
        clv_target,
        on="CustomerID",
        how="left"
    )

    # Customers who didn't purchase in prediction window
    clv_data["CLV"] = clv_data["CLV"].fillna(0)

    return clv_data
