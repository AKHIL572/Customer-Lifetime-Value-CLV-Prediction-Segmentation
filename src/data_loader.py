"""
Data loading utilities for the CLV Prediction project.

All file paths are resolved through config.py so this module never
hardcodes a filename that could drift out of sync with the rest of
the project.
"""

import logging

import pandas as pd

from src import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_raw_data(filename: str = config.RAW_DATA_FILE) -> pd.DataFrame:
    """
    Load the raw dataset from the Dataset folder.

    Parameters
    ----------
    filename : str
        Name of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset, unmodified.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the expected path.
    """
    file_path = config.DATASET_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Raw data file not found at {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8-sig")
    logger.info("Loaded raw data: %s rows, %s columns from %s", len(df), df.shape[1], file_path)
    return df


def save_clean_data(df: pd.DataFrame, filename: str = config.CLEANED_DATA_FILE) -> None:
    """
    Save a cleaned dataset to the Dataset folder.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset.
    filename : str
        Output filename.
    """
    output_path = config.DATASET_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info("Saved cleaned data: %s rows to %s", len(df), output_path)


def load_clean_data(filename: str = config.CLEANED_DATA_FILE) -> pd.DataFrame:
    """
    Load the cleaned, transaction-level dataset produced by the
    data-cleaning step (Notebook 1).

    Parameters
    ----------
    filename : str
        Name of the cleaned CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with InvoiceDate parsed as datetime.

    Raises
    ------
    FileNotFoundError
        If the cleaned dataset has not been produced yet.
    """
    file_path = config.DATASET_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {file_path}. "
            "Run the data-cleaning step (Notebook 1 or an equivalent cleaning "
            "script) before loading clean data."
        )

    df = pd.read_csv(file_path, parse_dates=["InvoiceDate"])
    logger.info("Loaded cleaned data: %s rows from %s", len(df), file_path)
    return df


def load_cancelled_orders(filename: str = config.CANCELLED_ORDERS_FILE) -> pd.DataFrame:
    """
    Load the cancelled/returned orders preserved separately during cleaning.
    Useful for future returns-behavior feature engineering.

    Returns
    -------
    pd.DataFrame
    """
    file_path = config.DATASET_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Cancelled orders file not found at {file_path}")

    df = pd.read_csv(file_path, parse_dates=["InvoiceDate"])
    logger.info("Loaded cancelled orders: %s rows from %s", len(df), file_path)
    return df
