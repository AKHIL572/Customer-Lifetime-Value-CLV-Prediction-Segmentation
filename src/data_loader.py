from pathlib import Path
import pandas as pd


# -----------------------------
# Define Project Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "Dataset"


# -----------------------------
# Load Raw Data
# -----------------------------
def load_raw_data(filename: str = "OnlineRetail.csv") -> pd.DataFrame:
    """
    Loads the raw dataset from the Dataset folder.

    Parameters
    ----------
    filename : str
        Name of the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataset.
    """
    file_path = DATASET_PATH / filename

    if not file_path.exists():
        raise FileNotFoundError(f"File not found at {file_path}")

    df = pd.read_csv(file_path)
    return df


# -----------------------------
# Save Cleaned Data
# -----------------------------
def save_clean_data(df: pd.DataFrame,
                    filename: str = "cleaned_online_retail_transactions.csv") -> None:
    """
    Saves cleaned dataset to Dataset folder.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset
    filename : str
        Output filename
    """
    output_path = DATASET_PATH / filename
    df.to_csv(output_path, index=False)


# -----------------------------
# Load Cleaned Data
# -----------------------------
def load_clean_data(
        filename: str = "cleaned_online_retail_transactions.csv"
) -> pd.DataFrame:
    """
    Loads cleaned dataset from Dataset folder.

    Parameters
    ----------
    filename : str
        Name of cleaned CSV file.

    Returns
    -------
    pd.DataFrame
    """
    file_path = DATASET_PATH / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {file_path}. "
            "Run preprocessing first."
        )

    df = pd.read_csv(file_path, parse_dates=["InvoiceDate"])
    return df
