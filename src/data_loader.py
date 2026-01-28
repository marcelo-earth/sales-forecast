"""Data loading utilities for the sales forecast project."""

import os
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def download_kaggle_data():
    """Download the Store Sales dataset from Kaggle.

    Requires kaggle.json credentials configured.
    See: https://www.kaggle.com/docs/api
    """
    import kaggle

    kaggle.api.competition_download_files(
        "store-sales-time-series-forecasting",
        path=RAW_DIR,
        unzip=True
    )
    print(f"Data downloaded to {RAW_DIR}")


def load_train_data() -> pd.DataFrame:
    """Load the training data."""
    filepath = RAW_DIR / "train.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Training data not found at {filepath}. "
            "Run download_kaggle_data() first."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


def load_test_data() -> pd.DataFrame:
    """Load the test data."""
    filepath = RAW_DIR / "test.csv"
    if not filepath.exists():
        raise FileNotFoundError(
            f"Test data not found at {filepath}. "
            "Run download_kaggle_data() first."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    return df


def load_stores() -> pd.DataFrame:
    """Load store metadata."""
    filepath = RAW_DIR / "stores.csv"
    return pd.read_csv(filepath)


def load_oil() -> pd.DataFrame:
    """Load oil prices data."""
    filepath = RAW_DIR / "oil.csv"
    return pd.read_csv(filepath, parse_dates=["date"])


def load_holidays() -> pd.DataFrame:
    """Load holidays data."""
    filepath = RAW_DIR / "holidays_events.csv"
    return pd.read_csv(filepath, parse_dates=["date"])


def load_transactions() -> pd.DataFrame:
    """Load transactions data."""
    filepath = RAW_DIR / "transactions.csv"
    return pd.read_csv(filepath, parse_dates=["date"])
