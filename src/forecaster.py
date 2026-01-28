"""Sales forecasting using AutoGluon TimeSeriesPredictor."""

from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

MODELS_DIR = Path(__file__).parent.parent / "models"


def prepare_timeseries_data(
    df: pd.DataFrame,
    item_id_columns: list[str] = ["store_nbr", "family"],
    target_column: str = "sales",
    timestamp_column: str = "date",
) -> TimeSeriesDataFrame:
    """Convert pandas DataFrame to AutoGluon TimeSeriesDataFrame.

    Args:
        df: Input dataframe with sales data
        item_id_columns: Columns that identify unique time series
        target_column: Column containing the target values
        timestamp_column: Column containing timestamps

    Returns:
        TimeSeriesDataFrame ready for AutoGluon
    """
    df = df.copy()

    # Create unique item_id from store and product family
    df["item_id"] = df[item_id_columns].astype(str).agg("-".join, axis=1)

    # Select and rename columns for AutoGluon format
    ts_df = df[["item_id", timestamp_column, target_column]].copy()
    ts_df = ts_df.rename(columns={timestamp_column: "timestamp", target_column: "target"})

    return TimeSeriesDataFrame.from_data_frame(
        ts_df,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def train_forecaster(
    train_data: TimeSeriesDataFrame,
    prediction_length: int = 16,
    model_path: str | Path | None = None,
    time_limit: int = 600,
    presets: str = "medium_quality",
) -> TimeSeriesPredictor:
    """Train an AutoGluon TimeSeriesPredictor.

    Args:
        train_data: Training data in TimeSeriesDataFrame format
        prediction_length: Number of time steps to forecast
        model_path: Path to save the trained model
        time_limit: Maximum training time in seconds
        presets: AutoGluon presets (fast_training, medium_quality, best_quality)

    Returns:
        Trained TimeSeriesPredictor
    """
    if model_path is None:
        model_path = MODELS_DIR / "sales_predictor"

    predictor = TimeSeriesPredictor(
        prediction_length=prediction_length,
        path=str(model_path),
        target="target",
        eval_metric="MASE",
    )

    predictor.fit(
        train_data,
        time_limit=time_limit,
        presets=presets,
    )

    return predictor


def load_forecaster(model_path: str | Path | None = None) -> TimeSeriesPredictor:
    """Load a previously trained forecaster.

    Args:
        model_path: Path to the saved model

    Returns:
        Loaded TimeSeriesPredictor
    """
    if model_path is None:
        model_path = MODELS_DIR / "sales_predictor"

    return TimeSeriesPredictor.load(str(model_path))


def make_predictions(
    predictor: TimeSeriesPredictor,
    data: TimeSeriesDataFrame,
) -> pd.DataFrame:
    """Generate forecasts using the trained predictor.

    Args:
        predictor: Trained TimeSeriesPredictor
        data: Data to generate predictions from

    Returns:
        DataFrame with predictions
    """
    predictions = predictor.predict(data)
    return predictions
