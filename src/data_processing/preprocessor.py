"""
Data preprocessing module for gaze tracking data.

Handles temporal alignment, resampling, and feature aggregation.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class GazeDataPreprocessor:
    """
    Preprocesses gaze tracking data for analysis.

    Handles:
    - Temporal alignment and synchronization
    - Resampling to fixed frequency
    - Feature aggregation over windows
    - Data validation
    """

    def __init__(self, sampling_rate: int = 30):
        """
        Initialize preprocessor.

        Args:
            sampling_rate: Target sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.sampling_period = 1.0 / sampling_rate

    def resample_data(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
        target_rate: int = 30,
    ) -> pd.DataFrame:
        """
        Resample gaze data to fixed sampling rate.

        Args:
            df: Input DataFrame with timestamp column
            time_col: Name of timestamp column
            target_rate: Target sampling rate in Hz

        Returns:
            Resampled DataFrame
        """
        if time_col not in df.columns:
            logger.warning(f"Timestamp column '{time_col}' not found, creating one")
            df = df.copy()
            df[time_col] = np.arange(len(df)) / self.sampling_rate
        else:
            df = df.copy()

        # Interpolate to fixed rate
        time_min = df[time_col].min()
        time_max = df[time_col].max()
        new_time = np.arange(time_min, time_max, 1.0 / target_rate)

        # Interpolate numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove(time_col) if time_col in numeric_cols else None

        resampled_data = {time_col: new_time}

        for col in numeric_cols:
            resampled_data[col] = np.interp(new_time, df[time_col], df[col])

        resampled_df = pd.DataFrame(resampled_data)

        logger.info(
            f"Resampled data from {len(df)} to {len(resampled_df)} samples at {target_rate}Hz"
        )

        return resampled_df

    def aggregate_features_windowed(
        self,
        df: pd.DataFrame,
        window_size: float = 1.0,
        step_size: Optional[float] = None,
        time_col: str = "timestamp",
        aggregation_funcs: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Aggregate features over sliding time windows.

        Args:
            df: Input DataFrame
            window_size: Window size in seconds
            step_size: Step size in seconds (defaults to window_size)
            time_col: Name of timestamp column
            aggregation_funcs: Dict of {column: [aggregation_functions]}

        Returns:
            Aggregated DataFrame with windowed statistics
        """
        if step_size is None:
            step_size = window_size

        if aggregation_funcs is None:
            aggregation_funcs = {
                col: ["mean", "std", "min", "max"]
                for col in df.select_dtypes(include=[np.number]).columns
                if col != time_col
            }

        logger.info(
            f"Aggregating features with {window_size}s windows, "
            f"{step_size}s step size"
        )

        time_min = df[time_col].min()
        time_max = df[time_col].max()

        windows = []
        window_start = time_min

        while window_start + window_size <= time_max:
            window_end = window_start + window_size

            # Get data in this window
            window_mask = (df[time_col] >= window_start) & (df[time_col] < window_end)
            window_data = df[window_mask]

            if len(window_data) > 0:
                row = {"window_start": window_start, "window_end": window_end}

                # Compute aggregations
                for col, funcs in aggregation_funcs.items():
                    if col in window_data.columns:
                        for func in funcs:
                            agg_key = f"{col}_{func}"
                            if func == "mean":
                                row[agg_key] = window_data[col].mean()
                            elif func == "std":
                                row[agg_key] = window_data[col].std()
                            elif func == "min":
                                row[agg_key] = window_data[col].min()
                            elif func == "max":
                                row[agg_key] = window_data[col].max()

                windows.append(row)

            window_start += step_size

        aggregated_df = pd.DataFrame(windows)
        logger.info(f"Created {len(aggregated_df)} aggregated windows")

        return aggregated_df

    def compute_temporal_derivatives(
        self,
        df: pd.DataFrame,
        columns: List[str],
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Compute temporal derivatives (velocity) for gaze coordinates.

        Args:
            df: Input DataFrame
            columns: Columns to compute derivatives for
            time_col: Name of timestamp column

        Returns:
            DataFrame with added derivative columns
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            # Compute derivative using forward differences
            time_diff = df[time_col].diff()
            value_diff = df[col].diff()

            # Avoid division by zero
            derivative = np.where(time_diff > 0, value_diff / time_diff, 0)

            df[f"{col}_velocity"] = derivative

        logger.info(f"Computed temporal derivatives for {len(columns)} columns")
        return df

    def compute_rolling_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        window: int = 30,
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Compute rolling window statistics.

        Args:
            df: Input DataFrame
            columns: Columns to compute statistics for
            window: Window size in samples
            time_col: Name of timestamp column

        Returns:
            DataFrame with rolling statistics
        """
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            df[f"{col}_rolling_mean"] = df[col].rolling(window=window, min_periods=1).mean()
            df[f"{col}_rolling_std"] = df[col].rolling(window=window, min_periods=1).std()

        logger.info(f"Computed rolling statistics for {len(columns)} columns")
        return df

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None,
        max_missing_rate: float = 0.1,
    ) -> Tuple[bool, Dict[str, str]]:
        """
        Validate data quality.

        Args:
            df: Input DataFrame
            required_cols: List of required columns
            max_missing_rate: Maximum allowed missing data rate

        Returns:
            Tuple of (is_valid, issues_dict)
        """
        issues = {}

        # Check required columns
        if required_cols:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                issues["missing_columns"] = f"Missing columns: {missing_cols}"

        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues["duplicates"] = f"Found {duplicates} duplicate rows"

        # Check missing data rate
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            missing_rate = df[col].isnull().sum() / len(df)
            if missing_rate > max_missing_rate:
                issues[f"missing_{col}"] = (
                    f"Missing rate {missing_rate:.2%} exceeds threshold"
                )

        is_valid = len(issues) == 0
        logger.info(f"Data validation: {'PASSED' if is_valid else 'FAILED'}")

        return is_valid, issues

    def preprocess_pipeline(
        self,
        df: pd.DataFrame,
        resample: bool = True,
        target_rate: int = 30,
        compute_velocities: bool = True,
        velocity_cols: Optional[List[str]] = None,
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Apply full preprocessing pipeline.

        Args:
            df: Input DataFrame
            resample: Whether to resample to fixed rate
            target_rate: Target sampling rate
            compute_velocities: Whether to compute velocity
            velocity_cols: Columns to compute velocity for
            validate: Whether to validate data

        Returns:
            Preprocessed DataFrame
        """
        logger.info("Starting preprocessing pipeline")

        if velocity_cols is None:
            velocity_cols = ["gaze_x", "gaze_y"]

        # Step 1: Validate
        if validate:
            is_valid, issues = self.validate_data_quality(df)
            if issues:
                logger.warning(f"Data validation issues: {issues}")

        # Step 2: Resample
        if resample:
            df = self.resample_data(df, target_rate=target_rate)

        # Step 3: Compute derivatives
        if compute_velocities:
            df = self.compute_temporal_derivatives(df, velocity_cols)

        # Step 4: Rolling statistics
        df = self.compute_rolling_statistics(df, velocity_cols)

        logger.info("Preprocessing pipeline completed")
        return df


if __name__ == "__main__":
    # Example usage
    preprocessor = GazeDataPreprocessor(sampling_rate=30)

    # Create sample data
    n_samples = 300
    time = np.arange(n_samples) / 30.0
    gaze_x = 640 + 200 * np.sin(2 * np.pi * time / 5)  # 5 Hz oscillation
    gaze_y = 360 + 150 * np.cos(2 * np.pi * time / 5)

    sample_df = pd.DataFrame({
        "timestamp": time,
        "gaze_x": gaze_x,
        "gaze_y": gaze_y,
    })

    # Apply preprocessing
    processed_df = preprocessor.preprocess_pipeline(sample_df)
    print("Preprocessed data shape:", processed_df.shape)
    print("Columns:", list(processed_df.columns))
