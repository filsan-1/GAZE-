"""
Data normalization module for gaze tracking data.

Normalizes gaze coordinates, head pose, and other features across datasets.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class GazeDataNormalizer:
    """
    Normalizes gaze data across multiple datasets.

    Handles:
    - Screen coordinate normalization (different resolutions)
    - Feature standardization (mean=0, std=1)
    - Head pose angle normalization
    - Outlier removal
    """

    def __init__(self, method: str = "standard"):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ("standard", "minmax")
        """
        self.method = method
        self.scalers = {}
        self.fit_params = {}

    def normalize_screen_coordinates(
        self,
        df: pd.DataFrame,
        gaze_x_col: str = "gaze_x",
        gaze_y_col: str = "gaze_y",
        screen_width: int = 1280,
        screen_height: int = 720,
    ) -> pd.DataFrame:
        """
        Normalize screen coordinates to [0, 1] range.

        Args:
            df: Input DataFrame
            gaze_x_col: Column name for gaze X coordinate
            gaze_y_col: Column name for gaze Y coordinate
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels

        Returns:
            DataFrame with normalized gaze coordinates
        """
        df = df.copy()

        if gaze_x_col in df.columns:
            df[gaze_x_col] = np.clip(df[gaze_x_col], 0, screen_width) / screen_width

        if gaze_y_col in df.columns:
            df[gaze_y_col] = np.clip(df[gaze_y_col], 0, screen_height) / screen_height

        logger.info(f"Normalized screen coordinates to [0, 1]")
        return df

    def normalize_head_pose(
        self,
        df: pd.DataFrame,
        pose_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize head pose angles to degrees.

        Args:
            df: Input DataFrame
            pose_cols: Column names for head pose (pitch, yaw, roll)

        Returns:
            DataFrame with normalized head pose
        """
        df = df.copy()

        if pose_cols is None:
            pose_cols = ["head_pose_x", "head_pose_y", "head_pose_z"]

        for col in pose_cols:
            if col in df.columns:
                # Clip to reasonable range (-90 to 90 degrees)
                df[col] = np.clip(df[col], -90, 90)

        logger.info("Normalized head pose angles")
        return df

    def standardize_features(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        fit: bool = False,
        scaler_name: str = "default",
    ) -> pd.DataFrame:
        """
        Standardize numeric features (zero mean, unit variance).

        Args:
            df: Input DataFrame
            feature_cols: Columns to standardize
            fit: Whether to fit scaler (True for training data)
            scaler_name: Name for this scaler instance

        Returns:
            DataFrame with standardized features
        """
        df = df.copy()

        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if fit:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers[scaler_name] = scaler
            logger.info(f"Fitted and standardized {len(feature_cols)} features")
        else:
            if scaler_name not in self.scalers:
                logger.warning(f"Scaler '{scaler_name}' not fitted, fitting now")
                scaler = StandardScaler()
                df[feature_cols] = scaler.fit_transform(df[feature_cols])
                self.scalers[scaler_name] = scaler
            else:
                df[feature_cols] = self.scalers[scaler_name].transform(df[feature_cols])
                logger.info(f"Applied existing scaler '{scaler_name}'")

        return df

    def remove_outliers_iqr(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        k: float = 1.5,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Remove outliers using Interquartile Range (IQR) method.

        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            k: IQR multiplier for outlier threshold

        Returns:
            Tuple of (cleaned DataFrame, number of removed samples)
        """
        df = df.copy()

        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        original_len = len(df)

        for col in columns:
            if col not in df.columns:
                continue

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR

            mask = (df[col] >= lower_bound) & (df[col] <= upper_bound)
            df = df[mask]

        removed_count = original_len - len(df)
        logger.info(f"Removed {removed_count} outlier samples using IQR method")

        return df, removed_count

    def impute_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "forward_fill",
    ) -> pd.DataFrame:
        """
        Handle missing values in dataset.

        Args:
            df: Input DataFrame
            method: Imputation method ("forward_fill", "mean", "drop")

        Returns:
            DataFrame with imputed values
        """
        df = df.copy()
        missing_count = df.isnull().sum().sum()

        if missing_count == 0:
            logger.info("No missing values found")
            return df

        if method == "forward_fill":
            df = df.fillna(method="ffill").fillna(method="bfill")
        elif method == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "drop":
            df = df.dropna()

        logger.info(f"Imputed {missing_count} missing values using '{method}' method")
        return df

    def normalize_full_pipeline(
        self,
        df: pd.DataFrame,
        screen_width: int = 1280,
        screen_height: int = 720,
        remove_outliers: bool = True,
        standardize: bool = True,
    ) -> pd.DataFrame:
        """
        Apply full normalization pipeline.

        Args:
            df: Input DataFrame
            screen_width: Screen width for coordinate normalization
            screen_height: Screen height for coordinate normalization
            remove_outliers: Whether to remove outliers
            standardize: Whether to standardize features

        Returns:
            Fully normalized DataFrame
        """
        logger.info("Starting normalization pipeline")

        # Step 1: Impute missing values
        df = self.impute_missing_values(df)

        # Step 2: Normalize screen coordinates
        df = self.normalize_screen_coordinates(df, screen_width, screen_height)

        # Step 3: Normalize head pose
        df = self.normalize_head_pose(df)

        # Step 4: Remove outliers
        if remove_outliers:
            df, _ = self.remove_outliers_iqr(df)

        # Step 5: Standardize features
        if standardize:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Don't standardize label columns
            exclude_cols = {"subject_id", "group", "diagnosis"}
            feature_cols = [c for c in numeric_cols if c not in exclude_cols]
            df = self.standardize_features(df, feature_cols, fit=True)

        logger.info("Normalization pipeline completed")
        return df


if __name__ == "__main__":
    # Example usage
    normalizer = GazeDataNormalizer()

    # Create sample data
    sample_data = {
        "gaze_x": [100, 640, 1280, -100, 2000],
        "gaze_y": [50, 360, 720, 100, 1000],
        "head_pose_x": [10, -5, 20, 100, -150],
        "eye_aspect_ratio": [0.5, 0.45, 0.55, 0.4, 0.6],
    }
    df = pd.DataFrame(sample_data)

    # Apply normalization
    normalized_df = normalizer.normalize_full_pipeline(df)
    print("Normalized data:")
    print(normalized_df)
