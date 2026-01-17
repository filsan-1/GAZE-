"""
Dataset loader for integrating multiple gaze datasets.

Supports MIT GazeCapture, Kaggle ASD datasets, and custom CSV formats.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, DATASET_CONFIGS, 
    PROJECT_ROOT
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetLoader:
    """
    Loads and manages multiple gaze-tracking datasets.
    
    Handles:
    - MIT GazeCapture dataset ingestion
    - Kaggle ASD gaze dataset loading
    - ASD vs TD comparison datasets
    - CSV-based custom data
    - Dataset merging and validation
    """

    def __init__(self):
        """Initialize the dataset loader."""
        self.datasets = {}
        self.metadata = {}
        self.loaded_datasets = []

    def load_csv_dataset(
        self,
        file_path: str,
        dataset_name: str,
        label_column: Optional[str] = None,
        required_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a CSV gaze dataset with validation.

        Args:
            file_path: Path to CSV file
            dataset_name: Name identifier for the dataset
            label_column: Column name containing labels (ASD/TD)
            required_columns: List of required columns for validation

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        logger.info(f"Loading CSV dataset: {dataset_name} from {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} samples from {dataset_name}")

        # Validate required columns
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in {dataset_name}: {missing_cols}"
                )

        # Add metadata
        self.metadata[dataset_name] = {
            "source": "csv",
            "file_path": str(file_path),
            "num_samples": len(df),
            "columns": list(df.columns),
            "label_column": label_column,
            "data_types": df.dtypes.to_dict(),
        }

        self.datasets[dataset_name] = df
        self.loaded_datasets.append(dataset_name)

        return df

    def create_synthetic_mit_gazecapture(
        self,
        num_samples: int = 1000,
        num_subjects: int = 100,
    ) -> pd.DataFrame:
        """
        Create synthetic MIT GazeCapture-like dataset for demonstration.

        Args:
            num_samples: Total number of samples
            num_subjects: Number of unique subjects

        Returns:
            Synthetic dataset DataFrame
        """
        logger.info(f"Generating synthetic MIT GazeCapture data: {num_samples} samples")

        np.random.seed(42)
        subjects = np.random.choice(num_subjects, num_samples)
        
        data = {
            "subject_id": subjects,
            "gaze_x": np.random.normal(640, 200, num_samples),  # Screen center ~640
            "gaze_y": np.random.normal(360, 150, num_samples),  # Screen center ~360
            "head_pose_x": np.random.normal(0, 15, num_samples),  # degrees
            "head_pose_y": np.random.normal(0, 15, num_samples),
            "head_pose_z": np.random.normal(0, 10, num_samples),
            "screen_width": np.full(num_samples, 1280),
            "screen_height": np.full(num_samples, 720),
            "timestamp": np.arange(num_samples) / 30.0,  # 30 Hz
        }

        df = pd.DataFrame(data)
        
        self.datasets["mit_gazecapture_synthetic"] = df
        self.metadata["mit_gazecapture_synthetic"] = {
            "source": "synthetic",
            "num_samples": num_samples,
            "num_subjects": num_subjects,
            "description": "Synthetic MIT GazeCapture-like dataset",
        }
        self.loaded_datasets.append("mit_gazecapture_synthetic")

        logger.info(f"Created synthetic MIT GazeCapture dataset")
        return df

    def create_synthetic_asd_comparison(
        self,
        num_asd_samples: int = 500,
        num_td_samples: int = 500,
    ) -> pd.DataFrame:
        """
        Create synthetic ASD vs TD comparison dataset.

        Simulates gaze pattern differences between ASD and TD groups.

        Args:
            num_asd_samples: Number of ASD group samples
            num_td_samples: Number of TD (typically developing) group samples

        Returns:
            Combined synthetic dataset with labels
        """
        logger.info(
            f"Generating synthetic ASD comparison data: "
            f"ASD={num_asd_samples}, TD={num_td_samples}"
        )

        np.random.seed(42)
        total_samples = num_asd_samples + num_td_samples

        # ASD-associated patterns
        asd_data = {
            "group": ["ASD"] * num_asd_samples,
            "fixation_duration_mean": np.random.normal(0.35, 0.12, num_asd_samples),  # Slightly longer
            "saccade_count_per_min": np.random.normal(25, 8, num_asd_samples),  # Fewer saccades
            "gaze_entropy": np.random.normal(2.1, 0.8, num_asd_samples),  # Lower entropy
            "roi_attention_eyes": np.random.normal(0.25, 0.15, num_asd_samples),  # Reduced eye gaze
            "roi_attention_mouth": np.random.normal(0.35, 0.15, num_asd_samples),  # More mouth gaze
            "blink_rate": np.random.normal(18, 6, num_asd_samples),  # Blinks per minute
            "eye_aspect_ratio_mean": np.random.normal(0.42, 0.08, num_asd_samples),
            "gaze_velocity_mean": np.random.normal(150, 40, num_asd_samples),  # deg/sec
            "pupil_diameter_variability": np.random.normal(0.15, 0.06, num_asd_samples),
        }

        # TD-associated patterns (baseline)
        td_data = {
            "group": ["TD"] * num_td_samples,
            "fixation_duration_mean": np.random.normal(0.28, 0.10, num_td_samples),
            "saccade_count_per_min": np.random.normal(35, 10, num_td_samples),
            "gaze_entropy": np.random.normal(2.8, 0.7, num_td_samples),
            "roi_attention_eyes": np.random.normal(0.42, 0.15, num_td_samples),
            "roi_attention_mouth": np.random.normal(0.18, 0.10, num_td_samples),
            "blink_rate": np.random.normal(20, 6, num_td_samples),
            "eye_aspect_ratio_mean": np.random.normal(0.45, 0.08, num_td_samples),
            "gaze_velocity_mean": np.random.normal(180, 45, num_td_samples),
            "pupil_diameter_variability": np.random.normal(0.18, 0.07, num_td_samples),
        }

        asd_df = pd.DataFrame(asd_data)
        td_df = pd.DataFrame(td_data)

        df = pd.concat([asd_df, td_df], ignore_index=True)
        df["subject_id"] = np.arange(len(df))
        df["timestamp"] = np.arange(len(df)) / 30.0

        self.datasets["asd_comparison_synthetic"] = df
        self.metadata["asd_comparison_synthetic"] = {
            "source": "synthetic",
            "num_asd": num_asd_samples,
            "num_td": num_td_samples,
            "description": "Synthetic ASD vs TD comparison dataset",
        }
        self.loaded_datasets.append("asd_comparison_synthetic")

        logger.info(f"Created synthetic ASD comparison dataset")
        return df

    def merge_datasets(
        self,
        dataset_names: List[str],
        output_name: str = "merged_dataset",
    ) -> pd.DataFrame:
        """
        Merge multiple loaded datasets into a single DataFrame.

        Args:
            dataset_names: List of dataset names to merge
            output_name: Name for the merged dataset

        Returns:
            Merged DataFrame

        Raises:
            ValueError: If dataset names don't exist
        """
        missing = set(dataset_names) - set(self.datasets.keys())
        if missing:
            raise ValueError(f"Datasets not found: {missing}")

        logger.info(f"Merging datasets: {dataset_names}")

        dfs_to_merge = [self.datasets[name] for name in dataset_names]
        merged_df = pd.concat(dfs_to_merge, ignore_index=True, axis=0)

        self.datasets[output_name] = merged_df
        self.metadata[output_name] = {
            "source": "merged",
            "merged_from": dataset_names,
            "num_samples": len(merged_df),
        }

        logger.info(f"Merged dataset created: {output_name} with {len(merged_df)} samples")
        return merged_df

    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """
        Retrieve a loaded dataset by name.

        Args:
            dataset_name: Name of the dataset

        Returns:
            Dataset DataFrame

        Raises:
            KeyError: If dataset doesn't exist
        """
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not loaded")
        return self.datasets[dataset_name]

    def list_datasets(self) -> List[str]:
        """Return list of loaded datasets."""
        return self.loaded_datasets.copy()

    def get_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get metadata for a specific dataset."""
        if dataset_name not in self.metadata:
            return {}
        return self.metadata[dataset_name].copy()

    def save_dataset(self, dataset_name: str, output_dir: Optional[Path] = None) -> Path:
        """
        Save a dataset to CSV.

        Args:
            dataset_name: Name of dataset to save
            output_dir: Output directory (defaults to PROCESSED_DATA_DIR)

        Returns:
            Path to saved file
        """
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found")

        if output_dir is None:
            output_dir = PROCESSED_DATA_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{dataset_name}.csv"
        self.datasets[dataset_name].to_csv(output_file, index=False)

        logger.info(f"Saved dataset '{dataset_name}' to {output_file}")
        return output_file

    def get_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get statistical summary of a dataset.

        Args:
            dataset_name: Name of dataset

        Returns:
            Dictionary of statistics
        """
        if dataset_name not in self.datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found")

        df = self.datasets[dataset_name]

        return {
            "num_samples": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_stats": df.describe().to_dict(),
        }


if __name__ == "__main__":
    # Example usage
    loader = DatasetLoader()

    # Create synthetic datasets
    loader.create_synthetic_mit_gazecapture(num_samples=1000)
    loader.create_synthetic_asd_comparison(num_asd_samples=500, num_td_samples=500)

    # Print loaded datasets
    print("Loaded datasets:", loader.list_datasets())

    # Get statistics
    for dataset_name in loader.list_datasets():
        stats = loader.get_statistics(dataset_name)
        print(f"\n{dataset_name}:")
        print(f"  Samples: {stats['num_samples']}")
        print(f"  Columns: {stats['num_columns']}")
