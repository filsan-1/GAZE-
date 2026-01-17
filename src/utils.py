"""
Utility functions for GAZE Research Platform.

Helper functions for data handling, visualization, and I/O operations.
"""

import logging
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
import json
import pickle

logger = logging.getLogger(__name__)


class DataUtils:
    """Utilities for data handling and I/O."""

    @staticmethod
    def save_features_csv(
        features: pd.DataFrame,
        filepath: Path,
        index: bool = False,
    ) -> None:
        """
        Save extracted features to CSV.
        
        Args:
            features: Features DataFrame.
            filepath: Output file path.
            index: Whether to save index.
        """
        features.to_csv(filepath, index=index)
        logger.info(f"Features saved to {filepath}")

    @staticmethod
    def load_features_csv(filepath: Path) -> pd.DataFrame:
        """Load features from CSV."""
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} feature rows from {filepath}")
        return df

    @staticmethod
    def save_session_log(
        session_data: Dict,
        filepath: Path,
    ) -> None:
        """Save session log as JSON."""
        # Convert numpy arrays to lists for JSON serialization
        session_serializable = {}
        for key, val in session_data.items():
            if isinstance(val, np.ndarray):
                session_serializable[key] = val.tolist()
            elif isinstance(val, list):
                # Check if list contains numpy arrays
                session_serializable[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in val
                ]
            else:
                session_serializable[key] = val
        
        with open(filepath, "w") as f:
            json.dump(session_serializable, f, indent=2)
        
        logger.info(f"Session log saved to {filepath}")

    @staticmethod
    def load_session_log(filepath: Path) -> Dict:
        """Load session log from JSON."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        logger.info(f"Session log loaded from {filepath}")
        return data


class ModelUtils:
    """Utilities for model management."""

    @staticmethod
    def list_saved_models(model_dir: Path) -> List[Path]:
        """List all saved models in directory."""
        return list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.pt"))

    @staticmethod
    def get_model_info(model_path: Path) -> Dict:
        """Get metadata about saved model."""
        return {
            "name": model_path.name,
            "path": str(model_path),
            "size_mb": model_path.stat().st_size / (1024 ** 2),
            "created": model_path.stat().st_ctime,
        }


class VisualizationUtils:
    """Utilities for visualization and reporting."""

    @staticmethod
    def create_summary_report(
        metrics: Dict,
        session_info: Dict,
    ) -> str:
        """
        Create text summary report.
        
        Args:
            metrics: Evaluation metrics.
            session_info: Session information.
            
        Returns:
            Formatted report string.
        """
        report = f"""
GAZE Session Report
{'='*50}

Session Information:
  - Duration: {session_info.get('duration', 'N/A')}
  - Frames captured: {session_info.get('num_frames', 'N/A')}
  - Sampling rate: {session_info.get('sampling_rate', 'N/A')} Hz

Gaze Metrics:
  - Fixation duration: {metrics.get('fixation_duration_mean', 0):.3f}s
  - Saccade count: {metrics.get('saccade_count', 0)}
  - Gaze entropy: {metrics.get('gaze_entropy', 0):.3f}
  - Mean velocity: {metrics.get('gaze_velocity_mean', 0):.1f} px/s

Classification:
  - ASD-like probability: {metrics.get('asd_probability', 0):.1f}%
  - Confidence: {metrics.get('confidence', 0):.1f}%
  - Risk tier: {metrics.get('risk_tier', 'Unknown')}

{'='*50}
Non-diagnostic research tool. Always consult qualified professionals.
        """
        return report

    @staticmethod
    def save_report(
        report: str,
        filepath: Path,
    ) -> None:
        """Save report to text file."""
        with open(filepath, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {filepath}")


class ConfigUtils:
    """Utilities for configuration management."""

    @staticmethod
    def print_config_summary(config_dict: Dict) -> None:
        """Print configuration summary."""
        print("\nConfiguration Summary")
        print("=" * 50)
        for key, value in config_dict.items():
            if isinstance(value, (list, dict)):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {value}")
        print("=" * 50 + "\n")


class ExperimentUtils:
    """Utilities for experiment tracking and reproducibility."""

    @staticmethod
    def log_hyperparameters(
        hyperparams: Dict,
        filepath: Path,
    ) -> None:
        """Log hyperparameters for reproducibility."""
        with open(filepath, "w") as f:
            json.dump(hyperparams, f, indent=2)
        logger.info(f"Hyperparameters logged to {filepath}")

    @staticmethod
    def create_experiment_metadata(
        experiment_name: str,
        description: str,
        model_type: str,
        dataset_info: Dict,
    ) -> Dict:
        """Create experiment metadata."""
        import datetime
        
        metadata = {
            "experiment_name": experiment_name,
            "description": description,
            "model_type": model_type,
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": dataset_info,
            "code_version": "1.0.0",
        }
        
        return metadata


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("GAZE Utilities Module Example")
    print("=" * 50)

    # Example data utils
    print("\n1. Data Utilities")
    data_utils = DataUtils()
    print("   ✓ DataUtils initialized")

    # Example model utils
    print("\n2. Model Utilities")
    model_utils = ModelUtils()
    print("   ✓ ModelUtils initialized")

    # Example visualization utils
    print("\n3. Visualization Utilities")
    vis_utils = VisualizationUtils()
    
    metrics = {
        "fixation_duration_mean": 0.25,
        "saccade_count": 42,
        "gaze_entropy": 2.5,
        "asd_probability": 68.5,
    }
    session_info = {
        "duration": "60.0s",
        "num_frames": 1800,
        "sampling_rate": 30,
    }
    
    report = vis_utils.create_summary_report(metrics, session_info)
    print(report)

    # Example config utils
    print("\n4. Configuration Utilities")
    config_utils = ConfigUtils()
    config = {"model_type": "random_forest", "num_features": 17}
    config_utils.print_config_summary(config)
