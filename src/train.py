"""
Training pipeline for GAZE Research Platform.

Handles dataset loading, merging, training, and evaluation.
"""

import logging
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import RandomForestGazeModel, NeuralGazeClassifier
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR,
    TRAIN_TEST_SPLIT, VALIDATION_SPLIT,
    HANDCRAFTED_FEATURES, CNN_EMBEDDING_DIM,
    CLASS_NAMES,
)

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages loading, preprocessing, and merging of multiple datasets.
    """

    def __init__(self):
        """Initialize dataset manager."""
        self.datasets = {}
        self.merged_data = None

    def load_csv_dataset(
        self,
        filepath: Path,
        dataset_name: str,
        label_column: str = "label",
    ) -> pd.DataFrame:
        """
        Load a CSV dataset.
        
        Args:
            filepath: Path to CSV file.
            dataset_name: Name of dataset.
            label_column: Column name with labels.
            
        Returns:
            Loaded DataFrame.
        """
        df = pd.read_csv(filepath)
        self.datasets[dataset_name] = df
        logger.info(f"Loaded {dataset_name}: {len(df)} samples")
        return df

    def create_synthetic_dataset(
        self,
        num_td: int = 100,
        num_asd: int = 100,
        num_features: int = 17,
    ) -> pd.DataFrame:
        """
        Create synthetic dataset for testing.
        
        Args:
            num_td: Number of TD samples.
            num_asd: Number of ASD samples.
            num_features: Number of features.
            
        Returns:
            Synthetic DataFrame.
        """
        # TD samples (label 0)
        td_data = np.random.randn(num_td, num_features)
        
        # ASD samples (label 1) - with slightly different distribution
        asd_data = np.random.randn(num_asd, num_features) + 0.5
        
        X = np.vstack([td_data, asd_data])
        y = np.hstack([np.zeros(num_td), np.ones(num_asd)])
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(num_features)])
        df["label"] = y
        
        logger.info(f"Created synthetic dataset: {len(df)} samples")
        return df

    def merge_datasets(self, normalize: bool = True) -> pd.DataFrame:
        """
        Merge multiple datasets.
        
        Args:
            normalize: Whether to normalize features across datasets.
            
        Returns:
            Merged DataFrame.
        """
        if not self.datasets:
            raise ValueError("No datasets loaded")
        
        dfs = list(self.datasets.values())
        merged = pd.concat(dfs, ignore_index=True)
        
        # Ensure label column exists
        if "label" not in merged.columns:
            raise ValueError("Datasets must have 'label' column")
        
        self.merged_data = merged
        logger.info(f"Merged dataset: {len(merged)} samples, {merged.shape[1]} features")
        
        return merged

    def split_train_val_test(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        val_size: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            data: Input data.
            test_size: Test set proportion.
            val_size: Validation set proportion (of remaining after test split).
            random_state: Random seed.
            
        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        # First split: train+val vs test
        train_val, test = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=data["label"],
        )
        
        # Second split: train vs val
        train, val = train_test_split(
            train_val,
            test_size=val_size / (1 - test_size),
            random_state=random_state,
            stratify=train_val["label"],
        )
        
        logger.info(f"Data split: train={len(train)}, val={len(val)}, test={len(test)}")
        
        return train, val, test


class ModelTrainer:
    """
    Training and evaluation pipeline for gaze models.
    """

    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize trainer.
        
        Args:
            model_type: "random_forest" or "neural_network".
        """
        self.model_type = model_type
        self.model = None
        self.training_history = {}
        self.best_metrics = None

    def train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Train Random Forest model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            
        Returns:
            Dictionary with training results.
        """
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestGazeModel()
        
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val)
        
        self.model.train(X_train, y_train, validation_data=val_data)
        
        # Evaluate
        results = {
            "train_accuracy": self.model.model.score(
                self.model.scaler.transform(X_train), y_train
            ),
        }
        
        if X_val is not None:
            results["val_accuracy"] = self.model.model.score(
                self.model.scaler.transform(X_val), y_val
            )
        
        self.best_metrics = results
        logger.info(f"Results: {results}")
        
        return results

    def train_neural_network(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        patience: int = 10,
    ) -> Dict:
        """
        Train neural network model.
        
        Args:
            X_train: Training features.
            y_train: Training labels.
            X_val: Validation features (optional).
            y_val: Validation labels (optional).
            epochs: Number of training epochs.
            batch_size: Batch size.
            patience: Early stopping patience.
            
        Returns:
            Dictionary with training results.
        """
        logger.info("Training Neural Network model...")
        
        self.model = NeuralGazeClassifier(input_dim=X_train.shape[1])
        
        val_data = None
        if X_val is not None and y_val is not None:
            val_data = (X_val, y_val)
        
        self.model.train(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=val_data,
            patience=patience,
        )
        
        self.best_metrics = {"model": "neural_network"}
        logger.info("Training completed")
        
        return self.best_metrics

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            X_test: Test features.
            y_test: Test labels.
            
        Returns:
            Dictionary with evaluation metrics.
        """
        # Predictions
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        
        logger.info(f"Test metrics: {metrics}")
        
        return metrics

    def save_model(self, filepath: Path):
        """Save trained model."""
        if self.model is None:
            raise RuntimeError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: Path):
        """Load trained model."""
        if self.model_type == "random_forest":
            self.model = RandomForestGazeModel()
        else:
            raise NotImplementedError
        
        self.model.load(filepath)
        logger.info(f"Model loaded from {filepath}")


class ExperimentLogger:
    """
    Log and visualize experiment results.
    """

    def __init__(self):
        """Initialize logger."""
        self.results = []

    def log_experiment(
        self,
        experiment_name: str,
        model_type: str,
        train_metrics: Dict,
        eval_metrics: Dict,
        dataset_info: Dict,
    ):
        """
        Log experiment results.
        
        Args:
            experiment_name: Name of experiment.
            model_type: Type of model.
            train_metrics: Training metrics.
            eval_metrics: Evaluation metrics.
            dataset_info: Dataset information.
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "experiment_name": experiment_name,
            "model_type": model_type,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "dataset_info": dataset_info,
        }
        
        self.results.append(result)
        logger.info(f"Experiment logged: {experiment_name}")

    def save_results(self, filepath: Path):
        """Save results to CSV."""
        results_df = pd.DataFrame([
            {
                "timestamp": r["timestamp"],
                "experiment": r["experiment_name"],
                "model": r["model_type"],
                **r["eval_metrics"],
            }
            for r in self.results
        ])
        
        results_df.to_csv(filepath, index=False)
        logger.info(f"Results saved to {filepath}")

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None,
    ):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches="tight")
            logger.info(f"Saved confusion matrix to {save_path}")
        
        plt.show()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("GAZE Training Pipeline Example")
    print("=" * 50)

    # Load/create datasets
    dm = DatasetManager()
    synthetic_data = dm.create_synthetic_dataset(num_td=150, num_asd=150, num_features=17)

    # Split data
    train, val, test = dm.split_train_val_test(synthetic_data)

    X_train = train.drop("label", axis=1).values
    y_train = train["label"].values
    X_val = val.drop("label", axis=1).values
    y_val = val["label"].values
    X_test = test.drop("label", axis=1).values
    y_test = test["label"].values

    # Train Random Forest
    print("\n1. Training Random Forest...")
    trainer_rf = ModelTrainer(model_type="random_forest")
    trainer_rf.train_random_forest(X_train, y_train, X_val, y_val)
    metrics_rf = trainer_rf.evaluate(X_test, y_test)
    print(f"   RF Test Accuracy: {metrics_rf['accuracy']:.3f}")

    # Train Neural Network
    print("\n2. Training Neural Network...")
    trainer_nn = ModelTrainer(model_type="neural_network")
    trainer_nn.train_neural_network(X_train, y_train, X_val, y_val, epochs=50)
    metrics_nn = trainer_nn.evaluate(X_test, y_test)
    print(f"   NN Test Accuracy: {metrics_nn['accuracy']:.3f}")

    # Log results
    print("\n3. Logging Results...")
    logger_exp = ExperimentLogger()
    logger_exp.log_experiment(
        "baseline_comparison",
        "random_forest",
        trainer_rf.best_metrics,
        metrics_rf,
        {"num_train": len(X_train), "num_test": len(X_test)},
    )
    logger_exp.save_results(LOGS_DIR / "experiment_results.csv")
