"""
Random Forest model for gaze pattern classification.

Trains and evaluates Random Forest classifier for ASD-associated gaze patterns.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


class RandomForestGazeModel:
    """
    Random Forest model for ASD-associated gaze pattern detection.

    Trains on labeled gaze feature data and outputs ASD likelihood scores.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 15, random_state: int = 42):
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
            random_state: Random seed
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
        )

        self.feature_names = None
        self.is_trained = False
        self.class_mapping = {0: "TD", 1: "ASD"}  # TD=Typically Developing, ASD

        logger.info("Initialized RandomForestGazeModel")

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, float]:
        """
        Train the model on gaze feature data.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Labels (0=TD, 1=ASD)
            feature_names: Names of features
            test_size: Fraction of data for testing
            random_state: Random seed

        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training Random Forest on {len(X)} samples with {X.shape[1]} features")

        # Store feature names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.feature_names = feature_names

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0

        metrics = {
            "train_accuracy": float(train_score),
            "test_accuracy": float(test_score),
            "auc": float(auc),
            "n_test_samples": int(len(X_test)),
        }

        logger.info(f"Training metrics: {metrics}")

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probabilities for each class
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        return self.model.predict_proba(X)

    def predict_asd_likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Predict ASD likelihood (0-100 scale).

        Args:
            X: Feature matrix of shape (n_samples, n_features) or (n_features,)

        Returns:
            ASD likelihood scores (0-100)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        # Handle single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Get probability for ASD class (class 1)
        proba = self.model.predict_proba(X)[:, 1]

        # Scale to 0-100
        scores = proba * 100

        return scores

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance scores.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary of {feature_name: importance}
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        importances = self.model.feature_importances_

        # Create dataframe
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        # Return top N
        top_features = importance_df.head(top_n)

        return dict(zip(top_features["feature"], top_features["importance"]))

    def save(self, filepath: str):
        """
        Save model to disk.

        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            logger.warning("Saving untrained model")

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "is_trained": self.is_trained,
            "class_mapping": self.class_mapping,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from disk.

        Args:
            filepath: Path to load model from
        """
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.is_trained = model_data["is_trained"]
        self.class_mapping = model_data.get("class_mapping", {0: "TD", 1: "ASD"})

        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Example usage
    from src.data_processing import DatasetLoader

    # Create synthetic data
    loader = DatasetLoader()
    df = loader.create_synthetic_asd_comparison(num_asd_samples=200, num_td_samples=200)

    # Prepare features and labels
    feature_cols = [
        "fixation_duration_mean", "saccade_count_per_min", "gaze_entropy",
        "roi_attention_eyes", "roi_attention_mouth", "blink_rate",
        "eye_aspect_ratio_mean", "gaze_velocity_mean", "pupil_diameter_variability"
    ]

    X = df[feature_cols].values
    y = (df["group"] == "ASD").astype(int).values

    # Train model
    model = RandomForestGazeModel()
    metrics = model.train(X, y, feature_names=feature_cols)

    print("Training metrics:", metrics)

    # Test prediction
    test_sample = X[0:1]
    asd_score = model.predict_asd_likelihood(test_sample)
    print(f"ASD likelihood: {asd_score[0]:.1f}%")

    # Feature importance
    importance = model.get_feature_importance(top_n=5)
    print("\nTop features:")
    for name, imp in importance.items():
        print(f"  {name}: {imp:.3f}")
