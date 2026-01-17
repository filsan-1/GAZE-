"""
Model definitions for GAZE Research Platform.

Implements RandomForest baseline and lightweight neural network classifier
for gaze pattern classification.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from config import (
    NN_HIDDEN_DIMS,
    NN_DROPOUT,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    CNN_EMBEDDING_DIM,
)

logger = logging.getLogger(__name__)


class RandomForestGazeModel:
    """
    Random Forest baseline classifier for gaze patterns.
    
    Combines handcrafted features for efficient classification.
    """

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: int = RF_MAX_DEPTH,
        random_state: int = 42,
    ):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in forest.
            max_depth: Maximum tree depth.
            random_state: Random seed.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info("RandomForestGazeModel initialized")

    def train(self, X: np.ndarray, y: np.ndarray, validation_data: Optional[Tuple] = None):
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (n_samples,).
            validation_data: Optional (X_val, y_val) for monitoring.
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info(f"Model trained on {len(X)} samples")
        
        # Evaluate on validation if provided
        if validation_data:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            val_accuracy = self.model.score(X_val_scaled, y_val)
            logger.info(f"Validation accuracy: {val_accuracy:.3f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Predicted labels.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix.
            
        Returns:
            Probability matrix (n_samples, 2).
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            feature_names: Optional list of feature names.
            
        Returns:
            Dictionary of feature importances.
        """
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        return dict(zip(feature_names, importances))

    def save(self, filepath: Path):
        """Save trained model to disk."""
        model_dict = {
            "model": self.model,
            "scaler": self.scaler,
        }
        joblib.dump(model_dict, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path):
        """Load trained model from disk."""
        model_dict = joblib.load(filepath)
        self.model = model_dict["model"]
        self.scaler = model_dict["scaler"]
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class NeuralGazeNetwork(nn.Module):
    """
    Lightweight neural network for gaze pattern classification.
    
    Processes both handcrafted features and CNN embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        dropout: float = NN_DROPOUT,
        num_classes: int = 2,
    ):
        """
        Initialize neural network.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout rate.
            num_classes: Number of output classes.
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = NN_HIDDEN_DIMS
        
        # Input layer
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class NeuralGazeClassifier:
    """
    Wrapper for neural network with training and evaluation utilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        """
        Initialize classifier.
        
        Args:
            input_dim: Input feature dimension.
            hidden_dims: Hidden layer dimensions.
            learning_rate: Learning rate.
            device: "cpu" or "cuda".
        """
        self.device = torch.device(device)
        self.model = NeuralGazeNetwork(input_dim, hidden_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(self.device))
        self.scaler = StandardScaler()
        self.is_trained = False
        logger.info(f"NeuralGazeClassifier initialized on {device}")

    def train_epoch(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> float:
        """
        Train for one epoch.
        
        Args:
            X: Feature matrix.
            y: Target labels.
            batch_size: Batch size.
            
        Returns:
            Average loss.
        """
        self.model.train()
        X_scaled = self.scaler.fit_transform(X)
        
        total_loss = 0.0
        num_batches = 0
        
        indices = np.random.permutation(len(X))
        for i in range(0, len(X), batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = torch.from_numpy(X_scaled[batch_indices]).float().to(self.device)
            y_batch = torch.from_numpy(y[batch_indices]).long().to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_data: Optional[Tuple] = None,
        patience: int = 10,
    ):
        """
        Train the neural network.
        
        Args:
            X: Feature matrix.
            y: Target labels.
            epochs: Number of epochs.
            batch_size: Batch size.
            validation_data: Optional (X_val, y_val).
            patience: Early stopping patience.
        """
        # Fit scaler on training data
        self.scaler.fit(X)
        
        best_val_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(X, y, batch_size)
            
            if validation_data:
                X_val, y_val = validation_data
                val_loss = self.evaluate(X_val, y_val)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
        
        self.is_trained = True
        logger.info("Training completed")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained first")
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probas = F.softmax(logits, dim=1)
        
        return probas.cpu().numpy()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate on validation data.
        
        Args:
            X: Feature matrix.
            y: Target labels.
            
        Returns:
            Average loss.
        """
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.from_numpy(X_scaled).float().to(self.device)
        y_tensor = torch.from_numpy(y).long().to(self.device)
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            loss = self.criterion(logits, y_tensor).item()
        
        return loss

    def save(self, filepath: Path):
        """Save model to disk."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler": self.scaler,
            "input_dim": self.model.input_dim,
            "num_classes": self.model.num_classes,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: Path):
        """Load model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler = checkpoint["scaler"]
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("GAZE Model Module Example")
    print("=" * 50)

    # Generate synthetic data
    X_train = np.random.randn(200, 17)
    y_train = np.random.randint(0, 2, 200)
    X_test = np.random.randn(50, 17)
    y_test = np.random.randint(0, 2, 50)

    # Example 1: Random Forest
    print("\n1. Random Forest Baseline")
    rf_model = RandomForestGazeModel()
    rf_model.train(X_train, y_train, validation_data=(X_test, y_test))
    rf_preds = rf_model.predict(X_test)
    print(f"   Predictions shape: {rf_preds.shape}")
    print(f"   Accuracy: {(rf_preds == y_test).mean():.3f}")

    # Example 2: Neural Network
    print("\n2. Neural Network Classifier")
    nn_model = NeuralGazeClassifier(input_dim=17, device="cpu")
    nn_model.train(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))
    nn_preds = nn_model.predict(X_test)
    print(f"   Predictions shape: {nn_preds.shape}")
    print(f"   Accuracy: {(nn_preds == y_test).mean():.3f}")
