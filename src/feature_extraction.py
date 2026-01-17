"""
Feature extraction module for GAZE Research Platform.

Extracts handcrafted gaze metrics and CNN embeddings from eye crops.
"""

import logging
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from pathlib import Path
from scipy import stats

from config import (
    HANDCRAFTED_FEATURES,
    CNN_MODEL_NAME,
    CNN_EMBEDDING_DIM,
    FREEZE_CNN_BACKBONE,
    EYE_CROP_SIZE,
)

logger = logging.getLogger(__name__)


class HandcraftedGazeFeatures:
    """
    Compute handcrafted gaze metrics from gaze trajectories.
    
    Extracts fixation stability, saccades, entropy, asymmetry, and other metrics.
    """

    def __init__(self, sampling_rate: int = 30):
        """
        Initialize feature extractor.
        
        Args:
            sampling_rate: Camera sampling rate in Hz.
        """
        self.sampling_rate = sampling_rate

    def compute_fixation_metrics(
        self,
        gaze_points: np.ndarray,
        fixation_threshold: float = 50.0,  # pixels
    ) -> Dict[str, float]:
        """
        Compute fixation-related metrics.
        
        Args:
            gaze_points: Array of gaze points (n_samples, 2).
            fixation_threshold: Distance threshold for fixation detection (pixels).
            
        Returns:
            Dictionary of fixation metrics.
        """
        if len(gaze_points) < 2:
            return {}

        # Compute gaze velocity
        velocity = np.linalg.norm(np.diff(gaze_points, axis=0), axis=1)
        
        # Detect fixations (low velocity)
        is_fixation = velocity < fixation_threshold
        fixation_events = np.diff(is_fixation.astype(int))
        fixation_starts = np.where(fixation_events == 1)[0]
        fixation_ends = np.where(fixation_events == -1)[0]

        if len(fixation_starts) == 0:
            return {
                "fixation_duration_mean": 0.0,
                "fixation_duration_std": 0.0,
                "fixation_count": 0,
            }

        # Compute fixation durations
        fixation_durations = []
        for start, end in zip(fixation_starts, fixation_ends):
            if start < end:
                duration = (end - start) / self.sampling_rate
                fixation_durations.append(duration)

        if len(fixation_durations) == 0:
            fixation_durations = [0.0]

        return {
            "fixation_duration_mean": float(np.mean(fixation_durations)),
            "fixation_duration_std": float(np.std(fixation_durations)),
            "fixation_count": int(len(fixation_durations)),
        }

    def compute_saccade_metrics(
        self,
        gaze_points: np.ndarray,
        saccade_threshold: float = 100.0,  # pixels/second
    ) -> Dict[str, float]:
        """
        Compute saccade-related metrics.
        
        Args:
            gaze_points: Array of gaze points (n_samples, 2).
            saccade_threshold: Velocity threshold for saccade detection.
            
        Returns:
            Dictionary of saccade metrics.
        """
        if len(gaze_points) < 2:
            return {}

        # Compute velocity and acceleration
        velocity = np.linalg.norm(np.diff(gaze_points, axis=0), axis=1)
        velocity_per_sec = velocity * self.sampling_rate

        # Detect saccades (high velocity)
        is_saccade = velocity_per_sec > saccade_threshold
        saccade_events = np.diff(is_saccade.astype(int))
        saccade_starts = np.where(saccade_events == 1)[0]
        saccade_ends = np.where(saccade_events == -1)[0]

        saccade_amplitudes = []
        saccade_velocities = []

        for start, end in zip(saccade_starts, saccade_ends):
            if start < end:
                amplitude = np.linalg.norm(gaze_points[end] - gaze_points[start])
                saccade_amplitudes.append(amplitude)
                
                duration = (end - start) / self.sampling_rate
                if duration > 0:
                    saccade_velocities.append(amplitude / duration)

        if len(saccade_velocities) == 0:
            saccade_velocities = [0.0]
        if len(saccade_amplitudes) == 0:
            saccade_amplitudes = [0.0]

        saccade_frequency = len(saccade_starts) * self.sampling_rate / len(gaze_points)

        return {
            "saccade_amplitude_mean": float(np.mean(saccade_amplitudes)),
            "saccade_amplitude_std": float(np.std(saccade_amplitudes)),
            "saccade_velocity_mean": float(np.mean(saccade_velocities)),
            "saccade_velocity_std": float(np.std(saccade_velocities)),
            "saccade_count": int(len(saccade_starts)),
            "saccade_frequency": float(saccade_frequency),
        }

    def compute_gaze_entropy(self, gaze_points: np.ndarray, bins: int = 20) -> float:
        """
        Compute gaze entropy (measure of gaze distribution randomness).
        
        Args:
            gaze_points: Array of gaze points (n_samples, 2).
            bins: Number of histogram bins.
            
        Returns:
            Entropy value.
        """
        if len(gaze_points) < 2:
            return 0.0

        # Create 2D histogram
        hist, _ = np.histogramdd(gaze_points, bins=bins)
        hist = hist.flatten()
        hist = hist[hist > 0]  # Remove zero bins
        hist = hist / hist.sum()  # Normalize

        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy)

    def compute_blink_metrics(self, eye_crops: List[np.ndarray]) -> Dict[str, float]:
        """
        Compute blink-related metrics from eye crops.
        
        Uses image variance as blink indicator (closed eyes have low variance).
        
        Args:
            eye_crops: List of eye crop images.
            
        Returns:
            Dictionary of blink metrics.
        """
        if len(eye_crops) == 0:
            return {"blink_rate": 0.0, "blink_duration_mean": 0.0}

        # Compute variance for each frame
        variances = []
        for crop in eye_crops:
            if crop is None or crop.size == 0:
                variances.append(0.0)
            else:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                variances.append(float(gray.var()))

        variances = np.array(variances)

        # Detect blinks (low variance)
        blink_threshold = np.percentile(variances, 10)
        is_blink = variances < blink_threshold

        blink_events = np.diff(is_blink.astype(int))
        blink_starts = np.where(blink_events == 1)[0]
        blink_ends = np.where(blink_events == -1)[0]

        if len(blink_starts) == 0:
            return {"blink_rate": 0.0, "blink_duration_mean": 0.0}

        blink_durations = []
        for start, end in zip(blink_starts, blink_ends):
            if start < end:
                duration = (end - start) / self.sampling_rate
                blink_durations.append(duration)

        blink_rate = len(blink_starts) * 60 / (len(eye_crops) / self.sampling_rate)

        return {
            "blink_rate": float(blink_rate),
            "blink_duration_mean": float(np.mean(blink_durations)) if blink_durations else 0.0,
        }

    def compute_asymmetry_metrics(
        self,
        left_eye_gaze: np.ndarray,
        right_eye_gaze: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute left/right eye asymmetry metrics.
        
        Args:
            left_eye_gaze: Left eye gaze points (n_samples, 2).
            right_eye_gaze: Right eye gaze points (n_samples, 2).
            
        Returns:
            Dictionary of asymmetry metrics.
        """
        if len(left_eye_gaze) == 0 or len(right_eye_gaze) == 0:
            return {"left_right_asymmetry": 0.0}

        # Compute mean gaze difference
        diff = left_eye_gaze - right_eye_gaze
        asymmetry = np.linalg.norm(diff, axis=1).mean()

        return {"left_right_asymmetry": float(asymmetry)}

    def compute_velocity_metrics(self, gaze_points: np.ndarray) -> Dict[str, float]:
        """
        Compute gaze velocity metrics.
        
        Args:
            gaze_points: Array of gaze points (n_samples, 2).
            
        Returns:
            Dictionary of velocity metrics.
        """
        if len(gaze_points) < 2:
            return {"gaze_velocity_mean": 0.0, "gaze_velocity_std": 0.0}

        velocity = np.linalg.norm(np.diff(gaze_points, axis=0), axis=1)
        velocity_per_sec = velocity * self.sampling_rate

        return {
            "gaze_velocity_mean": float(np.mean(velocity_per_sec)),
            "gaze_velocity_std": float(np.std(velocity_per_sec)),
        }

    def extract_all_handcrafted(
        self,
        gaze_points: np.ndarray,
        left_eye_gaze: Optional[np.ndarray] = None,
        right_eye_gaze: Optional[np.ndarray] = None,
        eye_crops: Optional[List[np.ndarray]] = None,
    ) -> pd.DataFrame:
        """
        Extract all handcrafted features in one call.
        
        Args:
            gaze_points: Combined gaze points (n_samples, 2).
            left_eye_gaze: Left eye gaze points (optional).
            right_eye_gaze: Right eye gaze points (optional).
            eye_crops: Eye crop images (optional, for blink detection).
            
        Returns:
            DataFrame with extracted features.
        """
        features = {}

        # Fixation metrics
        features.update(self.compute_fixation_metrics(gaze_points))

        # Saccade metrics
        features.update(self.compute_saccade_metrics(gaze_points))

        # Gaze entropy
        features["gaze_entropy"] = self.compute_gaze_entropy(gaze_points)

        # Blink metrics
        if eye_crops:
            features.update(self.compute_blink_metrics(eye_crops))

        # Asymmetry metrics
        if left_eye_gaze is not None and right_eye_gaze is not None:
            features.update(self.compute_asymmetry_metrics(left_eye_gaze, right_eye_gaze))

        # Velocity metrics
        features.update(self.compute_velocity_metrics(gaze_points))

        # Smooth pursuit (approximate as correlation with stimulus)
        features["smooth_pursuit_accuracy"] = 0.0  # Placeholder

        return pd.DataFrame([features])


class CNNEmbeddingExtractor:
    """
    Extract CNN embeddings from eye crops using MobileNetV2.
    
    Uses transfer learning with frozen backbone for efficient feature extraction.
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize CNN embedding extractor.
        
        Args:
            device: "cpu" or "cuda".
        """
        self.device = torch.device(device)
        self.model = self._load_mobilenet()
        logger.info(f"CNN model loaded on {device}")

    def _load_mobilenet(self) -> nn.Module:
        """Load pretrained MobileNetV2 and remove classification head."""
        model = models.mobilenet_v2(pretrained=True)
        
        # Remove classification head
        model = nn.Sequential(*list(model.children())[:-1])  # Remove avgpool and classifier
        
        if FREEZE_CNN_BACKBONE:
            for param in model.parameters():
                param.requires_grad = False
        
        model = model.to(self.device)
        model.eval()
        
        return model

    def preprocess_eye_crop(self, eye_crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess eye crop for CNN input.
        
        Args:
            eye_crop: Eye crop image (BGR, 224x224).
            
        Returns:
            Preprocessed tensor (1, 3, 224, 224).
        """
        # Convert BGR to RGB
        rgb_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        
        # Normalize to ImageNet standards
        rgb_crop = rgb_crop.astype(np.float32) / 255.0
        rgb_crop = (rgb_crop - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # Convert to tensor (C, H, W)
        tensor = torch.from_numpy(rgb_crop).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)

    def extract_embedding(self, eye_crop: np.ndarray) -> np.ndarray:
        """
        Extract CNN embedding from single eye crop.
        
        Args:
            eye_crop: Eye crop image.
            
        Returns:
            CNN embedding (1280-dimensional vector).
        """
        if eye_crop is None or eye_crop.size == 0:
            return np.zeros(CNN_EMBEDDING_DIM)

        try:
            tensor = self.preprocess_eye_crop(eye_crop)
            
            with torch.no_grad():
                embedding = self.model(tensor)
                embedding = embedding.view(embedding.size(0), -1)
                embedding = embedding.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")
            return np.zeros(CNN_EMBEDDING_DIM)

    def extract_embeddings_batch(
        self,
        eye_crops: List[np.ndarray],
    ) -> np.ndarray:
        """
        Extract embeddings from multiple eye crops.
        
        Args:
            eye_crops: List of eye crop images.
            
        Returns:
            Array of embeddings (n_samples, 1280).
        """
        embeddings = []
        for crop in eye_crops:
            embedding = self.extract_embedding(crop)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class FeatureExtractor:
    """
    Combined feature extraction: handcrafted metrics + CNN embeddings.
    """

    def __init__(self, use_cnn: bool = True, device: str = "cpu"):
        """
        Initialize feature extractor.
        
        Args:
            use_cnn: Whether to extract CNN embeddings.
            device: "cpu" or "cuda".
        """
        self.handcrafted_extractor = HandcraftedGazeFeatures()
        self.cnn_extractor = CNNEmbeddingExtractor(device) if use_cnn else None
        self.use_cnn = use_cnn

    def extract_combined_features(
        self,
        gaze_points: np.ndarray,
        left_eye_crops: List[np.ndarray],
        right_eye_crops: List[np.ndarray],
        left_eye_gaze: Optional[np.ndarray] = None,
        right_eye_gaze: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Extract all features (handcrafted + CNN).
        
        Args:
            gaze_points: Combined gaze points.
            left_eye_crops: List of left eye crops.
            right_eye_crops: List of right eye crops.
            left_eye_gaze: Left eye gaze points (optional).
            right_eye_gaze: Right eye gaze points (optional).
            
        Returns:
            Dictionary with all features.
        """
        # Handcrafted features
        handcrafted_df = self.handcrafted_extractor.extract_all_handcrafted(
            gaze_points,
            left_eye_gaze,
            right_eye_gaze,
            left_eye_crops + right_eye_crops,
        )

        features = handcrafted_df.to_dict(orient="records")[0]

        # CNN embeddings
        if self.use_cnn:
            left_embeddings = self.cnn_extractor.extract_embeddings_batch(left_eye_crops)
            right_embeddings = self.cnn_extractor.extract_embeddings_batch(right_eye_crops)
            
            features["left_eye_embedding_mean"] = left_embeddings.mean(axis=0)
            features["right_eye_embedding_mean"] = right_embeddings.mean(axis=0)

        return features


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("GAZE Feature Extraction Module Example")
    print("=" * 50)

    # Example handcrafted features
    gaze_points = np.random.rand(300, 2) * 1000
    handcrafted_extractor = HandcraftedGazeFeatures(sampling_rate=30)
    
    fixation_metrics = handcrafted_extractor.compute_fixation_metrics(gaze_points)
    print(f"Fixation metrics: {fixation_metrics}")

    saccade_metrics = handcrafted_extractor.compute_saccade_metrics(gaze_points)
    print(f"Saccade metrics: {saccade_metrics}")

    entropy = handcrafted_extractor.compute_gaze_entropy(gaze_points)
    print(f"Gaze entropy: {entropy:.3f}")
