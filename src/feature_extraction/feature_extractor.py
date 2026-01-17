"""
Gaze feature extraction module.

Computes fixation, saccade, entropy, and temporal gaze features.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import euclidean

from src.config import (
    FIXATION_VELOCITY_THRESHOLD,
    FIXATION_MIN_DURATION,
    SACCADE_MIN_DURATION,
)

logger = logging.getLogger(__name__)


class GazeFeatureExtractor:
    """
    Extracts comprehensive gaze tracking features.

    Computes:
    - Fixation duration and count
    - Saccade metrics (amplitude, velocity, count)
    - Gaze entropy and stability
    - Eye aspect ratio trends
    - Gaze dispersion
    """

    def __init__(self, sampling_rate: int = 30):
        """
        Initialize feature extractor.

        Args:
            sampling_rate: Sampling rate in Hz
        """
        self.sampling_rate = sampling_rate
        self.sample_period = 1.0 / sampling_rate

    def compute_fixations(
        self,
        gaze_points: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        velocity_threshold: float = FIXATION_VELOCITY_THRESHOLD,
        min_duration: float = FIXATION_MIN_DURATION,
    ) -> List[Dict[str, float]]:
        """
        Detect fixations from gaze points.

        A fixation is a period where gaze velocity is below threshold.

        Args:
            gaze_points: Array of shape (N, 2) with (x, y) coordinates
            velocities: Pre-computed gaze velocities
            velocity_threshold: Velocity threshold in deg/sec
            min_duration: Minimum fixation duration in seconds

        Returns:
            List of fixation dicts with center, duration, start_time
        """
        if len(gaze_points) < 2:
            return []

        # Compute velocities if not provided
        if velocities is None:
            velocities = self._compute_gaze_velocity(gaze_points)

        # Identify fixation periods (velocity < threshold)
        is_fixation = velocities < velocity_threshold
        fixation_groups = self._segment_boolean_array(is_fixation)

        fixations = []
        min_samples = int(min_duration * self.sampling_rate)

        for start_idx, end_idx in fixation_groups:
            duration = (end_idx - start_idx) * self.sample_period

            if duration >= min_duration:
                center = gaze_points[start_idx:end_idx].mean(axis=0)

                fixations.append({
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "duration": float(duration),
                    "start_time": float(start_idx * self.sample_period),
                })

        logger.info(f"Detected {len(fixations)} fixations")
        return fixations

    def compute_saccades(
        self,
        gaze_points: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        velocity_threshold: float = FIXATION_VELOCITY_THRESHOLD,
        min_duration: float = SACCADE_MIN_DURATION,
    ) -> List[Dict[str, float]]:
        """
        Detect saccades from gaze points.

        A saccade is a rapid eye movement between fixations.

        Args:
            gaze_points: Array of shape (N, 2) with (x, y) coordinates
            velocities: Pre-computed gaze velocities
            velocity_threshold: Velocity threshold in deg/sec
            min_duration: Minimum saccade duration in seconds

        Returns:
            List of saccade dicts with amplitude, duration, velocity
        """
        if len(gaze_points) < 2:
            return []

        # Compute velocities if not provided
        if velocities is None:
            velocities = self._compute_gaze_velocity(gaze_points)

        # Identify saccade periods (velocity >= threshold)
        is_saccade = velocities >= velocity_threshold
        saccade_groups = self._segment_boolean_array(is_saccade)

        saccades = []
        min_samples = int(min_duration * self.sampling_rate)

        for start_idx, end_idx in saccade_groups:
            num_samples = end_idx - start_idx

            if num_samples >= min_samples:
                start_point = gaze_points[start_idx]
                end_point = gaze_points[end_idx - 1]

                amplitude = euclidean(start_point, end_point)
                duration = num_samples * self.sample_period
                mean_velocity = np.mean(velocities[start_idx:end_idx])

                saccades.append({
                    "amplitude": float(amplitude),
                    "duration": float(duration),
                    "peak_velocity": float(np.max(velocities[start_idx:end_idx])),
                    "mean_velocity": float(mean_velocity),
                    "start_idx": int(start_idx),
                    "end_idx": int(end_idx),
                    "start_time": float(start_idx * self.sample_period),
                })

        logger.info(f"Detected {len(saccades)} saccades")
        return saccades

    def compute_gaze_entropy(self, gaze_points: np.ndarray, bins: int = 20) -> float:
        """
        Compute entropy of gaze distribution.

        Higher entropy indicates more distributed/random gaze patterns.

        Args:
            gaze_points: Array of shape (N, 2) with (x, y) coordinates
            bins: Number of bins for 2D histogram

        Returns:
            Entropy value
        """
        if len(gaze_points) < 2:
            return 0.0

        # Create 2D histogram
        x_range = (gaze_points[:, 0].min(), gaze_points[:, 0].max())
        y_range = (gaze_points[:, 1].min(), gaze_points[:, 1].max())

        hist, _, _ = np.histogram2d(
            gaze_points[:, 0], gaze_points[:, 1],
            bins=bins,
            range=[x_range, y_range]
        )

        # Normalize histogram to probability distribution
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0]  # Remove zero bins
        p = hist_flat / hist_flat.sum()

        # Compute entropy
        entropy = -np.sum(p * np.log2(p))

        return float(entropy)

    def compute_gaze_dispersion(self, gaze_points: np.ndarray) -> float:
        """
        Compute gaze dispersion (spread of gaze points).

        Args:
            gaze_points: Array of shape (N, 2) with (x, y) coordinates

        Returns:
            Dispersion value (variance)
        """
        if len(gaze_points) < 2:
            return 0.0

        # Compute centroid
        centroid = gaze_points.mean(axis=0)

        # Compute distances from centroid
        distances = np.linalg.norm(gaze_points - centroid, axis=1)

        # Dispersion as variance of distances
        dispersion = np.var(distances)

        return float(dispersion)

    def compute_roi_attention(
        self,
        gaze_points: np.ndarray,
        roi_regions: Dict[str, Tuple[float, float, float, float]],
    ) -> Dict[str, float]:
        """
        Compute proportion of gaze in each ROI.

        Args:
            gaze_points: Array of shape (N, 2) with (x, y) coordinates
            roi_regions: Dict of {roi_name: (x_min, y_min, x_max, y_max)}

        Returns:
            Dict of {roi_name: proportion}
        """
        roi_attention = {}

        for roi_name, (x_min, y_min, x_max, y_max) in roi_regions.items():
            # Check if gaze points fall within ROI
            in_roi = (
                (gaze_points[:, 0] >= x_min) & (gaze_points[:, 0] <= x_max) &
                (gaze_points[:, 1] >= y_min) & (gaze_points[:, 1] <= y_max)
            )

            proportion = in_roi.sum() / len(gaze_points) if len(gaze_points) > 0 else 0
            roi_attention[roi_name] = float(proportion)

        return roi_attention

    def extract_temporal_features(
        self,
        gaze_points: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        ear_values: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Extract temporal gaze statistics.

        Args:
            gaze_points: Array of shape (N, 2) with (x, y) coordinates
            velocities: Gaze velocities array
            ear_values: Eye aspect ratio values

        Returns:
            Dictionary of temporal features
        """
        features = {}

        # Gaze position features
        features["gaze_x_mean"] = float(gaze_points[:, 0].mean())
        features["gaze_y_mean"] = float(gaze_points[:, 1].mean())
        features["gaze_x_std"] = float(gaze_points[:, 0].std())
        features["gaze_y_std"] = float(gaze_points[:, 1].std())

        # Velocity features
        if velocities is not None:
            features["gaze_velocity_mean"] = float(velocities.mean())
            features["gaze_velocity_std"] = float(velocities.std())
            features["gaze_velocity_max"] = float(velocities.max())

        # Blink/EAR features
        if ear_values is not None:
            features["ear_mean"] = float(ear_values.mean())
            features["ear_std"] = float(ear_values.std())
            features["blink_count"] = int((ear_values < 0.2).sum())

        return features

    def extract_all_features(
        self,
        gaze_points: np.ndarray,
        ear_values: Optional[np.ndarray] = None,
        roi_regions: Optional[Dict[str, Tuple]] = None,
    ) -> Dict[str, float]:
        """
        Extract comprehensive feature set.

        Args:
            gaze_points: Array of shape (N, 2) with (x, y) coordinates
            ear_values: Eye aspect ratio values
            roi_regions: Dictionary of ROI regions

        Returns:
            Comprehensive feature dictionary
        """
        features = {}

        # Compute velocities
        velocities = self._compute_gaze_velocity(gaze_points)

        # Fixation features
        fixations = self.compute_fixations(gaze_points, velocities)
        features["fixation_count"] = float(len(fixations))
        if fixations:
            features["fixation_duration_mean"] = np.mean([f["duration"] for f in fixations])
            features["fixation_duration_max"] = np.max([f["duration"] for f in fixations])
        else:
            features["fixation_duration_mean"] = 0.0
            features["fixation_duration_max"] = 0.0

        # Saccade features
        saccades = self.compute_saccades(gaze_points, velocities)
        features["saccade_count"] = float(len(saccades))
        features["saccade_count_per_min"] = (len(saccades) / len(gaze_points)) * 60 * self.sampling_rate
        if saccades:
            features["saccade_amplitude_mean"] = np.mean([s["amplitude"] for s in saccades])
            features["saccade_velocity_mean"] = np.mean([s["mean_velocity"] for s in saccades])
        else:
            features["saccade_amplitude_mean"] = 0.0
            features["saccade_velocity_mean"] = 0.0

        # Entropy and dispersion
        features["gaze_entropy"] = self.compute_gaze_entropy(gaze_points)
        features["gaze_dispersion"] = self.compute_gaze_dispersion(gaze_points)

        # Temporal features
        temporal_features = self.extract_temporal_features(gaze_points, velocities, ear_values)
        features.update(temporal_features)

        # ROI attention
        if roi_regions:
            roi_attention = self.compute_roi_attention(gaze_points, roi_regions)
            for roi_name, attention in roi_attention.items():
                features[f"roi_attention_{roi_name}"] = attention

        return features

    def _compute_gaze_velocity(self, gaze_points: np.ndarray) -> np.ndarray:
        """
        Compute gaze velocity from points.

        Args:
            gaze_points: Array of shape (N, 2)

        Returns:
            Array of velocities
        """
        diffs = np.diff(gaze_points, axis=0)
        velocities = np.linalg.norm(diffs, axis=1) / self.sample_period

        # Pad first value
        velocities = np.concatenate([[0], velocities])

        return velocities

    def _segment_boolean_array(
        self,
        boolean_array: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Segment boolean array into consecutive True regions.

        Args:
            boolean_array: Boolean array

        Returns:
            List of (start_idx, end_idx) tuples
        """
        # Find transitions
        diff = np.diff(np.concatenate([[False], boolean_array, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        return list(zip(starts, ends))


if __name__ == "__main__":
    # Example usage
    extractor = GazeFeatureExtractor(sampling_rate=30)

    # Create synthetic gaze data
    n_samples = 300
    t = np.arange(n_samples) / 30.0
    gaze_x = 640 + 200 * np.sin(2 * np.pi * t / 5)
    gaze_y = 360 + 150 * np.cos(2 * np.pi * t / 5)

    gaze_points = np.column_stack([gaze_x, gaze_y])

    # Extract features
    features = extractor.extract_all_features(gaze_points)

    print("Extracted features:")
    for key, value in sorted(features.items()):
        print(f"  {key}: {value:.3f}")
