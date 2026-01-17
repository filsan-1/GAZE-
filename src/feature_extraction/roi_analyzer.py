"""
ROI (Region of Interest) analysis for gaze attention.

Analyzes facial attention patterns based on gaze data.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ROIAnalyzer:
    """
    Analyzes gaze attention distribution across facial ROIs.

    ROIs include:
    - Eyes
    - Mouth
    - Nose
    - Off-face (background)
    """

    # Standard facial ROI definitions (relative coordinates 0-1)
    DEFAULT_ROIS = {
        "eyes": (0.25, 0.25, 0.75, 0.45),      # Upper face
        "nose": (0.35, 0.35, 0.65, 0.55),      # Center face
        "mouth": (0.25, 0.55, 0.75, 0.80),     # Lower face
        "off_face": (-1.0, -1.0, 2.0, 2.0),   # Background
    }

    def __init__(self, roi_definitions: Optional[Dict[str, Tuple]] = None):
        """
        Initialize ROI analyzer.

        Args:
            roi_definitions: Custom ROI definitions (name: (x_min, y_min, x_max, y_max))
        """
        self.roi_definitions = roi_definitions or self.DEFAULT_ROIS
        logger.info(f"Initialized ROI analyzer with {len(self.roi_definitions)} ROIs")

    def get_roi_points(
        self,
        gaze_points: np.ndarray,
        roi_name: str,
    ) -> np.ndarray:
        """
        Get gaze points that fall within a specific ROI.

        Args:
            gaze_points: Array of shape (N, 2) with normalized coordinates [0, 1]
            roi_name: Name of ROI

        Returns:
            Array of points in the ROI
        """
        if roi_name not in self.roi_definitions:
            raise ValueError(f"Unknown ROI: {roi_name}")

        x_min, y_min, x_max, y_max = self.roi_definitions[roi_name]

        mask = (
            (gaze_points[:, 0] >= x_min) & (gaze_points[:, 0] <= x_max) &
            (gaze_points[:, 1] >= y_min) & (gaze_points[:, 1] <= y_max)
        )

        return gaze_points[mask]

    def compute_roi_attention_distribution(
        self,
        gaze_points: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute attention distribution across all ROIs.

        Args:
            gaze_points: Array of shape (N, 2) with normalized coordinates [0, 1]

        Returns:
            Dict of {roi_name: proportion}
        """
        distribution = {}

        for roi_name in self.roi_definitions:
            roi_points = self.get_roi_points(gaze_points, roi_name)
            proportion = len(roi_points) / len(gaze_points) if len(gaze_points) > 0 else 0
            distribution[roi_name] = float(proportion)

        return distribution

    def compute_roi_fixation_time(
        self,
        gaze_points: np.ndarray,
        fixations: List[Dict],
    ) -> Dict[str, float]:
        """
        Compute total fixation time in each ROI.

        Args:
            gaze_points: Array of shape (N, 2)
            fixations: List of fixation dicts with start_idx, end_idx, duration

        Returns:
            Dict of {roi_name: total_fixation_time}
        """
        roi_fixation_times = {roi: 0.0 for roi in self.roi_definitions}

        for fixation in fixations:
            start_idx = fixation["start_idx"]
            end_idx = fixation["end_idx"]
            duration = fixation["duration"]

            fixation_points = gaze_points[start_idx:end_idx]

            for roi_name in self.roi_definitions:
                roi_points = self.get_roi_points(fixation_points, roi_name)

                if len(roi_points) > 0:
                    # Proportion of this fixation in this ROI
                    proportion = len(roi_points) / len(fixation_points)
                    roi_fixation_times[roi_name] += duration * proportion

        return roi_fixation_times

    def compute_roi_transition_matrix(
        self,
        gaze_points: np.ndarray,
        window_size: int = 5,
    ) -> np.ndarray:
        """
        Compute transition matrix between ROIs.

        Args:
            gaze_points: Array of shape (N, 2)
            window_size: Window size for ROI assignment

        Returns:
            Transition matrix (num_rois x num_rois)
        """
        roi_names = list(self.roi_definitions.keys())
        n_rois = len(roi_names)

        # Assign each gaze point to an ROI
        roi_sequence = []
        for i in range(0, len(gaze_points), window_size):
            window = gaze_points[i:i + window_size]
            if len(window) == 0:
                continue

            # Find which ROI has the most points
            max_roi = None
            max_count = -1

            for roi_name in roi_names:
                count = len(self.get_roi_points(window, roi_name))
                if count > max_count:
                    max_count = count
                    max_roi = roi_name

            if max_roi is not None:
                roi_sequence.append(roi_names.index(max_roi))

        # Build transition matrix
        transition_matrix = np.zeros((n_rois, n_rois))

        for i in range(len(roi_sequence) - 1):
            from_roi = roi_sequence[i]
            to_roi = roi_sequence[i + 1]
            transition_matrix[from_roi, to_roi] += 1

        # Normalize rows
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = np.divide(
            transition_matrix,
            row_sums[:, np.newaxis],
            where=row_sums[:, np.newaxis] != 0
        )

        return transition_matrix

    def compute_roi_entropy(
        self,
        gaze_points: np.ndarray,
    ) -> float:
        """
        Compute entropy of ROI attention distribution.

        Higher entropy indicates more distributed attention.

        Args:
            gaze_points: Array of shape (N, 2)

        Returns:
            Entropy value
        """
        distribution = self.compute_roi_attention_distribution(gaze_points)
        proportions = np.array(list(distribution.values()))

        # Remove zero values
        proportions = proportions[proportions > 0]

        if len(proportions) == 0:
            return 0.0

        # Shannon entropy
        entropy = -np.sum(proportions * np.log2(proportions))

        return float(entropy)

    def get_roi_summary(
        self,
        gaze_points: np.ndarray,
        fixations: Optional[List[Dict]] = None,
    ) -> Dict[str, float]:
        """
        Get comprehensive ROI analysis summary.

        Args:
            gaze_points: Array of shape (N, 2)
            fixations: Optional list of fixation dicts

        Returns:
            Comprehensive ROI summary
        """
        summary = {}

        # Attention distribution
        attention = self.compute_roi_attention_distribution(gaze_points)
        for roi_name, prop in attention.items():
            summary[f"roi_{roi_name}_attention"] = prop

        # Fixation time if available
        if fixations:
            fixation_times = self.compute_roi_fixation_time(gaze_points, fixations)
            for roi_name, time in fixation_times.items():
                summary[f"roi_{roi_name}_fixation_time"] = time

        # ROI entropy
        summary["roi_attention_entropy"] = self.compute_roi_entropy(gaze_points)

        return summary


if __name__ == "__main__":
    # Example usage
    analyzer = ROIAnalyzer()

    # Create synthetic gaze data (normalized [0, 1])
    n_samples = 300
    t = np.arange(n_samples) / 30.0

    # Gaze moves between eyes and mouth
    gaze_x = np.where(t % 2 < 1, 0.4, 0.5)  # Eyes at 0.4, nose at 0.5
    gaze_y = np.where(t % 2 < 1, 0.35, 0.65)  # Eyes at 0.35, mouth at 0.65

    gaze_points = np.column_stack([gaze_x, gaze_y])

    # Analyze
    summary = analyzer.get_roi_summary(gaze_points)

    print("ROI Analysis Summary:")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value:.3f}")
