"""
Gaze rendering and visualization module.

Renders gaze vectors, fixations, and ROI attention overlays.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

from src.config import LEFT_EYE_INDICES, RIGHT_EYE_INDICES

logger = logging.getLogger(__name__)


class GazeRenderer:
    """
    Renders gaze tracking overlays on video frames.

    Renders:
    - Gaze vector (from eye to estimated gaze point)
    - Fixation point
    - Eye aspect ratio indicator
    - ROI attention heatmap
    - Gaze trail (recent gaze history)
    """

    def __init__(self, max_trail_length: int = 30):
        """
        Initialize gaze renderer.

        Args:
            max_trail_length: Maximum number of gaze points to display as trail
        """
        self.max_trail_length = max_trail_length
        self.gaze_trail = []

    def render_gaze_point(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[float, float],
        radius: int = 15,
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw gaze point on frame.

        Args:
            frame: Input frame
            gaze_point: (gaze_x, gaze_y) coordinates
            radius: Circle radius
            color: BGR color tuple
            thickness: Line thickness

        Returns:
            Frame with gaze point drawn
        """
        frame = frame.copy()

        x, y = int(gaze_point[0]), int(gaze_point[1])

        # Draw circle
        cv2.circle(frame, (x, y), radius, color, thickness)

        # Draw crosshair
        cv2.line(frame, (x - 10, y), (x + 10, y), color, 1)
        cv2.line(frame, (x, y - 10), (x, y + 10), color, 1)

        return frame

    def render_gaze_vector(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        gaze_point: Tuple[float, float],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw gaze vector from eye center to gaze point.

        Args:
            frame: Input frame
            landmarks: Facial landmarks
            gaze_point: Target gaze point
            color: BGR color
            thickness: Line thickness

        Returns:
            Frame with gaze vector
        """
        frame = frame.copy()

        # Get eye centers
        left_eye = landmarks[LEFT_EYE_INDICES].mean(axis=0)
        right_eye = landmarks[RIGHT_EYE_INDICES].mean(axis=0)
        eye_center = (left_eye + right_eye) / 2

        # Draw vector from eye center to gaze point
        start = (int(eye_center[0]), int(eye_center[1]))
        end = (int(gaze_point[0]), int(gaze_point[1]))

        cv2.arrowedLine(frame, start, end, color, thickness, tipLength=0.3)

        return frame

    def render_gaze_trail(
        self,
        frame: np.ndarray,
        gaze_point: Tuple[float, float],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw trail of recent gaze points.

        Args:
            frame: Input frame
            gaze_point: Current gaze point to add to trail
            color: BGR color
            thickness: Line thickness

        Returns:
            Frame with gaze trail
        """
        frame = frame.copy()

        # Add current point to trail
        self.gaze_trail.append(gaze_point)

        # Limit trail length
        if len(self.gaze_trail) > self.max_trail_length:
            self.gaze_trail.pop(0)

        # Draw trail as connected points
        if len(self.gaze_trail) > 1:
            points = np.array(self.gaze_trail, dtype=np.int32)

            # Draw polyline with fading effect
            for i in range(len(points) - 1):
                # Fade color as trail goes back in time
                alpha = (i + 1) / len(points)
                faded_color = tuple(int(c * alpha) for c in color)

                cv2.line(frame, tuple(points[i]), tuple(points[i + 1]), faded_color, thickness)

        return frame

    def render_eye_aspect_ratio(
        self,
        frame: np.ndarray,
        ear_left: float,
        ear_right: float,
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 0.7,
    ) -> np.ndarray:
        """
        Display Eye Aspect Ratio values.

        Args:
            frame: Input frame
            ear_left: Left eye aspect ratio
            ear_right: Right eye aspect ratio
            position: Text position (x, y)
            font_scale: Font scale

        Returns:
            Frame with EAR text
        """
        frame = frame.copy()

        text = f"EAR L:{ear_left:.2f} R:{ear_right:.2f}"
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
        )

        return frame

    def render_roi_heatmap(
        self,
        frame: np.ndarray,
        roi_attention: Dict[str, float],
        position: Tuple[int, int] = (10, 60),
        bar_width: int = 20,
        bar_height: int = 100,
    ) -> np.ndarray:
        """
        Draw ROI attention heatmap.

        Args:
            frame: Input frame
            roi_attention: Dictionary of {roi_name: attention_proportion}
            position: Top-left position
            bar_width: Width of each bar
            bar_height: Height of each bar

        Returns:
            Frame with attention bars
        """
        frame = frame.copy()

        x, y = position
        roi_names = list(roi_attention.keys())

        for i, (roi_name, attention) in enumerate(roi_attention.items()):
            # Scale attention to bar height
            bar_height_scaled = int(attention * bar_height)

            # Color based on attention level
            color = (0, int(255 * attention), int(255 * (1 - attention)))

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x + i * (bar_width + 5), y),
                (x + i * (bar_width + 5) + bar_width, y + bar_height_scaled),
                color,
                -1,
            )

            # Draw label
            cv2.putText(
                frame,
                roi_name,
                (x + i * (bar_width + 5), y + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return frame

    def render_fixation_point(
        self,
        frame: np.ndarray,
        fixation_point: Tuple[float, float],
        fixation_duration: float = 0.0,
        radius: int = 30,
        color: Tuple[int, int, int] = (255, 0, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw fixation point and duration.

        Args:
            frame: Input frame
            fixation_point: Center of fixation
            fixation_duration: Duration of fixation in seconds
            radius: Fixation circle radius
            color: BGR color
            thickness: Line thickness

        Returns:
            Frame with fixation indicator
        """
        frame = frame.copy()

        x, y = int(fixation_point[0]), int(fixation_point[1])

        # Draw circle
        cv2.circle(frame, (x, y), radius, color, thickness)

        # Draw duration text if provided
        if fixation_duration > 0:
            text = f"Fix: {fixation_duration:.2f}s"
            cv2.putText(
                frame,
                text,
                (x - 40, y - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        return frame

    def render_stimulus_target(
        self,
        frame: np.ndarray,
        stimulus_position: Tuple[float, float],
        radius: int = 15,
        color: Tuple[int, int, int] = (0, 0, 255),
        filled: bool = True,
    ) -> np.ndarray:
        """
        Draw the stimulus target (red dot).

        Args:
            frame: Input frame
            stimulus_position: (x, y) coordinates
            radius: Dot radius
            color: BGR color
            filled: Whether to fill the circle

        Returns:
            Frame with stimulus
        """
        frame = frame.copy()

        x, y = int(stimulus_position[0]), int(stimulus_position[1])

        if filled:
            cv2.circle(frame, (x, y), radius, color, -1)
        else:
            cv2.circle(frame, (x, y), radius, color, 2)

        return frame

    def clear_trail(self):
        """Clear gaze trail history."""
        self.gaze_trail = []


if __name__ == "__main__":
    # Example usage
    renderer = GazeRenderer()

    # Create dummy frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Render gaze point
    frame = renderer.render_gaze_point(frame, (640, 360))

    # Render stimulus
    frame = renderer.render_stimulus_target(frame, (300, 200))

    print("Rendering test completed")
