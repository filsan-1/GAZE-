"""
MediaPipe-based facial landmark and iris detection.

Detects facial landmarks and iris position for gaze estimation.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe not installed. Install with: pip install mediapipe")

from src.config import (
    MEDIAPIPE_DETECTION_CONFIDENCE,
    MEDIAPIPE_TRACKING_CONFIDENCE,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)

logger = logging.getLogger(__name__)


class MediaPipeDetector:
    """
    MediaPipe Face Mesh-based gaze tracking detector.

    Detects:
    - 468 facial landmarks
    - Iris position
    - Eye state (open/closed)
    - Head pose approximation
    """

    def __init__(
        self,
        detection_confidence: float = MEDIAPIPE_DETECTION_CONFIDENCE,
        tracking_confidence: float = MEDIAPIPE_TRACKING_CONFIDENCE,
    ):
        """
        Initialize MediaPipe detector.

        Args:
            detection_confidence: Minimum confidence for face detection
            tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.last_landmarks = None
        logger.info("MediaPipe detector initialized")

    def detect_landmarks(
        self,
        frame: np.ndarray,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Detect facial landmarks in frame.

        Args:
            frame: Input frame (BGR format, numpy array)

        Returns:
            Dictionary with landmarks or None if no face detected
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Convert to image coordinates
        h, w = frame.shape[:2]
        landmark_points = np.array([
            [lm.x * w, lm.y * h, lm.z * w] for lm in landmarks
        ])

        self.last_landmarks = landmark_points

        return {
            "landmarks": landmark_points,
            "left_eye": landmark_points[LEFT_EYE_INDICES],
            "right_eye": landmark_points[RIGHT_EYE_INDICES],
            "iris_left": self._extract_iris_points(landmark_points, is_right=False),
            "iris_right": self._extract_iris_points(landmark_points, is_right=True),
            "face_detected": True,
        }

    def _extract_iris_points(
        self,
        landmarks: np.ndarray,
        is_right: bool = False,
    ) -> np.ndarray:
        """
        Extract iris landmark points.

        Args:
            landmarks: All facial landmarks
            is_right: Whether to extract right iris (True) or left (False)

        Returns:
            Iris points array
        """
        # Iris landmarks: left (468-471), right (472-475)
        if is_right:
            iris_indices = [472, 473, 474, 475]
        else:
            iris_indices = [468, 469, 470, 471]

        return landmarks[iris_indices]

    def estimate_gaze_point(
        self,
        landmarks: np.ndarray,
        frame_width: int,
        frame_height: int,
    ) -> Tuple[float, float]:
        """
        Estimate gaze point on screen.

        Uses iris position relative to eye region.

        Args:
            landmarks: Facial landmark points
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels

        Returns:
            Tuple of (gaze_x, gaze_y) in pixels
        """
        # Get iris centers
        left_iris = landmarks[468:472]
        right_iris = landmarks[472:476]

        left_iris_center = left_iris.mean(axis=0)
        right_iris_center = right_iris.mean(axis=0)

        # Average the two eyes
        gaze_point = (left_iris_center + right_iris_center) / 2

        # Clip to frame bounds
        gaze_x = np.clip(gaze_point[0], 0, frame_width)
        gaze_y = np.clip(gaze_point[1], 0, frame_height)

        return float(gaze_x), float(gaze_y)

    def get_head_pose(
        self,
        landmarks: np.ndarray,
    ) -> Dict[str, float]:
        """
        Estimate head pose from facial landmarks.

        Computes approximate pitch, yaw, roll angles.

        Args:
            landmarks: Facial landmark points

        Returns:
            Dictionary with head_pitch, head_yaw, head_roll (degrees)
        """
        # Use specific landmarks for head pose estimation
        nose_tip = landmarks[1]  # Tip of nose
        chin = landmarks[152]  # Chin
        left_eye = landmarks[33]  # Left eye outer corner
        right_eye = landmarks[263]  # Right eye outer corner

        # Compute vectors
        vertical_vector = chin - nose_tip  # Pitch vector
        horizontal_vector = right_eye - left_eye  # Yaw vector

        # Angles (simplified)
        pitch = np.arctan2(vertical_vector[1], vertical_vector[2]) * 180 / np.pi
        yaw = np.arctan2(horizontal_vector[0], horizontal_vector[2]) * 180 / np.pi
        roll = np.arctan2(vertical_vector[0], vertical_vector[1]) * 180 / np.pi

        return {
            "pitch": float(pitch),
            "yaw": float(yaw),
            "roll": float(roll),
        }

    def compute_eye_aspect_ratio(
        self,
        landmarks: np.ndarray,
        is_right: bool = False,
    ) -> float:
        """
        Compute Eye Aspect Ratio (EAR) for blink detection.

        Args:
            landmarks: Facial landmarks
            is_right: Whether to compute for right eye (True) or left (False)

        Returns:
            Eye Aspect Ratio value
        """
        if is_right:
            # Right eye indices
            eye_indices = RIGHT_EYE_INDICES
        else:
            # Left eye indices
            eye_indices = LEFT_EYE_INDICES

        eye_points = landmarks[eye_indices]

        # Compute vertical distances (P2-P6, P3-P5)
        vertical_dist_1 = np.linalg.norm(eye_points[1] - eye_points[5])
        vertical_dist_2 = np.linalg.norm(eye_points[2] - eye_points[4])

        # Compute horizontal distance (P1-P4)
        horizontal_dist = np.linalg.norm(eye_points[0] - eye_points[3])

        # EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
        ear = (vertical_dist_1 + vertical_dist_2) / (2 * horizontal_dist) if horizontal_dist > 0 else 0

        return float(ear)

    def is_blink(
        self,
        landmarks: np.ndarray,
        threshold: float = 0.2,
    ) -> bool:
        """
        Detect if eyes are closed (blink).

        Args:
            landmarks: Facial landmarks
            threshold: EAR threshold for closed eyes

        Returns:
            True if eyes are closed
        """
        left_ear = self.compute_eye_aspect_ratio(landmarks, is_right=False)
        right_ear = self.compute_eye_aspect_ratio(landmarks, is_right=True)

        return (left_ear + right_ear) / 2 < threshold

    def draw_landmarks(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        draw_iris: bool = True,
        draw_face_mesh: bool = True,
    ) -> np.ndarray:
        """
        Draw facial landmarks on frame.

        Args:
            frame: Input frame
            landmarks: Facial landmarks
            draw_iris: Whether to draw iris points
            draw_face_mesh: Whether to draw face mesh connections

        Returns:
            Frame with drawn landmarks
        """
        frame = frame.copy()

        # Draw iris points
        if draw_iris:
            for i, point in enumerate(landmarks[468:476]):
                cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)

        # Draw specific facial features
        # Eyes
        for eye_idx in LEFT_EYE_INDICES:
            pt = landmarks[eye_idx]
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), 1)

        for eye_idx in RIGHT_EYE_INDICES:
            pt = landmarks[eye_idx]
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), 1)

        return frame

    def release(self):
        """Release MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()


if __name__ == "__main__":
    # Example usage
    detector = MediaPipeDetector()

    # Simulate with dummy frame
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # In real usage, pass actual camera frame
    # result = detector.detect_landmarks(frame)
    # if result:
    #     print("Face detected, landmarks shape:", result["landmarks"].shape)

    detector.release()
