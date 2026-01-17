"""
Preprocessing module for GAZE Research Platform.

Handles face/eye detection, landmark extraction, normalization, and preparation
of data for feature extraction and model training.
"""

import logging
from typing import Tuple, Optional, Dict, List
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import json

from config import (
    CAMERA_RESOLUTION,
    EYE_CROP_SIZE,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    FACE_DETECTION_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MediaPipeFaceDetector:
    """
    Face and eye landmark detection using MediaPipe FaceMesh.
    
    Detects facial landmarks and extracts eye regions for analysis.
    """

    def __init__(self, static_image_mode: bool = False, max_num_faces: int = 1):
        """
        Initialize MediaPipe FaceMesh detector.
        
        Args:
            static_image_mode: Whether to process static images or video.
            max_num_faces: Maximum number of faces to detect.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=FACE_DETECTION_THRESHOLD,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        logger.info("MediaPipe FaceMesh initialized")

    def detect_landmarks(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect facial landmarks in a frame.
        
        Args:
            frame: Input image (BGR format from OpenCV).
            
        Returns:
            Dictionary with landmarks and face info, or None if no face detected.
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks or len(results.multi_face_landmarks) == 0:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape

        # Convert normalized coordinates to pixel coordinates
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark
        ])

        return {
            "landmarks": landmarks,
            "frame_shape": (h, w),
            "num_landmarks": len(landmarks),
        }

    def extract_eye_crops(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract left and right eye crops from frame.
        
        Args:
            frame: Input image.
            landmarks: Facial landmarks from detect_landmarks().
            
        Returns:
            Tuple of (left_eye_crop, right_eye_crop), or (None, None) if extraction fails.
        """
        try:
            # Get eye landmark indices
            left_eye_points = landmarks[LEFT_EYE_INDICES][:, :2].astype(np.int32)
            right_eye_points = landmarks[RIGHT_EYE_INDICES][:, :2].astype(np.int32)

            # Compute bounding boxes with padding
            left_eye_crop = self._extract_eye_crop(frame, left_eye_points)
            right_eye_crop = self._extract_eye_crop(frame, right_eye_points)

            return left_eye_crop, right_eye_crop

        except Exception as e:
            logger.warning(f"Eye crop extraction failed: {e}")
            return None, None

    @staticmethod
    def _extract_eye_crop(
        frame: np.ndarray,
        eye_points: np.ndarray,
        padding: int = 20,
    ) -> Optional[np.ndarray]:
        """
        Extract single eye crop from frame.
        
        Args:
            frame: Input image.
            eye_points: 2D eye landmark points.
            padding: Padding around eye region.
            
        Returns:
            Resized eye crop or None if extraction fails.
        """
        h, w = frame.shape[:2]
        x_min = max(0, int(eye_points[:, 0].min()) - padding)
        x_max = min(w, int(eye_points[:, 0].max()) + padding)
        y_min = max(0, int(eye_points[:, 1].min()) - padding)
        y_max = min(h, int(eye_points[:, 1].max()) + padding)

        if x_max <= x_min or y_max <= y_min:
            return None

        eye_crop = frame[y_min:y_max, x_min:x_max]
        
        # Resize to standard size
        eye_crop_resized = cv2.resize(eye_crop, EYE_CROP_SIZE, interpolation=cv2.INTER_LINEAR)
        
        return eye_crop_resized

    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw facial landmarks on frame.
        
        Args:
            frame: Input image.
            landmarks: Facial landmarks.
            
        Returns:
            Frame with drawn landmarks.
        """
        frame_copy = frame.copy()
        
        # Draw landmarks as circles
        for landmark in landmarks[:, :2].astype(np.int32):
            cv2.circle(frame_copy, tuple(landmark), 2, (0, 255, 0), -1)
        
        return frame_copy


class GazeNormalizer:
    """
    Normalize gaze coordinates and features across datasets.
    
    Ensures consistent coordinate systems and handles missing values robustly.
    """

    def __init__(self, screen_width: float = 1920, screen_height: float = 1080):
        """
        Initialize normalizer.
        
        Args:
            screen_width: Assumed screen width for normalization.
            screen_height: Assumed screen height for normalization.
        """
        self.screen_width = screen_width
        self.screen_height = screen_height

    def normalize_gaze_coordinates(
        self,
        gaze_x: np.ndarray,
        gaze_y: np.ndarray,
        origin: str = "top-left",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize gaze coordinates to [0, 1] range.
        
        Args:
            gaze_x: X coordinates (raw pixel or normalized).
            gaze_y: Y coordinates (raw pixel or normalized).
            origin: Coordinate origin ("top-left" or "center").
            
        Returns:
            Tuple of normalized (gaze_x, gaze_y) in [0, 1] range.
        """
        # Ensure arrays
        gaze_x = np.asarray(gaze_x, dtype=np.float32)
        gaze_y = np.asarray(gaze_y, dtype=np.float32)

        # Handle missing values (NaN)
        valid_mask = ~(np.isnan(gaze_x) | np.isnan(gaze_y))
        
        # Normalize to [0, 1]
        x_norm = np.clip(gaze_x / self.screen_width, 0, 1)
        y_norm = np.clip(gaze_y / self.screen_height, 0, 1)

        # If center origin, shift coordinates
        if origin == "center":
            x_norm = (x_norm - 0.5) * 2
            y_norm = (y_norm - 0.5) * 2

        return x_norm, y_norm

    def handle_missing_landmarks(
        self,
        landmarks: np.ndarray,
        method: str = "interpolate",
    ) -> np.ndarray:
        """
        Handle missing or invalid landmarks robustly.
        
        Args:
            landmarks: Landmark array (n_frames, n_landmarks, 3).
            method: "interpolate", "forward_fill", or "remove".
            
        Returns:
            Processed landmarks with missing values handled.
        """
        landmarks_clean = landmarks.copy()
        
        if method == "interpolate":
            # Linear interpolation for missing values
            for i in range(landmarks.shape[1]):
                for j in range(landmarks.shape[2]):
                    col = landmarks[:, i, j]
                    mask = np.isnan(col)
                    if mask.any():
                        col[mask] = np.interp(
                            np.flatnonzero(mask),
                            np.flatnonzero(~mask),
                            col[~mask]
                        )
                        landmarks_clean[:, i, j] = col
        
        elif method == "forward_fill":
            # Forward fill missing values
            for i in range(landmarks.shape[1]):
                for j in range(landmarks.shape[2]):
                    col = landmarks[:, i, j]
                    mask = np.isnan(col)
                    if mask.any():
                        col[mask] = np.nan_to_num(col, nan=col[~mask][0] if (~mask).any() else 0)
                        landmarks_clean[:, i, j] = col
        
        # Replace any remaining NaN with 0
        landmarks_clean = np.nan_to_num(landmarks_clean, nan=0.0)
        
        return landmarks_clean

    def normalize_pupil_diameter(
        self,
        pupil_diameter: np.ndarray,
    ) -> np.ndarray:
        """
        Normalize pupil diameter to [0, 1] range.
        
        Args:
            pupil_diameter: Pupil diameter values.
            
        Returns:
            Normalized pupil diameter.
        """
        # Typical pupil diameter range: 2-8mm, normalize to [0, 1]
        pupil_norm = np.clip((pupil_diameter - 2.0) / 6.0, 0, 1)
        return pupil_norm


class DataPreprocessor:
    """
    End-to-end data preprocessing pipeline.
    
    Combines face detection, landmark extraction, and normalization.
    """

    def __init__(self):
        """Initialize preprocessor with detector and normalizer."""
        self.detector = MediaPipeFaceDetector()
        self.normalizer = GazeNormalizer()

    def process_frame(
        self,
        frame: np.ndarray,
        normalize: bool = True,
    ) -> Optional[Dict]:
        """
        Process a single frame end-to-end.
        
        Args:
            frame: Input frame (BGR).
            normalize: Whether to normalize coordinates.
            
        Returns:
            Dictionary with processed data or None if processing fails.
        """
        # Detect landmarks
        landmark_data = self.detector.detect_landmarks(frame)
        if landmark_data is None:
            return None

        landmarks = landmark_data["landmarks"]
        
        # Extract eye crops
        left_eye, right_eye = self.detector.extract_eye_crops(frame, landmarks)
        if left_eye is None or right_eye is None:
            return None

        # Compute gaze point (average of eye landmarks)
        left_eye_center = landmarks[LEFT_EYE_INDICES][:, :2].mean(axis=0)
        right_eye_center = landmarks[RIGHT_EYE_INDICES][:, :2].mean(axis=0)
        gaze_point = (left_eye_center + right_eye_center) / 2

        result = {
            "landmarks": landmarks,
            "gaze_point": gaze_point,
            "left_eye_crop": left_eye,
            "right_eye_crop": right_eye,
            "frame_shape": landmark_data["frame_shape"],
        }

        if normalize:
            # Normalize coordinates
            gaze_x_norm, gaze_y_norm = self.normalizer.normalize_gaze_coordinates(
                gaze_point[0],
                gaze_point[1],
            )
            result["gaze_point_normalized"] = np.array([gaze_x_norm, gaze_y_norm])

        return result

    def process_frames_batch(
        self,
        frames: List[np.ndarray],
    ) -> List[Dict]:
        """
        Process multiple frames.
        
        Args:
            frames: List of input frames (BGR).
            
        Returns:
            List of processed frame data.
        """
        results = []
        for frame in frames:
            result = self.process_frame(frame)
            if result is not None:
                results.append(result)
        
        logger.info(f"Processed {len(results)}/{len(frames)} frames successfully")
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Example: Process a single frame from webcam
    print("GAZE Preprocessing Module Example")
    print("=" * 50)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam not available. Skipping example.")
    else:
        preprocessor = DataPreprocessor()
        
        ret, frame = cap.read()
        if ret:
            result = preprocessor.process_frame(frame)
            if result:
                print(f"✓ Successfully processed frame")
                print(f"  - Landmarks shape: {result['landmarks'].shape}")
                print(f"  - Gaze point: {result['gaze_point']}")
                print(f"  - Left eye crop shape: {result['left_eye_crop'].shape}")
                print(f"  - Right eye crop shape: {result['right_eye_crop'].shape}")
            else:
                print("✗ Failed to detect face/landmarks")
        
        cap.release()
