"""
Configuration module for GAZE application.

Contains all global settings, dataset configurations, and model parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
DOCS_DIR = PROJECT_ROOT / "docs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# CAMERA & MEDIAPIPE SETTINGS
# ============================================================================

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# MediaPipe Face Mesh confidence thresholds
MEDIAPIPE_DETECTION_CONFIDENCE = 0.7
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# ============================================================================
# STIMULUS SETTINGS
# ============================================================================

STIMULUS_RADIUS = 15  # pixels
STIMULUS_COLOR = (0, 0, 255)  # BGR: Red
STIMULUS_SPEED = 100  # pixels per second

# Stimulus trajectory modes
STIMULUS_MODES = {
    "linear": "Horizontal linear movement",
    "circular": "Smooth circular motion",
    "random": "Semi-random Brownian motion",
    "static": "Stationary center position",
}

# ============================================================================
# GAZE TRACKING SETTINGS
# ============================================================================

# Eye landmarks (MediaPipe Face Mesh indices)
LEFT_EYE_INDICES = [33, 133, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
RIGHT_EYE_INDICES = [263, 362, 386, 385, 384, 398, 362, 381, 380, 379, 375, 374, 390, 249]

# Face ROI landmarks for attention analysis
FACE_ROIS = {
    "left_eye": [33, 133],  # Eye corners
    "right_eye": [263, 362],
    "nose": [1, 4, 5],  # Nose tip and center
    "mouth": [61, 291, 78, 308],  # Mouth corners
}

# ============================================================================
# FIXATION & SACCADE DETECTION
# ============================================================================

FIXATION_VELOCITY_THRESHOLD = 30  # deg/sec
FIXATION_MIN_DURATION = 0.1  # seconds
SACCADE_MIN_DURATION = 0.01  # seconds

# ============================================================================
# FEATURE EXTRACTION SETTINGS
# ============================================================================

# Sampling window for temporal statistics
FEATURE_WINDOW_SIZE = 30  # seconds
FEATURE_SAMPLING_RATE = 30  # Hz

# Gaze history buffer
GAZE_HISTORY_MAX_SAMPLES = 3000  # ~100 seconds at 30 Hz

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

DATASET_CONFIGS = {
    "mit_gazecapture": {
        "name": "MIT GazeCapture",
        "description": "Large-scale mobile device gaze tracking dataset",
        "file_pattern": "gazecapture_*.csv",
        "features": ["gaze_x", "gaze_y", "head_pose_x", "head_pose_y", "head_pose_z"],
        "label_column": None,  # Unlabeled reference dataset
    },
    "kaggle_asd": {
        "name": "Kaggle ASD Gaze Dataset",
        "description": "ASD vs typically developing gaze patterns",
        "file_pattern": "asd_gaze_*.csv",
        "features": [
            "gaze_x", "gaze_y", "fixation_duration", "saccade_amplitude",
            "eye_aspect_ratio", "pupil_diameter", "blink_rate"
        ],
        "label_column": "diagnosis",  # "ASD" or "TD"
    },
    "asd_comparison": {
        "name": "ASD vs TD Comparison Dataset",
        "description": "Controlled comparison of ASD and typically developing gaze",
        "file_pattern": "asd_td_comparison_*.csv",
        "features": [
            "gaze_x", "gaze_y", "fixation_duration", "saccade_count",
            "gaze_entropy", "roi_attention_eyes", "roi_attention_mouth"
        ],
        "label_column": "group",  # "ASD" or "TD"
    },
}

# ============================================================================
# MODEL SETTINGS
# ============================================================================

MODEL_TYPE = "random_forest"  # Options: "random_forest", "pytorch_neural_net"

MODEL_CONFIGS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "pytorch_neural_net": {
        "input_size": 50,  # Number of input features
        "hidden_sizes": [128, 64, 32],
        "output_size": 1,  # Probability of ASD-associated pattern
        "dropout_rate": 0.3,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 50,
        "device": "cpu",  # Will use cuda if available
    },
}

# ============================================================================
# SCORING & THRESHOLDS
# ============================================================================

# ASD-associated gaze likelihood score thresholds (0-100)
SCORE_THRESHOLDS = {
    "low": (0, 33),  # Low ASD-associated pattern
    "moderate": (33, 67),  # Moderate ASD-associated pattern
    "elevated": (67, 100),  # Elevated ASD-associated pattern
}

# ============================================================================
# UI & VISUALIZATION SETTINGS
# ============================================================================

STREAMLIT_CONFIG = {
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#FF6B6B",
        "backgroundColor": "#F0F2F6",
        "secondaryBackgroundColor": "#E0E6F2",
        "textColor": "#262730",
        "font": "sans serif",
    },
}

# ============================================================================
# LOGGING & DEBUGGING
# ============================================================================

LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
SAVE_INTERMEDIATE_RESULTS = True
DEBUG_MODE = False

# ============================================================================
# ETHICAL & DISCLAIMER SETTINGS
# ============================================================================

ETHICAL_DISCLAIMER = """
**⚠️ IMPORTANT: NON-DIAGNOSTIC RESEARCH TOOL**

This application is a **research prototype** designed for exploratory analysis 
and educational purposes ONLY. It is **NOT** a diagnostic tool and does **NOT** 
diagnose autism spectrum disorder or any medical condition.

**Key Limitations:**
- Gaze patterns alone cannot diagnose ASD
- Results reflect statistical associations in reference populations
- Individual variation is substantial and clinically meaningful
- This tool is not validated for clinical decision-making
- Always consult qualified healthcare professionals for diagnosis

**Responsible Use:**
- Use only for research, education, and exploratory analysis
- Do not make clinical claims based on outputs
- Communicate limitations to any participants
- Maintain data privacy and obtain informed consent
"""

# ============================================================================
# FEATURE IMPORTANCE EXPLANATIONS
# ============================================================================

FEATURE_EXPLANATIONS = {
    "gaze_entropy": "Measure of gaze randomness. Higher entropy may indicate less stable attention.",
    "fixation_duration_mean": "Average duration of fixations. Reduced fixation switching may be ASD-associated.",
    "saccade_count": "Number of rapid eye movements. Saccade patterns differ across groups.",
    "gaze_dispersion": "Spread of gaze points. Excessive dispersion may indicate inattention.",
    "eye_aspect_ratio": "Pupil-to-eye-opening ratio. Used for blink detection.",
    "roi_attention_eyes": "Proportion of time looking at facial eyes. May be reduced in ASD.",
    "roi_attention_mouth": "Proportion of time looking at mouth. May be increased in ASD.",
    "blink_rate": "Frequency of eye blinks. May differ across groups.",
    "pupil_diameter_variability": "Changes in pupil size. Reflects cognitive load and engagement.",
    "gaze_velocity": "Speed of eye movements. Related to attention and stimulus tracking.",
}

# ============================================================================
# REFERENCE POPULATION SETTINGS
# ============================================================================

# Percentile calculation for contextualization
PERCENTILE_BINS = [0, 5, 10, 25, 50, 75, 90, 95, 100]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_config_summary() -> Dict[str, Any]:
    """Get a summary of active configuration."""
    return {
        "camera": {
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fps": CAMERA_FPS,
        },
        "model_type": MODEL_TYPE,
        "datasets_available": list(DATASET_CONFIGS.keys()),
        "project_root": str(PROJECT_ROOT),
        "data_directory": str(DATA_DIR),
    }


if __name__ == "__main__":
    import json

    config_summary = get_config_summary()
    print(json.dumps(config_summary, indent=2))
