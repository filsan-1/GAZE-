"""
Configuration file for GAZE Research Platform.

Centralized settings for paths, hyperparameters, and constants.
"""

from pathlib import Path
from typing import Dict, List

# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
SUPPORTED_DATASETS = ["MIT_GazeCapture", "Kaggle_ASD", "Custom_CSV"]
DEFAULT_DATASET = "Custom_CSV"

# ============================================================================
# CAMERA AND PREPROCESSING SETTINGS
# ============================================================================
CAMERA_RESOLUTION = (640, 480)  # (width, height)
CAMERA_FPS = 30
EYE_CROP_SIZE = (224, 224)  # Input size for CNN
FACE_DETECTION_THRESHOLD = 0.5

# MediaPipe face mesh landmarks
NUM_FACE_LANDMARKS = 468
LEFT_EYE_INDICES = list(range(133, 145))  # Left eye landmarks in FaceMesh
RIGHT_EYE_INDICES = list(range(362, 374))  # Right eye landmarks in FaceMesh

# ============================================================================
# FEATURE EXTRACTION SETTINGS
# ============================================================================
HANDCRAFTED_FEATURES = [
    "fixation_duration_mean",
    "fixation_duration_std",
    "fixation_count",
    "gaze_dispersion",
    "gaze_entropy",
    "saccade_amplitude_mean",
    "saccade_amplitude_std",
    "saccade_velocity_mean",
    "saccade_velocity_std",
    "saccade_count",
    "saccade_frequency",
    "blink_rate",
    "blink_duration_mean",
    "left_right_asymmetry",
    "gaze_velocity_mean",
    "gaze_velocity_std",
    "smooth_pursuit_accuracy",
]

# CNN embedding extraction
CNN_MODEL_NAME = "mobilenet_v2"  # Transfer learning backbone
CNN_EMBEDDING_DIM = 1280  # MobileNetV2 output dimension
FREEZE_CNN_BACKBONE = True  # Transfer learning

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================
# Random Forest baseline
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE = 42

# Neural network classifier
NN_HIDDEN_DIMS = [256, 128, 64]
NN_DROPOUT = 0.3
NN_LEARNING_RATE = 0.001
NN_BATCH_SIZE = 32
NN_EPOCHS = 100
NN_EARLY_STOPPING_PATIENCE = 10
NN_RANDOM_STATE = 42

# Training general
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
CLASS_WEIGHT_BALANCE = True  # Handle class imbalance
RANDOM_STATE = 42

# ============================================================================
# EVALUATION METRICS
# ============================================================================
EVAL_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "confusion_matrix",
]

# ============================================================================
# WEBCAM AND UI SETTINGS
# ============================================================================
STREAMLIT_THEME = "light"
MAX_TRACKING_DURATION = 300  # seconds
STIMULUS_SPEED = 2  # pixels per frame
STIMULUS_RADIUS = 15  # pixels
STIMULUS_COLOR = (0, 0, 255)  # BGR: red

# ============================================================================
# ETHICAL AND DISCLAIMER SETTINGS
# ============================================================================
ETHICAL_DISCLAIMER = """
⚠️ **CRITICAL DISCLAIMER**

This application is **NOT a diagnostic tool** and cannot diagnose autism or any 
medical condition. Gaze patterns alone are insufficient for clinical assessment.

This system is designed for:
- Research exploration
- Educational demonstration
- Academic study only

**This is not a substitute for professional clinical evaluation.**
Always consult qualified healthcare professionals for diagnosis.

By using this tool, you acknowledge that:
1. Results are probabilistic patterns, not diagnoses
2. Results should NOT influence clinical decisions
3. Data is processed locally for privacy protection
"""

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "gaze_research.log"

# ============================================================================
# DATASET STATISTICS (Reference for normalization)
# ============================================================================
DATASET_NORMALIZATION_PARAMS = {
    "gaze_x_mean": 0.5,
    "gaze_x_std": 0.15,
    "gaze_y_mean": 0.5,
    "gaze_y_std": 0.15,
    "pupil_diameter_mean": 4.0,
    "pupil_diameter_std": 1.0,
}

# ============================================================================
# CLASS LABELS
# ============================================================================
CLASS_LABELS = {
    0: "Typical Development (TD)",
    1: "Autism Spectrum Disorder (ASD)",
}

CLASS_NAMES = ["TD", "ASD"]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def get_config_summary() -> Dict:
    """Get a summary of key configuration parameters."""
    return {
        "project_root": str(PROJECT_ROOT),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "camera_resolution": CAMERA_RESOLUTION,
        "eye_crop_size": EYE_CROP_SIZE,
        "num_handcrafted_features": len(HANDCRAFTED_FEATURES),
        "cnn_embedding_dim": CNN_EMBEDDING_DIM,
        "nn_hidden_dims": NN_HIDDEN_DIMS,
        "train_test_split": TRAIN_TEST_SPLIT,
        "class_labels": CLASS_LABELS,
    }
