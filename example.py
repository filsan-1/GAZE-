"""
Comprehensive example demonstrating GAZE Research Platform usage.

Shows end-to-end workflow: preprocessing, feature extraction, training, and evaluation.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Setup paths
GAZE_DIR = Path(__file__).parent
sys.path.insert(0, str(GAZE_DIR))

from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR,
    HANDCRAFTED_FEATURES, CLASS_LABELS,
)
from src.preprocessing import MediaPipeFaceDetector, DataPreprocessor
from src.feature_extraction import HandcraftedGazeFeatures, FeatureExtractor
from src.model import RandomForestGazeModel, NeuralGazeClassifier
from src.train import DatasetManager, ModelTrainer, ExperimentLogger
from src.utils import DataUtils, VisualizationUtils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def example_1_preprocessing():
    """Example 1: Face detection and preprocessing from webcam."""
    print("\n" + "="*70)
    print("EXAMPLE 1: PREPROCESSING - Face Detection & Gaze Extraction")
    print("="*70)
    
    import cv2
    
    preprocessor = DataPreprocessor()
    logger.info("DataPreprocessor initialized")
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.warning("Webcam not available - skipping live example")
        return
    
    print("\nCapturing 10 frames from webcam...")
    frames_processed = 0
    
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result = preprocessor.process_frame(frame)
        
        if result is not None:
            print(f"  ✓ Frame {_+1} processed successfully")
            print(f"    - Landmarks: {result['landmarks'].shape}")
            print(f"    - Gaze point: {result['gaze_point']}")
            print(f"    - Normalized gaze: {result.get('gaze_point_normalized', 'N/A')}")
            print(f"    - Left eye crop: {result['left_eye_crop'].shape}")
            print(f"    - Right eye crop: {result['right_eye_crop'].shape}")
            frames_processed += 1
    
    cap.release()
    print(f"\n✓ Successfully processed {frames_processed} frames")


def example_2_feature_extraction():
    """Example 2: Feature extraction from synthetic gaze data."""
    print("\n" + "="*70)
    print("EXAMPLE 2: FEATURE EXTRACTION - Handcrafted Metrics")
    print("="*70)
    
    # Generate synthetic gaze trajectory
    print("\nGenerating synthetic gaze trajectory...")
    t = np.linspace(0, 4*np.pi, 300)
    gaze_x = 500 + 300 * np.cos(t) + np.random.randn(300) * 20
    gaze_y = 300 + 200 * np.sin(t) + np.random.randn(300) * 15
    gaze_points = np.column_stack([gaze_x, gaze_y])
    
    # Extract handcrafted features
    extractor = HandcraftedGazeFeatures(sampling_rate=30)
    
    print("\nExtracting metrics...")
    fixation_metrics = extractor.compute_fixation_metrics(gaze_points)
    print(f"  Fixation metrics: {fixation_metrics}")
    
    saccade_metrics = extractor.compute_saccade_metrics(gaze_points)
    print(f"  Saccade metrics: {saccade_metrics}")
    
    entropy = extractor.compute_gaze_entropy(gaze_points)
    print(f"  Gaze entropy: {entropy:.3f}")
    
    velocity = extractor.compute_velocity_metrics(gaze_points)
    print(f"  Velocity metrics: {velocity}")
    
    print("\n✓ Feature extraction completed")


def example_3_model_training():
    """Example 3: Training Random Forest and Neural Network models."""
    print("\n" + "="*70)
    print("EXAMPLE 3: MODEL TRAINING - Random Forest & Neural Network")
    print("="*70)
    
    # Generate synthetic dataset
    print("\nGenerating synthetic ASD vs TD dataset...")
    np.random.seed(42)
    
    # TD samples (label 0)
    X_td = np.random.randn(150, 17)
    y_td = np.zeros(150)
    
    # ASD samples (label 1) - shifted distribution
    X_asd = np.random.randn(150, 17) + 0.5
    y_asd = np.ones(150)
    
    X = np.vstack([X_td, X_asd])
    y = np.hstack([y_td, y_asd])
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Train Random Forest
    print("\n1. Training Random Forest Classifier...")
    trainer_rf = ModelTrainer(model_type="random_forest")
    trainer_rf.train_random_forest(X_train, y_train, X_val, y_val)
    metrics_rf = trainer_rf.evaluate(X_test, y_test)
    
    print(f"  Results:")
    for metric, value in metrics_rf.items():
        print(f"    {metric}: {value:.3f}")
    
    # Train Neural Network
    print("\n2. Training Neural Network Classifier...")
    trainer_nn = ModelTrainer(model_type="neural_network")
    trainer_nn.train_neural_network(
        X_train, y_train, X_val, y_val,
        epochs=50, batch_size=16, patience=10
    )
    metrics_nn = trainer_nn.evaluate(X_test, y_test)
    
    print(f"  Results:")
    for metric, value in metrics_nn.items():
        print(f"    {metric}: {value:.3f}")
    
    # Save models
    print("\n3. Saving models...")
    trainer_rf.save_model(MODELS_DIR / "rf_gaze_model.pkl")
    trainer_nn.save_model(MODELS_DIR / "nn_gaze_model.pt")
    print("  ✓ Models saved")
    
    return trainer_rf, trainer_nn, X_test, y_test


def example_4_session_analysis():
    """Example 4: End-to-end session analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 4: SESSION ANALYSIS - Complete Workflow")
    print("="*70)
    
    # Create synthetic gaze session data
    print("\nSimulating gaze tracking session...")
    n_frames = 300
    
    # Gaze trajectory (normalized coordinates)
    t = np.linspace(0, 4*np.pi, n_frames)
    gaze_x_norm = 0.5 + 0.3 * np.cos(t) + np.random.randn(n_frames) * 0.05
    gaze_y_norm = 0.5 + 0.3 * np.sin(t) + np.random.randn(n_frames) * 0.05
    gaze_points_norm = np.column_stack([gaze_x_norm, gaze_y_norm])
    
    # Feature extraction
    print("Extracting features...")
    feature_extractor = HandcraftedGazeFeatures(sampling_rate=30)
    
    # Convert to pixels
    gaze_points_px = gaze_points_norm * np.array([1920, 1080])
    
    # Compute all metrics
    features = {
        **feature_extractor.compute_fixation_metrics(gaze_points_px),
        **feature_extractor.compute_saccade_metrics(gaze_points_px),
        "gaze_entropy": feature_extractor.compute_gaze_entropy(gaze_points_px),
        **feature_extractor.compute_velocity_metrics(gaze_points_px),
    }
    
    print(f"\nExtracted Features:")
    for name, value in features.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.3f}" if isinstance(value, float) else f"  {name}: {value}")
    
    # Create feature vector for classification
    feature_vector = np.array([
        features.get("fixation_duration_mean", 0),
        features.get("fixation_duration_std", 0),
        features.get("fixation_count", 0),
        features.get("gaze_entropy", 0),
        features.get("saccade_amplitude_mean", 0),
        features.get("saccade_amplitude_std", 0),
        features.get("saccade_velocity_mean", 0),
        features.get("saccade_velocity_std", 0),
        features.get("saccade_count", 0),
        features.get("saccade_frequency", 0),
        features.get("blink_rate", 0),
        features.get("left_right_asymmetry", 0),
        features.get("gaze_velocity_mean", 0),
        features.get("gaze_velocity_std", 0),
        0, 0, 0,  # Padding to match training features
    ]).reshape(1, -1)
    
    # Scoring with trained model
    print("\nClassifying gaze pattern...")
    try:
        trainer_rf, _, _, _ = example_3_model_training()
        probabilities = trainer_rf.model.predict_proba(
            trainer_rf.model.scaler.transform(feature_vector)
        )
        
        td_prob = probabilities[0, 0] * 100
        asd_prob = probabilities[0, 1] * 100
        
        print(f"  TD probability: {td_prob:.1f}%")
        print(f"  ASD probability: {asd_prob:.1f}%")
        
        # Determine risk tier
        if asd_prob < 33:
            tier = "LOW"
        elif asd_prob < 67:
            tier = "MODERATE"
        else:
            tier = "ELEVATED"
        
        print(f"  Risk tier: {tier}")
        
    except Exception as e:
        logger.warning(f"Classification skipped: {e}")
    
    # Generate report
    print("\nGenerating report...")
    vis_utils = VisualizationUtils()
    report = vis_utils.create_summary_report(
        {
            "fixation_duration_mean": features.get("fixation_duration_mean", 0),
            "saccade_count": features.get("saccade_count", 0),
            "gaze_entropy": features.get("gaze_entropy", 0),
            "asd_probability": asd_prob,
            "confidence": 75.0,
            "risk_tier": tier,
        },
        {
            "duration": "10.0s",
            "num_frames": n_frames,
            "sampling_rate": 30,
        }
    )
    
    # Save report
    report_path = LOGS_DIR / "example_session_report.txt"
    vis_utils.save_report(report, report_path)
    print(f"  ✓ Report saved to {report_path}")


def main():
    """Run all examples."""
    print("\n" + "="*70)
    print("GAZE RESEARCH PLATFORM - COMPREHENSIVE EXAMPLES")
    print("="*70)
    
    try:
        # Example 1: Preprocessing
        example_1_preprocessing()
        
        # Example 2: Feature Extraction
        example_2_feature_extraction()
        
        # Example 3: Model Training
        example_3_model_training()
        
        # Example 4: Session Analysis
        example_4_session_analysis()
        
        print("\n" + "="*70)
        print("✓ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Example failed: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
