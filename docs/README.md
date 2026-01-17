# Comprehensive Documentation for GAZE Research Platform

## Overview

**GAZE** is a research-grade Python application for analyzing gaze patterns associated with neurodevelopmental differences. This system is **explicitly non-diagnostic** and designed for research, education, and exploratory analysis only.

### Key Disclaimer

⚠️ **CRITICAL**: This application is **NOT** a diagnostic tool. It does not diagnose autism spectrum disorder (ASD) or any medical condition. Gaze patterns alone cannot diagnose neurodevelopmental conditions. Always consult qualified healthcare professionals for clinical assessment.

---

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [Architecture Overview](#architecture-overview)
3. [Core Components](#core-components)
4. [Feature Extraction](#feature-extraction)
5. [Model Training](#model-training)
6. [Usage Guide](#usage-guide)
7. [Dataset Integration](#dataset-integration)
8. [Ethical Considerations](#ethical-considerations)
9. [Limitations & Caveats](#limitations--caveats)
10. [API Reference](#api-reference)

---

## Installation & Setup

### Requirements

- Python 3.8+
- OpenCV 4.5+
- MediaPipe 0.10+
- scikit-learn 1.0+
- PyTorch 2.0+ (optional, for neural network models)
- Streamlit 1.20+

### Installation

```bash
# Clone repository
git clone https://github.com/filsan-1/GAZE-.git
cd GAZE-

# Install dependencies
pip install -r requirements.txt

# Run Streamlit UI
streamlit run ui/app.py

# Or run command-line analysis
python -m src.models.random_forest_model
```

---

## Architecture Overview

```
GAZE/
├── src/
│   ├── config.py                    # Global configuration
│   ├── data_processing/             # Data ingestion & preprocessing
│   │   ├── dataset_loader.py
│   │   ├── normalizer.py
│   │   └── preprocessor.py
│   ├── gaze_tracking/               # Real-time gaze detection
│   │   ├── mediapipe_detector.py
│   │   ├── gaze_renderer.py
│   │   └── stimulus_generator.py
│   ├── feature_extraction/          # Feature computation
│   │   ├── feature_extractor.py
│   │   └── roi_analyzer.py
│   └── models/                      # ML models & scoring
│       ├── random_forest_model.py
│       └── asd_scorer.py
├── ui/                              # Streamlit web interface
│   └── app.py
├── data/                            # Data storage
│   ├── raw/                         # Raw dataset files
│   └── processed/                   # Preprocessed data
├── models/                          # Trained model files
├── results/                         # Analysis outputs
└── docs/                            # Documentation
```

---

## Core Components

### 1. Configuration (`src/config.py`)

Central configuration for all parameters:

```python
from src.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT,
    FIXATION_VELOCITY_THRESHOLD,
    STIMULUS_MODES,
    DATASET_CONFIGS,
    FEATURE_EXPLANATIONS,
)
```

**Key settings:**
- Camera resolution (1280×720)
- MediaPipe confidence thresholds
- Fixation detection parameters
- ROI definitions
- Model hyperparameters

### 2. Data Processing Pipeline

#### Dataset Loader
```python
from src.data_processing import DatasetLoader

loader = DatasetLoader()

# Create synthetic datasets for demo
mit_df = loader.create_synthetic_mit_gazecapture(num_samples=1000)
asd_df = loader.create_synthetic_asd_comparison(
    num_asd_samples=500,
    num_td_samples=500
)

# Load custom CSV
df = loader.load_csv_dataset(
    "path/to/data.csv",
    "my_dataset",
    label_column="diagnosis"
)

# Merge datasets
merged = loader.merge_datasets(
    ["dataset1", "dataset2"],
    output_name="combined"
)
```

#### Data Normalizer
```python
from src.data_processing import GazeDataNormalizer

normalizer = GazeDataNormalizer(method="standard")

normalized_df = normalizer.normalize_full_pipeline(
    df,
    screen_width=1280,
    screen_height=720,
    remove_outliers=True,
    standardize=True
)
```

#### Data Preprocessor
```python
from src.data_processing import GazeDataPreprocessor

preprocessor = GazeDataPreprocessor(sampling_rate=30)

processed_df = preprocessor.preprocess_pipeline(
    df,
    resample=True,
    target_rate=30,
    compute_velocities=True,
    validate=True
)
```

### 3. Real-Time Gaze Tracking

#### MediaPipe Detector
```python
from src.gaze_tracking import MediaPipeDetector

detector = MediaPipeDetector(
    detection_confidence=0.7,
    tracking_confidence=0.5
)

# Process frame
landmarks = detector.detect_landmarks(frame)

if landmarks:
    # Get gaze point
    gaze_x, gaze_y = detector.estimate_gaze_point(
        landmarks["landmarks"],
        frame_width=1280,
        frame_height=720
    )

    # Get head pose
    head_pose = detector.get_head_pose(landmarks["landmarks"])

    # Compute eye aspect ratio (for blink detection)
    ear_left = detector.compute_eye_aspect_ratio(
        landmarks["landmarks"],
        is_right=False
    )

detector.release()
```

#### Gaze Renderer
```python
from src.gaze_tracking import GazeRenderer

renderer = GazeRenderer(max_trail_length=30)

# Render various overlays
frame = renderer.render_gaze_point(frame, gaze_point, radius=15)
frame = renderer.render_gaze_vector(frame, landmarks, gaze_point)
frame = renderer.render_gaze_trail(frame, gaze_point)
frame = renderer.render_fixation_point(frame, fixation_point, duration=0.5)
frame = renderer.render_stimulus_target(frame, stimulus_pos, radius=15)
```

#### Stimulus Generator
```python
from src.gaze_tracking import StimulusGenerator

stimulus = StimulusGenerator(
    screen_width=1280,
    screen_height=720,
    speed=100
)

# Get position for different trajectory modes
pos_linear = stimulus.get_position(mode="linear", t=elapsed_time)
pos_circular = stimulus.get_position(mode="circular", t=elapsed_time)
pos_random = stimulus.get_position(mode="random", dt=time_delta)
pos_static = stimulus.get_position(mode="static")
```

---

## Feature Extraction

### Gaze Features

#### Fixations
- **Definition**: Periods where gaze velocity is below threshold (~30 deg/sec)
- **Metrics**: Duration, count per minute, spatial stability
- **ASD Association**: May show longer, less frequent fixations

```python
fixations = extractor.compute_fixations(
    gaze_points,
    velocities=None,
    velocity_threshold=30,  # deg/sec
    min_duration=0.1  # seconds
)
```

#### Saccades
- **Definition**: Rapid eye movements between fixations
- **Metrics**: Amplitude (degrees), velocity, count
- **ASD Association**: May show reduced saccade count and velocity

```python
saccades = extractor.compute_saccades(
    gaze_points,
    velocity_threshold=30,
    min_duration=0.01
)
```

#### Gaze Entropy
- **Definition**: Randomness/disorder of gaze distribution
- **Range**: 0 (focused) to log(N) (random)
- **ASD Association**: May show lower entropy (more focused, less variable gaze)

```python
entropy = extractor.compute_gaze_entropy(gaze_points, bins=20)
```

#### Gaze Dispersion
- **Definition**: Spread of gaze points around centroid
- **Metric**: Variance of distances from gaze center
- **ASD Association**: May show excessive dispersion (less stable attention)

```python
dispersion = extractor.compute_gaze_dispersion(gaze_points)
```

#### Eye Aspect Ratio (EAR)
- **Definition**: Ratio of eye opening to width
- **Use**: Blink detection (EAR < 0.2 indicates closed eye)
- **Metrics**: Mean EAR, blink rate, blink duration

```python
ear_left = detector.compute_eye_aspect_ratio(landmarks, is_right=False)
ear_right = detector.compute_eye_aspect_ratio(landmarks, is_right=True)
```

#### ROI Attention Distribution
- **Definition**: Proportion of time looking at facial regions
- **ROIs**: Eyes, nose, mouth, off-face
- **ASD Association**: May show reduced eye region gaze, increased mouth gaze

```python
roi_analyzer = ROIAnalyzer()
roi_attention = roi_analyzer.compute_roi_attention_distribution(gaze_points)
# Returns: {'eyes': 0.35, 'mouth': 0.25, 'nose': 0.15, 'off_face': 0.25}
```

### Comprehensive Feature Extraction

```python
from src.feature_extraction import GazeFeatureExtractor

extractor = GazeFeatureExtractor(sampling_rate=30)

features = extractor.extract_all_features(
    gaze_points,
    ear_values=ear_array,
    roi_regions={
        "eyes": (0.25, 0.25, 0.75, 0.45),
        "mouth": (0.25, 0.55, 0.75, 0.80),
    }
)

# Returns 30+ metrics:
# - fixation_count, fixation_duration_mean, fixation_duration_max
# - saccade_count, saccade_amplitude_mean, saccade_velocity_mean
# - gaze_entropy, gaze_dispersion
# - gaze_x_mean, gaze_y_mean, gaze_x_std, gaze_y_std
# - gaze_velocity_mean, gaze_velocity_std
# - ear_mean, ear_std, blink_count
# - roi_attention_eyes, roi_attention_mouth, etc.
```

---

## Model Training

### Random Forest Classifier

```python
from src.models import RandomForestGazeModel
import numpy as np

# Prepare training data
X_features = np.array([...])  # shape: (n_samples, n_features)
y_labels = np.array([...])     # 0=TD, 1=ASD

# Train model
model = RandomForestGazeModel(
    n_estimators=100,
    max_depth=15,
    random_state=42
)

metrics = model.train(
    X_features,
    y_labels,
    feature_names=feature_column_names,
    test_size=0.2
)

print(metrics)
# {'train_accuracy': 0.82, 'test_accuracy': 0.79, 'auc': 0.85}

# Save model
model.save("models/asd_gaze_model.pkl")

# Load model
model.load("models/asd_gaze_model.pkl")

# Predict ASD likelihood (0-100)
test_features = np.array([...])
scores = model.predict_asd_likelihood(test_features)
# returns: [45.3, 72.1, 28.9, ...]

# Feature importance
importance = model.get_feature_importance(top_n=10)
# {'fixation_duration_mean': 0.15, 'gaze_entropy': 0.12, ...}
```

### ASD Likelihood Scoring

```python
from src.models import ASDLikelihoodScorer

# Create scorer with reference population
reference_scores = np.random.beta(2, 5, 1000) * 100
scorer = ASDLikelihoodScorer(reference_scores=reference_scores)

# Score interpretation
score = 65.3

percentile = scorer.score_to_percentile(score)
# 75.3 (higher than 75% of reference population)

tier = scorer.score_to_tier(score)
# {'tier': 'moderate', 'description': '...', 'score_range': '33-67'}

confidence = scorer.compute_confidence(score)
# 42.1 (lower confidence near decision boundaries)

# Full report
report = scorer.generate_report(
    score=score,
    features=feature_dict,
    feature_importance=importance_dict,
    percentile=percentile
)

print(report)
```

---

## Usage Guide

### Complete Analysis Pipeline

```python
import numpy as np
from src.data_processing import DatasetLoader, GazeDataNormalizer, GazeDataPreprocessor
from src.feature_extraction import GazeFeatureExtractor, ROIAnalyzer
from src.models import RandomForestGazeModel, ASDLikelihoodScorer

# 1. Load datasets
loader = DatasetLoader()
asd_df = loader.create_synthetic_asd_comparison(
    num_asd_samples=300,
    num_td_samples=300
)

# 2. Normalize data
normalizer = GazeDataNormalizer()
normalized_df = normalizer.normalize_full_pipeline(asd_df)

# 3. Preprocess data
preprocessor = GazeDataPreprocessor(sampling_rate=30)
processed_df = preprocessor.preprocess_pipeline(normalized_df)

# 4. Extract features
feature_cols = [
    "fixation_duration_mean", "saccade_count_per_min",
    "gaze_entropy", "roi_attention_eyes", "roi_attention_mouth",
    "blink_rate", "eye_aspect_ratio_mean"
]

X = processed_df[feature_cols].values
y = (processed_df["group"] == "ASD").astype(int).values

# 5. Train model
model = RandomForestGazeModel()
metrics = model.train(X, y, feature_names=feature_cols)

# 6. Score new samples
new_sample = X[0:1]
score = model.predict_asd_likelihood(new_sample)[0]
importance = model.get_feature_importance(top_n=10)

# 7. Interpret results
scorer = ASDLikelihoodScorer(reference_scores=X[:, 0])  # Use first feature as reference
percentile = scorer.score_to_percentile(score)
tier = scorer.score_to_tier(score)
report = scorer.generate_report(
    score, {"feature": value}, importance, percentile
)

print(f"Score: {score:.1f}%")
print(f"Percentile: {percentile:.1f}th")
print(f"Tier: {tier['tier']}")
print(report)
```

### Streamlit Web Interface

```bash
streamlit run ui/app.py
```

Navigate to `http://localhost:8501`

---

## Dataset Integration

### MIT GazeCapture

```python
# Create synthetic MIT GazeCapture-like data
mit_df = loader.create_synthetic_mit_gazecapture(
    num_samples=10000,
    num_subjects=100
)

# Columns: subject_id, gaze_x, gaze_y, head_pose_x/y/z,
#          screen_width, screen_height, timestamp
```

### ASD Comparison Datasets

```python
# Create synthetic ASD vs TD comparison data
asd_df = loader.create_synthetic_asd_comparison(
    num_asd_samples=500,
    num_td_samples=500
)

# Key features:
# - fixation_duration_mean: ASD ~0.35s, TD ~0.28s
# - saccade_count_per_min: ASD ~25, TD ~35
# - gaze_entropy: ASD ~2.1, TD ~2.8
# - roi_attention_eyes: ASD ~0.25, TD ~0.42
# - roi_attention_mouth: ASD ~0.35, TD ~0.18
```

### Custom CSV Import

```python
# Load custom dataset
custom_df = loader.load_csv_dataset(
    "path/to/custom_data.csv",
    "my_dataset",
    label_column="diagnosis",  # Optional
    required_columns=["gaze_x", "gaze_y", "timestamp"]
)
```

---

## Ethical Considerations

### Non-Diagnostic Framework

This system operates under strict ethical guidelines:

1. **No Clinical Claims**: Does not diagnose or clinically assess ASD
2. **Statistical Pattern Analysis**: Identifies behavioral pattern associations only
3. **Research Context**: Suitable for exploratory research and education
4. **Individual Variation**: Acknowledges substantial variation within groups
5. **Informed Consent**: Requires explicit participant consent for data collection

### Bias and Limitations

⚠️ **Known Limitations:**

- **Dataset Bias**: Training data may not represent diverse populations
- **Confounding Factors**: Gaze patterns influenced by attention, fatigue, task, etc.
- **Clinical Validity**: Not validated for diagnostic use
- **Generalization**: Limited generalization across ages, cultures, neurotypes
- **Model Uncertainty**: 20-25% error rate on test data

### Privacy

- ✅ All data stored locally
- ✅ No external communication required
- ✅ Complete local control over data
- ✅ Easy data deletion and privacy management

---

## Limitations & Caveats

### Technical Limitations

1. **Camera Dependency**: Requires frontal face visibility
2. **Lighting**: Performance degrades in poor lighting
3. **Glasses/Contacts**: May interfere with iris detection
4. **Head Movement**: High head movement reduces accuracy
5. **Sample Size**: Models trained on limited synthetic data

### Methodological Caveats

1. **Group Overlap**: Substantial overlap between ASD and TD distributions
2. **Heterogeneity**: ASD is heterogeneous; gaze patterns vary widely
3. **Confounds**: Attention, fatigue, task engagement affect gaze
4. **Development**: Gaze patterns change with age and development
5. **Context**: Stimulus context significantly influences gaze behavior

### Statistical Notes

- Feature importance estimates may be unstable with small samples
- Percentile rankings depend on reference population characteristics
- Confidence intervals not reported; all outputs are point estimates
- No significance testing or p-values provided

---

## API Reference

### MediaPipeDetector

```python
detector = MediaPipeDetector(detection_confidence=0.7, tracking_confidence=0.5)

# Detect landmarks
result = detector.detect_landmarks(frame)
# Returns: {'landmarks': (468, 3), 'left_eye': (14, 3), 'right_eye': (14, 3),
#           'iris_left': (4, 3), 'iris_right': (4, 3), 'face_detected': True}

# Estimate gaze
gaze_x, gaze_y = detector.estimate_gaze_point(landmarks, frame_width, frame_height)

# Head pose
pose = detector.get_head_pose(landmarks)  # Returns: {pitch, yaw, roll}

# Eye aspect ratio
ear = detector.compute_eye_aspect_ratio(landmarks, is_right=False)

# Blink detection
is_blink = detector.is_blink(landmarks, threshold=0.2)

detector.release()
```

### GazeFeatureExtractor

```python
extractor = GazeFeatureExtractor(sampling_rate=30)

# Fixations
fixations = extractor.compute_fixations(gaze_points, velocities, threshold, duration)

# Saccades
saccades = extractor.compute_saccades(gaze_points, velocities, threshold, duration)

# Entropy
entropy = extractor.compute_gaze_entropy(gaze_points, bins=20)

# Dispersion
dispersion = extractor.compute_gaze_dispersion(gaze_points)

# ROI attention
roi_attn = extractor.compute_roi_attention(gaze_points, roi_regions)

# All features
all_features = extractor.extract_all_features(gaze_points, ear_values, roi_regions)
```

### RandomForestGazeModel

```python
model = RandomForestGazeModel(n_estimators=100, max_depth=15)

# Training
metrics = model.train(X, y, feature_names, test_size=0.2)

# Prediction
scores = model.predict_asd_likelihood(X)  # 0-100 scale
probabilities = model.predict_proba(X)    # 0-1 scale

# Importance
importance = model.get_feature_importance(top_n=10)

# Persistence
model.save("path/to/model.pkl")
model.load("path/to/model.pkl")
```

### ASDLikelihoodScorer

```python
scorer = ASDLikelihoodScorer(reference_scores=scores_array)

# Percentile
percentile = scorer.score_to_percentile(score)

# Tier classification
tier = scorer.score_to_tier(score)

# Confidence
confidence = scorer.compute_confidence(score, model_uncertainty=None)

# Interpretation
interp = scorer.generate_interpretation(score, percentile, importance)

# Full report
report = scorer.generate_report(score, features, importance, percentile)
```

---

## Citation

If you use this platform for research or education, please cite:

```bibtex
@software{gaze_research_2025,
  title={GAZE: Research-Grade Gaze Pattern Analysis Platform},
  author={Research Team},
  year={2025},
  url={https://github.com/filsan-1/GAZE-}
}
```

---

## Support & Contact

For questions, issues, or contributions:

- 📧 Email: [contact details]
- 📝 Issues: GitHub Issues
- 🤝 Contributions: Pull Requests welcome

---

## License

MIT License - See LICENSE file for details

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Status**: Research Prototype
