# GAZE Research Platform

Research-grade Python application for analyzing gaze patterns in neurodevelopmental research. GAZE enables comprehensive analysis of eye-tracking data for exploratory and educational studies.

## Important Notice

⚠️ **This application is NOT a diagnostic tool.** It does not diagnose autism or any medical condition. Gaze patterns alone cannot determine clinical status. This system is designed for research, educational, and analytical purposes only.

## Features

- **Real-Time Gaze Tracking**: MediaPipe facial landmark detection with iris tracking
- **Comprehensive Metrics**: Fixation duration, saccade analysis, gaze entropy, pupil dynamics, and ROI attention
- **Machine Learning**: Random Forest-based gaze pattern classification with confidence scoring
- **Web Interface**: Interactive Streamlit UI with real-time visualization and stimulus presentation
- **Dataset Integration**: Support for CSV-based data, MIT GazeCapture datasets, and custom formats
- **Privacy-First**: All data processing is local; no external transmission
- **Reproducible Pipeline**: Modular, well-documented code suitable for research environments

## Quick Start

### Installation

```bash
git clone https://github.com/filsan-1/GAZE-.git
cd GAZE-
pip install -r requirements.txt
```

Requires Python 3.8+

### Run Demo

```bash
python main.py
```

Executes the complete pipeline:
1. Dataset generation
2. Data normalization and preprocessing
3. Feature extraction
4. Model training
5. Results analysis

### Launch Web Interface

```bash
streamlit run ui/app.py
```

Open http://localhost:8501

## Project Structure

```
GAZE/
├── src/
│   ├── config.py
│   ├── data_processing/       # Loading, normalization, preprocessing
│   ├── feature_extraction/    # 30+ gaze metrics
│   ├── gaze_tracking/         # MediaPipe, stimulus, rendering
│   └── models/                # Random Forest, ASD scorer
├── ui/app.py                  # Streamlit interface
├── main.py                    # Demo pipeline
├── requirements.txt
├── data/                      # Datasets
├── models/                    # Trained models
└── results/                   # Outputs
```

## Gaze Metrics

The system extracts 30+ metrics including:
- Fixation: duration, count, stability
- Saccade: count, amplitude, velocity
- Entropy: dispersion, randomness
- ROI Attention: eyes, mouth, nose distribution
- Eye Dynamics: aspect ratio, blink rate, asymmetry

## Model

- **Algorithm**: Random Forest Classifier
- **Output**: ASD-like likelihood score (0-100%) with confidence and percentile rank
- **Training**: Synthetic ASD vs typically developing comparison data

## Ethical Framework

✅ **Non-Diagnostic** - Research tool only
✅ **Privacy-Preserving** - Local data processing
✅ **Transparent** - Feature importance explanations
✅ **Inclusive** - Acknowledges dataset bias limitations

## Limitations

- Dataset bias; may not generalize to all populations
- Environmental factors affect measurements
- High natural variation in gaze patterns
- Cannot replace clinical assessment
- Trained on synthetic reference data

## Citation

```bibtex
@software{gaze_research_2025,
  title={GAZE: Research-Grade Gaze Pattern Analysis Platform},
  year={2025},
  url={https://github.com/filsan-1/GAZE-}
}
```

---

**Version**: 1.0.0 | **Status**: Research Prototype | **License**: MIT
