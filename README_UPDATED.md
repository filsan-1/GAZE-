# GAZE Research Platform

## 👁️ Research-Grade Gaze Pattern Analysis for Neurodevelopmental Research

GAZE is a comprehensive Python application for analyzing gaze patterns associated with autism spectrum disorder (ASD) and other neurodevelopmental differences. **This system is explicitly non-diagnostic** and designed for research, education, and exploratory analysis only.

### 🎯 Key Features

- **Real-Time Gaze Tracking**: MediaPipe-based facial landmark detection and iris tracking
- **Comprehensive Features**: 30+ gaze metrics including fixations, saccades, entropy, and ROI attention
- **Machine Learning**: Random Forest classifier for pattern recognition and ASD-associated gaze likelihood scoring
- **Interactive UI**: Streamlit-based web interface with real-time visualization
- **Multi-Dataset Support**: Integrates MIT GazeCapture, Kaggle ASD datasets, and custom CSV data
- **Ethical Framework**: Non-diagnostic, privacy-preserving, locally-stored data
- **Reproducible**: Fully documented, modular, well-tested code

### ⚠️ Critical Disclaimer

**This application is NOT a diagnostic tool.** It does not diagnose autism or any medical condition. Gaze patterns alone cannot determine autism status. Always consult qualified healthcare professionals for clinical assessment.

---

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Usage Examples](#usage-examples)
- [Core Components](#core-components)
- [Ethical Considerations](#ethical-considerations)
- [Limitations](#limitations)
- [Citation](#citation)

---

## 💾 Installation

### Requirements

- Python 3.8+
- pip or conda
- ~500MB disk space for dependencies

### Setup

```bash
# Clone repository
git clone https://github.com/filsan-1/GAZE-.git
cd GAZE-

# Install dependencies
pip install -r requirements.txt

# Or use conda
conda create -n gaze python=3.14
conda activate gaze
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Run Demo Pipeline

```bash
python main.py
```

This demonstrates the complete analysis workflow:
- Dataset generation
- Data normalization
- Feature extraction
- Model training
- Result interpretation

### 2. Launch Web Interface

```bash
streamlit run ui/app.py
```

Open browser to `http://localhost:8501`

### 3. Load Your Data

```python
from src.data_processing import DatasetLoader

loader = DatasetLoader()
df = loader.load_csv_dataset(
    "path/to/your/data.csv",
    "my_dataset",
    label_column="diagnosis"
)
```

---

## 🏗️ Architecture

```
GAZE/
├── src/                              # Core modules
│   ├── config.py                     # Central configuration
│   ├── data_processing/              # Data handling
│   │   ├── dataset_loader.py
│   │   ├── normalizer.py
│   │   └── preprocessor.py
│   ├── gaze_tracking/                # Real-time gaze
│   │   ├── mediapipe_detector.py
│   │   ├── gaze_renderer.py
│   │   └── stimulus_generator.py
│   ├── feature_extraction/           # 30+ metrics
│   │   ├── feature_extractor.py
│   │   └── roi_analyzer.py
│   └── models/                       # ML pipeline
│       ├── random_forest_model.py
│       └── asd_scorer.py
├── ui/app.py                         # Streamlit interface
├── main.py                           # Demo pipeline
├── requirements.txt                  # Dependencies
├── setup.py                          # Installation
├── data/                             # Datasets
├── models/                           # Model checkpoints
├── results/                          # Outputs
└── docs/README.md                    # Full documentation
```

---

## 📖 Full Documentation

See [docs/README.md](docs/README.md) for:
- Complete API reference
- Feature explanations
- Usage examples
- Model details
- Ethical guidelines

---

## ⚠️ Ethical Safeguards

✅ **Non-Diagnostic**
- For research and education only
- Does not diagnose autism

✅ **Privacy-Preserving**
- All data stored locally
- No external uploads

✅ **Transparent**
- Feature importance explanations
- Documented limitations

---

## 📊 Performance

- **Test Accuracy**: ~79%
- **AUC-ROC**: ~0.85
- **Training Samples**: 600 (synthetic)
- **Features**: 30+

---

## 🎓 Citation

```bibtex
@software{gaze_research_2025,
  title={GAZE: Research-Grade Gaze Pattern Analysis Platform},
  author={Research Team},
  year={2025},
  url={https://github.com/filsan-1/GAZE-}
}
```

---

**Version**: 1.0.0 | **Status**: Research Prototype | **License**: MIT
