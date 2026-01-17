#!/usr/bin/env python3
"""
GAZE Quick Start Guide

This script helps you get started with the GAZE research platform.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import verify_dependencies, create_directories, print_config


def print_header():
    """Print welcome header."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   👁️  GAZE RESEARCH PLATFORM - QUICK START                  ║
║                                                              ║
║   Research-Grade Gaze Pattern Analysis for ASD Research    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_instructions():
    """Print usage instructions."""
    print("""
════════════════════════════════════════════════════════════════
QUICK START GUIDE
════════════════════════════════════════════════════════════════

✅ OPTION 1: Run Demo Pipeline
────────────────────────────────────────────────────────────────
This runs a complete analysis with synthetic data:

    python main.py

Expected output: Analysis report with ASD likelihood scores
Time: ~5 seconds

════════════════════════════════════════════════════════════════

✅ OPTION 2: Launch Web Interface
────────────────────────────────────────────────────────────────
Interactive dashboard for exploration:

    streamlit run ui/app.py

Then open: http://localhost:8501

Features:
  • 📺 Live gaze visualization (demo mode)
  • 📊 Real-time metrics and statistics
  • 📈 Feature importance analysis
  • 📋 Comprehensive reports

════════════════════════════════════════════════════════════════

✅ OPTION 3: Programmatic Usage
────────────────────────────────────────────────────────────────
Use GAZE as a library in your code:

    from src.data_processing import DatasetLoader
    from src.models import RandomForestGazeModel
    
    # Load or create datasets
    loader = DatasetLoader()
    df = loader.create_synthetic_asd_comparison()
    
    # Train model
    model = RandomForestGazeModel()
    model.train(X, y, feature_names)
    
    # Predict and score
    score = model.predict_asd_likelihood(sample)

See docs/README.md for detailed examples.

════════════════════════════════════════════════════════════════

✅ OPTION 4: Load Your Own Data
────────────────────────────────────────────────────────────────
Analyze your own gaze datasets:

    from src.data_processing import DatasetLoader
    
    loader = DatasetLoader()
    df = loader.load_csv_dataset(
        "your_data.csv",
        "dataset_name",
        label_column="diagnosis"
    )

CSV file should contain columns like:
  • gaze_x, gaze_y: Gaze coordinates
  • timestamp: Time information
  • diagnosis: Optional labels (ASD/TD)

════════════════════════════════════════════════════════════════

📚 DOCUMENTATION
────────────────────────────────────────────────────────────────
Complete documentation:  docs/README.md

Topics covered:
  • Architecture and design
  • Feature extraction details
  • Model training and evaluation
  • API reference
  • Ethical considerations
  • Limitations and caveats

════════════════════════════════════════════════════════════════

📁 PROJECT STRUCTURE
────────────────────────────────────────────────────────────────
src/
  ├── config.py                 Central configuration
  ├── data_processing/          Data loading & preprocessing
  ├── gaze_tracking/            Real-time gaze detection
  ├── feature_extraction/       30+ gaze metrics
  └── models/                   ML models & scoring

ui/
  └── app.py                    Streamlit web interface

data/
  ├── raw/                      Input datasets
  └── processed/                Preprocessed data

models/                         Trained model checkpoints
results/                        Analysis outputs
docs/                          Complete documentation

════════════════════════════════════════════════════════════════

⚠️  IMPORTANT ETHICAL NOTES
════════════════════════════════════════════════════════════════

This application:
  ✗ Does NOT diagnose autism or any medical condition
  ✓ Is for RESEARCH and EDUCATIONAL purposes ONLY
  ✓ Identifies statistical gaze patterns only
  ✓ Stores all data LOCALLY on your machine
  ✓ Is TRANSPARENT about limitations

When using this system:
  • Clearly communicate limitations to participants
  • Obtain informed consent for data collection
  • Do NOT make clinical claims based on outputs
  • Always consult qualified professionals for diagnosis

════════════════════════════════════════════════════════════════

🎓 CITATION
════════════════════════════════════════════════════════════════

If you publish research using GAZE, please cite:

@software{gaze_research_2025,
  title={GAZE: Research-Grade Gaze Pattern Analysis Platform},
  author={Research Team},
  year={2025},
  url={https://github.com/filsan-1/GAZE-}
}

════════════════════════════════════════════════════════════════

Ready to begin? Run one of the commands above! 🚀

Questions? See docs/README.md or create a GitHub issue.

════════════════════════════════════════════════════════════════
    """)


def main():
    """Main entry point."""
    print_header()

    print("Verifying dependencies...\n")
    if not verify_dependencies():
        print("\n⚠️  Some dependencies are missing!")
        print("Install with: pip install -r requirements.txt")
        return 1

    print("\nCreating project directories...\n")
    create_directories()

    print("\nConfiguration Summary:\n")
    print_config()

    print_instructions()

    print("✨ Setup complete! Ready to use GAZE platform.\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
