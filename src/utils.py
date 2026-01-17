"""
Utility functions for GAZE application.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from src.config import get_config_summary

logger = logging.getLogger(__name__)


def print_config():
    """Print current configuration."""
    config = get_config_summary()
    print(json.dumps(config, indent=2))


def create_directories():
    """Create all necessary project directories."""
    from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR

    for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ensured: {directory}")


def verify_dependencies():
    """Verify all required dependencies are installed."""
    dependencies = {
        "numpy": "Scientific computing",
        "pandas": "Data handling",
        "opencv": "Computer vision",
        "mediapipe": "Face detection",
        "sklearn": "Machine learning",
        "streamlit": "Web UI",
        "torch": "Deep learning (optional)",
    }

    missing = []

    for package, description in dependencies.items():
        try:
            if package == "opencv":
                import cv2
            elif package == "sklearn":
                import sklearn
            elif package == "torch":
                try:
                    import torch
                except ImportError:
                    logger.warning(f"Optional package '{package}' not available")
                    continue
            else:
                __import__(package)

            logger.info(f"✓ {package}: {description}")
        except ImportError:
            logger.error(f"✗ {package}: {description} - NOT INSTALLED")
            missing.append(package)

    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False

    logger.info("\nAll dependencies verified!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("GAZE Platform Verification")
    print("=" * 70 + "\n")

    print("Configuration:")
    print_config()

    print("\nCreating directories...")
    create_directories()

    print("\nVerifying dependencies...")
    verify_dependencies()

    print("\n" + "=" * 70)
    print("Setup complete! Ready to use GAZE platform.")
    print("=" * 70)
