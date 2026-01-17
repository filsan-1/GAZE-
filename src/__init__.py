"""
GAZE Research Application - Gaze Pattern Analysis for ASD Research
===================================================================

A research-grade Python gaze-tracking application for analyzing gaze patterns
associated with Autism Spectrum Disorder (ASD). This system is explicitly 
non-diagnostic and intended ONLY for research and educational purposes.

DISCLAIMER:
-----------
This application is NOT a diagnostic tool and does NOT diagnose autism or
any medical condition. It is designed for research and educational analysis
only. All outputs represent statistical patterns relative to reference
populations and should NOT be interpreted as clinical diagnoses.

Modules:
--------
- data_processing: Dataset integration, normalization, and preprocessing
- gaze_tracking: Real-time MediaPipe-based gaze detection and stimulus rendering
- feature_extraction: Comprehensive gaze feature computation
- models: ASD likelihood estimation models and training pipelines
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__license__ = "MIT"

import os
import sys

# Ensure src directory is in path
sys.path.insert(0, os.path.dirname(__file__))
