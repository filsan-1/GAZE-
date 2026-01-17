"""
Streamlit web UI for GAZE application.

Interactive real-time gaze tracking and analysis interface.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
import cv2
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px

from src.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    STIMULUS_MODES, ETHICAL_DISCLAIMER,
    PROJECT_ROOT, MODEL_DIR, RESULTS_DIR,
    FEATURE_EXPLANATIONS,
)
from src.gaze_tracking import (
    MediaPipeDetector,
    GazeRenderer,
    StimulusGenerator,
)
from src.feature_extraction import GazeFeatureExtractor, ROIAnalyzer
from src.models import RandomForestGazeModel, ASDLikelihoodScorer
from src.data_processing import DatasetLoader, GazeDataNormalizer, GazeDataPreprocessor

logger = logging.getLogger(__name__)


class StreamlitGazeApp:
    """
    Streamlit application for GAZE research platform.
    """

    def __init__(self):
        """Initialize the Streamlit application."""
        self.detector = None
        self.renderer = None
        self.stimulus_gen = None
        self.feature_extractor = None
        self.roi_analyzer = None
        self.model = None
        self.scorer = None

        self._initialize_session_state()
        self._load_models()

    def _initialize_session_state(self):
        """Initialize Streamlit session state."""
        if "gaze_history" not in st.session_state:
            st.session_state.gaze_history = []

        if "fixation_history" not in st.session_state:
            st.session_state.fixation_history = []

        if "roi_attention_history" not in st.session_state:
            st.session_state.roi_attention_history = []

        if "session_started" not in st.session_state:
            st.session_state.session_started = False

    def _load_models(self):
        """Load trained models."""
        try:
            model_path = MODEL_DIR / "asd_gaze_model.pkl"
            if model_path.exists():
                self.model = RandomForestGazeModel()
                self.model.load(str(model_path))
                logger.info("Loaded trained model")
            else:
                logger.warning("No trained model found, will use demo mode")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

        # Initialize scorer with dummy reference data for demo
        reference_scores = np.random.beta(2, 5, 1000) * 100
        self.scorer = ASDLikelihoodScorer(reference_scores=reference_scores)

    def render_header(self):
        """Render application header."""
        col1, col2 = st.columns([4, 1])

        with col1:
            st.title("👁️ GAZE Research Platform")
            st.markdown(
                "**Research-Grade Gaze Pattern Analysis for ASD Research**"
            )

        st.markdown("---")

    def render_ethical_disclaimer(self):
        """Render ethical disclaimer."""
        st.warning(ETHICAL_DISCLAIMER, icon="⚠️")

    def render_demo_mode(self):
        """Render demo mode interface."""
        st.info(
            "🧪 **DEMO MODE**: Using simulated gaze data. "
            "In production, this would connect to real-time camera feed."
        )

        # Demo stimulus generator
        if self.stimulus_gen is None:
            self.stimulus_gen = StimulusGenerator(
                screen_width=CAMERA_WIDTH,
                screen_height=CAMERA_HEIGHT,
            )

        # Stimulus mode selection
        stimulus_mode = st.selectbox(
            "Select Stimulus Trajectory",
            list(STIMULUS_MODES.keys()),
            format_func=lambda x: f"{x.title()} - {STIMULUS_MODES[x]}",
        )

        # Create demo visualization
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📺 Live Gaze Visualization")

            # Create frame with stimulus
            frame = np.ones((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8) * 200

            # Get stimulus position
            elapsed_time = time.time() % 10  # 10 second loop
            stimulus_pos = self.stimulus_gen.get_position(
                mode=stimulus_mode,
                t=elapsed_time,
                dt=1.0 / CAMERA_FPS,
            )

            # Draw stimulus
            cv2.circle(
                frame,
                (int(stimulus_pos[0]), int(stimulus_pos[1])),
                15,
                (0, 0, 255),
                -1,
            )

            # Draw simulated gaze point (following stimulus)
            gaze_x = stimulus_pos[0] + np.random.normal(0, 30)
            gaze_y = stimulus_pos[1] + np.random.normal(0, 30)

            gaze_point = (
                np.clip(gaze_x, 0, CAMERA_WIDTH),
                np.clip(gaze_y, 0, CAMERA_HEIGHT),
            )

            cv2.circle(frame, (int(gaze_point[0]), int(gaze_point[1])), 12, (0, 255, 255), 2)

            # Display frame
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        with col2:
            st.subheader("📊 Real-Time Metrics")

            # Simulated metrics
            col_left, col_right = st.columns(2)

            with col_left:
                st.metric("Fixation Duration", "0.23 s")
                st.metric("Saccade Count", "42")

            with col_right:
                st.metric("Gaze Accuracy", "87.3%")
                st.metric("Blink Rate", "18/min")

    def render_demo_analysis(self):
        """Render demo analysis results."""
        st.subheader("📈 Analysis Results")

        # Create synthetic gaze data
        loader = DatasetLoader()
        df = loader.create_synthetic_asd_comparison(
            num_asd_samples=100,
            num_td_samples=100,
        )

        # Extract features
        feature_cols = [
            "fixation_duration_mean",
            "saccade_count_per_min",
            "gaze_entropy",
            "roi_attention_eyes",
            "roi_attention_mouth",
            "blink_rate",
            "eye_aspect_ratio_mean",
            "gaze_velocity_mean",
            "pupil_diameter_variability",
        ]

        if self.model is None:
            # Train a model on the fly for demo
            from src.models import RandomForestGazeModel

            self.model = RandomForestGazeModel()
            X = df[feature_cols].values
            y = (df["group"] == "ASD").astype(int).values

            self.model.train(X, y, feature_names=feature_cols)

        # Get sample and predict
        sample_asd = df[df["group"] == "ASD"].iloc[0][feature_cols].values
        sample_td = df[df["group"] == "TD"].iloc[0][feature_cols].values

        score_asd = self.model.predict_asd_likelihood(sample_asd.reshape(1, -1))[0]
        score_td = self.model.predict_asd_likelihood(sample_td.reshape(1, -1))[0]

        # Display scores
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("ASD-Like Score", f"{score_asd:.1f}%")

        with col2:
            st.metric("Percentile", f"{self.scorer.score_to_percentile(score_asd):.1f}th")

        with col3:
            tier = self.scorer.score_to_tier(score_asd)
            st.metric("Risk Tier", tier["tier"].upper())

        # Feature importance
        st.subheader("🎯 Key Contributing Features")

        importance = self.model.get_feature_importance(top_n=10)

        importance_df = pd.DataFrame(
            list(importance.items()),
            columns=["Feature", "Importance"],
        )

        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance for ASD-Associated Patterns",
        )

        st.plotly_chart(fig, use_container_width=True)

        # ROI attention visualization
        st.subheader("👀 Attention Distribution")

        roi_data = {
            "ROI": ["Eyes", "Mouth", "Nose", "Off-Face"],
            "Attention %": [25, 35, 20, 20],
        }

        fig = go.Figure(data=[
            go.Pie(labels=roi_data["ROI"], values=roi_data["Attention %"])
        ])

        st.plotly_chart(fig, use_container_width=True)

    def render_data_explorer(self):
        """Render data exploration interface."""
        st.subheader("📊 Dataset Explorer")

        # Load datasets
        loader = DatasetLoader()

        col1, col2 = st.columns(2)

        with col1:
            datasets_to_load = st.multiselect(
                "Select Datasets to Explore",
                ["MIT GazeCapture (Synthetic)", "ASD Comparison (Synthetic)"],
                default=["ASD Comparison (Synthetic)"],
            )

        # Load selected datasets
        if "MIT GazeCapture (Synthetic)" in datasets_to_load:
            mit_df = loader.create_synthetic_mit_gazecapture(num_samples=500)
            st.write(f"**MIT GazeCapture**: {len(mit_df)} samples")
            st.dataframe(mit_df.head())

        if "ASD Comparison (Synthetic)" in datasets_to_load:
            asd_df = loader.create_synthetic_asd_comparison(
                num_asd_samples=250,
                num_td_samples=250,
            )

            st.write(f"**ASD Comparison**: {len(asd_df)} samples ({asd_df['group'].value_counts().to_dict()})")

            # Plot distribution
            fig = px.box(
                asd_df,
                y="fixation_duration_mean",
                x="group",
                title="Fixation Duration by Group",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(asd_df.describe())

    def render_about(self):
        """Render about section."""
        st.subheader("ℹ️ About This Research Platform")

        st.markdown("""
        ### System Architecture

        **GAZE** is a research-grade application for analyzing gaze patterns associated with 
        neurodevelopmental differences. The system includes:

        - **MediaPipe Face Mesh**: Real-time facial landmark detection
        - **Feature Extraction**: Comprehensive gaze metrics (fixations, saccades, entropy, ROI attention)
        - **Machine Learning**: Random Forest classifier for pattern recognition
        - **Interactive Visualization**: Real-time overlays and analysis dashboards

        ### Key Features

        ✅ **Non-Diagnostic**: Research and educational use only  
        ✅ **Privacy-Preserving**: All data stored locally  
        ✅ **Reproducible**: Fully documented and open methods  
        ✅ **Comprehensive**: Covers fixations, saccades, entropy, ROI analysis  
        ✅ **Transparent**: Feature importance explanations  

        ### Dataset Support

        - MIT GazeCapture
        - Kaggle ASD Gaze Datasets
        - Custom CSV imports
        - Cross-dataset validation

        ### Citation

        If you use this platform for research, please cite:
        ```
        GAZE Research Platform v1.0 (2025)
        ```
        """)

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("📚 [Documentation](https://github.com/filsan-1/GAZE-)")

        with col2:
            st.warning("⚠️ [Non-Diagnostic Disclaimer](#ethical-disclaimer)")

        with col3:
            st.success("📧 Contact for inquiries")

    def run(self):
        """Run the Streamlit application."""
        # Page config
        st.set_page_config(
            page_title="GAZE Research Platform",
            page_icon="👁️",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Sidebar navigation
        st.sidebar.title("Navigation")

        page = st.sidebar.radio(
            "Select Page",
            ["Home", "Live Tracking", "Data Explorer", "Model Analysis", "About"],
            index=0,
        )

        # Header
        self.render_header()

        # Content based on page selection
        if page == "Home":
            self.render_ethical_disclaimer()

            st.markdown("""
            ### Welcome to GAZE Research Platform

            This application enables researchers and educators to analyze gaze patterns
            and attention behaviors associated with neurodevelopmental differences.

            **Key Capabilities:**
            - 👁️ Real-time gaze tracking with MediaPipe
            - 📊 Comprehensive feature extraction (30+ metrics)
            - 🤖 Machine learning-based pattern recognition
            - 📈 Interactive data visualization and reporting
            - 🔬 Support for multiple datasets and cross-validation

            **Ethical Foundation:**
            - Explicitly non-diagnostic
            - Privacy-preserving (local data storage)
            - Transparent methodology and results
            - Comprehensive documentation and limitations
            """)

            # Demo section
            st.markdown("---")
            st.subheader("🧪 Interactive Demo")

            self.render_demo_mode()
            self.render_demo_analysis()

        elif page == "Live Tracking":
            st.subheader("👁️ Real-Time Gaze Tracking")

            self.render_ethical_disclaimer()

            st.info(
                "🚀 **COMING SOON**: Real-time webcam integration. "
                "Currently demonstrating with simulated data."
            )

            self.render_demo_mode()

        elif page == "Data Explorer":
            st.subheader("📊 Dataset Management and Exploration")

            self.render_ethical_disclaimer()

            self.render_data_explorer()

        elif page == "Model Analysis":
            st.subheader("🤖 Model Training and Analysis")

            self.render_ethical_disclaimer()

            # Model training interface
            st.info("Placeholder for model training interface")

            self.render_demo_analysis()

        elif page == "About":
            self.render_ethical_disclaimer()

            self.render_about()

        # Footer
        st.markdown("---")

        st.markdown("""
        <div style="text-align: center; color: gray; font-size: 0.85em;">
        <p>GAZE Research Platform v1.0 | Non-Diagnostic Research Tool</p>
        <p>For research and educational purposes only | Always consult qualified healthcare professionals</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main entry point."""
    logging.basicConfig(level=logging.INFO)

    app = StreamlitGazeApp()
    app.run()


if __name__ == "__main__":
    main()
