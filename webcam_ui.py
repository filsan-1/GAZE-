"""
Streamlit Webcam UI for GAZE Research Platform.

Interactive real-time gaze tracking with live video, eye crops, stimulus tracking,
and gaze pattern analysis.
"""

import logging
import sys
import os
from pathlib import Path
import time
import numpy as np
import cv2
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    ETHICAL_DISCLAIMER, CLASS_NAMES, CAMERA_RESOLUTION,
    EYE_CROP_SIZE, CAMERA_FPS,
)
from src.preprocessing import DataPreprocessor
from src.feature_extraction import FeatureExtractor, HandcraftedGazeFeatures
from src.model import RandomForestGazeModel

logger = logging.getLogger(__name__)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="GAZE Research Platform",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
@st.cache_resource
def load_preprocessor():
    """Load preprocessor (cached)."""
    return DataPreprocessor()


@st.cache_resource
def load_feature_extractor():
    """Load feature extractor (cached)."""
    return FeatureExtractor(use_cnn=False)  # CPU-friendly


@st.cache_resource
def load_model():
    """Load trained model (cached)."""
    from config import MODELS_DIR
    model = RandomForestGazeModel()
    model_path = MODELS_DIR / "asd_gaze_model.pkl"
    
    if model_path.exists():
        model.load(model_path)
        return model
    else:
        st.warning("⚠️ No trained model found. Using random model.")
        return None


def init_session_state():
    """Initialize session state variables."""
    if "tracking_data" not in st.session_state:
        st.session_state.tracking_data = {
            "gaze_points": [],
            "left_eye_crops": [],
            "right_eye_crops": [],
            "timestamps": [],
            "fixation_metrics": {},
        }
    
    if "is_tracking" not in st.session_state:
        st.session_state.is_tracking = False
    
    if "session_results" not in st.session_state:
        st.session_state.session_results = None


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def draw_red_dot_stimulus(
    frame: np.ndarray,
    elapsed_time: float,
    pattern: str = "linear",
    speed: float = 2.0,
    radius: int = 15,
) -> np.ndarray:
    """
    Draw red dot stimulus on frame.
    
    Args:
        frame: Input frame.
        elapsed_time: Time since start (seconds).
        pattern: "linear", "circular", or "random".
        speed: Movement speed (pixels per frame).
        radius: Stimulus radius.
        
    Returns:
        Frame with drawn stimulus.
    """
    h, w = frame.shape[:2]
    
    if pattern == "linear":
        # Horizontal oscillation
        x = w // 2 + int(np.sin(elapsed_time * speed) * (w // 4))
        y = h // 2
    
    elif pattern == "circular":
        # Circular motion
        x = int(w // 2 + (w // 4) * np.cos(elapsed_time * speed))
        y = int(h // 2 + (h // 4) * np.sin(elapsed_time * speed))
    
    else:  # random
        # Pseudo-random smooth motion
        x = int(w // 2 + (w // 4) * np.sin(elapsed_time * speed * 0.7))
        y = int(h // 2 + (h // 4) * np.cos(elapsed_time * speed * 0.5))
    
    # Clip to frame bounds
    x = np.clip(x, radius, w - radius)
    y = np.clip(y, radius, h - radius)
    
    # Draw stimulus
    cv2.circle(frame, (x, y), radius, (0, 0, 255), -1)  # Red circle
    cv2.circle(frame, (x, y), radius + 3, (0, 0, 255), 2)  # Border
    
    return frame


def compute_tracking_accuracy(
    gaze_points: np.ndarray,
    stimulus_points: np.ndarray,
) -> float:
    """
    Compute tracking accuracy (inverse of mean distance to stimulus).
    
    Args:
        gaze_points: Gaze points (n_samples, 2).
        stimulus_points: Stimulus points (n_samples, 2).
        
    Returns:
        Accuracy score (0-100).
    """
    if len(gaze_points) == 0:
        return 0.0
    
    distances = np.linalg.norm(gaze_points - stimulus_points, axis=1)
    max_distance = np.sqrt(CAMERA_RESOLUTION[0]**2 + CAMERA_RESOLUTION[1]**2) / 2
    
    accuracy = max(0, 100 * (1 - distances.mean() / max_distance))
    return float(accuracy)


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_header():
    """Render page header."""
    st.markdown("# 👁️ GAZE Research Platform")
    st.markdown("**Real-Time Gaze Tracking for Neurodevelopmental Research**")
    st.markdown("---")


def render_disclaimer():
    """Render ethical disclaimer."""
    with st.expander("⚠️ Important Disclaimer (Click to Read)", expanded=False):
        st.warning(ETHICAL_DISCLAIMER)


def render_webcam_tracking():
    """Render real-time webcam tracking interface."""
    st.subheader("📹 Live Webcam Gaze Tracking")
    
    col_settings, col_video = st.columns([1, 3])
    
    # Settings panel
    with col_settings:
        st.markdown("### Tracking Settings")
        
        # Start/Stop buttons
        col_start, col_stop = st.columns(2)
        with col_start:
            start_tracking = st.button("▶️ Start", key="btn_start_tracking")
        with col_stop:
            stop_tracking = st.button("⏹️ Stop", key="btn_stop_tracking")
        
        if start_tracking:
            st.session_state.is_tracking = True
        if stop_tracking:
            st.session_state.is_tracking = False
        
        # Stimulus settings
        st.markdown("#### Stimulus Settings")
        pattern = st.selectbox(
            "Trajectory",
            ["linear", "circular", "random"],
            help="Pattern for red dot movement"
        )
        
        speed = st.slider(
            "Speed",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.1
        )
        
        # Display settings
        st.markdown("#### Display Options")
        show_stimulus = st.checkbox("Red Dot Stimulus", value=True)
        show_landmarks = st.checkbox("Face Landmarks", value=False)
        show_eye_crops = st.checkbox("Eye Crops", value=True)
        
        # Session info
        st.markdown("#### Session Info")
        session_duration = st.slider(
            "Duration (seconds)",
            min_value=10,
            max_value=300,
            value=60,
            step=10
        )
    
    # Video panel
    with col_video:
        if st.session_state.is_tracking:
            st.info("🔴 **RECORDING** - Starting webcam...")
            
            # Initialize components
            preprocessor = load_preprocessor()
            extractor = load_feature_extractor()
            model = load_model()
            
            # Placeholders
            frame_placeholder = st.empty()
            metrics_placeholder = st.empty()
            eye_crops_placeholder = st.empty()
            
            # Open webcam
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Webcam not accessible")
                st.session_state.is_tracking = False
            else:
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
                
                frame_count = 0
                start_time = time.time()
                session_data = {
                    "gaze_points": [],
                    "timestamps": [],
                    "left_crops": [],
                    "right_crops": [],
                }
                
                stimulus_points = []
                
                while st.session_state.is_tracking:
                    elapsed = time.time() - start_time
                    
                    # Check session duration
                    if elapsed > session_duration:
                        st.success("✅ Session completed!")
                        st.session_state.is_tracking = False
                        break
                    
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from webcam")
                        break
                    
                    # Flip frame for selfie view
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    result = preprocessor.process_frame(frame)
                    
                    if result is not None:
                        # Get gaze point
                        gaze_point = result["gaze_point_normalized"]
                        session_data["gaze_points"].append(gaze_point)
                        session_data["timestamps"].append(elapsed)
                        
                        # Store eye crops
                        session_data["left_crops"].append(result["left_eye_crop"])
                        session_data["right_crops"].append(result["right_eye_crop"])
                        
                        # Draw landmarks
                        if show_landmarks and result["landmarks"] is not None:
                            landmarks = result["landmarks"][:, :2].astype(np.int32)
                            for lm in landmarks:
                                cv2.circle(frame, tuple(lm), 2, (0, 255, 0), -1)
                    
                    # Draw stimulus
                    if show_stimulus:
                        frame = draw_red_dot_stimulus(
                            frame, elapsed, pattern, speed
                        )
                        # Track stimulus position for accuracy calculation
                        h, w = frame.shape[:2]
                        if pattern == "linear":
                            stim_x = w // 2 + int(np.sin(elapsed * speed) * (w // 4))
                            stim_y = h // 2
                        else:
                            stim_x = w // 2
                            stim_y = h // 2
                        stimulus_points.append(np.array([stim_x, stim_y]))
                    
                    # Display frame
                    frame_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        use_column_width=True
                    )
                    
                    # Display eye crops
                    if show_eye_crops and result is not None:
                        left_crop = result["left_eye_crop"]
                        right_crop = result["right_eye_crop"]
                        
                        if left_crop is not None and right_crop is not None:
                            eye_img = np.hstack([
                                cv2.cvtColor(left_crop, cv2.COLOR_BGR2RGB),
                                cv2.cvtColor(right_crop, cv2.COLOR_BGR2RGB),
                            ])
                            eye_crops_placeholder.image(eye_img, width=400)
                    
                    # Update metrics
                    with metrics_placeholder.container():
                        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                        with col_m1:
                            st.metric("Frames", frame_count)
                        with col_m2:
                            st.metric("Duration", f"{elapsed:.1f}s")
                        with col_m3:
                            st.metric("FPS", f"{frame_count / max(elapsed, 0.1):.1f}")
                        with col_m4:
                            # Compute tracking accuracy if stimulus points available
                            if stimulus_points and len(session_data["gaze_points"]) > 0:
                                gaze_px = np.array(session_data["gaze_points"]) * np.array(CAMERA_RESOLUTION)
                                stimulus_px = np.array(stimulus_points)
                                acc = compute_tracking_accuracy(gaze_px, stimulus_px)
                                st.metric("Accuracy", f"{acc:.1f}%")
                    
                    frame_count += 1
                    time.sleep(1.0 / CAMERA_FPS)
                
                cap.release()
                
                # Store session data
                if len(session_data["gaze_points"]) > 0:
                    st.session_state.tracking_data = session_data
                    st.success(f"✅ Captured {len(session_data['gaze_points'])} frames")
                    
                    # Show analysis button
                    if st.button("📊 Analyze Session"):
                        render_session_analysis(session_data)
        
        else:
            st.info("Click 'Start' to begin live tracking")
            
            # Show placeholder
            placeholder_img = np.ones((CAMERA_RESOLUTION[1], CAMERA_RESOLUTION[0], 3), dtype=np.uint8) * 200
            cv2.putText(placeholder_img, "Waiting for start...", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            frame_placeholder = st.empty()
            frame_placeholder.image(cv2.cvtColor(placeholder_img, cv2.COLOR_BGR2RGB))


def render_session_analysis(session_data: dict):
    """Render analysis of tracking session."""
    st.subheader("📊 Session Analysis")
    
    gaze_points = np.array(session_data["gaze_points"])
    
    if len(gaze_points) == 0:
        st.warning("No valid gaze data to analyze")
        return
    
    # Feature extraction
    feature_extractor = HandcraftedGazeFeatures(sampling_rate=CAMERA_FPS)
    
    # Convert normalized coordinates to pixels
    gaze_px = gaze_points * np.array(CAMERA_RESOLUTION)
    
    # Extract metrics
    fixation_metrics = feature_extractor.compute_fixation_metrics(gaze_px)
    saccade_metrics = feature_extractor.compute_saccade_metrics(gaze_px)
    entropy = feature_extractor.compute_gaze_entropy(gaze_px)
    velocity_metrics = feature_extractor.compute_velocity_metrics(gaze_px)
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Fixation Duration",
            f"{fixation_metrics.get('fixation_duration_mean', 0):.2f}s"
        )
    with col2:
        st.metric(
            "Saccade Count",
            f"{saccade_metrics.get('saccade_count', 0)}"
        )
    with col3:
        st.metric(
            "Gaze Entropy",
            f"{entropy:.2f}"
        )
    with col4:
        st.metric(
            "Mean Velocity",
            f"{velocity_metrics.get('gaze_velocity_mean', 0):.1f}px/s"
        )
    
    # Plot gaze trajectory
    st.markdown("#### Gaze Trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gaze_points[:, 0],
        y=1 - gaze_points[:, 1],  # Flip y for display
        mode="lines+markers",
        name="Gaze Path",
        line=dict(color="red", width=2),
        marker=dict(size=4),
    ))
    fig.update_layout(
        title="Gaze Trajectory Over Time",
        xaxis_title="Normalized X",
        yaxis_title="Normalized Y",
        hovermode="closest",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_demo_mode():
    """Render demo mode with simulated data."""
    st.subheader("🧪 Demo Mode")
    
    st.info("Simulated gaze data for testing without webcam")
    
    # Generate synthetic data
    num_samples = st.slider("Samples", 100, 1000, 300)
    pattern = st.selectbox("Pattern", ["linear", "circular", "smooth"])
    
    if st.button("Generate & Analyze Demo Data"):
        # Generate synthetic gaze points
        t = np.linspace(0, 2 * np.pi, num_samples)
        
        if pattern == "linear":
            gaze_x = 0.5 + 0.3 * np.sin(t)
            gaze_y = 0.5 * np.ones_like(t)
        elif pattern == "circular":
            gaze_x = 0.5 + 0.3 * np.cos(t)
            gaze_y = 0.5 + 0.3 * np.sin(t)
        else:
            gaze_x = 0.5 + 0.3 * np.sin(t) + 0.1 * np.random.randn(num_samples) * 0.05
            gaze_y = 0.5 + 0.3 * np.cos(t) + 0.1 * np.random.randn(num_samples) * 0.05
        
        gaze_points_norm = np.column_stack([gaze_x, gaze_y])
        
        # Add to session
        st.session_state.tracking_data["gaze_points"] = gaze_points_norm
        
        st.success("✅ Demo data generated")
        
        # Analyze
        render_session_analysis({
            "gaze_points": gaze_points_norm,
            "left_crops": [],
            "right_crops": [],
        })


def render_about():
    """Render about page."""
    st.subheader("ℹ️ About GAZE")
    
    st.markdown("""
    ### GAZE Research Platform
    
    **Purpose:** Research-grade gaze tracking for neurodevelopmental research
    
    **Key Features:**
    - Real-time webcam-based gaze tracking using MediaPipe
    - Handcrafted gaze metrics (fixations, saccades, entropy)
    - CNN embeddings from eye regions using MobileNetV2
    - Machine learning models (Random Forest + Neural Network)
    - Privacy-preserving local data processing
    - Non-diagnostic research tool
    
    ### Technology Stack
    - **MediaPipe:** Facial landmark detection
    - **OpenCV:** Image processing
    - **PyTorch:** Deep learning
    - **scikit-learn:** Machine learning
    - **Streamlit:** Web interface
    
    ### Important Notes
    - **NOT a diagnostic tool** - cannot diagnose autism or any condition
    - Results are probabilistic patterns, not clinical assessments
    - Always consult qualified healthcare professionals for clinical evaluation
    - All data is processed locally for privacy
    
    ### Citation
    ```
    @software{gaze_research_2026,
        title={GAZE: Research-Grade Gaze Tracking Platform},
        year={2026}
    }
    ```
    """)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    """Main application."""
    logging.basicConfig(level=logging.INFO)
    
    init_session_state()
    
    render_header()
    render_disclaimer()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio(
            "Select Page",
            ["📹 Webcam Tracking", "🧪 Demo", "ℹ️ About"],
            label_visibility="collapsed"
        )
    
    if page == "📹 Webcam Tracking":
        render_webcam_tracking()
    elif page == "🧪 Demo":
        render_demo_mode()
    elif page == "ℹ️ About":
        render_about()


if __name__ == "__main__":
    main()
