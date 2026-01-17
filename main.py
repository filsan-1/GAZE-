"""
Main entry point for GAZE Research Platform.

Example usage and demonstration of core functionality.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import DatasetLoader, GazeDataNormalizer, GazeDataPreprocessor
from src.feature_extraction import GazeFeatureExtractor, ROIAnalyzer
from src.models import RandomForestGazeModel, ASDLikelihoodScorer
from src.config import MODEL_DIR, RESULTS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """
    Run complete GAZE analysis pipeline.
    
    Demonstrates:
    1. Dataset loading and synthetic data generation
    2. Data normalization and preprocessing
    3. Feature extraction
    4. Model training
    5. Result interpretation and reporting
    """

    print("\n" + "=" * 70)
    print("GAZE RESEARCH PLATFORM - DEMO PIPELINE")
    print("=" * 70 + "\n")

    # =========================================================================
    # STEP 1: LOAD DATASETS
    # =========================================================================
    print("[1/6] Loading and generating datasets...\n")

    loader = DatasetLoader()

    # Create synthetic ASD comparison dataset
    logger.info("Generating ASD vs TD comparison dataset...")
    asd_df = loader.create_synthetic_asd_comparison(
        num_asd_samples=300,
        num_td_samples=300,
    )

    print(f"✓ Loaded dataset: {len(asd_df)} samples")
    print(f"  Groups: ASD={len(asd_df[asd_df['group']=='ASD'])}, "
          f"TD={len(asd_df[asd_df['group']=='TD'])}")
    print(f"  Features: {len(asd_df.columns)}\n")

    # =========================================================================
    # STEP 2: NORMALIZE DATA
    # =========================================================================
    print("[2/6] Normalizing and preprocessing data...\n")

    normalizer = GazeDataNormalizer(method="standard")
    normalized_df = normalizer.normalize_full_pipeline(
        asd_df,
        screen_width=1280,
        screen_height=720,
        remove_outliers=True,
        standardize=False,  # Will standardize after feature extraction
    )

    print(f"✓ Data normalized")
    print(f"  Samples remaining: {len(normalized_df)}\n")

    # =========================================================================
    # STEP 3: EXTRACT FEATURES
    # =========================================================================
    print("[3/6] Extracting gaze features...\n")

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

    logger.info(f"Using {len(feature_cols)} features for modeling")

    # Verify all features exist
    available_features = [col for col in feature_cols if col in normalized_df.columns]
    print(f"✓ Feature columns available: {len(available_features)}/{len(feature_cols)}")
    print(f"  Features: {', '.join(available_features[:5])}...")
    print()

    # =========================================================================
    # STEP 4: PREPARE TRAINING DATA
    # =========================================================================
    print("[4/6] Preparing training data...\n")

    X = normalized_df[available_features].values
    y = (normalized_df["group"] == "ASD").astype(int).values

    # Standardize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"  ASD samples: {(y == 1).sum()}")
    print(f"  TD samples: {(y == 0).sum()}\n")

    # =========================================================================
    # STEP 5: TRAIN MODEL
    # =========================================================================
    print("[5/6] Training Random Forest model...\n")

    model = RandomForestGazeModel(n_estimators=100, max_depth=15)

    metrics = model.train(
        X, y,
        feature_names=available_features,
        test_size=0.2,
        random_state=42
    )

    print(f"✓ Model trained successfully")
    print(f"  Train accuracy: {metrics['train_accuracy']:.1%}")
    print(f"  Test accuracy: {metrics['test_accuracy']:.1%}")
    print(f"  AUC-ROC: {metrics['auc']:.3f}\n")

    # Feature importance
    importance = model.get_feature_importance(top_n=5)
    print("  Top features:")
    for feat, imp in list(importance.items())[:5]:
        print(f"    • {feat}: {imp:.3f}")
    print()

    # =========================================================================
    # STEP 6: SCORE AND INTERPRET RESULTS
    # =========================================================================
    print("[6/6] Scoring and generating reports...\n")

    # Create scorer with reference population
    reference_scores = model.predict_asd_likelihood(X) * 100
    scorer = ASDLikelihoodScorer(reference_scores=reference_scores)

    # Score sample cases
    sample_indices = {
        "ASD Case": np.where(y == 1)[0][0],
        "TD Case": np.where(y == 0)[0][0],
    }

    results_summary = []

    for case_name, idx in sample_indices.items():
        sample_features = X[idx:idx + 1]
        score = model.predict_asd_likelihood(sample_features)[0]
        percentile = scorer.score_to_percentile(score)
        tier = scorer.score_to_tier(score)
        confidence = scorer.compute_confidence(score)

        results_summary.append({
            "Case": case_name,
            "True Group": "ASD" if y[idx] == 1 else "TD",
            "ASD Score": f"{score:.1f}%",
            "Percentile": f"{percentile:.1f}th",
            "Tier": tier["tier"],
            "Confidence": f"{confidence:.1f}%",
        })

        print(f"✓ {case_name}")
        print(f"  ASD-Like Score: {score:.1f}%")
        print(f"  Percentile Rank: {percentile:.1f}th")
        print(f"  Risk Tier: {tier['tier'].upper()}")
        print(f"  Confidence: {confidence:.1f}%\n")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    print("[SAVING RESULTS]\n")

    # Save model
    model_path = MODEL_DIR / "asd_gaze_model_demo.pkl"
    model.save(str(model_path))
    print(f"✓ Model saved to: {model_path}")

    # Save results summary
    results_df = pd.DataFrame(results_summary)
    results_path = RESULTS_DIR / "analysis_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✓ Results saved to: {results_path}\n")

    # =========================================================================
    # GENERATE COMPREHENSIVE REPORT
    # =========================================================================
    print("=" * 70)
    print("SAMPLE ANALYSIS REPORT")
    print("=" * 70 + "\n")

    # Get feature data for a sample
    sample_idx = np.where(y == 1)[0][0]
    sample_data = X[sample_idx]
    sample_score = model.predict_asd_likelihood(sample_data.reshape(1, -1))[0]
    sample_percentile = scorer.score_to_percentile(sample_score)

    # Create features dict from available features
    sample_features_dict = {
        feat: normalized_df[feat].iloc[sample_idx]
        for feat in available_features
    }

    # Get importance for this sample
    importance_dict = model.get_feature_importance(top_n=10)

    report = scorer.generate_report(
        score=sample_score,
        features=sample_features_dict,
        feature_importance=importance_dict,
        percentile=sample_percentile,
    )

    print(report)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"""
Platform Status:
  ✓ Data loading and preprocessing
  ✓ Feature extraction (9 metrics)
  ✓ Model training (Random Forest, 100 trees)
  ✓ Score prediction and interpretation
  ✓ Results export

Model Performance:
  • Test Accuracy: {metrics['test_accuracy']:.1%}
  • AUC-ROC: {metrics['auc']:.3f}
  • Feature Count: {len(available_features)}

Outputs Generated:
  • Model: {model_path}
  • Results: {results_path}
  • Report: Console output above

Next Steps:
  1. Run Streamlit UI: streamlit run ui/app.py
  2. Train on your own data: see docs/README.md
  3. Integrate live gaze tracking: see src/gaze_tracking/
  4. Customize features: see src/feature_extraction/

⚠️  DISCLAIMER: This system is for research and education only.
    It is NOT a diagnostic tool. Always consult qualified
    professionals for clinical assessment.
    """)

    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
