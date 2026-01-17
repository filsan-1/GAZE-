"""
ASD Likelihood Scorer for converting model outputs to interpretable scores.

Produces ASD likelihood scores, percentile rankings, and risk tiers.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class ASDLikelihoodScorer:
    """
    Converts model outputs to ASD likelihood scores and interpretations.

    Provides:
    - Probability to 0-100 score conversion
    - Percentile ranking relative to reference population
    - Risk tier classification (Low/Moderate/Elevated)
    - Feature-based explanations
    """

    # Score tier definitions
    TIERS = {
        "low": {"range": (0, 33), "description": "Low ASD-associated gaze pattern"},
        "moderate": {"range": (33, 67), "description": "Moderate ASD-associated gaze pattern"},
        "elevated": {"range": (67, 100), "description": "Elevated ASD-associated gaze pattern"},
    }

    def __init__(self, reference_scores: Optional[np.ndarray] = None):
        """
        Initialize scorer.

        Args:
            reference_scores: Array of reference population scores for percentile calculation
        """
        self.reference_scores = reference_scores
        self.percentile_bins = np.percentile(
            reference_scores, np.linspace(0, 100, 101)
        ) if reference_scores is not None else None

        logger.info("Initialized ASDLikelihoodScorer")

    def score_to_percentile(self, score: float) -> float:
        """
        Convert ASD likelihood score to percentile rank.

        Args:
            score: ASD likelihood score (0-100)

        Returns:
            Percentile rank (0-100)
        """
        if self.reference_scores is None:
            # Default: assume uniform distribution
            return float(score)

        percentile = stats.percentileofscore(self.reference_scores, score)
        return float(percentile)

    def score_to_tier(self, score: float) -> Dict[str, str]:
        """
        Classify score into risk tier.

        Args:
            score: ASD likelihood score (0-100)

        Returns:
            Dict with tier_name and description
        """
        for tier_name, tier_info in self.TIERS.items():
            min_score, max_score = tier_info["range"]
            if min_score <= score < max_score:
                return {
                    "tier": tier_name,
                    "description": tier_info["description"],
                    "score_range": f"{min_score}-{max_score}",
                }

        # Edge case: score exactly 100
        return {
            "tier": "elevated",
            "description": self.TIERS["elevated"]["description"],
            "score_range": f"{self.TIERS['elevated']['range'][0]}-100",
        }

    def compute_confidence(
        self,
        score: float,
        model_uncertainty: Optional[float] = None,
    ) -> float:
        """
        Compute confidence score for the prediction.

        Args:
            score: ASD likelihood score
            model_uncertainty: Model uncertainty estimate

        Returns:
            Confidence score (0-100)
        """
        # Confidence is higher for extreme scores
        # and lower near decision boundaries (33, 67)

        boundaries = [33, 67]
        min_distance_to_boundary = min(abs(score - b) for b in boundaries)

        # Normalize distance: max distance is 50 (from 0 to 50 or 50 to 100)
        boundary_confidence = (min_distance_to_boundary / 50) * 100

        # Incorporate model uncertainty if available
        if model_uncertainty is not None:
            overall_confidence = 0.7 * boundary_confidence + 0.3 * (100 - model_uncertainty)
        else:
            overall_confidence = boundary_confidence

        return float(np.clip(overall_confidence, 0, 100))

    def generate_interpretation(
        self,
        score: float,
        percentile: float,
        feature_importance: Optional[Dict[str, float]] = None,
        top_n_features: int = 5,
    ) -> Dict[str, str]:
        """
        Generate human-readable interpretation of scores.

        Args:
            score: ASD likelihood score
            percentile: Percentile rank
            feature_importance: Dictionary of feature importance scores
            top_n_features: Number of top features to include in explanation

        Returns:
            Dictionary with interpretation components
        """
        tier_info = self.score_to_tier(score)

        interpretation = {
            "score": f"{score:.1f}%",
            "percentile": f"{percentile:.1f}th percentile",
            "tier": tier_info["tier"],
            "tier_description": tier_info["description"],
        }

        # Feature-based explanation
        if feature_importance:
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_n_features]

            feature_explanation = []
            for feature_name, importance in sorted_features:
                if importance > 0:
                    feature_explanation.append(f"increased {feature_name}")
                else:
                    feature_explanation.append(f"decreased {feature_name}")

            interpretation["key_features"] = ", ".join(feature_explanation)

        return interpretation

    def generate_report(
        self,
        score: float,
        features: Dict[str, float],
        feature_importance: Optional[Dict[str, float]] = None,
        percentile: Optional[float] = None,
    ) -> str:
        """
        Generate comprehensive text report.

        Args:
            score: ASD likelihood score
            features: Dictionary of computed features
            feature_importance: Dictionary of feature importance scores
            percentile: Percentile rank

        Returns:
            Formatted text report
        """
        if percentile is None:
            percentile = self.score_to_percentile(score)

        tier_info = self.score_to_tier(score)
        confidence = self.compute_confidence(score)

        report = []
        report.append("=" * 70)
        report.append("GAZE PATTERN ANALYSIS REPORT")
        report.append("=" * 70)
        report.append("")

        # Disclaimer
        report.append("⚠️  DISCLAIMER:")
        report.append("This report is for research and educational purposes ONLY.")
        report.append("It is NOT a diagnostic tool and does NOT diagnose autism.")
        report.append("")

        # Score summary
        report.append("RESULTS SUMMARY:")
        report.append(f"  ASD-Associated Gaze Likelihood: {score:.1f}%")
        report.append(f"  Percentile Rank: {percentile:.1f}th percentile")
        report.append(f"  Risk Tier: {tier_info['tier'].upper()}")
        report.append(f"  Confidence: {confidence:.1f}%")
        report.append("")

        # Tier interpretation
        report.append("INTERPRETATION:")
        report.append(f"  {tier_info['description']}")
        report.append("")

        # Top contributing features
        if feature_importance:
            report.append("KEY GAZE PATTERN FEATURES:")
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]

            for feature_name, importance in sorted_features:
                direction = "↑" if importance > 0 else "↓"
                report.append(f"  {direction} {feature_name} ({abs(importance):.3f})")

            report.append("")

        # Feature values
        report.append("DETAILED GAZE METRICS:")
        for feature_name in sorted(features.keys()):
            report.append(f"  {feature_name}: {features[feature_name]:.3f}")

        report.append("")
        report.append("=" * 70)
        report.append("For inquiries, contact the research team.")
        report.append("=" * 70)

        return "\n".join(report)

    @staticmethod
    def create_reference_population(
        reference_data: np.ndarray,
        percentiles: List[int] = [5, 25, 50, 75, 95],
    ) -> Dict[int, float]:
        """
        Create reference population statistics.

        Args:
            reference_data: Array of reference population scores
            percentiles: Percentiles to compute

        Returns:
            Dictionary of {percentile: value}
        """
        reference_stats = {}

        for p in percentiles:
            reference_stats[p] = np.percentile(reference_data, p)

        return reference_stats


if __name__ == "__main__":
    # Example usage
    # Create reference population
    reference_scores = np.random.beta(2, 5, 1000) * 100  # TD-biased distribution

    scorer = ASDLikelihoodScorer(reference_scores=reference_scores)

    # Score a sample
    sample_score = 65.3
    percentile = scorer.score_to_percentile(sample_score)
    tier = scorer.score_to_tier(sample_score)

    print(f"Score: {sample_score:.1f}%")
    print(f"Percentile: {percentile:.1f}th")
    print(f"Tier: {tier['tier']}")

    # Generate report
    features = {
        "fixation_duration_mean": 0.35,
        "gaze_entropy": 2.1,
        "roi_attention_eyes": 0.25,
    }

    feature_importance = {
        "fixation_duration_mean": 0.15,
        "gaze_entropy": -0.12,
        "roi_attention_eyes": -0.18,
    }

    report = scorer.generate_report(sample_score, features, feature_importance, percentile)
    print("\n" + report)
