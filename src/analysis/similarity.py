"""
Similarity algorithms for comparing applications.

Implements proportional similarity where the focus is on the ratio of matching
features to total applicable features, not absolute feature counts.
"""
from typing import Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Support both relative and absolute imports
try:
    from ..core.application import Application
except ImportError:
    from core.application import Application


class SimilarityCalculator:
    """
    Calculate various similarity metrics between applications.
    """

    @staticmethod
    def proportional_similarity(app1: Application, app2: Application) -> float:
        """
        Calculate proportional similarity between two applications.

        This metric ensures that two simple apps with high overlap score higher
        than two complex apps with lower proportional overlap.

        Algorithm:
        1. Find dimensions where at least one app has a score > 0 (union)
        2. For each dimension, calculate similarity of scores (normalized)
        3. Weight by the minimum of the two scores (emphasizes shared features)
        4. Normalize by the count of applicable dimensions

        Args:
            app1: First application
            app2: Second application

        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        # Get all dimensions where at least one app has a non-zero score
        all_dims = set(app1.scores.keys()) | set(app2.scores.keys())

        if not all_dims:
            return 0.0

        # Calculate weighted similarity across dimensions
        total_similarity = 0.0
        total_weight = 0.0

        for dim in all_dims:
            score1 = app1.get_score(dim)
            score2 = app2.get_score(dim)

            # Skip if both are zero (shouldn't happen given our union)
            if score1 == 0 and score2 == 0:
                continue

            # Calculate similarity for this dimension (1 - normalized difference)
            max_score = max(score1, score2)
            if max_score > 0:
                dim_similarity = 1 - abs(score1 - score2) / max_score
            else:
                dim_similarity = 0.0

            # Weight by minimum score (emphasizes shared active features)
            weight = min(score1, score2) + 0.1  # Add small constant to avoid zero weight
            total_similarity += dim_similarity * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_similarity / total_weight

    @staticmethod
    def jaccard_similarity(app1: Application, app2: Application) -> float:
        """
        Calculate Jaccard similarity based on active dimensions.

        Jaccard = |intersection| / |union|

        This is purely based on which dimensions are active (non-zero),
        not the magnitude of scores.

        Args:
            app1: First application
            app2: Second application

        Returns:
            Jaccard similarity between 0 and 1
        """
        dims1 = set(app1.active_dimensions)
        dims2 = set(app2.active_dimensions)

        if not dims1 and not dims2:
            return 1.0  # Both empty = identical

        intersection = len(dims1 & dims2)
        union = len(dims1 | dims2)

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def cosine_similarity_score(
        app1: Application, app2: Application, all_dimensions: list
    ) -> float:
        """
        Calculate cosine similarity between two applications.

        Args:
            app1: First application
            app2: Second application
            all_dimensions: List of all dimension indices to consider

        Returns:
            Cosine similarity between -1 and 1 (typically 0 to 1 for non-negative scores)
        """
        vec1 = app1.get_score_vector(all_dimensions).reshape(1, -1)
        vec2 = app2.get_score_vector(all_dimensions).reshape(1, -1)

        return cosine_similarity(vec1, vec2)[0, 0]

    @staticmethod
    def weighted_proportional_similarity(
        app1: Application, app2: Application
    ) -> Tuple[float, dict]:
        """
        Calculate proportional similarity with detailed breakdown per dimension.

        Returns both the overall similarity and a dictionary with per-dimension details.

        Args:
            app1: First application
            app2: Second application

        Returns:
            Tuple of (overall_similarity, dimension_details)
            where dimension_details is a dict with keys:
                - 'dimension': dimension index
                - 'score1': score for app1
                - 'score2': score for app2
                - 'similarity': similarity for this dimension
                - 'weight': weight applied
        """
        all_dims = set(app1.scores.keys()) | set(app2.scores.keys())

        if not all_dims:
            return 0.0, {}

        dimension_details = []
        total_similarity = 0.0
        total_weight = 0.0

        for dim in sorted(all_dims):
            score1 = app1.get_score(dim)
            score2 = app2.get_score(dim)

            if score1 == 0 and score2 == 0:
                continue

            max_score = max(score1, score2)
            dim_similarity = 1 - abs(score1 - score2) / max_score if max_score > 0 else 0.0

            weight = min(score1, score2) + 0.1
            total_similarity += dim_similarity * weight
            total_weight += weight

            dimension_details.append(
                {
                    "dimension": dim,
                    "score1": score1,
                    "score2": score2,
                    "similarity": dim_similarity,
                    "weight": weight,
                }
            )

        overall_similarity = total_similarity / total_weight if total_weight > 0 else 0.0

        return overall_similarity, dimension_details
