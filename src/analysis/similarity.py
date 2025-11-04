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

        IMPORTANT: Similarity is biased towards applications WITH features.
        Two applications with all zeros are NOT similar (returns 0.0).

        Algorithm:
        1. Identify shared active dimensions (both apps have score > 0)
        2. Identify asymmetric dimensions (only one app has score > 0)
        3. Calculate similarity weighted heavily by shared active features
        4. Penalize for asymmetric dimensions (mismatched features)
        5. Return 0 if no shared active dimensions exist

        Args:
            app1: First application
            app2: Second application

        Returns:
            Similarity score between 0 and 1 (1 = identical)
        """
        # Get active dimensions for each app
        active1 = set(app1.active_dimensions)
        active2 = set(app2.active_dimensions)

        # If both apps have no features, they are NOT similar
        if not active1 and not active2:
            return 0.0

        # Find shared and asymmetric dimensions
        shared_dims = active1 & active2  # Both have non-zero scores
        asymmetric_dims = (active1 | active2) - shared_dims  # Only one has non-zero

        # If no shared active dimensions, similarity is very low
        if not shared_dims:
            # Return a very small similarity based on how different they are
            # More asymmetric dimensions = less similar
            total_dims = len(active1 | active2)
            return 0.1 / (1 + total_dims)  # Approaches 0 as difference grows

        # Calculate similarity for shared active dimensions
        shared_similarity = 0.0
        shared_weight = 0.0

        for dim in shared_dims:
            score1 = app1.get_score(dim)
            score2 = app2.get_score(dim)

            # Calculate how similar the scores are (1 = identical, 0 = max difference)
            max_score = max(score1, score2)
            dim_similarity = 1 - abs(score1 - score2) / max_score

            # Weight by the magnitude of shared investment in this dimension
            # Use min to emphasize that both must have significant scores
            weight = min(score1, score2)
            shared_similarity += dim_similarity * weight
            shared_weight += weight

        # Calculate base similarity from shared dimensions
        if shared_weight > 0:
            base_similarity = shared_similarity / shared_weight
        else:
            base_similarity = 0.0

        # Apply penalty for asymmetric dimensions
        # The more dimensions they DON'T share, the less similar they are
        total_active_dims = len(active1 | active2)
        shared_ratio = len(shared_dims) / total_active_dims if total_active_dims > 0 else 0

        # Final similarity combines:
        # 1. How similar the shared dimensions are (base_similarity)
        # 2. What proportion of their dimensions are shared (shared_ratio)
        # This ensures apps must have both high score similarity AND high dimension overlap
        final_similarity = base_similarity * (0.5 + 0.5 * shared_ratio)

        return final_similarity

    @staticmethod
    def jaccard_similarity(app1: Application, app2: Application) -> float:
        """
        Calculate Jaccard similarity based on active dimensions.

        Jaccard = |intersection| / |union|

        This is purely based on which dimensions are active (non-zero),
        not the magnitude of scores.

        IMPORTANT: Two applications with no features are NOT similar (returns 0.0).

        Args:
            app1: First application
            app2: Second application

        Returns:
            Jaccard similarity between 0 and 1
        """
        dims1 = set(app1.active_dimensions)
        dims2 = set(app2.active_dimensions)

        # If both apps have no features, they are NOT similar
        if not dims1 and not dims2:
            return 0.0

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

        IMPORTANT: Similarity is biased towards applications WITH features.
        Two applications with all zeros are NOT similar (returns 0.0).

        Args:
            app1: First application
            app2: Second application

        Returns:
            Tuple of (overall_similarity, dimension_details)
            where dimension_details is a list of dicts with keys:
                - 'dimension': dimension index
                - 'score1': score for app1
                - 'score2': score for app2
                - 'similarity': similarity for this dimension
                - 'weight': weight applied
                - 'shared': boolean indicating if both apps have this feature
        """
        # Get active dimensions for each app
        active1 = set(app1.active_dimensions)
        active2 = set(app2.active_dimensions)

        # If both apps have no features, they are NOT similar
        if not active1 and not active2:
            return 0.0, []

        # Find shared and all relevant dimensions
        shared_dims = active1 & active2
        all_relevant_dims = active1 | active2

        # If no shared active dimensions, similarity is very low
        if not shared_dims:
            total_dims = len(all_relevant_dims)
            similarity = 0.1 / (1 + total_dims)

            # Still provide dimension details
            dimension_details = []
            for dim in sorted(all_relevant_dims):
                score1 = app1.get_score(dim)
                score2 = app2.get_score(dim)
                dimension_details.append({
                    "dimension": dim,
                    "score1": score1,
                    "score2": score2,
                    "similarity": 0.0,
                    "weight": 0.0,
                    "shared": False,
                })
            return similarity, dimension_details

        # Calculate similarity for shared dimensions
        dimension_details = []
        shared_similarity = 0.0
        shared_weight = 0.0

        # Process all relevant dimensions
        for dim in sorted(all_relevant_dims):
            score1 = app1.get_score(dim)
            score2 = app2.get_score(dim)
            is_shared = dim in shared_dims

            if is_shared:
                # Both apps have this dimension
                max_score = max(score1, score2)
                dim_similarity = 1 - abs(score1 - score2) / max_score
                weight = min(score1, score2)
                shared_similarity += dim_similarity * weight
                shared_weight += weight
            else:
                # Only one app has this dimension (asymmetric)
                dim_similarity = 0.0
                weight = 0.0

            dimension_details.append({
                "dimension": dim,
                "score1": score1,
                "score2": score2,
                "similarity": dim_similarity,
                "weight": weight,
                "shared": is_shared,
            })

        # Calculate base similarity from shared dimensions
        if shared_weight > 0:
            base_similarity = shared_similarity / shared_weight
        else:
            base_similarity = 0.0

        # Apply penalty for asymmetric dimensions
        total_active_dims = len(all_relevant_dims)
        shared_ratio = len(shared_dims) / total_active_dims if total_active_dims > 0 else 0

        # Final similarity
        overall_similarity = base_similarity * (0.5 + 0.5 * shared_ratio)

        return overall_similarity, dimension_details
