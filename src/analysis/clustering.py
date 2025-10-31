"""
Clustering engine for identifying similar applications.
"""
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN

# Support both relative and absolute imports
try:
    from ..core.feature_matrix import FeatureMatrix
    from .similarity import SimilarityCalculator
except ImportError:
    from core.feature_matrix import FeatureMatrix
    from analysis.similarity import SimilarityCalculator


class ClusteringEngine:
    """
    Performs clustering analysis on applications to identify similar groups.
    """

    def __init__(self, feature_matrix: FeatureMatrix):
        """
        Initialize the clustering engine.

        Args:
            feature_matrix: FeatureMatrix containing applications to cluster
        """
        self.feature_matrix = feature_matrix
        self.similarity_calculator = SimilarityCalculator()
        self._similarity_matrix = None

    def calculate_similarity_matrix(self, method: str = "proportional") -> np.ndarray:
        """
        Calculate pairwise similarity matrix for all applications.

        Args:
            method: Similarity method to use ("proportional", "jaccard", "cosine")

        Returns:
            Symmetric similarity matrix of shape (n_apps, n_apps)
        """
        app_names = self.feature_matrix.get_application_names()
        n_apps = len(app_names)
        similarity_matrix = np.zeros((n_apps, n_apps))

        for i in range(n_apps):
            for j in range(i, n_apps):
                app1 = self.feature_matrix.get_application(app_names[i])
                app2 = self.feature_matrix.get_application(app_names[j])

                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    if method == "proportional":
                        sim = self.similarity_calculator.proportional_similarity(
                            app1, app2
                        )
                    elif method == "jaccard":
                        sim = self.similarity_calculator.jaccard_similarity(app1, app2)
                    elif method == "cosine":
                        sim = self.similarity_calculator.cosine_similarity_score(
                            app1, app2, self.feature_matrix.all_dimensions
                        )
                    else:
                        raise ValueError(f"Unknown similarity method: {method}")

                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim  # Symmetric

        self._similarity_matrix = similarity_matrix
        return similarity_matrix

    def get_proximity_rankings(
        self, method: str = "proportional", top_n: int = None
    ) -> List[Tuple[str, str, float]]:
        """
        Get ranked list of application pairs by similarity.

        Args:
            method: Similarity method to use
            top_n: Optional limit on number of pairs to return

        Returns:
            List of tuples (app1_name, app2_name, similarity_score) sorted by similarity
        """
        if self._similarity_matrix is None:
            self.calculate_similarity_matrix(method)

        app_names = self.feature_matrix.get_application_names()
        n_apps = len(app_names)

        # Extract upper triangle (excluding diagonal) as a list of pairs
        pairs = []
        for i in range(n_apps):
            for j in range(i + 1, n_apps):
                similarity = self._similarity_matrix[i, j]
                pairs.append((app_names[i], app_names[j], similarity))

        # Sort by similarity (descending)
        pairs.sort(key=lambda x: x[2], reverse=True)

        if top_n:
            pairs = pairs[:top_n]

        return pairs

    def get_similar_apps(
        self, app_name: str, method: str = "proportional", top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get the most similar applications to a given application.

        Args:
            app_name: Name of the application to find similarities for
            method: Similarity method to use
            top_n: Number of similar apps to return

        Returns:
            List of tuples (similar_app_name, similarity_score) sorted by similarity
        """
        if self._similarity_matrix is None:
            self.calculate_similarity_matrix(method)

        app_names = self.feature_matrix.get_application_names()

        if app_name not in app_names:
            raise ValueError(f"Application '{app_name}' not found in feature matrix")

        app_idx = app_names.index(app_name)
        similarities = self._similarity_matrix[app_idx]

        # Create list of (app_name, similarity) excluding self
        similar_apps = [
            (app_names[i], similarities[i])
            for i in range(len(app_names))
            if i != app_idx
        ]

        # Sort by similarity (descending)
        similar_apps.sort(key=lambda x: x[1], reverse=True)

        return similar_apps[:top_n]

    def hierarchical_clustering(
        self, n_clusters: int = None, method: str = "proportional"
    ) -> Dict[str, int]:
        """
        Perform hierarchical clustering on applications.

        Args:
            n_clusters: Number of clusters to form (if None, uses distance threshold)
            method: Similarity method to use

        Returns:
            Dictionary mapping application names to cluster labels
        """
        if self._similarity_matrix is None:
            self.calculate_similarity_matrix(method)

        # Convert similarity to distance
        distance_matrix = 1 - self._similarity_matrix

        # Perform clustering
        if n_clusters:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, metric="precomputed", linkage="average"
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric="precomputed",
                linkage="average",
            )

        labels = clusterer.fit_predict(distance_matrix)

        # Map application names to cluster labels
        app_names = self.feature_matrix.get_application_names()
        return {app_names[i]: int(labels[i]) for i in range(len(app_names))}

    def get_similarity_dataframe(self, method: str = "proportional") -> pd.DataFrame:
        """
        Get similarity matrix as a pandas DataFrame with app names as indices.

        Args:
            method: Similarity method to use

        Returns:
            DataFrame with similarity matrix
        """
        if self._similarity_matrix is None:
            self.calculate_similarity_matrix(method)

        app_names = self.feature_matrix.get_application_names()
        return pd.DataFrame(
            self._similarity_matrix, index=app_names, columns=app_names
        )

    def get_detailed_comparison(
        self, app1_name: str, app2_name: str
    ) -> Tuple[float, pd.DataFrame]:
        """
        Get detailed comparison between two applications.

        Args:
            app1_name: Name of first application
            app2_name: Name of second application

        Returns:
            Tuple of (overall_similarity, details_dataframe)
        """
        app1 = self.feature_matrix.get_application(app1_name)
        app2 = self.feature_matrix.get_application(app2_name)

        if app1 is None:
            raise ValueError(f"Application '{app1_name}' not found")
        if app2 is None:
            raise ValueError(f"Application '{app2_name}' not found")

        similarity, details = (
            self.similarity_calculator.weighted_proportional_similarity(app1, app2)
        )

        # Convert details to DataFrame with dimension names
        if details:
            for detail in details:
                dim_idx = detail["dimension"]
                detail["dimension_name"] = self.feature_matrix.get_dimension_name(
                    dim_idx
                )

            df = pd.DataFrame(details)
            df = df[
                [
                    "dimension",
                    "dimension_name",
                    "score1",
                    "score2",
                    "similarity",
                    "weight",
                ]
            ]
        else:
            df = pd.DataFrame()

        return similarity, df
