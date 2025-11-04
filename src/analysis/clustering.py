"""
Clustering engine for identifying similar applications.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

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

    def auto_cluster(
        self, method: str = "proportional", min_clusters: int = 2, max_clusters: int = None
    ) -> Tuple[Dict[str, int], Dict[str, any]]:
        """
        Automatically determine optimal number of clusters and perform clustering.

        Uses silhouette analysis to find the optimal number of clusters that
        naturally groups similar applications without requiring manual specification.

        Args:
            method: Similarity method to use
            min_clusters: Minimum number of clusters to consider
            max_clusters: Maximum number of clusters to consider (default: n_apps // 2)

        Returns:
            Tuple of (cluster_assignments, metadata)
            - cluster_assignments: Dict mapping app names to cluster labels
            - metadata: Dict with keys:
                - 'n_clusters': Optimal number of clusters found
                - 'silhouette_score': Quality metric (higher is better)
                - 'method_used': Algorithm used
                - 'all_scores': List of scores for each k tried
        """
        if self._similarity_matrix is None:
            self.calculate_similarity_matrix(method)

        n_apps = len(self.feature_matrix.get_application_names())

        # Set reasonable bounds
        if max_clusters is None:
            max_clusters = min(n_apps - 1, n_apps // 2 + 2)

        max_clusters = min(max_clusters, n_apps - 1)
        min_clusters = max(2, min_clusters)

        if min_clusters >= n_apps:
            # Too few apps to cluster
            app_names = self.feature_matrix.get_application_names()
            return (
                {name: 0 for name in app_names},
                {
                    'n_clusters': 1,
                    'silhouette_score': 0.0,
                    'method_used': 'single_cluster',
                    'all_scores': []
                }
            )

        # Try different numbers of clusters and evaluate
        best_score = -1
        best_k = min_clusters
        all_scores = []

        distance_matrix = 1 - self._similarity_matrix

        for k in range(min_clusters, max_clusters + 1):
            try:
                # Perform clustering
                clusterer = AgglomerativeClustering(
                    n_clusters=k,
                    metric="precomputed",
                    linkage="average"
                )
                labels = clusterer.fit_predict(distance_matrix)

                # Calculate silhouette score (measures cluster quality)
                score = silhouette_score(distance_matrix, labels, metric="precomputed")
                all_scores.append((k, score))

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                # Skip if clustering fails for this k
                all_scores.append((k, -1))
                continue

        # Perform final clustering with optimal k
        clusterer = AgglomerativeClustering(
            n_clusters=best_k,
            metric="precomputed",
            linkage="average"
        )
        labels = clusterer.fit_predict(distance_matrix)

        # Map to application names
        app_names = self.feature_matrix.get_application_names()
        cluster_assignments = {app_names[i]: int(labels[i]) for i in range(len(app_names))}

        metadata = {
            'n_clusters': best_k,
            'silhouette_score': best_score,
            'method_used': 'silhouette_optimization',
            'all_scores': all_scores
        }

        return cluster_assignments, metadata

    def auto_cluster_threshold(
        self, method: str = "proportional", similarity_threshold: float = None
    ) -> Tuple[Dict[str, int], Dict[str, any]]:
        """
        Automatically cluster based on similarity threshold.

        Applications are grouped together if their similarity exceeds the threshold.
        This creates natural clusters without specifying a number.

        Args:
            method: Similarity method to use
            similarity_threshold: Minimum similarity for grouping (default: auto-detect)

        Returns:
            Tuple of (cluster_assignments, metadata)
            - cluster_assignments: Dict mapping app names to cluster labels
            - metadata: Dict with keys:
                - 'n_clusters': Number of clusters formed
                - 'threshold_used': Similarity threshold applied
                - 'method_used': Algorithm used
        """
        if self._similarity_matrix is None:
            self.calculate_similarity_matrix(method)

        # Auto-detect threshold if not provided
        if similarity_threshold is None:
            similarity_threshold = self._find_optimal_threshold()

        # Convert similarity threshold to distance threshold
        distance_threshold = 1 - similarity_threshold

        # Perform hierarchical clustering with distance threshold
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average"
        )

        distance_matrix = 1 - self._similarity_matrix
        labels = clusterer.fit_predict(distance_matrix)

        # Map to application names
        app_names = self.feature_matrix.get_application_names()
        cluster_assignments = {app_names[i]: int(labels[i]) for i in range(len(app_names))}

        n_clusters = len(set(labels))

        metadata = {
            'n_clusters': n_clusters,
            'threshold_used': similarity_threshold,
            'distance_threshold': distance_threshold,
            'method_used': 'threshold_based'
        }

        return cluster_assignments, metadata

    def _find_optimal_threshold(self) -> float:
        """
        Find optimal similarity threshold based on the distribution of similarities.

        Uses the gap between high and low similarities to determine a natural cutoff.

        Returns:
            Optimal similarity threshold
        """
        # Get all pairwise similarities (upper triangle, excluding diagonal)
        n = len(self._similarity_matrix)
        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                similarities.append(self._similarity_matrix[i, j])

        similarities = np.array(similarities)

        if len(similarities) == 0:
            return 0.5

        # Sort similarities
        sorted_sims = np.sort(similarities)

        # Find the largest gap in similarities
        # This represents the natural boundary between "similar" and "dissimilar"
        gaps = np.diff(sorted_sims)

        if len(gaps) == 0:
            return 0.5

        # Find the gap in the upper half (we want to find similar apps)
        upper_half_start = len(sorted_sims) // 2
        if upper_half_start < len(gaps):
            largest_gap_idx = upper_half_start + np.argmax(gaps[upper_half_start:])
            threshold = (sorted_sims[largest_gap_idx] + sorted_sims[largest_gap_idx + 1]) / 2
        else:
            # Use median if no clear gap
            threshold = np.median(similarities)

        # Ensure threshold is reasonable (between 0.3 and 0.8)
        threshold = max(0.3, min(0.8, threshold))

        return threshold

    def dbscan_clustering(
        self, method: str = "proportional", eps: float = None, min_samples: int = 2
    ) -> Tuple[Dict[str, int], Dict[str, any]]:
        """
        Perform DBSCAN clustering (density-based clustering).

        DBSCAN automatically finds clusters based on density and can identify outliers.
        Applications that don't fit well into any cluster are marked as noise (-1).

        Args:
            method: Similarity method to use
            eps: Maximum distance for points to be neighbors (auto-detected if None)
            min_samples: Minimum points to form a dense region

        Returns:
            Tuple of (cluster_assignments, metadata)
            - cluster_assignments: Dict mapping app names to cluster labels (-1 = noise)
            - metadata: Dict with keys:
                - 'n_clusters': Number of clusters found (excluding noise)
                - 'n_noise': Number of applications marked as noise
                - 'eps_used': Distance parameter used
                - 'method_used': Algorithm used
        """
        if self._similarity_matrix is None:
            self.calculate_similarity_matrix(method)

        distance_matrix = 1 - self._similarity_matrix

        # Auto-detect eps if not provided
        if eps is None:
            # Use mean of nearest neighbor distances
            nearest_neighbor_dists = []
            for i in range(len(distance_matrix)):
                dists = distance_matrix[i].copy()
                dists[i] = np.inf  # Exclude self
                nearest_neighbor_dists.append(np.min(dists))
            eps = np.mean(nearest_neighbor_dists)

        # Perform DBSCAN
        clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="precomputed"
        )
        labels = clusterer.fit_predict(distance_matrix)

        # Map to application names
        app_names = self.feature_matrix.get_application_names()
        cluster_assignments = {app_names[i]: int(labels[i]) for i in range(len(app_names))}

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        metadata = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'eps_used': eps,
            'method_used': 'dbscan'
        }

        return cluster_assignments, metadata

    def compare_clustering_methods(
        self, method: str = "proportional"
    ) -> pd.DataFrame:
        """
        Compare different automatic clustering methods.

        Runs multiple clustering approaches and compares their results to help
        understand how the data naturally groups.

        Args:
            method: Similarity method to use

        Returns:
            DataFrame comparing different clustering methods
        """
        results = []

        # Method 1: Silhouette-optimized
        try:
            clusters1, meta1 = self.auto_cluster(method=method)
            results.append({
                'Method': 'Silhouette-Optimized',
                'N Clusters': meta1['n_clusters'],
                'Quality Score': f"{meta1['silhouette_score']:.4f}",
                'Description': 'Optimal based on silhouette analysis'
            })
        except Exception as e:
            results.append({
                'Method': 'Silhouette-Optimized',
                'N Clusters': 'Error',
                'Quality Score': 'N/A',
                'Description': str(e)
            })

        # Method 2: Threshold-based
        try:
            clusters2, meta2 = self.auto_cluster_threshold(method=method)
            results.append({
                'Method': 'Threshold-Based',
                'N Clusters': meta2['n_clusters'],
                'Quality Score': f"threshold={meta2['threshold_used']:.4f}",
                'Description': f'Groups apps with similarity > {meta2["threshold_used"]:.2f}'
            })
        except Exception as e:
            results.append({
                'Method': 'Threshold-Based',
                'N Clusters': 'Error',
                'Quality Score': 'N/A',
                'Description': str(e)
            })

        # Method 3: DBSCAN
        try:
            clusters3, meta3 = self.dbscan_clustering(method=method)
            results.append({
                'Method': 'DBSCAN',
                'N Clusters': meta3['n_clusters'],
                'Quality Score': f"{meta3['n_noise']} outliers",
                'Description': f'Density-based, eps={meta3["eps_used"]:.4f}'
            })
        except Exception as e:
            results.append({
                'Method': 'DBSCAN',
                'N Clusters': 'Error',
                'Quality Score': 'N/A',
                'Description': str(e)
            })

        return pd.DataFrame(results)

    def analyze_cluster_features(
        self, cluster_assignments: Dict[str, int], cluster_id: int
    ) -> Dict[str, any]:
        """
        Analyze the most significant features for a specific cluster.

        Identifies which dimensions are most characteristic of the applications
        in this cluster - i.e., the features that define this group.

        Args:
            cluster_assignments: Dict mapping app names to cluster IDs
            cluster_id: The cluster to analyze

        Returns:
            Dict with keys:
                - 'applications': List of application names in cluster
                - 'size': Number of applications
                - 'significant_features': List of (dimension, score, name) tuples
                - 'avg_scores': Average scores for each dimension
                - 'active_count': Count of apps with non-zero score per dimension
        """
        # Get applications in this cluster
        app_names = [name for name, cid in cluster_assignments.items() if cid == cluster_id]

        if not app_names:
            return {
                'applications': [],
                'size': 0,
                'significant_features': [],
                'avg_scores': {},
                'active_count': {}
            }

        # Calculate statistics for each dimension
        all_dims = self.feature_matrix.all_dimensions
        dim_stats = {}

        for dim in all_dims:
            scores = []
            active_count = 0

            for app_name in app_names:
                app = self.feature_matrix.get_application(app_name)
                score = app.get_score(dim)
                scores.append(score)
                if score > 0:
                    active_count += 1

            avg_score = sum(scores) / len(scores) if scores else 0
            dim_stats[dim] = {
                'avg_score': avg_score,
                'active_count': active_count,
                'active_ratio': active_count / len(app_names) if app_names else 0
            }

        # Identify significant features
        # A feature is significant if:
        # 1. Many apps in the cluster have it (high active_ratio)
        # 2. The average score is high
        significant_features = []

        for dim, stats in dim_stats.items():
            # Weight by both presence and magnitude
            significance = stats['active_ratio'] * stats['avg_score']

            if significance > 0:
                dim_name = self.feature_matrix.get_dimension_name(dim)
                significant_features.append({
                    'dimension': dim,
                    'dimension_name': dim_name,
                    'avg_score': stats['avg_score'],
                    'active_count': stats['active_count'],
                    'active_ratio': stats['active_ratio'],
                    'significance': significance
                })

        # Sort by significance
        significant_features.sort(key=lambda x: x['significance'], reverse=True)

        return {
            'applications': app_names,
            'size': len(app_names),
            'significant_features': significant_features,
            'avg_scores': {dim: stats['avg_score'] for dim, stats in dim_stats.items()},
            'active_count': {dim: stats['active_count'] for dim, stats in dim_stats.items()}
        }

    def get_cluster_summary(
        self, cluster_assignments: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Get a summary of all clusters with their characteristics.

        Args:
            cluster_assignments: Dict mapping app names to cluster IDs

        Returns:
            DataFrame with cluster summaries
        """
        cluster_ids = sorted(set(cluster_assignments.values()))
        summaries = []

        for cluster_id in cluster_ids:
            analysis = self.analyze_cluster_features(cluster_assignments, cluster_id)

            # Get top 3 features
            top_features = analysis['significant_features'][:3]
            feature_names = [f['dimension_name'] for f in top_features]
            feature_str = ', '.join(feature_names) if feature_names else 'No strong features'

            cluster_label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Outliers"

            summaries.append({
                'Cluster': cluster_label,
                'Size': analysis['size'],
                'Top Features': feature_str,
                'Applications': ', '.join(sorted(analysis['applications'])[:3]) +
                               (f' +{analysis["size"]-3} more' if analysis['size'] > 3 else '')
            })

        return pd.DataFrame(summaries)
