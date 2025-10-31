"""
Visualization utilities for displaying analysis results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from typing import Dict, List, Tuple
import pandas as pd


class Visualizer:
    """
    Creates visualizations for clustering and similarity analysis.
    """

    @staticmethod
    def plot_similarity_heatmap(
        similarity_matrix: np.ndarray, app_names: List[str], figsize: Tuple[int, int] = (12, 10)
    ) -> Figure:
        """
        Create a heatmap of the similarity matrix.

        Args:
            similarity_matrix: Square matrix of pairwise similarities
            app_names: List of application names (labels)
            figsize: Figure size in inches

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            similarity_matrix,
            xticklabels=app_names,
            yticklabels=app_names,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            square=True,
            ax=ax,
            cbar_kws={"label": "Similarity Score"},
        )

        ax.set_title("Application Similarity Heatmap", fontsize=16, pad=20)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_cluster_distribution(
        clusters: Dict[str, int], figsize: Tuple[int, int] = (10, 6)
    ) -> Figure:
        """
        Create a bar chart showing the distribution of applications across clusters.

        Args:
            clusters: Dictionary mapping app names to cluster IDs
            figsize: Figure size in inches

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Count applications per cluster
        cluster_counts = {}
        for cluster_id in clusters.values():
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

        # Sort by cluster ID
        cluster_ids = sorted(cluster_counts.keys())
        counts = [cluster_counts[cid] for cid in cluster_ids]

        # Create bar chart
        bars = ax.bar(
            [f"Cluster {cid}" for cid in cluster_ids],
            counts,
            color=sns.color_palette("husl", len(cluster_ids)),
        )

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=12,
            )

        ax.set_xlabel("Cluster", fontsize=12)
        ax.set_ylabel("Number of Applications", fontsize=12)
        ax.set_title("Application Distribution Across Clusters", fontsize=16, pad=20)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_top_similarities(
        rankings: List[Tuple[str, str, float]],
        top_n: int = 15,
        figsize: Tuple[int, int] = (12, 8),
    ) -> Figure:
        """
        Create a horizontal bar chart of top similar application pairs.

        Args:
            rankings: List of (app1, app2, similarity) tuples
            top_n: Number of top pairs to show
            figsize: Figure size in inches

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Take top N rankings
        top_rankings = rankings[:top_n]

        # Create labels and scores
        labels = [f"{app1} ↔ {app2}" for app1, app2, _ in top_rankings]
        scores = [score for _, _, score in top_rankings]

        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        colors = sns.color_palette("RdYlGn", len(labels))

        bars = ax.barh(y_pos, scores, color=colors)

        # Add score labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            ax.text(
                score + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}",
                va="center",
                fontsize=10,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()  # Highest similarity at top
        ax.set_xlabel("Similarity Score", fontsize=12)
        ax.set_title(f"Top {top_n} Most Similar Application Pairs", fontsize=16, pad=20)
        ax.set_xlim(0, 1.1)
        plt.tight_layout()

        return fig

    @staticmethod
    def plot_dimension_comparison(
        comparison_df: pd.DataFrame, app1_name: str, app2_name: str, figsize: Tuple[int, int] = (12, 8)
    ) -> Figure:
        """
        Create a grouped bar chart comparing two applications across dimensions.

        Args:
            comparison_df: DataFrame with dimension comparison details
            app1_name: Name of first application
            app2_name: Name of second application
            figsize: Figure size in inches

        Returns:
            Matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Extract data
        dimensions = comparison_df["dimension_name"].tolist()
        scores1 = comparison_df["score1"].tolist()
        scores2 = comparison_df["score2"].tolist()

        # Set up bar positions
        x = np.arange(len(dimensions))
        width = 0.35

        # Create grouped bars
        bars1 = ax.bar(x - width / 2, scores1, width, label=app1_name, alpha=0.8)
        bars2 = ax.bar(x + width / 2, scores2, width, label=app2_name, alpha=0.8)

        # Customize
        ax.set_xlabel("Dimensions", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(
            f"Dimension Comparison: {app1_name} vs {app2_name}", fontsize=16, pad=20
        )
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions, rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()

        return fig

    @staticmethod
    def save_figure(fig: Figure, filepath: str, dpi: int = 300):
        """
        Save a figure to file.

        Args:
            fig: Matplotlib Figure object
            filepath: Output file path
            dpi: Resolution in dots per inch
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
