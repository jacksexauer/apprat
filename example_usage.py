#!/usr/bin/env python3
"""
Example script demonstrating programmatic usage of apprat core functionality.

This script shows how to use the apprat library without the GUI.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import after path is set
import core.csv_loader as csv_loader
import analysis.clustering as clustering
import analysis.similarity as similarity

CSVLoader = csv_loader.CSVLoader
ClusteringEngine = clustering.ClusteringEngine
SimilarityCalculator = similarity.SimilarityCalculator


def main():
    """Run example analysis."""
    print("=" * 60)
    print("apprat - Application Rationalization Tool")
    print("Programmatic Usage Example")
    print("=" * 60)
    print()

    # Load data
    print("Loading sample data...")
    data_dir = Path(__file__).parent / "data"
    matrix_file = data_dir / "sample_applications.csv"
    mapping_file = data_dir / "sample_dimensions.csv"

    feature_matrix = CSVLoader.load_feature_matrix(
        str(matrix_file), str(mapping_file)
    )

    print(f"✓ Loaded {len(feature_matrix)} applications")
    print(f"✓ Found {len(feature_matrix.all_dimensions)} dimensions")
    print()

    # Display applications
    print("Applications:")
    for app_name in feature_matrix.get_application_names():
        app = feature_matrix.get_application(app_name)
        print(f"  - {app_name} ({app.num_active_dimensions} active dimensions)")
    print()

    # Create clustering engine
    engine = ClusteringEngine(feature_matrix)

    # Calculate similarity rankings
    print("Calculating similarity rankings...")
    rankings = engine.get_proximity_rankings(method="proportional", top_n=10)

    print()
    print("Top 10 Most Similar Application Pairs:")
    print("-" * 60)
    for i, (app1, app2, score) in enumerate(rankings, 1):
        print(f"{i:2d}. {app1:20s} ↔ {app2:20s}  Score: {score:.4f}")
    print()

    # Find similar apps for a specific application
    target_app = "CRM System"
    print(f"Finding applications similar to '{target_app}'...")
    similar_apps = engine.get_similar_apps(target_app, method="proportional", top_n=5)

    print()
    print(f"Top 5 Applications Similar to '{target_app}':")
    print("-" * 60)
    for i, (app_name, score) in enumerate(similar_apps, 1):
        print(f"{i}. {app_name:30s}  Similarity: {score:.4f}")
    print()

    # Detailed comparison
    app1_name = rankings[0][0]
    app2_name = rankings[0][1]
    print(f"Detailed comparison: {app1_name} vs {app2_name}")
    print("-" * 60)

    similarity, details_df = engine.get_detailed_comparison(app1_name, app2_name)
    print(f"Overall Similarity: {similarity:.4f}")
    print()
    print("Dimension-by-dimension breakdown:")
    print(details_df.to_string(index=False))
    print()

    # Clustering
    print("Performing hierarchical clustering (3 clusters)...")
    clusters = engine.hierarchical_clustering(n_clusters=3, method="proportional")

    print()
    print("Cluster Assignments:")
    print("-" * 60)

    # Group by cluster
    cluster_groups = {}
    for app_name, cluster_id in clusters.items():
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(app_name)

    for cluster_id in sorted(cluster_groups.keys()):
        apps = sorted(cluster_groups[cluster_id])
        print(f"\nCluster {cluster_id} ({len(apps)} applications):")
        for app_name in apps:
            print(f"  - {app_name}")

    print()
    print("=" * 60)
    print("Analysis complete!")
    print()
    print("For GUI interface, run: python run.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
