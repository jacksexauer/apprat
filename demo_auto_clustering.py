#!/usr/bin/env python3
"""
Demonstration of automatic clustering functionality.

Shows how the application can dynamically determine the optimal number of clusters
without requiring manual specification.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import core.csv_loader as csv_loader
import analysis.clustering as clustering

CSVLoader = csv_loader.CSVLoader
ClusteringEngine = clustering.ClusteringEngine


def print_cluster_results(clusters, metadata, title):
    """Print clustering results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")

    # Print metadata
    print(f"\n📊 Clustering Metadata:")
    for key, value in metadata.items():
        if key == 'all_scores':
            continue  # Skip the detailed scores list
        print(f"  {key}: {value}")

    # Group by cluster
    cluster_groups = {}
    for app_name, cluster_id in clusters.items():
        if cluster_id not in cluster_groups:
            cluster_groups[cluster_id] = []
        cluster_groups[cluster_id].append(app_name)

    # Print clusters
    print(f"\n🗂️  Cluster Distribution:")
    print(f"  Total applications: {len(clusters)}")
    print(f"  Total clusters: {len(cluster_groups)}")

    print(f"\n📋 Cluster Assignments:")
    for cluster_id in sorted(cluster_groups.keys()):
        apps = sorted(cluster_groups[cluster_id])
        cluster_label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Outliers"
        print(f"\n  {cluster_label} ({len(apps)} applications):")
        for app_name in apps:
            print(f"    - {app_name}")


def main():
    """Run automatic clustering demonstrations."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "AUTOMATIC CLUSTERING DEMONSTRATION" + " "*19 + "║")
    print("║" + " "*10 + "No Need to Specify Number of Clusters!" + " "*17 + "║")
    print("╚" + "="*68 + "╝")

    # Load sample data
    print("\n📂 Loading sample data...")
    data_dir = Path(__file__).parent / "data"
    matrix_file = data_dir / "sample_applications.csv"
    mapping_file = data_dir / "sample_dimensions.csv"

    feature_matrix = CSVLoader.load_feature_matrix(
        str(matrix_file), str(mapping_file)
    )

    print(f"✓ Loaded {len(feature_matrix)} applications")
    print(f"✓ Found {len(feature_matrix.all_dimensions)} dimensions")

    # Create clustering engine
    engine = ClusteringEngine(feature_matrix)

    # Method 1: Silhouette-optimized clustering
    print("\n" + "="*70)
    print("METHOD 1: Silhouette-Optimized Clustering")
    print("="*70)
    print("\n🔍 Analyzes cluster quality across different numbers of clusters")
    print("   Uses silhouette score to find the optimal grouping")
    print("\n⏳ Calculating optimal clusters...")

    clusters1, metadata1 = engine.auto_cluster(method="proportional")
    print_cluster_results(clusters1, metadata1, "✅ Silhouette-Optimized Results")

    # Show the scores for different k values
    if 'all_scores' in metadata1:
        print(f"\n📈 Silhouette Scores for Different Cluster Counts:")
        for k, score in metadata1['all_scores'][:10]:  # Show first 10
            indicator = " ← BEST" if k == metadata1['n_clusters'] else ""
            print(f"  k={k}: {score:.4f}{indicator}")

    # Method 2: Threshold-based clustering
    print("\n\n" + "="*70)
    print("METHOD 2: Threshold-Based Clustering")
    print("="*70)
    print("\n🔍 Groups applications with similarity above a threshold")
    print("   Automatically detects the threshold from similarity distribution")
    print("\n⏳ Finding natural groupings...")

    clusters2, metadata2 = engine.auto_cluster_threshold(method="proportional")
    print_cluster_results(clusters2, metadata2, "✅ Threshold-Based Results")

    print(f"\n💡 Interpretation:")
    print(f"   Applications are grouped if their similarity ≥ {metadata2['threshold_used']:.3f}")
    print(f"   This creates {metadata2['n_clusters']} natural groups")

    # Method 3: DBSCAN clustering
    print("\n\n" + "="*70)
    print("METHOD 3: DBSCAN (Density-Based) Clustering")
    print("="*70)
    print("\n🔍 Finds clusters based on density")
    print("   Can identify outliers that don't fit any cluster")
    print("\n⏳ Detecting density-based clusters...")

    clusters3, metadata3 = engine.dbscan_clustering(method="proportional")
    print_cluster_results(clusters3, metadata3, "✅ DBSCAN Results")

    if metadata3['n_noise'] > 0:
        print(f"\n⚠️  Note: {metadata3['n_noise']} applications identified as outliers")
        print(f"   These applications don't strongly belong to any cluster")

    # Comparison
    print("\n\n" + "="*70)
    print("COMPARISON OF AUTOMATIC CLUSTERING METHODS")
    print("="*70)

    comparison_df = engine.compare_clustering_methods(method="proportional")
    print("\n" + comparison_df.to_string(index=False))

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)

    print("\n✅ All three methods automatically determined cluster counts!")
    print("\n📊 Results:")
    print(f"  • Silhouette-Optimized: {metadata1['n_clusters']} clusters")
    print(f"  • Threshold-Based:      {metadata2['n_clusters']} clusters")
    print(f"  • DBSCAN:               {metadata3['n_clusters']} clusters")

    print("\n💡 Which method to use?")
    print("\n  🏆 Silhouette-Optimized (Recommended)")
    print("     ✓ Best overall cluster quality")
    print("     ✓ Works well for most datasets")
    print("     ✓ Balances cluster size and coherence")

    print("\n  🎯 Threshold-Based")
    print("     ✓ Creates natural, intuitive groups")
    print("     ✓ Good for business-driven clustering")
    print("     ✓ Easy to explain: 'apps with >X% similarity'")

    print("\n  🔬 DBSCAN")
    print("     ✓ Identifies outliers/anomalies")
    print("     ✓ Good for finding unique applications")
    print("     ✓ No assumption about cluster sizes")

    print("\n🎯 Business Value:")
    print("   ✓ No need to guess the number of clusters")
    print("   ✓ Data-driven cluster formation")
    print("   ✓ More objective consolidation recommendations")
    print("   ✓ Multiple perspectives on natural groupings")

    print("\n" + "="*70)
    print("For GUI interface with automatic clustering, run: python run.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
