#!/usr/bin/env python3
"""
Demonstration of cluster feature analysis.

Shows how clusters are characterized by their significant features,
helping understand why applications are grouped together.
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


def print_cluster_analysis(engine, clusters, cluster_id):
    """Print detailed analysis of a cluster."""
    analysis = engine.analyze_cluster_features(clusters, cluster_id)

    cluster_label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Outliers"

    print(f"\n{'='*70}")
    print(f"{cluster_label} - Detailed Analysis")
    print(f"{'='*70}")

    # Applications
    print(f"\n📱 Applications ({analysis['size']} total):")
    for app_name in sorted(analysis['applications']):
        print(f"  • {app_name}")

    # Significant features
    print(f"\n🔑 Significant Features:")
    print(f"{'='*70}")
    features = analysis['significant_features'][:10]  # Top 10

    if features:
        print(f"{'Rank':<6} {'Feature':<25} {'Avg Score':<12} {'Coverage':<15} {'Significance':<12}")
        print(f"{'-'*70}")

        for rank, feature in enumerate(features, 1):
            coverage = f"{feature['active_count']}/{analysis['size']} ({feature['active_ratio']*100:.0f}%)"
            print(
                f"{rank:<6} "
                f"{feature['dimension_name']:<25} "
                f"{feature['avg_score']:<12.2f} "
                f"{coverage:<15} "
                f"{feature['significance']:<12.2f}"
            )

        print(f"\n💡 Interpretation:")
        print(f"  This cluster is characterized by:")

        # Top 3 features
        for i, feature in enumerate(features[:3], 1):
            coverage_pct = feature['active_ratio'] * 100
            print(f"    {i}. {feature['dimension_name']} "
                  f"(avg: {feature['avg_score']:.1f}, {coverage_pct:.0f}% of apps)")

    else:
        print("  No significant features identified for this cluster.")


def main():
    """Run cluster analysis demonstration."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*18 + "CLUSTER FEATURE ANALYSIS" + " "*26 + "║")
    print("║" + " "*12 + "Understanding Why Apps Are Grouped" + " "*22 + "║")
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

    # Perform automatic clustering
    print("\n🔍 Performing automatic clustering...")
    clusters, metadata = engine.auto_cluster(method="proportional")

    print(f"\n✅ Found {metadata['n_clusters']} clusters automatically")
    print(f"   Quality score (silhouette): {metadata['silhouette_score']:.4f}")

    # Get cluster summary
    print("\n" + "="*70)
    print("CLUSTER OVERVIEW")
    print("="*70)

    summary = engine.get_cluster_summary(clusters)
    print("\n" + summary.to_string(index=False))

    # Detailed analysis for each cluster
    cluster_ids = sorted(set(clusters.values()))

    print("\n\n" + "="*70)
    print("DETAILED CLUSTER ANALYSIS")
    print("="*70)

    for cluster_id in cluster_ids:
        print_cluster_analysis(engine, clusters, cluster_id)

    # Example: Comparing clusters
    print("\n\n" + "="*70)
    print("CLUSTER COMPARISON")
    print("="*70)

    print("\n🔬 Comparing Cluster 0 vs Cluster 1:")
    print("\nCluster 0 (HR-Focused):")
    analysis0 = engine.analyze_cluster_features(clusters, 0)
    top_features0 = [f['dimension_name'] for f in analysis0['significant_features'][:3]]
    print(f"  Top features: {', '.join(top_features0)}")
    print(f"  Applications: {', '.join(sorted(analysis0['applications']))}")

    print("\nCluster 1 (Customer/Sales-Focused):")
    analysis1 = engine.analyze_cluster_features(clusters, 1)
    top_features1 = [f['dimension_name'] for f in analysis1['significant_features'][:3]]
    print(f"  Top features: {', '.join(top_features1)}")
    print(f"  Applications: {', '.join(sorted(analysis1['applications']))}")

    print("\n💡 Key Differences:")
    unique_to_0 = set(top_features0) - set(top_features1)
    unique_to_1 = set(top_features1) - set(top_features0)

    if unique_to_0:
        print(f"  • Cluster 0 emphasizes: {', '.join(unique_to_0)}")
    if unique_to_1:
        print(f"  • Cluster 1 emphasizes: {', '.join(unique_to_1)}")

    # Summary
    print("\n\n" + "="*70)
    print("BUSINESS VALUE")
    print("="*70)

    print("\n✅ Benefits of Cluster Feature Analysis:")
    print("\n  1. 🎯 Understanding Clusters")
    print("     → See which features define each group")
    print("     → Understand why applications are grouped together")

    print("\n  2. 📊 Informed Decisions")
    print("     → Identify consolidation opportunities based on shared features")
    print("     → Prioritize which clusters to rationalize first")

    print("\n  3. 🔍 Gap Analysis")
    print("     → Identify missing capabilities within clusters")
    print("     → Find opportunities for feature standardization")

    print("\n  4. 💬 Stakeholder Communication")
    print("     → Explain clustering results with concrete features")
    print("     → Justify consolidation recommendations")

    print("\n📋 Example Business Use Cases:")
    print("\n  Scenario 1: Cluster 0 (HR-focused apps)")
    print("    Finding: High scores in User Management, Compliance, Workflow")
    print("    Action: → Consider consolidating into single HR platform")
    print("    Value: → Reduce redundancy, improve user experience")

    print("\n  Scenario 2: Cluster 1 (Customer-facing apps)")
    print("    Finding: High scores in Cloud Native, Reporting, API Integration")
    print("    Action: → Standardize on common tech stack")
    print("    Value: → Simplify integration, reduce maintenance")

    print("\n" + "="*70)
    print("For GUI interface with cluster analysis, run: python run.py")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
