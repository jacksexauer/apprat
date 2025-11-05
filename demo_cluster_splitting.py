#!/usr/bin/env python3
"""
Demonstration of cluster splitting functionality.

Shows how users can refine clustering results by splitting clusters that contain
distinct sub-groups of applications.
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


def print_cluster_composition(engine, clusters, cluster_id):
    """Print detailed composition of a cluster."""
    analysis = engine.analyze_cluster_features(clusters, cluster_id)

    # Handle string cluster IDs (including hierarchical ones)
    cluster_label = f"Cluster {cluster_id}" if str(cluster_id) != "-1" else "Outliers"

    print(f"\n{'='*70}")
    print(f"{cluster_label} - {analysis['size']} applications")
    print(f"{'='*70}")

    print(f"\n📱 Applications:")
    for app_name in sorted(analysis['applications']):
        print(f"  • {app_name}")

    print(f"\n🔑 Top Features:")
    for i, feature in enumerate(analysis['significant_features'][:5], 1):
        coverage = feature['active_ratio'] * 100
        print(f"  {i}. {feature['dimension_name']:<25} "
              f"(avg: {feature['avg_score']:.2f}, {coverage:.0f}% coverage)")


def main():
    """Run cluster splitting demonstration."""
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "CLUSTER SPLITTING DEMO" + " "*26 + "║")
    print("║" + " "*14 + "Refine Clusters with Intelligent Splitting" + " "*12 + "║")
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

    # Perform initial clustering
    print("\n" + "="*70)
    print("STEP 1: Initial Automatic Clustering")
    print("="*70)

    clusters, metadata = engine.auto_cluster(method="proportional")

    print(f"\n✅ Found {metadata['n_clusters']} clusters automatically")
    print(f"   Quality score: {metadata['silhouette_score']:.4f}")

    # Show cluster overview (with natural sorting for hierarchical IDs)
    print("\n📊 Cluster Overview:")

    def natural_sort_key(cluster_id):
        """Helper for natural sorting of cluster IDs."""
        parts = str(cluster_id).split('.')
        return tuple(int(p) if p.lstrip('-').isdigit() else p for p in parts)

    for cluster_id in sorted(set(clusters.values()), key=natural_sort_key):
        analysis = engine.analyze_cluster_features(clusters, cluster_id)
        top_features = [f['dimension_name'] for f in analysis['significant_features'][:2]]
        features_str = ', '.join(top_features)
        print(f"  Cluster {cluster_id}: {analysis['size']} apps - {features_str}")

    # Let's examine Cluster 1 (Customer/Sales) which might benefit from splitting
    cluster_to_split = "1"  # Using string ID

    print("\n\n" + "="*70)
    print(f"STEP 2: Examining Cluster {cluster_to_split} for Potential Split")
    print("="*70)

    print_cluster_composition(engine, clusters, cluster_to_split)

    print(f"\n💭 Analysis:")
    print(f"   This cluster has {len([n for n, c in clusters.items() if str(c) == str(cluster_to_split)])} applications.")
    print(f"   While they share some features, there might be distinct sub-groups.")
    print(f"   Let's split it to see if we can identify more cohesive groups.")
    print(f"\n✨ Note: The split will create hierarchical sub-clusters named {cluster_to_split}.1 and {cluster_to_split}.2")

    # Perform the split
    print("\n\n" + "="*70)
    print(f"STEP 3: Splitting Cluster {cluster_to_split}")
    print("="*70)

    print(f"\n⏳ Running intelligent split algorithm...")
    print(f"   • Calculating similarity matrix for cluster applications")
    print(f"   • Running hierarchical clustering with k=2")
    print(f"   • Finding optimal division...")

    new_clusters, split_metadata = engine.split_cluster(
        clusters, cluster_to_split, method="proportional"
    )

    print(f"\n✅ Split Complete!")
    print(f"\n📊 Split Results:")
    print(f"   Original: Cluster {split_metadata['original_cluster']} "
          f"({split_metadata['size_1'] + split_metadata['size_2']} apps)")
    print(f"   →  New Cluster {split_metadata['new_cluster_1']}: "
          f"{split_metadata['size_1']} apps")
    print(f"   →  New Cluster {split_metadata['new_cluster_2']}: "
          f"{split_metadata['size_2']} apps")

    # Show the two new clusters
    print("\n\n" + "="*70)
    print("STEP 4: Analyzing the Split Results")
    print("="*70)

    print_cluster_composition(engine, new_clusters, split_metadata['new_cluster_1'])
    print_cluster_composition(engine, new_clusters, split_metadata['new_cluster_2'])

    # Compare the two new clusters
    print("\n\n" + "="*70)
    print("STEP 5: Comparing New Sub-Clusters")
    print("="*70)

    analysis1 = engine.analyze_cluster_features(new_clusters, split_metadata['new_cluster_1'])
    analysis2 = engine.analyze_cluster_features(new_clusters, split_metadata['new_cluster_2'])

    top_features1 = set([f['dimension_name'] for f in analysis1['significant_features'][:3]])
    top_features2 = set([f['dimension_name'] for f in analysis2['significant_features'][:3]])

    shared = top_features1 & top_features2
    unique_1 = top_features1 - top_features2
    unique_2 = top_features2 - top_features1

    print(f"\n🔍 Feature Comparison:")
    if shared:
        print(f"\n  Shared Features:")
        for feature in shared:
            print(f"    • {feature}")

    if unique_1:
        print(f"\n  Unique to Cluster {split_metadata['new_cluster_1']}:")
        for feature in unique_1:
            print(f"    • {feature}")

    if unique_2:
        print(f"\n  Unique to Cluster {split_metadata['new_cluster_2']}:")
        for feature in unique_2:
            print(f"    • {feature}")

    # Show final cluster count
    print("\n\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)

    print(f"\n📊 Before Split: {metadata['n_clusters']} clusters")
    print(f"📊 After Split:  {len(set(new_clusters.values()))} clusters")

    print(f"\n📋 All Clusters After Split (with hierarchical naming):")
    for cluster_id in sorted(set(new_clusters.values()), key=natural_sort_key):
        analysis = engine.analyze_cluster_features(new_clusters, cluster_id)
        apps_str = ', '.join(sorted(analysis['applications']))
        print(f"\n  Cluster {cluster_id} ({analysis['size']} apps):")
        print(f"    Applications: {apps_str}")
        top_features = [f['dimension_name'] for f in analysis['significant_features'][:3]]
        print(f"    Key Features: {', '.join(top_features)}")

    # Benefits explanation
    print("\n\n" + "="*70)
    print("BENEFITS OF CLUSTER SPLITTING")
    print("="*70)

    print("\n✅ More Granular Groupings")
    print("   → Identify distinct sub-groups within broader categories")
    print("   → Example: Split 'Customer Apps' into 'CRM/Sales' and 'Operations'")

    print("\n✅ Better Consolidation Targets")
    print("   → Smaller, more cohesive groups make better merge candidates")
    print("   → Easier to justify consolidation with tighter feature alignment")

    print("\n✅ Flexible Refinement")
    print("   → Start with automatic clustering for overview")
    print("   → Manually refine where needed with intelligent splitting")
    print("   → Undo button allows experimentation")

    print("\n✅ Data-Driven Decisions")
    print("   → Split algorithm uses actual similarity data")
    print("   → Not just removing outliers - finding natural divisions")
    print("   → Considers full dimensionality of applications")

    # Use cases
    print("\n\n" + "="*70)
    print("EXAMPLE USE CASES")
    print("="*70)

    print("\n📌 Use Case 1: Different Product Lines")
    print("   Scenario: Cluster contains apps from multiple business units")
    print("   Action:   Split to separate by product line")
    print("   Benefit:  Consolidation can respect organizational boundaries")

    print("\n📌 Use Case 2: Technology Stack Differences")
    print("   Scenario: Cluster has apps on different technology stacks")
    print("   Action:   Split to separate legacy from modern apps")
    print("   Benefit:  Plan migration separately from consolidation")

    print("\n📌 Use Case 3: User Base Segmentation")
    print("   Scenario: Cluster has internal and external-facing apps")
    print("   Action:   Split to separate by user type")
    print("   Benefit:  Different security and UX requirements")

    print("\n" + "="*70)
    print("For GUI interface with cluster splitting, run: python run.py")
    print("In the GUI, click 'Split Cluster' to split the selected cluster")
    print("Use 'Undo Split' to revert if needed")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
