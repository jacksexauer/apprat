# Automatic Clustering

## Overview

The apprat tool now features **automatic clustering** that dynamically determines the optimal number of clusters based on the data, eliminating the need to specify an arbitrary cluster count.

## Why Automatic Clustering?

**The Problem**: Traditional clustering requires you to specify how many clusters you want. But how do you know the right number? Guessing can lead to:
- Too few clusters: Unrelated applications grouped together
- Too many clusters: Natural groups split apart
- Arbitrary decisions without data-driven justification

**The Solution**: Automatic clustering analyzes your data and finds the natural groupings automatically.

## Three Automatic Methods

### 1. Silhouette-Optimized (Recommended) 🏆

**How it works**:
- Tries different numbers of clusters (2 to n/2)
- Calculates silhouette score for each configuration
- Selects the number that maximizes cluster quality

**Silhouette Score**: Measures how well-separated and cohesive clusters are
- +1.0: Perfect clusters
- 0.0: Overlapping clusters
- -1.0: Wrong clusters

**Best for**:
- General-purpose clustering
- When you want the best overall quality
- Balanced cluster sizes

**Example**:
```python
clusters, metadata = engine.auto_cluster(method="proportional")
# Result: 3 clusters (silhouette score: 0.366)
```

### 2. Threshold-Based 🎯

**How it works**:
- Analyzes the distribution of pairwise similarities
- Finds the natural gap between "similar" and "dissimilar"
- Groups applications with similarity above the threshold

**Best for**:
- Business-driven clustering
- When you want intuitive, explainable groups
- "Apps with >80% similarity should be grouped"

**Example**:
```python
clusters, metadata = engine.auto_cluster_threshold(method="proportional")
# Result: 8 clusters (threshold: 0.800)
# Interpretation: Apps grouped if similarity ≥ 80%
```

**Custom threshold**:
```python
clusters, metadata = engine.auto_cluster_threshold(
    method="proportional",
    similarity_threshold=0.7
)
# Group apps with ≥70% similarity
```

### 3. DBSCAN (Density-Based) 🔬

**How it works**:
- Finds dense regions in the similarity space
- Forms clusters around these dense regions
- Identifies outliers that don't fit any cluster

**Best for**:
- Finding anomalies/unique applications
- When cluster sizes are very different
- Identifying apps that don't belong to any group

**Example**:
```python
clusters, metadata = engine.dbscan_clustering(method="proportional")
# Result: 2 clusters + 8 outliers
# Outliers (cluster_id = -1) don't fit any group
```

## Comparison Results

Using the sample dataset (12 applications):

| Method | Clusters | Interpretation |
|--------|----------|----------------|
| **Silhouette** | 3 | Balanced groups with best overall quality |
| **Threshold** | 8 | Tight groups with >80% similarity |
| **DBSCAN** | 2 + 8 outliers | Two dense groups, others unique |

### Silhouette-Optimized: 3 Clusters
- **Cluster 0**: HR-focused (HR Management, Employee Portal, Payroll, Customer Portal)
- **Cluster 1**: Customer/Sales (CRM, Email Marketing, Inventory, Orders, Supply Chain)
- **Cluster 2**: Analytics (Analytics Platform, Data Warehouse, Sales Dashboard)

### Threshold-Based: 8 Clusters
- Pairs of very similar apps (>80% similarity)
- More granular groupings
- Stricter consolidation candidates

### DBSCAN: 2 Clusters + Outliers
- **Cluster 0**: Analytics Platform ↔ Data Warehouse (very similar)
- **Cluster 1**: HR Management ↔ Employee Portal (very similar)
- **Outliers**: 8 apps that don't have strong density connections

## Usage

### In Python Code

```python
from core.csv_loader import CSVLoader
from analysis.clustering import ClusteringEngine

# Load data
feature_matrix = CSVLoader.load_feature_matrix("apps.csv", "dimensions.csv")
engine = ClusteringEngine(feature_matrix)

# Method 1: Silhouette-optimized (recommended)
clusters, metadata = engine.auto_cluster(method="proportional")
print(f"Found {metadata['n_clusters']} clusters")
print(f"Quality score: {metadata['silhouette_score']:.3f}")

# Method 2: Threshold-based
clusters, metadata = engine.auto_cluster_threshold(method="proportional")
print(f"Found {metadata['n_clusters']} clusters")
print(f"Threshold: {metadata['threshold_used']:.3f}")

# Method 3: DBSCAN
clusters, metadata = engine.dbscan_clustering(method="proportional")
print(f"Found {metadata['n_clusters']} clusters")
print(f"Outliers: {metadata['n_noise']}")

# Compare all methods
comparison = engine.compare_clustering_methods(method="proportional")
print(comparison)
```

### In the GUI

1. Load your CSV files
2. Go to the "Clusters" tab
3. **Mode dropdown**: Select "Automatic" (default)
4. **Auto Method dropdown**: Choose your preferred method
   - "Silhouette (Best Quality)" - Recommended
   - "Threshold (Natural Groups)" - Business-friendly
   - "DBSCAN (Density-Based)" - Finds outliers
5. **Similarity dropdown**: Choose similarity method
6. Click "Calculate Clusters"

The application will:
- Automatically determine the optimal number of clusters
- Display the results
- Show a popup with clustering metadata

### Using the Demo Script

```bash
python3 demo_auto_clustering.py
```

Shows all three methods with sample data and detailed explanations.

## Advanced Options

### Controlling Cluster Count Range

```python
# Limit the search range for silhouette optimization
clusters, metadata = engine.auto_cluster(
    method="proportional",
    min_clusters=2,   # Minimum clusters to try
    max_clusters=8    # Maximum clusters to try
)
```

### Custom Threshold

```python
# Use a specific similarity threshold
clusters, metadata = engine.auto_cluster_threshold(
    method="proportional",
    similarity_threshold=0.65  # Group if ≥65% similar
)
```

### DBSCAN Parameters

```python
# Control density parameters
clusters, metadata = engine.dbscan_clustering(
    method="proportional",
    eps=0.2,          # Maximum distance for neighbors
    min_samples=3     # Minimum points to form cluster
)
```

## Understanding the Results

### Metadata Returned

All methods return `(clusters, metadata)`:

**clusters**: Dict mapping app names to cluster IDs
```python
{
    'App A': 0,
    'App B': 0,
    'App C': 1,
    ...
}
```

**metadata**: Dict with method-specific information

**Silhouette-optimized**:
```python
{
    'n_clusters': 3,
    'silhouette_score': 0.366,
    'method_used': 'silhouette_optimization',
    'all_scores': [(2, 0.331), (3, 0.366), ...]  # All k values tried
}
```

**Threshold-based**:
```python
{
    'n_clusters': 8,
    'threshold_used': 0.800,
    'distance_threshold': 0.200,
    'method_used': 'threshold_based'
}
```

**DBSCAN**:
```python
{
    'n_clusters': 2,
    'n_noise': 8,          # Number of outliers
    'eps_used': 0.176,
    'method_used': 'dbscan'
}
```

## Interpreting Outliers (DBSCAN)

Applications with `cluster_id = -1` are outliers:
- They don't fit well into any cluster
- They may be unique/specialized applications
- They could be candidates for:
  - Standalone applications (keep as-is)
  - Custom solutions (don't consolidate)
  - Further investigation (understand uniqueness)

## Best Practices

1. **Start with Silhouette-Optimized**
   - Most reliable for general use
   - Good balance of quality and interpretability

2. **Use Threshold for Business Alignment**
   - When you need to explain: "Apps with >X% similarity"
   - Set threshold based on business risk tolerance
   - Higher threshold = more conservative consolidation

3. **Use DBSCAN for Anomaly Detection**
   - Identify truly unique applications
   - Find apps that don't fit standard patterns
   - Useful for portfolio assessment

4. **Compare Multiple Methods**
   - Different methods provide different perspectives
   - Agreement across methods = strong evidence
   - Disagreement = investigate further

## Migration from Manual Clustering

If you were using manual clustering:

**Before**:
```python
clusters = engine.hierarchical_clustering(n_clusters=3, method="proportional")
```

**After** (recommended):
```python
clusters, metadata = engine.auto_cluster(method="proportional")
# Automatically finds optimal n_clusters
```

**Still available** (if you need manual control):
```python
clusters = engine.hierarchical_clustering(n_clusters=3, method="proportional")
# Manual mode still supported
```

## Benefits

✅ **Data-Driven**: Clusters based on actual similarities, not guesses

✅ **Objective**: Reduces subjective decision-making

✅ **Reproducible**: Same data = same clusters

✅ **Explainable**: Clear metrics and thresholds

✅ **Flexible**: Multiple methods for different use cases

✅ **Efficient**: No trial-and-error with different k values

## Technical Details

### Silhouette Score Calculation

For each cluster configuration:
1. Calculate silhouette coefficient for each application
2. Average across all applications
3. Higher score = better separation

### Threshold Detection

1. Calculate all pairwise similarities
2. Sort similarities
3. Find largest gap in upper half of distribution
4. Set threshold at the gap midpoint
5. Constrain to reasonable range (0.3-0.8)

### DBSCAN Epsilon Auto-Detection

1. Calculate distance to nearest neighbor for each app
2. Use mean of nearest neighbor distances as epsilon
3. This adapts to the density of your data

## Troubleshooting

**Too many clusters**:
- Try silhouette method (it optimizes for balance)
- Increase threshold for threshold-based method
- Reduce eps for DBSCAN

**Too few clusters**:
- Lower threshold for threshold-based method
- Check if data actually has distinct groups
- Review similarity scores - may be naturally similar

**All outliers in DBSCAN**:
- Data is sparse/diverse (good to know!)
- Reduce min_samples parameter
- Try threshold-based method instead

## Further Reading

- `demo_auto_clustering.py` - Interactive demonstration
- `src/analysis/clustering.py` - Implementation details
- `tests/test_core.py` - Usage examples
