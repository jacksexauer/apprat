## Cluster Splitting Feature

## Overview

Added **intelligent cluster splitting** that allows users to divide a cluster into two sub-clusters based on the natural distribution of applications across dimensions. Uses **hierarchical naming** (X → X.1 and X.2) to show the relationship between original and split clusters. Includes **undo functionality** to revert splits if needed.

## Why Cluster Splitting?

### The Problem

Automatic clustering produces good overall groupings, but sometimes:
- A cluster contains distinct sub-groups that should be separate
- Two very different applications end up together because they're both dissimilar from everything else
- You want more granular control over consolidation decisions

**Example**: Databricks and Salesforce might cluster together in a portfolio of legacy mainframe apps (they're both modern, cloud-based). But you want them in separate clusters to recognize their different purposes.

### The Solution

**Intelligent cluster splitting** automatically finds the best way to divide a cluster into two sub-clusters:
- ✅ Not just "remove the outlier"
- ✅ Finds natural divisions based on similarity distribution
- ✅ Considers all dimensions, not just a single axis
- ✅ Can split evenly, unevenly, or isolate one app - depends on the data

## How It Works

### Splitting Algorithm

1. **Extract cluster applications**: Get all apps in the selected cluster
2. **Calculate sub-similarity matrix**: Compute similarities between just these apps (treats the cluster as the entire plane)
3. **Run hierarchical clustering**: Apply AgglomerativeClustering with k=2
4. **Find optimal division**: Algorithm determines best way to split based on distances
5. **Create new clusters**: Assign apps to two hierarchical cluster IDs (X.1 and X.2)

**Key Points**:
- The algorithm treats the cluster being split as the entire plane - similarities are calculated only among applications within that cluster
- Uses hierarchical naming: splitting Cluster 0 creates Clusters 0.1 and 0.2, splitting Cluster 1 creates 1.1 and 1.2
- The algorithm considers the full similarity distribution, not just the least similar app
- It finds the natural "cut point" within the cluster based on application relationships

### Split Patterns

The algorithm can produce different split patterns based on data:

| Pattern | Example | When It Happens |
|---------|---------|----------------|
| **Even Split** | 3 + 3 | Cluster has two clear sub-groups of similar size |
| **Uneven Split** | 4 + 2 | One sub-group is more cohesive/larger |
| **Isolate One** | 5 + 1 | One app is clearly different from the rest |
| **Pairs** | 1 + 1 | Only 2 apps in cluster (edge case) |

The algorithm **decides automatically** based on similarity patterns.

## Using Cluster Splitting

### In the GUI

1. **Run clustering first**:
   - Load your data
   - Go to Clusters tab
   - Click "Calculate Clusters"

2. **Select a cluster to split**:
   - Click on a cluster in the left list
   - Review its applications and features

3. **Split the cluster**:
   - Click "Split Cluster" button
   - Algorithm runs automatically
   - Popup shows results

4. **Review the split**:
   - Two new clusters appear in the list
   - Each has its own feature analysis
   - Check if the split makes sense

5. **Undo if needed**:
   - Click "Undo Split" to revert
   - Can undo multiple splits
   - History stack tracks changes

### In Python Code

```python
from core.csv_loader import CSVLoader
from analysis.clustering import ClusteringEngine

# Load and cluster
feature_matrix = CSVLoader.load_feature_matrix("apps.csv", "dims.csv")
engine = ClusteringEngine(feature_matrix)
clusters, metadata = engine.auto_cluster(method="proportional")

# Split a specific cluster (use string ID)
cluster_to_split = "1"  # Cluster IDs are strings
new_clusters, split_metadata = engine.split_cluster(
    clusters,
    cluster_to_split,
    method="proportional"
)

print(f"Split cluster {split_metadata['original_cluster']} into:")
print(f"  - Cluster {split_metadata['new_cluster_1']}: {split_metadata['size_1']} apps")
print(f"  - Cluster {split_metadata['new_cluster_2']}: {split_metadata['size_2']} apps")
# Output: Split cluster 1 into:
#           - Cluster 1.1: 3 apps
#           - Cluster 1.2: 2 apps

# See which apps went where
print(f"\nGroup 1: {', '.join(split_metadata['group_1'])}")
print(f"Group 2: {', '.join(split_metadata['group_2'])}")
```

### Demo Script

```bash
python3 demo_cluster_splitting.py
```

Shows:
- Initial clustering (3 clusters)
- Selection of cluster to split
- Split execution
- Analysis of results
- Comparison of new sub-clusters

## UI Features

### Split Cluster Button

**Location**: Clusters tab, top controls (after "Calculate Clusters")

**Enabled When**: Clusters have been calculated

**Tooltip**: "Split the selected cluster into two sub-clusters"

**Behavior**:
1. Validates selection (must have cluster selected)
2. Checks cluster size (must have ≥2 apps)
3. Runs split algorithm
4. Shows popup with results
5. Updates display automatically

### Undo Split Button

**Location**: Next to "Split Cluster" button

**Enabled When**: At least one split has been performed

**Tooltip**: "Undo the last cluster split"

**Behavior**:
1. Restores previous cluster state from history
2. Refreshes display
3. Disables itself if no more history
4. Shows confirmation popup

### History Stack

Automatically maintains history of cluster states:
- **Initial state**: After "Calculate Clusters"
- **Each split**: Pushes current state before splitting
- **Undo**: Pops from stack
- **Recalculate**: Clears history (fresh start)

## Split Results

### Metadata Returned

```python
{
    'original_cluster': '1',
    'new_cluster_1': '1.1',
    'new_cluster_2': '1.2',
    'size_1': 3,
    'size_2': 2,
    'group_1': ['App A', 'App B', 'App C'],
    'group_2': ['App D', 'App E'],
    'method_used': 'hierarchical_split'
}
```

**Note**: Cluster IDs are now strings to support hierarchical naming (e.g., "1", "1.1", "1.2").

### Result Popup

Shows:
- Original cluster size
- Two new cluster IDs with hierarchical naming
- Complete list of apps in each new cluster
- Clear visual separation

Example:
```
Cluster Split Complete

Original: Cluster 1 (5 apps)

Split into:
  • Cluster 1.1 (3 apps)
  • Cluster 1.2 (2 apps)

Cluster 1.1:
  - CRM System
  - Email Marketing
  - Order Management

Cluster 1.2:
  - Inventory System
  - Supply Chain
```

## Example Scenarios

### Scenario 1: Different Business Units

**Situation**: Cluster contains apps from Sales and Marketing

**Split Result**:
- Cluster A (3 apps): Sales apps (CRM, Sales Dashboard, Orders)
- Cluster B (2 apps): Marketing apps (Email Marketing, Campaign Manager)

**Benefit**: Consolidation can respect organizational boundaries

### Scenario 2: Technology Stack Split

**Situation**: Cluster mixes legacy and modern apps

**Split Result**:
- Cluster A (4 apps): Modern cloud-native apps
- Cluster B (2 apps): Legacy on-premises apps

**Benefit**: Plan migration separately from consolidation

### Scenario 3: Data vs. Transactional

**Situation**: Cluster has data warehouses and operational apps

**Split Result**:
- Cluster A (2 apps): Analytics/BI platforms (high on Data Analytics)
- Cluster B (3 apps): Operational systems (high on Real-time Processing)

**Benefit**: Different architecture patterns and requirements

### Scenario 4: User Segmentation

**Situation**: Cluster contains internal and customer-facing apps

**Split Result**:
- Cluster A (3 apps): Customer-facing (high on User Management, Mobile)
- Cluster B (2 apps): Internal tools (high on Workflow Automation)

**Benefit**: Different security and UX considerations

## Comparison: Split vs. Reclustering

| Aspect | Split Cluster | Recalculate Clusters |
|--------|---------------|---------------------|
| **Scope** | One cluster | All applications |
| **Approach** | Refine existing | Start fresh |
| **History** | Maintained (undo available) | Cleared |
| **Other clusters** | Unchanged | May change |
| **Use case** | Targeted refinement | Major restructure |

**When to split**: You like most clusters but one needs refinement
**When to recalculate**: You want to try a different clustering approach entirely

## Best Practices

### 1. Start with Automatic Clustering

Don't manually split from the beginning. Let the algorithm do initial clustering, then refine:
```
Auto Cluster (3 clusters) → Review → Split Cluster 1 → 4 clusters
```

### 2. Check Feature Analysis First

Before splitting, review the cluster's feature analysis:
- Are there obviously different feature profiles?
- Do apps have very different coverage percentages?
- Are there gaps in key features?

### 3. Split Based on Business Logic

Split when you see:
- Different business units/product lines
- Mixed legacy and modern apps
- Internal vs. external facing
- Data vs. operational systems

### 4. Use Undo Liberally

Experiment! The undo button lets you try splits and revert:
```
Split → Review → Not quite right → Undo → Try different approach
```

### 5. Analyze Split Results

After splitting, examine both new clusters:
- Look at their top features
- See what makes them different
- Verify the split makes business sense

## Technical Details

### Algorithm: Hierarchical Clustering

```python
# Pseudocode
def split_cluster(cluster_apps):
    # 1. Build similarity matrix for just these apps
    sim_matrix = calculate_similarities(cluster_apps)

    # 2. Convert to distance
    dist_matrix = 1 - sim_matrix

    # 3. Run hierarchical clustering with k=2
    clusterer = AgglomerativeClustering(
        n_clusters=2,
        metric="precomputed",
        linkage="average"
    )
    labels = clusterer.fit_predict(dist_matrix)

    # 4. Assign to new cluster IDs
    return new_assignments
```

### Complexity

- **Time**: O(n²) where n = apps in cluster
- **Space**: O(n²) for similarity matrix
- **Typical**: <100ms for 10-20 apps

### Cluster ID Assignment

New clusters use **hierarchical naming** based on the original cluster ID.

**Format**: Splitting cluster X creates clusters X.1 and X.2

**Examples**:
- Split Cluster 0 → Clusters 0.1 and 0.2
- Split Cluster 1 → Clusters 1.1 and 1.2
- Split Cluster 0.1 → Clusters 0.1.1 and 0.1.2

**Sorting**: Clusters are naturally sorted: "0", "0.1", "0.2", "1", "1.1", "1.2", "2"

**Before Split**:
```
Clusters: 0, 1, 2
```

**After Splitting Cluster 1**:
```
Clusters: 0, 1.1, 1.2, 2
```

Note: Original cluster 1 no longer exists after split - it's replaced by 1.1 and 1.2.

## Limitations

### Cannot Split

- **Single app cluster**: Need ≥2 apps to split
- **No clusters calculated**: Must run clustering first

### Split Patterns

The algorithm always produces **2 clusters**, not 3 or more. To split into 3:
1. Split into 2
2. Select one of the results
3. Split again

### Undo Depth

- Unlimited undo stack (memory permitting)
- Stack cleared when recalculating clusters
- Each split adds to history

## Files Modified

### Core Logic
- ✅ `src/analysis/clustering.py`
  - Added `split_cluster()` method (~100 lines)

### UI
- ✅ `src/ui/main_window.py`
  - Added "Split Cluster" button
  - Added "Undo Split" button
  - Added history stack management
  - Added `split_selected_cluster()` method
  - Added `undo_cluster_split()` method
  - Added `refresh_cluster_display()` method

### Demo & Documentation
- ✅ `demo_cluster_splitting.py` (NEW, ~180 lines)
- ✅ `CLUSTER_SPLITTING.md` (NEW, this file)

## Testing

### Demo Script
```bash
python3 demo_cluster_splitting.py
```

Expected output:
- ✅ Initial clustering (3 clusters)
- ✅ Split cluster 1
- ✅ Results show 4 clusters
- ✅ Analysis of new sub-clusters
- ✅ Feature comparison

### Manual Testing (GUI)

1. Load sample data
2. Calculate clusters (automatic)
3. Select Cluster 1
4. Click "Split Cluster"
5. Verify:
   - ✅ Popup shows split details
   - ✅ List updates with 2 new clusters
   - ✅ Original cluster removed
   - ✅ Undo button enabled
6. Click "Undo Split"
7. Verify:
   - ✅ Returns to 3 clusters
   - ✅ Original cluster restored
   - ✅ Undo button disabled

## Benefits Summary

| Benefit | Description |
|---------|-------------|
| **Granular Control** | Refine automatic clustering results without starting over |
| **Data-Driven** | Split based on actual similarity patterns, not guesses |
| **Flexible** | Try different splits and undo if needed |
| **Intelligent** | Considers all dimensions, finds natural divisions |
| **Business-Aligned** | Split based on business logic and organizational needs |
| **Reversible** | Undo button allows experimentation |

## Use Case Summary

✅ **Separate business units** within a functional cluster

✅ **Split technology stacks** (legacy vs. modern)

✅ **Distinguish user types** (internal vs. external)

✅ **Separate architecture patterns** (data vs. transactional)

✅ **Refine consolidation targets** for better alignment

✅ **Isolate outliers** that don't belong with main group

## Conclusion

Cluster splitting provides **intelligent, data-driven refinement** of automatic clustering results. Users can:

- 🎯 Split clusters that contain distinct sub-groups
- 🔍 Find natural divisions based on similarity patterns
- 📊 Get detailed analysis of split results
- ↩️ Undo splits to try different approaches
- 🎨 Refine clustering without starting over

This feature bridges the gap between **automatic clustering** (fast, objective) and **manual clustering** (flexible, business-aligned), giving users the best of both worlds! 🎉
