# Similarity Algorithm Update

## Summary

Updated the similarity scoring algorithm to **bias towards applications WITH features** rather than applications without features. This ensures that two applications with all zeros are NOT considered similar.

## Key Changes

### 1. Proportional Similarity Algorithm

**Previous Behavior:**
- Used `min(score1, score2) + 0.1` for weighting
- Added 0.1 constant gave weight even when one score was 0
- Didn't strongly penalize asymmetric features (one app has it, other doesn't)

**New Behavior:**
- Only counts dimensions where **BOTH** apps have non-zero scores (shared active features)
- Explicitly penalizes asymmetric dimensions (where only one app has the feature)
- Returns 0.0 for apps with no features at all
- Returns very low similarity (< 0.1) for apps with no shared active dimensions

### 2. Jaccard Similarity

**Previous Behavior:**
- Returned 1.0 when both apps had no features (treating empty as "identical")

**New Behavior:**
- Returns 0.0 when both apps have no features (empty apps are NOT similar)

## Algorithm Details

### Improved Proportional Similarity

```python
# Step 1: Find shared active dimensions (both > 0)
shared_dims = active_dimensions_app1 ∩ active_dimensions_app2

# Step 2: If no shared dimensions, return very low similarity
if no shared_dims:
    return 0.1 / (1 + total_dimensions)  # Approaches 0

# Step 3: Calculate similarity for shared dimensions only
for each shared dimension:
    dim_similarity = 1 - |score1 - score2| / max(score1, score2)
    weight = min(score1, score2)  # No artificial 0.1 added!

# Step 4: Calculate shared ratio penalty
shared_ratio = num_shared_dims / num_total_active_dims

# Step 5: Combine base similarity with shared ratio
final_similarity = base_similarity × (0.5 + 0.5 × shared_ratio)
```

### Key Formula Components

1. **Base Similarity**: How similar are the scores in shared dimensions?
   - 1.0 = identical scores in all shared dimensions
   - 0.0 = maximum difference in shared dimensions

2. **Shared Ratio**: What proportion of dimensions are shared?
   - 1.0 = all dimensions are shared
   - 0.5 = half of dimensions are shared
   - 0.0 = no shared dimensions

3. **Final Similarity**: Combines both factors
   - Requires BOTH high score similarity AND high dimension overlap
   - Formula: `base_similarity × (0.5 + 0.5 × shared_ratio)`

## Example Comparisons

### Example 1: Apps with No Features
```
App A: {0: 0, 1: 0, 2: 0}
App B: {0: 0, 1: 0, 2: 0}

Old Similarity: Variable (depending on implementation)
New Similarity: 0.0 ✓

Rationale: No features means not similar - they're just both undefined
```

### Example 2: Apps with No Shared Features
```
App A: {0: 5, 1: 3}
App B: {2: 4, 3: 2}

Old Similarity: ~0.1-0.2 (got some credit for having features)
New Similarity: 0.025 (very low) ✓

Rationale: No shared dimensions means fundamentally different applications
```

### Example 3: Apps with Partial Overlap
```
App A: {0: 5, 1: 3, 2: 4}
App B: {0: 5, 1: 3, 3: 2}

Shared dimensions: {0, 1}
Total dimensions: {0, 1, 2, 3} = 4
Shared ratio: 2/4 = 0.5

Base similarity: ~1.0 (identical scores in shared dims)
Final similarity: 1.0 × (0.5 + 0.5 × 0.5) = 0.75 ✓

Rationale: High similarity in shared features, but only half of dimensions overlap
```

### Example 4: Apps with Full Overlap
```
App A: {0: 5, 1: 3, 2: 4}
App B: {0: 5, 1: 3, 2: 4}

Shared dimensions: {0, 1, 2}
Shared ratio: 3/3 = 1.0

Base similarity: 1.0 (identical scores)
Final similarity: 1.0 × (0.5 + 0.5 × 1.0) = 1.0 ✓

Rationale: Identical applications
```

## Impact on Results

### Sample Data Results

Using the sample dataset (12 applications, 10 dimensions):

**Top Similar Pairs:**
1. HR Management ↔ Employee Portal: **93.54%**
   - Share 9 active dimensions with very similar scores
   - Clear consolidation candidate

2. Analytics Platform ↔ Data Warehouse: **84.68%**
   - Share 7 active dimensions focused on data/analytics
   - Both specialized in data processing

3. Inventory System ↔ Supply Chain: **82.19%**
   - Share 9 active dimensions in operational areas
   - Similar operational focus

**Clustering Behavior:**
- Cluster 0: HR-focused apps (HR Management, Employee Portal, Payroll, Customer Portal)
- Cluster 1: Customer/sales apps (CRM, Email Marketing, Orders, Inventory, Supply Chain)
- Cluster 2: Analytics apps (Analytics Platform, Data Warehouse, Sales Dashboard)

## Benefits

1. **More Accurate Consolidation Recommendations**
   - Only suggests consolidation when apps truly share features
   - Doesn't suggest consolidation just because both apps are undefined

2. **Better Clustering**
   - Apps group by shared active features, not absence of features
   - More meaningful clusters for business decisions

3. **Intuitive Results**
   - Empty/undefined apps don't appear as "similar"
   - Similarity scores reflect actual feature overlap

4. **Proportional Fairness Maintained**
   - Still ensures simple apps with high overlap > complex apps with low overlap
   - Still normalizes by applicable features, not absolute counts

## Testing

All tests pass, including new tests specifically for:
- Apps with all zeros return similarity of 0.0
- Apps with no shared active dimensions return very low similarity
- Partial overlap is scored appropriately
- Identical apps still return 1.0

Run tests:
```bash
PYTHONPATH=/Users/jacksexauer/Projects/apprat/src python3 -m pytest tests/test_core.py -v
```

## Backward Compatibility

This is a **breaking change** in similarity calculation. Previous similarity scores will differ from new scores. However, this is an improvement in accuracy and should produce more meaningful results.

If you need the old behavior, you can:
1. Use the `cosine_similarity_score` method instead
2. Modify the algorithm to add back the 0.1 constant in weighting

## Files Changed

- `src/analysis/similarity.py`: Updated all three similarity methods
  - `proportional_similarity()`: Complete rewrite
  - `jaccard_similarity()`: Returns 0.0 for empty apps
  - `weighted_proportional_similarity()`: Updated to match new algorithm

- `tests/test_core.py`: Added new tests for zero-feature behavior

## Next Steps

The algorithm now correctly handles:
- ✅ Apps with no features (returns 0.0)
- ✅ Apps with no shared features (returns very low similarity)
- ✅ Apps with partial overlap (proportional scoring)
- ✅ Apps with full overlap (returns 1.0)

The similarity scoring is now more aligned with business intuition: **applications are similar when they share features, not when they both lack features**.
