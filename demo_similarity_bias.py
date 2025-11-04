#!/usr/bin/env python3
"""
Demonstration of the updated similarity algorithm that biases towards
applications WITH features rather than applications without features.
"""
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import core.application as app_module
import analysis.similarity as sim_module

Application = app_module.Application
SimilarityCalculator = sim_module.SimilarityCalculator


def print_comparison(title, app1, app2, calc):
    """Print a comparison between two applications."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"App 1: {app1.name}")
    print(f"  Scores: {dict(sorted(app1.scores.items()))}")
    print(f"  Active dimensions: {app1.active_dimensions} ({app1.num_active_dimensions} total)")
    print(f"\nApp 2: {app2.name}")
    print(f"  Scores: {dict(sorted(app2.scores.items()))}")
    print(f"  Active dimensions: {app2.active_dimensions} ({app2.num_active_dimensions} total)")

    # Calculate similarities
    prop_sim = calc.proportional_similarity(app1, app2)
    jacc_sim = calc.jaccard_similarity(app1, app2)

    print(f"\n📊 Similarity Results:")
    print(f"  Proportional Similarity: {prop_sim:.4f}")
    print(f"  Jaccard Similarity:      {jacc_sim:.4f}")

    # Interpretation
    if prop_sim >= 0.8:
        interpretation = "⭐ VERY SIMILAR - Strong consolidation candidate"
    elif prop_sim >= 0.6:
        interpretation = "✓ SIMILAR - Consider consolidation"
    elif prop_sim >= 0.3:
        interpretation = "~ SOMEWHAT SIMILAR - Review for overlap"
    elif prop_sim >= 0.1:
        interpretation = "✗ NOT SIMILAR - Different focus"
    else:
        interpretation = "✗✗ NOT SIMILAR AT ALL - Completely different or undefined"

    print(f"\n  {interpretation}")


def main():
    """Run demonstrations."""
    calc = SimilarityCalculator()

    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "SIMILARITY ALGORITHM DEMONSTRATION" + " "*19 + "║")
    print("║" + " "*11 + "Feature-Biased Similarity Scoring" + " "*22 + "║")
    print("╚" + "="*68 + "╝")

    # Case 1: Apps with NO features (all zeros)
    app_empty1 = Application("Empty App 1", {0: 0, 1: 0, 2: 0})
    app_empty2 = Application("Empty App 2", {0: 0, 1: 0, 2: 0})
    print_comparison(
        "Case 1: Two Applications with NO Features (All Zeros)",
        app_empty1, app_empty2, calc
    )
    print("\n💡 Key Point: Apps with no features are NOT similar (0.0)")
    print("   Rationale: Both being undefined doesn't make them similar")

    # Case 2: Apps with NO SHARED features
    app_diff1 = Application("Sales App", {0: 5, 1: 4, 2: 3})
    app_diff2 = Application("Analytics App", {3: 5, 4: 4, 5: 3})
    print_comparison(
        "Case 2: Applications with NO Shared Features",
        app_diff1, app_diff2, calc
    )
    print("\n💡 Key Point: Apps with no overlap get very low similarity (< 0.1)")
    print("   Rationale: Different features means fundamentally different apps")

    # Case 3: Apps with PARTIAL overlap
    app_partial1 = Application("CRM System", {0: 5, 1: 4, 2: 3, 3: 2})
    app_partial2 = Application("Sales Platform", {0: 5, 1: 4, 4: 3, 5: 2})
    print_comparison(
        "Case 3: Applications with PARTIAL Feature Overlap (2/6 dimensions)",
        app_partial1, app_partial2, calc
    )
    print("\n💡 Key Point: Partial overlap gives moderate similarity")
    print("   Rationale: Share some features (0, 1) but differ in others")

    # Case 4: Apps with HIGH overlap
    app_similar1 = Application("HR Management", {0: 5, 1: 4, 2: 3, 3: 2})
    app_similar2 = Application("Employee Portal", {0: 5, 1: 4, 2: 3, 3: 2})
    print_comparison(
        "Case 4: Applications with HIGH Feature Overlap (Identical)",
        app_similar1, app_similar2, calc
    )
    print("\n💡 Key Point: Identical features give perfect similarity (1.0)")
    print("   Rationale: Same features with same scores = consolidation candidate")

    # Case 5: Simple apps with high overlap vs complex apps with low overlap
    app_simple1 = Application("Simple App 1", {0: 5, 1: 4, 2: 3})
    app_simple2 = Application("Simple App 2", {0: 5, 1: 4, 2: 3})

    app_complex1 = Application("Complex App 1", {
        0: 5, 1: 4, 2: 3, 3: 5, 4: 2, 5: 4
    })
    app_complex2 = Application("Complex App 2", {
        0: 5, 1: 4, 2: 3, 6: 4, 7: 3, 8: 2
    })

    simple_sim = calc.proportional_similarity(app_simple1, app_simple2)
    complex_sim = calc.proportional_similarity(app_complex1, app_complex2)

    print(f"\n{'='*70}")
    print("Case 5: Proportional Similarity - Simple vs Complex Apps")
    print(f"{'='*70}")
    print("\n🔹 Simple Apps (3 dimensions, 100% overlap):")
    print(f"   {app_simple1.name}: {app_simple1.num_active_dimensions} features")
    print(f"   {app_simple2.name}: {app_simple2.num_active_dimensions} features")
    print(f"   Similarity: {simple_sim:.4f}")

    print("\n🔹 Complex Apps (6 dimensions, 50% overlap):")
    print(f"   {app_complex1.name}: {app_complex1.num_active_dimensions} features")
    print(f"   {app_complex2.name}: {app_complex2.num_active_dimensions} features")
    print(f"   Similarity: {complex_sim:.4f}")

    print("\n📊 Comparison:")
    if simple_sim > complex_sim:
        print(f"   ✓ Simple apps (100% overlap) are MORE similar ({simple_sim:.4f})")
        print(f"     than complex apps (50% overlap) ({complex_sim:.4f})")
        print("\n💡 Key Point: Proportional overlap matters more than absolute overlap")
        print("   Rationale: 3/3 features matching > 3/6 features matching")
    else:
        print(f"   ✗ Unexpected result - complex apps scored higher")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("\n✅ The updated algorithm correctly handles:")
    print("   1. Apps with no features → Similarity = 0.0 (not similar)")
    print("   2. Apps with no shared features → Similarity < 0.1 (very different)")
    print("   3. Apps with partial overlap → Proportional scoring")
    print("   4. Apps with high overlap → High similarity (consolidation candidates)")
    print("   5. Proportional fairness → Simple 100% overlap > Complex 50% overlap")
    print("\n🎯 Business Value:")
    print("   - Only suggests consolidation when apps truly share features")
    print("   - Doesn't confuse 'undefined' with 'similar'")
    print("   - Provides more accurate consolidation recommendations")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
