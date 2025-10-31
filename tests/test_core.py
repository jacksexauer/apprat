"""
Unit tests for core functionality.
"""
import pytest
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from core.application import Application
from core.feature_matrix import FeatureMatrix
from core.csv_loader import CSVLoader
from analysis.similarity import SimilarityCalculator
from analysis.clustering import ClusteringEngine


class TestApplication:
    """Test Application class."""

    def test_create_application(self):
        """Test creating an application."""
        scores = {0: 5.0, 1: 3.0, 2: 0.0, 3: 4.0}
        app = Application("Test App", scores)

        assert app.name == "Test App"
        assert app.get_score(0) == 5.0
        assert app.get_score(2) == 0.0
        assert app.get_score(99) == 0.0  # Non-existent dimension

    def test_active_dimensions(self):
        """Test active dimensions detection."""
        scores = {0: 5.0, 1: 0.0, 2: 3.0, 3: 0.0, 4: 1.0}
        app = Application("Test App", scores)

        active = app.active_dimensions
        assert len(active) == 3
        assert 0 in active
        assert 2 in active
        assert 4 in active
        assert 1 not in active
        assert 3 not in active


class TestFeatureMatrix:
    """Test FeatureMatrix class."""

    def test_create_matrix(self):
        """Test creating a feature matrix."""
        matrix = FeatureMatrix()
        assert len(matrix) == 0

    def test_add_applications(self):
        """Test adding applications to matrix."""
        matrix = FeatureMatrix()

        app1 = Application("App 1", {0: 5.0, 1: 3.0})
        app2 = Application("App 2", {0: 4.0, 2: 2.0})

        matrix.add_application(app1)
        matrix.add_application(app2)

        assert len(matrix) == 2
        assert matrix.get_application("App 1") == app1
        assert matrix.get_application("App 2") == app2

    def test_dimension_mappings(self):
        """Test dimension mappings."""
        matrix = FeatureMatrix()
        matrix.add_dimension_mapping(0, "Cloud Native")
        matrix.add_dimension_mapping(1, "Mobile Support")

        assert matrix.get_dimension_name(0) == "Cloud Native"
        assert matrix.get_dimension_name(1) == "Mobile Support"
        assert matrix.get_dimension_name(99) == "Dimension 99"


class TestSimilarityCalculator:
    """Test SimilarityCalculator class."""

    def test_proportional_similarity_identical(self):
        """Test proportional similarity with identical apps."""
        app1 = Application("App 1", {0: 5.0, 1: 3.0, 2: 4.0})
        app2 = Application("App 2", {0: 5.0, 1: 3.0, 2: 4.0})

        calc = SimilarityCalculator()
        similarity = calc.proportional_similarity(app1, app2)

        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_proportional_similarity_different(self):
        """Test proportional similarity with different apps."""
        app1 = Application("App 1", {0: 5.0, 1: 0.0})
        app2 = Application("App 2", {0: 0.0, 1: 5.0})

        calc = SimilarityCalculator()
        similarity = calc.proportional_similarity(app1, app2)

        # Should be low similarity since no overlap
        assert similarity < 0.5

    def test_jaccard_similarity(self):
        """Test Jaccard similarity."""
        app1 = Application("App 1", {0: 5.0, 1: 3.0, 2: 4.0})
        app2 = Application("App 2", {0: 2.0, 1: 1.0})

        calc = SimilarityCalculator()
        similarity = calc.jaccard_similarity(app1, app2)

        # Intersection: {0, 1}, Union: {0, 1, 2}
        # Jaccard = 2/3
        assert similarity == pytest.approx(2/3, rel=0.01)


class TestClusteringEngine:
    """Test ClusteringEngine class."""

    def test_similarity_matrix(self):
        """Test similarity matrix calculation."""
        matrix = FeatureMatrix()
        matrix.add_application(Application("App 1", {0: 5.0, 1: 3.0}))
        matrix.add_application(Application("App 2", {0: 4.0, 1: 3.0}))
        matrix.add_application(Application("App 3", {0: 1.0, 2: 5.0}))

        engine = ClusteringEngine(matrix)
        sim_matrix = engine.calculate_similarity_matrix()

        # Check shape
        assert sim_matrix.shape == (3, 3)

        # Check diagonal is 1.0
        assert sim_matrix[0, 0] == 1.0
        assert sim_matrix[1, 1] == 1.0
        assert sim_matrix[2, 2] == 1.0

        # Check symmetry
        assert sim_matrix[0, 1] == sim_matrix[1, 0]

    def test_proximity_rankings(self):
        """Test proximity rankings."""
        matrix = FeatureMatrix()
        matrix.add_application(Application("App 1", {0: 5.0, 1: 3.0}))
        matrix.add_application(Application("App 2", {0: 4.0, 1: 3.0}))
        matrix.add_application(Application("App 3", {0: 1.0, 2: 5.0}))

        engine = ClusteringEngine(matrix)
        rankings = engine.get_proximity_rankings(top_n=2)

        # Should get 2 pairs
        assert len(rankings) == 2

        # Each ranking is (app1, app2, score)
        assert len(rankings[0]) == 3

        # Scores should be sorted descending
        assert rankings[0][2] >= rankings[1][2]


def test_csv_loader_integration():
    """Integration test for CSV loading."""
    # This test requires the sample CSV files
    data_dir = Path(__file__).parent.parent / "data"
    matrix_file = data_dir / "sample_applications.csv"
    mapping_file = data_dir / "sample_dimensions.csv"

    if matrix_file.exists() and mapping_file.exists():
        matrix = CSVLoader.load_feature_matrix(
            str(matrix_file), str(mapping_file)
        )

        assert len(matrix) > 0
        assert len(matrix.all_dimensions) > 0
        assert len(matrix.dimension_names) > 0
