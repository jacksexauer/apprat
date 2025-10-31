"""
Feature matrix management for the apprat tool.
"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# Support both relative and absolute imports
try:
    from .application import Application
except ImportError:
    from core.application import Application


class FeatureMatrix:
    """
    Manages the feature matrix containing all applications and their dimension scores.
    """

    def __init__(self):
        """Initialize an empty FeatureMatrix."""
        self.applications: Dict[str, Application] = {}
        self.dimension_names: Dict[int, str] = {}
        self._all_dimensions: Optional[List[int]] = None

    def add_application(self, application: Application) -> None:
        """
        Add an application to the matrix.

        Args:
            application: Application instance to add
        """
        self.applications[application.name] = application
        self._all_dimensions = None  # Reset cache

    def add_dimension_mapping(self, index: int, name: str) -> None:
        """
        Add a dimension mapping.

        Args:
            index: Dimension index
            name: Human-readable dimension name
        """
        self.dimension_names[index] = name
        self._all_dimensions = None  # Reset cache

    def set_dimension_mappings(self, mappings: Dict[int, str]) -> None:
        """
        Set all dimension mappings at once.

        Args:
            mappings: Dictionary of dimension index to name
        """
        self.dimension_names = mappings.copy()
        self._all_dimensions = None  # Reset cache

    @property
    def all_dimensions(self) -> List[int]:
        """
        Get sorted list of all dimension indices.

        Returns:
            Sorted list of all dimension indices
        """
        if self._all_dimensions is None:
            # Get all dimensions from both mappings and application scores
            dims_from_mappings = set(self.dimension_names.keys())
            dims_from_apps = set()
            for app in self.applications.values():
                dims_from_apps.update(app.scores.keys())
            self._all_dimensions = sorted(dims_from_mappings | dims_from_apps)
        return self._all_dimensions

    def get_dimension_name(self, index: int) -> str:
        """
        Get the name for a dimension index.

        Args:
            index: Dimension index

        Returns:
            Dimension name, or "Dimension {index}" if not mapped
        """
        return self.dimension_names.get(index, f"Dimension {index}")

    def get_application(self, name: str) -> Optional[Application]:
        """
        Get an application by name.

        Args:
            name: Application name

        Returns:
            Application instance or None if not found
        """
        return self.applications.get(name)

    def get_application_names(self) -> List[str]:
        """
        Get list of all application names.

        Returns:
            List of application names
        """
        return list(self.applications.keys())

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the feature matrix to a pandas DataFrame.

        Returns:
            DataFrame with applications as rows and dimensions as columns
        """
        data = []
        for app_name, app in self.applications.items():
            row = {"Application": app_name}
            for dim in self.all_dimensions:
                dim_name = self.get_dimension_name(dim)
                row[f"{dim}: {dim_name}"] = app.get_score(dim)
            data.append(row)

        return pd.DataFrame(data)

    def get_score_matrix(self) -> np.ndarray:
        """
        Get the raw score matrix as a numpy array.

        Returns:
            2D numpy array with shape (num_applications, num_dimensions)
        """
        app_names = self.get_application_names()
        matrix = np.zeros((len(app_names), len(self.all_dimensions)))

        for i, app_name in enumerate(app_names):
            app = self.applications[app_name]
            matrix[i] = app.get_score_vector(self.all_dimensions)

        return matrix

    def __len__(self) -> int:
        """Get the number of applications in the matrix."""
        return len(self.applications)

    def __repr__(self) -> str:
        return (
            f"FeatureMatrix(applications={len(self.applications)}, "
            f"dimensions={len(self.all_dimensions)})"
        )
