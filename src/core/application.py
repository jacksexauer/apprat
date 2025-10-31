"""
Application data model for the apprat tool.
"""
from typing import Dict, List, Optional
import numpy as np


class Application:
    """
    Represents a single application with its feature scores.
    """

    def __init__(self, name: str, scores: Dict[int, float]):
        """
        Initialize an Application.

        Args:
            name: The name/identifier of the application
            scores: Dictionary mapping dimension indices to scores
        """
        self.name = name
        self.scores = scores
        self._active_dimensions = None

    @property
    def active_dimensions(self) -> List[int]:
        """
        Get list of dimension indices where this application has non-zero scores.

        Returns:
            List of dimension indices with non-zero scores
        """
        if self._active_dimensions is None:
            self._active_dimensions = [
                dim for dim, score in self.scores.items() if score > 0
            ]
        return self._active_dimensions

    @property
    def num_active_dimensions(self) -> int:
        """
        Get count of dimensions where this application has non-zero scores.

        Returns:
            Number of active (non-zero) dimensions
        """
        return len(self.active_dimensions)

    def get_score(self, dimension: int) -> float:
        """
        Get the score for a specific dimension.

        Args:
            dimension: The dimension index

        Returns:
            The score for that dimension (0 if not present)
        """
        return self.scores.get(dimension, 0.0)

    def get_score_vector(self, all_dimensions: List[int]) -> np.ndarray:
        """
        Get a score vector for all specified dimensions.

        Args:
            all_dimensions: List of all dimension indices to include

        Returns:
            Numpy array of scores in the order of all_dimensions
        """
        return np.array([self.get_score(dim) for dim in all_dimensions])

    def __repr__(self) -> str:
        return f"Application(name='{self.name}', dimensions={self.num_active_dimensions})"

    def __str__(self) -> str:
        return self.name
