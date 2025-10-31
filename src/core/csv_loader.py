"""
CSV loader for importing application data and dimension mappings.
"""
from typing import Tuple
import pandas as pd

# Support both relative and absolute imports
try:
    from .application import Application
    from .feature_matrix import FeatureMatrix
except ImportError:
    from core.application import Application
    from core.feature_matrix import FeatureMatrix


class CSVLoader:
    """
    Handles loading application matrix and dimension mapping from CSV files.
    """

    @staticmethod
    def load_dimension_mapping(filepath: str) -> dict:
        """
        Load dimension mapping from CSV file.

        Expected CSV format:
        Index,Dimension
        0,Cloud Native
        1,Mobile Support
        ...

        Args:
            filepath: Path to the dimension mapping CSV file

        Returns:
            Dictionary mapping dimension index (int) to name (str)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the CSV format is invalid
        """
        try:
            df = pd.read_csv(filepath)

            # Check for required columns (flexible naming)
            if len(df.columns) < 2:
                raise ValueError(
                    "Dimension mapping CSV must have at least 2 columns "
                    "(index and dimension name)"
                )

            # Use first column as index, second as name
            index_col = df.columns[0]
            name_col = df.columns[1]

            # Build mapping dictionary
            mapping = {}
            for _, row in df.iterrows():
                idx = int(row[index_col])
                name = str(row[name_col])
                mapping[idx] = name

            return mapping

        except FileNotFoundError:
            raise FileNotFoundError(f"Dimension mapping file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading dimension mapping: {str(e)}")

    @staticmethod
    def load_application_matrix(filepath: str) -> Tuple[list, list, list]:
        """
        Load application matrix from CSV file.

        Expected CSV format:
        Application,0,1,2,3,4
        App A,5,3,0,4,2
        App B,4,3,1,5,2
        ...

        Args:
            filepath: Path to the application matrix CSV file

        Returns:
            Tuple of (app_names, dimension_indices, scores_matrix)
            - app_names: List of application names
            - dimension_indices: List of dimension indices (as integers)
            - scores_matrix: 2D list of scores

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the CSV format is invalid
        """
        try:
            df = pd.read_csv(filepath)

            # First column should be application names
            if len(df.columns) < 2:
                raise ValueError(
                    "Application matrix CSV must have at least 2 columns "
                    "(application name and at least one dimension)"
                )

            app_col = df.columns[0]
            app_names = df[app_col].tolist()

            # Remaining columns are dimension indices
            dimension_cols = df.columns[1:]
            dimension_indices = [int(col) for col in dimension_cols]

            # Get the score matrix
            scores_matrix = df[dimension_cols].values.tolist()

            return app_names, dimension_indices, scores_matrix

        except FileNotFoundError:
            raise FileNotFoundError(f"Application matrix file not found: {filepath}")
        except Exception as e:
            raise ValueError(f"Error loading application matrix: {str(e)}")

    @staticmethod
    def load_feature_matrix(
        matrix_filepath: str, mapping_filepath: str = None
    ) -> FeatureMatrix:
        """
        Load a complete FeatureMatrix from CSV files.

        Args:
            matrix_filepath: Path to application matrix CSV
            mapping_filepath: Optional path to dimension mapping CSV

        Returns:
            Populated FeatureMatrix instance

        Raises:
            FileNotFoundError: If required files don't exist
            ValueError: If CSV formats are invalid
        """
        feature_matrix = FeatureMatrix()

        # Load dimension mapping if provided
        if mapping_filepath:
            mappings = CSVLoader.load_dimension_mapping(mapping_filepath)
            feature_matrix.set_dimension_mappings(mappings)

        # Load application matrix
        app_names, dim_indices, scores_matrix = CSVLoader.load_application_matrix(
            matrix_filepath
        )

        # Create Application objects and add to feature matrix
        for app_name, scores_row in zip(app_names, scores_matrix):
            scores_dict = {
                dim_idx: float(score)
                for dim_idx, score in zip(dim_indices, scores_row)
            }
            app = Application(name=app_name, scores=scores_dict)
            feature_matrix.add_application(app)

        return feature_matrix
