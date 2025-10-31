"""
Main window for the apprat desktop application.
"""
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QComboBox,
    QSpinBox,
    QMessageBox,
    QHeaderView,
)
from PyQt6.QtCore import Qt
from typing import Optional

# Support both relative and absolute imports
try:
    from ..core.feature_matrix import FeatureMatrix
    from ..core.csv_loader import CSVLoader
    from ..analysis.clustering import ClusteringEngine
except ImportError:
    from core.feature_matrix import FeatureMatrix
    from core.csv_loader import CSVLoader
    from analysis.clustering import ClusteringEngine


class MainWindow(QMainWindow):
    """
    Main application window for apprat.
    """

    def __init__(self):
        super().__init__()
        self.feature_matrix: Optional[FeatureMatrix] = None
        self.clustering_engine: Optional[ClusteringEngine] = None
        self.matrix_filepath: Optional[str] = None
        self.mapping_filepath: Optional[str] = None

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("apprat - Application Rationalization Tool")
        self.setMinimumSize(1000, 700)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Add file loading section
        main_layout.addLayout(self.create_file_loading_section())

        # Add tabs for different views
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Tab 1: Application Matrix View
        self.matrix_tab = QWidget()
        self.matrix_layout = QVBoxLayout(self.matrix_tab)
        self.matrix_table = QTableWidget()
        self.matrix_layout.addWidget(self.matrix_table)
        self.tabs.addTab(self.matrix_tab, "Application Matrix")

        # Tab 2: Similarity Rankings
        self.similarity_tab = QWidget()
        self.similarity_layout = QVBoxLayout(self.similarity_tab)
        self.similarity_layout.addLayout(self.create_similarity_controls())
        self.similarity_table = QTableWidget()
        self.similarity_layout.addWidget(self.similarity_table)
        self.tabs.addTab(self.similarity_tab, "Similarity Rankings")

        # Tab 3: Clusters
        self.cluster_tab = QWidget()
        self.cluster_layout = QVBoxLayout(self.cluster_tab)
        self.cluster_layout.addLayout(self.create_cluster_controls())
        self.cluster_table = QTableWidget()
        self.cluster_layout.addWidget(self.cluster_table)
        self.tabs.addTab(self.cluster_tab, "Clusters")

        # Status bar
        self.statusBar().showMessage("Ready. Please load application data.")

    def create_file_loading_section(self) -> QHBoxLayout:
        """Create the file loading section."""
        layout = QHBoxLayout()

        # Matrix file selection
        layout.addWidget(QLabel("Application Matrix:"))
        self.matrix_label = QLabel("No file selected")
        layout.addWidget(self.matrix_label)
        matrix_btn = QPushButton("Browse...")
        matrix_btn.clicked.connect(self.load_matrix_file)
        layout.addWidget(matrix_btn)

        # Mapping file selection
        layout.addWidget(QLabel("Dimension Mapping:"))
        self.mapping_label = QLabel("No file selected")
        layout.addWidget(self.mapping_label)
        mapping_btn = QPushButton("Browse...")
        mapping_btn.clicked.connect(self.load_mapping_file)
        layout.addWidget(mapping_btn)

        # Load button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data)
        self.load_btn.setEnabled(False)
        layout.addWidget(self.load_btn)

        return layout

    def create_similarity_controls(self) -> QHBoxLayout:
        """Create controls for similarity analysis."""
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Similarity Method:"))
        self.similarity_method = QComboBox()
        self.similarity_method.addItems(["proportional", "jaccard", "cosine"])
        layout.addWidget(self.similarity_method)

        layout.addWidget(QLabel("Show Top:"))
        self.top_n_spinner = QSpinBox()
        self.top_n_spinner.setRange(5, 1000)
        self.top_n_spinner.setValue(20)
        layout.addWidget(self.top_n_spinner)

        self.calculate_similarity_btn = QPushButton("Calculate Similarity")
        self.calculate_similarity_btn.clicked.connect(self.calculate_similarity)
        self.calculate_similarity_btn.setEnabled(False)
        layout.addWidget(self.calculate_similarity_btn)

        layout.addStretch()
        return layout

    def create_cluster_controls(self) -> QHBoxLayout:
        """Create controls for clustering analysis."""
        layout = QHBoxLayout()

        layout.addWidget(QLabel("Number of Clusters:"))
        self.n_clusters_spinner = QSpinBox()
        self.n_clusters_spinner.setRange(2, 20)
        self.n_clusters_spinner.setValue(3)
        layout.addWidget(self.n_clusters_spinner)

        layout.addWidget(QLabel("Method:"))
        self.cluster_method = QComboBox()
        self.cluster_method.addItems(["proportional", "jaccard", "cosine"])
        layout.addWidget(self.cluster_method)

        self.calculate_clusters_btn = QPushButton("Calculate Clusters")
        self.calculate_clusters_btn.clicked.connect(self.calculate_clusters)
        self.calculate_clusters_btn.setEnabled(False)
        layout.addWidget(self.calculate_clusters_btn)

        layout.addStretch()
        return layout

    def load_matrix_file(self):
        """Open file dialog to select application matrix CSV."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Application Matrix CSV", "", "CSV Files (*.csv)"
        )
        if filepath:
            self.matrix_filepath = filepath
            self.matrix_label.setText(filepath.split("/")[-1])
            self.check_files_loaded()

    def load_mapping_file(self):
        """Open file dialog to select dimension mapping CSV."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Dimension Mapping CSV", "", "CSV Files (*.csv)"
        )
        if filepath:
            self.mapping_filepath = filepath
            self.mapping_label.setText(filepath.split("/")[-1])
            self.check_files_loaded()

    def check_files_loaded(self):
        """Enable load button if matrix file is selected."""
        self.load_btn.setEnabled(self.matrix_filepath is not None)

    def load_data(self):
        """Load data from selected CSV files."""
        try:
            self.statusBar().showMessage("Loading data...")

            # Load feature matrix
            self.feature_matrix = CSVLoader.load_feature_matrix(
                self.matrix_filepath, self.mapping_filepath
            )

            # Create clustering engine
            self.clustering_engine = ClusteringEngine(self.feature_matrix)

            # Display matrix
            self.display_matrix()

            # Enable analysis buttons
            self.calculate_similarity_btn.setEnabled(True)
            self.calculate_clusters_btn.setEnabled(True)

            self.statusBar().showMessage(
                f"Loaded {len(self.feature_matrix)} applications with "
                f"{len(self.feature_matrix.all_dimensions)} dimensions"
            )

            QMessageBox.information(
                self,
                "Success",
                f"Successfully loaded {len(self.feature_matrix)} applications!",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data:\n{str(e)}")
            self.statusBar().showMessage("Error loading data")

    def display_matrix(self):
        """Display the application matrix in the table."""
        if not self.feature_matrix:
            return

        df = self.feature_matrix.to_dataframe()

        self.matrix_table.setRowCount(len(df))
        self.matrix_table.setColumnCount(len(df.columns))
        self.matrix_table.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                item = QTableWidgetItem(str(row[col]))
                self.matrix_table.setItem(i, j, item)

        self.matrix_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def calculate_similarity(self):
        """Calculate and display similarity rankings."""
        if not self.clustering_engine:
            return

        try:
            self.statusBar().showMessage("Calculating similarities...")

            method = self.similarity_method.currentText()
            top_n = self.top_n_spinner.value()

            rankings = self.clustering_engine.get_proximity_rankings(method, top_n)

            # Display in table
            self.similarity_table.setRowCount(len(rankings))
            self.similarity_table.setColumnCount(3)
            self.similarity_table.setHorizontalHeaderLabels(
                ["Application 1", "Application 2", "Similarity Score"]
            )

            for i, (app1, app2, score) in enumerate(rankings):
                self.similarity_table.setItem(i, 0, QTableWidgetItem(app1))
                self.similarity_table.setItem(i, 1, QTableWidgetItem(app2))
                self.similarity_table.setItem(i, 2, QTableWidgetItem(f"{score:.4f}"))

            self.similarity_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )

            self.statusBar().showMessage(f"Calculated similarities using {method} method")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate similarities:\n{str(e)}")

    def calculate_clusters(self):
        """Calculate and display clusters."""
        if not self.clustering_engine:
            return

        try:
            self.statusBar().showMessage("Calculating clusters...")

            method = self.cluster_method.currentText()
            n_clusters = self.n_clusters_spinner.value()

            clusters = self.clustering_engine.hierarchical_clustering(n_clusters, method)

            # Group by cluster
            cluster_groups = {}
            for app_name, cluster_id in clusters.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(app_name)

            # Display in table
            total_rows = sum(len(apps) for apps in cluster_groups.values())
            self.cluster_table.setRowCount(total_rows)
            self.cluster_table.setColumnCount(2)
            self.cluster_table.setHorizontalHeaderLabels(["Cluster", "Application"])

            row = 0
            for cluster_id in sorted(cluster_groups.keys()):
                apps = sorted(cluster_groups[cluster_id])
                for app_name in apps:
                    self.cluster_table.setItem(row, 0, QTableWidgetItem(f"Cluster {cluster_id}"))
                    self.cluster_table.setItem(row, 1, QTableWidgetItem(app_name))
                    row += 1

            self.cluster_table.horizontalHeader().setSectionResizeMode(
                QHeaderView.ResizeMode.ResizeToContents
            )

            self.statusBar().showMessage(
                f"Calculated {len(cluster_groups)} clusters using {method} method"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate clusters:\n{str(e)}")
