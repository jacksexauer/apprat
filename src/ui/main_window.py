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
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextEdit,
    QGroupBox,
)
from PyQt6.QtCore import Qt
from typing import Optional, Dict

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

        # Tab 3: Clusters (with split view)
        self.cluster_tab = QWidget()
        self.cluster_layout = QVBoxLayout(self.cluster_tab)
        self.cluster_layout.addLayout(self.create_cluster_controls())

        # Create splitter for cluster list and details
        cluster_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Cluster list
        self.cluster_list = QListWidget()
        self.cluster_list.currentItemChanged.connect(self.on_cluster_selected)
        cluster_splitter.addWidget(self.cluster_list)

        # Right side: Cluster details
        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)

        # Applications in cluster
        apps_group = QGroupBox("Applications in Cluster")
        apps_layout = QVBoxLayout(apps_group)
        self.cluster_apps_list = QListWidget()
        apps_layout.addWidget(self.cluster_apps_list)
        detail_layout.addWidget(apps_group)

        # Significant features
        features_group = QGroupBox("Significant Features")
        features_layout = QVBoxLayout(features_group)
        self.cluster_features_table = QTableWidget()
        self.cluster_features_table.setColumnCount(4)
        self.cluster_features_table.setHorizontalHeaderLabels([
            "Feature", "Avg Score", "Apps with Feature", "Significance"
        ])
        self.cluster_features_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        features_layout.addWidget(self.cluster_features_table)
        detail_layout.addWidget(features_group)

        cluster_splitter.addWidget(detail_widget)
        cluster_splitter.setStretchFactor(0, 1)  # Cluster list takes 1 part
        cluster_splitter.setStretchFactor(1, 2)  # Details takes 2 parts

        self.cluster_layout.addWidget(cluster_splitter)
        self.tabs.addTab(self.cluster_tab, "Clusters")

        # Store cluster data for later use
        self.current_clusters: Optional[Dict[str, str]] = None
        self.cluster_history: list = []  # Stack for undo functionality

        # Status bar
        self.statusBar().showMessage("Ready. Please load application data.")

    def _natural_sort_key(self, cluster_id: str):
        """
        Generate a sort key for natural sorting of cluster IDs.

        Handles hierarchical cluster names like "0", "0.1", "0.2", "1", "1.1", "1.2".

        Args:
            cluster_id: Cluster ID string to generate sort key for

        Returns:
            Tuple that can be used for sorting
        """
        parts = str(cluster_id).split('.')
        key = []
        for part in parts:
            try:
                key.append(int(part))
            except ValueError:
                key.append(part)
        return tuple(key)

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

        # Clustering mode
        layout.addWidget(QLabel("Mode:"))
        self.cluster_mode = QComboBox()
        self.cluster_mode.addItems(["Automatic", "Manual"])
        self.cluster_mode.currentTextChanged.connect(self.on_cluster_mode_changed)
        layout.addWidget(self.cluster_mode)

        # Number of clusters (for manual mode)
        self.n_clusters_label = QLabel("Clusters:")
        layout.addWidget(self.n_clusters_label)
        self.n_clusters_spinner = QSpinBox()
        self.n_clusters_spinner.setRange(2, 20)
        self.n_clusters_spinner.setValue(3)
        layout.addWidget(self.n_clusters_spinner)

        # Auto-cluster method
        self.auto_method_label = QLabel("Auto Method:")
        layout.addWidget(self.auto_method_label)
        self.auto_cluster_method = QComboBox()
        self.auto_cluster_method.addItems([
            "Silhouette (Best Quality)",
            "Threshold (Natural Groups)",
            "DBSCAN (Density-Based)"
        ])
        layout.addWidget(self.auto_cluster_method)

        # Similarity method
        layout.addWidget(QLabel("Similarity:"))
        self.cluster_method = QComboBox()
        self.cluster_method.addItems(["proportional", "jaccard", "cosine"])
        layout.addWidget(self.cluster_method)

        # Calculate button
        self.calculate_clusters_btn = QPushButton("Calculate Clusters")
        self.calculate_clusters_btn.clicked.connect(self.calculate_clusters)
        self.calculate_clusters_btn.setEnabled(False)
        layout.addWidget(self.calculate_clusters_btn)

        # Add separator
        layout.addWidget(QLabel(" | "))

        # Split cluster button
        self.split_cluster_btn = QPushButton("Split Cluster")
        self.split_cluster_btn.clicked.connect(self.split_selected_cluster)
        self.split_cluster_btn.setEnabled(False)
        self.split_cluster_btn.setToolTip("Split the selected cluster into two sub-clusters")
        layout.addWidget(self.split_cluster_btn)

        # Undo button
        self.undo_split_btn = QPushButton("Undo Split")
        self.undo_split_btn.clicked.connect(self.undo_cluster_split)
        self.undo_split_btn.setEnabled(False)
        self.undo_split_btn.setToolTip("Undo the last cluster split")
        layout.addWidget(self.undo_split_btn)

        layout.addStretch()

        # Initially set to automatic mode
        self.on_cluster_mode_changed("Automatic")

        return layout

    def on_cluster_mode_changed(self, mode: str):
        """Handle cluster mode change."""
        is_manual = mode == "Manual"
        self.n_clusters_label.setVisible(is_manual)
        self.n_clusters_spinner.setVisible(is_manual)
        self.auto_method_label.setVisible(not is_manual)
        self.auto_cluster_method.setVisible(not is_manual)

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

            # Clear cached similarity matrix to force recalculation with new method
            self.clustering_engine._similarity_matrix = None

            rankings = self.clustering_engine.get_proximity_rankings(method, top_n)

            # Clear and refresh the table
            self.similarity_table.clearContents()
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
            mode = self.cluster_mode.currentText()

            # Determine which clustering method to use
            if mode == "Automatic":
                auto_method = self.auto_cluster_method.currentText()

                if "Silhouette" in auto_method:
                    clusters, metadata = self.clustering_engine.auto_cluster(method=method)
                    status_msg = (
                        f"Automatically found {metadata['n_clusters']} clusters "
                        f"(silhouette score: {metadata['silhouette_score']:.3f})"
                    )
                elif "Threshold" in auto_method:
                    clusters, metadata = self.clustering_engine.auto_cluster_threshold(method=method)
                    status_msg = (
                        f"Automatically found {metadata['n_clusters']} clusters "
                        f"(threshold: {metadata['threshold_used']:.3f})"
                    )
                elif "DBSCAN" in auto_method:
                    clusters, metadata = self.clustering_engine.dbscan_clustering(method=method)
                    status_msg = (
                        f"Found {metadata['n_clusters']} clusters with "
                        f"{metadata['n_noise']} outliers (DBSCAN)"
                    )
                else:
                    clusters, metadata = self.clustering_engine.auto_cluster(method=method)
                    status_msg = f"Automatically found {metadata['n_clusters']} clusters"
            else:
                # Manual mode
                n_clusters = self.n_clusters_spinner.value()
                clusters = self.clustering_engine.hierarchical_clustering(n_clusters, method)
                status_msg = f"Created {n_clusters} clusters using {method} method"

            # Store clusters for later use and in history
            self.current_clusters = clusters
            self.cluster_history = [clusters.copy()]  # Reset history with new clustering

            # Group by cluster
            cluster_groups = {}
            for app_name, cluster_id in clusters.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(app_name)

            # Populate cluster list (use natural sorting for hierarchical IDs)
            self.cluster_list.clear()
            for cluster_id in sorted(cluster_groups.keys(), key=self._natural_sort_key):
                # Handle string cluster IDs (including hierarchical ones like "0.1")
                cluster_label = f"Cluster {cluster_id}" if str(cluster_id) != "-1" else "Outliers"
                apps_count = len(cluster_groups[cluster_id])

                # Analyze cluster features
                analysis = self.clustering_engine.analyze_cluster_features(clusters, cluster_id)
                top_features = analysis['significant_features'][:2]
                feature_names = [f['dimension_name'] for f in top_features]
                features_preview = ', '.join(feature_names) if feature_names else 'No features'

                # Create list item with cluster info
                item_text = f"{cluster_label} ({apps_count} apps) - {features_preview}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.ItemDataRole.UserRole, cluster_id)  # Store cluster_id
                self.cluster_list.addItem(item)

            # Select first cluster by default
            if self.cluster_list.count() > 0:
                self.cluster_list.setCurrentRow(0)

            # Enable split button now that we have clusters
            self.split_cluster_btn.setEnabled(True)
            self.undo_split_btn.setEnabled(False)  # No history yet

            self.statusBar().showMessage(status_msg)

            # Show detailed information for automatic clustering
            if mode == "Automatic":
                info_msg = f"Clustering Results:\n\n{status_msg}\n\n"
                info_msg += f"Cluster Distribution:\n"
                for cluster_id in sorted(cluster_groups.keys(), key=self._natural_sort_key):
                    cluster_label = f"Cluster {cluster_id}" if str(cluster_id) != "-1" else "Outliers"
                    info_msg += f"  {cluster_label}: {len(cluster_groups[cluster_id])} applications\n"

                QMessageBox.information(self, "Automatic Clustering Complete", info_msg)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate clusters:\n{str(e)}")

    def on_cluster_selected(self, current, previous):
        """Handle cluster selection to show details."""
        if not current or not self.current_clusters or not self.clustering_engine:
            return

        try:
            # Get cluster ID from item data
            cluster_id = current.data(Qt.ItemDataRole.UserRole)

            # Analyze cluster features
            analysis = self.clustering_engine.analyze_cluster_features(
                self.current_clusters, cluster_id
            )

            # Populate applications list
            self.cluster_apps_list.clear()
            for app_name in sorted(analysis['applications']):
                self.cluster_apps_list.addItem(app_name)

            # Populate features table
            features = analysis['significant_features']
            self.cluster_features_table.setRowCount(len(features))

            for row, feature in enumerate(features):
                # Feature name
                self.cluster_features_table.setItem(
                    row, 0, QTableWidgetItem(feature['dimension_name'])
                )

                # Average score
                self.cluster_features_table.setItem(
                    row, 1, QTableWidgetItem(f"{feature['avg_score']:.2f}")
                )

                # Apps with feature
                apps_with = f"{feature['active_count']}/{analysis['size']}"
                percentage = f"({feature['active_ratio']*100:.0f}%)"
                self.cluster_features_table.setItem(
                    row, 2, QTableWidgetItem(f"{apps_with} {percentage}")
                )

                # Significance
                self.cluster_features_table.setItem(
                    row, 3, QTableWidgetItem(f"{feature['significance']:.2f}")
                )

            self.cluster_features_table.resizeColumnsToContents()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display cluster details:\n{str(e)}")

    def split_selected_cluster(self):
        """Split the currently selected cluster into two sub-clusters."""
        if not self.current_clusters or not self.clustering_engine:
            return

        # Get selected cluster
        current_item = self.cluster_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a cluster to split.")
            return

        cluster_id = current_item.data(Qt.ItemDataRole.UserRole)

        # Check if cluster has enough apps
        cluster_apps = [name for name, cid in self.current_clusters.items() if cid == cluster_id]
        if len(cluster_apps) < 2:
            QMessageBox.warning(
                self, "Cannot Split",
                f"Cluster has only {len(cluster_apps)} application(s). Need at least 2 to split."
            )
            return

        try:
            self.statusBar().showMessage("Splitting cluster...")

            # Save current state to history before splitting
            self.cluster_history.append(self.current_clusters.copy())

            # Perform the split
            method = self.cluster_method.currentText()
            new_clusters, metadata = self.clustering_engine.split_cluster(
                self.current_clusters, cluster_id, method
            )

            # Update current clusters
            self.current_clusters = new_clusters

            # Refresh the display
            self.refresh_cluster_display()

            # Enable undo button
            self.undo_split_btn.setEnabled(True)

            # Show info about the split
            info_msg = f"Cluster Split Complete\n\n"
            info_msg += f"Original: Cluster {metadata['original_cluster']} ({len(cluster_apps)} apps)\n\n"
            info_msg += f"Split into:\n"
            info_msg += f"  • Cluster {metadata['new_cluster_1']} ({metadata['size_1']} apps)\n"
            info_msg += f"  • Cluster {metadata['new_cluster_2']} ({metadata['size_2']} apps)\n\n"

            # Show which apps went where
            if 'group_1' in metadata and 'group_2' in metadata:
                info_msg += f"\nCluster {metadata['new_cluster_1']}:\n"
                for app in sorted(metadata['group_1']):
                    info_msg += f"  - {app}\n"

                info_msg += f"\nCluster {metadata['new_cluster_2']}:\n"
                for app in sorted(metadata['group_2']):
                    info_msg += f"  - {app}\n"

            QMessageBox.information(self, "Cluster Split", info_msg)

            self.statusBar().showMessage(
                f"Split cluster {cluster_id} into {metadata['new_cluster_1']} "
                f"and {metadata['new_cluster_2']}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to split cluster:\n{str(e)}")

    def undo_cluster_split(self):
        """Undo the last cluster split."""
        if len(self.cluster_history) < 2:
            QMessageBox.warning(self, "Nothing to Undo", "No cluster splits to undo.")
            return

        try:
            # Remove current state and restore previous
            self.cluster_history.pop()  # Remove current
            self.current_clusters = self.cluster_history[-1].copy()

            # Refresh the display
            self.refresh_cluster_display()

            # Disable undo if no more history
            if len(self.cluster_history) <= 1:
                self.undo_split_btn.setEnabled(False)

            self.statusBar().showMessage("Cluster split undone")

            QMessageBox.information(
                self, "Undo Complete",
                "The last cluster split has been undone."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to undo split:\n{str(e)}")

    def refresh_cluster_display(self):
        """Refresh the cluster list display with current clusters."""
        if not self.current_clusters or not self.clustering_engine:
            return

        # Group by cluster
        cluster_groups = {}
        for app_name, cluster_id in self.current_clusters.items():
            if cluster_id not in cluster_groups:
                cluster_groups[cluster_id] = []
            cluster_groups[cluster_id].append(app_name)

        # Remember current selection
        current_item = self.cluster_list.currentItem()
        selected_cluster_id = None
        if current_item:
            selected_cluster_id = current_item.data(Qt.ItemDataRole.UserRole)

        # Populate cluster list (use natural sorting for hierarchical IDs)
        self.cluster_list.clear()
        for cluster_id in sorted(cluster_groups.keys(), key=self._natural_sort_key):
            # Handle string cluster IDs (including hierarchical ones like "0.1")
            cluster_label = f"Cluster {cluster_id}" if str(cluster_id) != "-1" else "Outliers"
            apps_count = len(cluster_groups[cluster_id])

            # Analyze cluster features
            analysis = self.clustering_engine.analyze_cluster_features(
                self.current_clusters, cluster_id
            )
            top_features = analysis['significant_features'][:2]
            feature_names = [f['dimension_name'] for f in top_features]
            features_preview = ', '.join(feature_names) if feature_names else 'No features'

            # Create list item with cluster info
            item_text = f"{cluster_label} ({apps_count} apps) - {features_preview}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, cluster_id)
            self.cluster_list.addItem(item)

        # Try to restore selection or select first
        selection_found = False
        if selected_cluster_id is not None:
            for i in range(self.cluster_list.count()):
                item = self.cluster_list.item(i)
                if item.data(Qt.ItemDataRole.UserRole) == selected_cluster_id:
                    self.cluster_list.setCurrentRow(i)
                    selection_found = True
                    break

        if not selection_found and self.cluster_list.count() > 0:
            self.cluster_list.setCurrentRow(0)
