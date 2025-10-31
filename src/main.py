"""
Main entry point for the apprat application.
"""
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    """Launch the apprat desktop application."""
    app = QApplication(sys.argv)
    app.setApplicationName("apprat")
    app.setOrganizationName("apprat")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
