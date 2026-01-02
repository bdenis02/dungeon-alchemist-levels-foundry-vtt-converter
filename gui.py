import sys
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QLabel, QCheckBox, QRadioButton,
                               QComboBox, QLineEdit, QPushButton, QStatusBar,
                               QToolBar, QGroupBox, QSpinBox)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QSettings
from main import check_and_convert, get_basepath_and_dirpath_from_filename

random.seed(42)


class DropArea(QLabel):
    def __init__(self):
        super().__init__("Drag and Drop File Here")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "border: 2px dashed #666; margin: 10px; padding: 20px; border-radius: 5px;")
        self.setAcceptDrops(True)
        self.file_path = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.file_path = files[0]
            self.setText(f"File Loaded: {self.file_path}")
        else:
            self.setText(f"File load error")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Organization and Application names are used to create the config file path
        self.settings = QSettings("Mogusha", "Dungeon Alchemist Levels FoundryVTT Converter")
        self.setWindowTitle("PySide6 Persistent App")
        self.resize(600, 500)

        # UI Construction
        self.setup_ui()

        # Load Defaults on Startup
        self.load_settings()

    def setup_ui(self):
        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction(QAction("Open", self))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Tile Save Path Location
        input_layout = QHBoxLayout()
        self.input_tile_save_path = QLineEdit()
        self.input_tile_save_path.setPlaceholderText(
            "local path in FoundryVTT to save this scene assets {basepath} for original filename")
        input_layout.addWidget(QLabel("FoundryVTT tile save path:"))
        input_layout.addWidget(self.input_tile_save_path)
        main_layout.addLayout(input_layout)

        # Ground Floor Index
        self.ground_floor_selector = QSpinBox()
        self.ground_floor_selector.setRange(-50, 50)
        self.ground_floor_selector.setSingleStep(1)
        ground_floor_spin_layout = QHBoxLayout()
        ground_floor_spin_layout.addWidget(QLabel("Ground Floor:"))
        ground_floor_spin_layout.addWidget(self.ground_floor_selector)
        main_layout.addLayout(ground_floor_spin_layout)

        # Floor Height
        self.floor_height_selector = QSpinBox()
        self.floor_height_selector.setRange(1, 100)
        self.floor_height_selector.setSingleStep(1)
        floor_height_spin_layout = QHBoxLayout()
        floor_height_spin_layout.addWidget(QLabel("Floor Height:"))
        floor_height_spin_layout.addWidget(self.floor_height_selector)
        main_layout.addLayout(floor_height_spin_layout)

        # Seed
        self.seed_selector = QSpinBox()
        self.seed_selector.setSingleStep(1)
        seed_spin_layout = QHBoxLayout()
        seed_spin_layout.addWidget(QLabel("Seed:"))
        seed_spin_layout.addWidget(self.seed_selector)
        main_layout.addLayout(seed_spin_layout)

        # Drag & Drop
        self.drop_zone = DropArea()
        main_layout.addWidget(self.drop_zone)

        # Big Process Button
        self.process_button = QPushButton("PROCESS")
        self.process_button.setMinimumHeight(50)
        self.process_button.setStyleSheet("background-color: #2c3e50; color: white;")
        self.process_button.clicked.connect(self.process_map)
        main_layout.addWidget(self.process_button)

        self.setStatusBar(QStatusBar(self))

    # --- Persistence Logic ---

    def load_settings(self):
        """Retrieves values from the local config file."""
        # The second argument is the 'default' value if the key doesn't exist yet
        self.input_tile_save_path.setText(
            self.settings.value(
                "tile_save_path",
                "assets/scenes/{basepath}"))

        val = int(self.settings.value("ground_floor", 0))
        self.ground_floor_selector.setValue(val)

        val = int(self.settings.value("floor_height", 10))
        self.floor_height_selector.setValue(val)

        val = int(self.settings.value("seed", 42))
        self.seed_selector.setValue(val)

        self.statusBar().showMessage("Settings loaded from local storage.")

    def save_settings(self):
        """Writes current UI state to the local config file."""
        self.settings.setValue("tile_save_path", self.input_tile_save_path.text())
        self.settings.setValue("ground_floor", self.ground_floor_selector.value())
        self.settings.setValue("floor_height", self.floor_height_selector.value())
        self.settings.setValue("seed", self.seed_selector.value())
        print(f"Settings saved to {QSettings.fileName(self.settings)}")

    def closeEvent(self, event):
        """Triggered automatically when the window is closed."""
        self.save_settings()
        event.accept()

    def process_map(self):
        if self.drop_zone.file_path is None:
            self.statusBar().showMessage("No files selected!")
            return
        filepath = self.drop_zone.file_path
        tile_save_path = self.input_tile_save_path.text()
        if "{basepath}" in tile_save_path:
            basepath, dirpath = get_basepath_and_dirpath_from_filename(filepath)
            tile_save_path = tile_save_path.format(basepath=basepath)
        random.seed(self.seed_selector.value())
        check_and_convert(
            filepath,
            tile_save_path,
            self.ground_floor_selector.value(),
            self.floor_height_selector.value())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
