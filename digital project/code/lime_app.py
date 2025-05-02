# lime_app.py
import os
import numpy as np
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QComboBox, QPushButton, QFileDialog, QGroupBox, QGridLayout,
    QCheckBox, QMessageBox, QFrame, QScrollArea, QDialog, QProgressBar,
    QLineEdit, QTabWidget, QInputDialog
)
from lime.lime_image import LimeImageExplainer
from style import StyleSheet
from modern_frame import ModernFrame
from image_views import ImageViews
from progress_dialog import ProgressDialog
from lime_explainer import LimeExplanationGenerator, ExplanationWorker
from novellime import DPPLimeView  # Import the Novel LIME implementation
from model_handler import ModelHandler
from model_dialogs import ModelConfigDialog, ClassSetConfigDialog


class VideoProcessingWorker(QThread):
    """Worker thread for processing video files."""
    finished = pyqtSignal(list)  # Signal emitting list of frame paths
    progress = pyqtSignal(int, int)  # Signal for progress updates
    error = pyqtSignal(str)

    def __init__(self, video_path, output_dir):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir

    def run(self):
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_paths = []

            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                frame_path = os.path.join(self.output_dir, f'frame_{frame_idx:04d}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
                self.progress.emit(frame_idx + 1, total_frames)

            cap.release()
            self.finished.emit(frame_paths)
        except Exception as e:
            self.error.emit(str(e))


class LimeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced LIME Explanations")
        self.setGeometry(100, 100, 1800, 1000)
        self.setStyleSheet(StyleSheet.get_stylesheet())

        self.dataset = []
        self.current_image_index = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.explanation_images = []
        self.explanation_labels = []
        self.current_explanation_index = 0
        self.current_image_type = "original"
        self.class_checkboxes = {}

        # Initialize model handler first
        self.model_handler = ModelHandler()
        
        # Then initialize the explanation generator
        self.explanation_generator = LimeExplanationGenerator(
            dataset=self.dataset,
            current_image_index=self.current_image_index,
            model_handler=self.model_handler
        )
        
        self.explanation_worker = None
        self.video_worker = None

        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left Panel: Controls
        control_frame = ModernFrame()
        control_layout = QVBoxLayout()
        control_layout.setSpacing(10)
        control_layout.setContentsMargins(10, 10, 10, 10)
        control_frame.setLayout(control_layout)

        # Model Management
        model_manage_group = QGroupBox("Model Management")
        model_manage_layout = QVBoxLayout()
        
        model_sel_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_selector = QComboBox()
        self.update_model_selector()
        model_sel_layout.addWidget(model_label)
        model_sel_layout.addWidget(self.model_selector)
        model_manage_layout.addLayout(model_sel_layout)
        
        model_btn_layout = QHBoxLayout()
        add_model_btn = QPushButton("Add Model")
        edit_model_btn = QPushButton("Edit Model")
        remove_model_btn = QPushButton("Remove Model")
        
        add_model_btn.clicked.connect(self.add_model)
        edit_model_btn.clicked.connect(self.edit_model)
        remove_model_btn.clicked.connect(self.remove_model)
        
        model_btn_layout.addWidget(add_model_btn)
        model_btn_layout.addWidget(edit_model_btn)
        model_btn_layout.addWidget(remove_model_btn)
        model_manage_layout.addLayout(model_btn_layout)
        
        class_btn_layout = QHBoxLayout()
        add_class_set_btn = QPushButton("Add Class Set")
        edit_class_set_btn = QPushButton("Edit Class Set")
        remove_class_set_btn = QPushButton("Remove Class Set")
        
        add_class_set_btn.clicked.connect(self.add_class_set)
        edit_class_set_btn.clicked.connect(self.edit_class_set)
        remove_class_set_btn.clicked.connect(self.remove_class_set)
        
        class_btn_layout.addWidget(add_class_set_btn)
        class_btn_layout.addWidget(edit_class_set_btn)
        class_btn_layout.addWidget(remove_class_set_btn)
        model_manage_layout.addLayout(class_btn_layout)
        
        model_manage_group.setLayout(model_manage_layout)
        control_layout.addWidget(model_manage_group)
        
        # Class Selection
        class_label = QLabel("Select Classes:")
        class_label.setStyleSheet("font-weight: bold;")
        self.class_scroll = QScrollArea()
        self.class_scroll.setWidgetResizable(True)
        self.class_scroll.setFrameShape(QFrame.NoFrame)
        self.class_scroll.setMinimumHeight(150)
        
        # Create container widget for class checkboxes
        self.class_container = QWidget()
        self.class_container_layout = QVBoxLayout()
        self.class_container.setLayout(self.class_container_layout)
        
        # Create the groupbox for classes
        self.class_groupbox = QGroupBox("Available Classes")
        self.class_grid = QGridLayout()
        self.class_grid.setSpacing(8)
        self.class_grid.setContentsMargins(10, 15, 10, 10)
        self.class_groupbox.setLayout(self.class_grid)
        
        # Add a placeholder message
        self.class_placeholder = QLabel("Select a model to view available classes")
        self.class_placeholder.setAlignment(Qt.AlignCenter)
        self.class_grid.addWidget(self.class_placeholder, 0, 0, 1, 3)
        
        # Add the groupbox to the container layout
        self.class_container_layout.addWidget(self.class_groupbox)
        
        # Set the container as the scrollable widget
        self.class_scroll.setWidget(self.class_container)
        
        control_layout.addWidget(class_label)
        control_layout.addWidget(self.class_scroll)

        # Select All/None Buttons
        select_buttons_layout = QHBoxLayout()
        select_all_button = QPushButton("Select All")
        select_none_button = QPushButton("Select None")
        select_all_button.clicked.connect(self.select_all_classes)
        select_none_button.clicked.connect(self.deselect_all_classes)
        select_buttons_layout.addWidget(select_all_button)
        select_buttons_layout.addWidget(select_none_button)
        control_layout.addLayout(select_buttons_layout)

        # LIME Parameters Input
        lime_params_group = QGroupBox("LIME Parameters")
        lime_params_layout = QGridLayout()

        lime_params_layout.addWidget(QLabel("Number of Samples:"), 0, 0)
        self.num_samples_input = QLineEdit("1000")
        lime_params_layout.addWidget(self.num_samples_input, 0, 1)

        lime_params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_input = QLineEdit("10")
        lime_params_layout.addWidget(self.batch_size_input, 1, 1)

        # Add other LIME parameter inputs here as needed
        lime_params_layout.addWidget(QLabel("Positive Only:"), 2, 0)
        self.positive_only_checkbox = QCheckBox()
        self.positive_only_checkbox.setChecked(True)  # Default to True
        lime_params_layout.addWidget(self.positive_only_checkbox, 2, 1)

        lime_params_layout.addWidget(QLabel("Number of Features:"), 3, 0)
        self.num_features_input = QLineEdit("5")
        lime_params_layout.addWidget(self.num_features_input, 3, 1)

        lime_params_layout.addWidget(QLabel("Hide Rest:"), 4, 0)
        self.hide_rest_checkbox = QCheckBox()
        lime_params_layout.addWidget(self.hide_rest_checkbox, 4, 1)

        lime_params_layout.addWidget(QLabel("Grey Color (R,G,B):"), 5, 0)
        self.grey_color_input = QLineEdit("0.5,0.5,0.5")
        lime_params_layout.addWidget(self.grey_color_input, 5, 1)

        lime_params_layout.addWidget(QLabel("Distance Metric:"), 6, 0)
        self.distance_metric_input = QComboBox()
        self.distance_metric_input.addItems(["cosine", "euclidean", "manhattan", "l1", "l2"])
        self.distance_metric_input.setCurrentText("cosine")
        lime_params_layout.addWidget(self.distance_metric_input, 6, 1)


        lime_params_layout.addWidget(QLabel("Kernel Width:"), 7, 0)
        self.kernel_width_input = QLineEdit("0.25")
        lime_params_layout.addWidget(self.kernel_width_input, 7, 1)

        lime_params_layout.addWidget(QLabel("Interpretable Model Type:"), 8, 0)
        self.interpretable_model_type_input = QComboBox()
        self.interpretable_model_type_input.addItems(["linear", "ridge", "lasso", "tree", "random_forest"])
        self.interpretable_model_type_input.setCurrentText("linear")
        lime_params_layout.addWidget(self.interpretable_model_type_input, 8, 1)


        # Add heatmap visualization toggle
        lime_params_layout.addWidget(QLabel("Use Heatmap Visualization:"), 9, 0)
        self.heatmap_visualization_checkbox = QCheckBox()
        lime_params_layout.addWidget(self.heatmap_visualization_checkbox, 9, 1)

        # Add heatmap opacity control
        lime_params_layout.addWidget(QLabel("Heatmap Opacity (0-1):"), 10, 0)
        self.heatmap_opacity_input = QLineEdit("0.7")
        lime_params_layout.addWidget(self.heatmap_opacity_input, 10, 1)

        lime_params_group.setLayout(lime_params_layout)
        control_layout.addWidget(lime_params_group)

        # Action Buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout()

        self.load_button = QPushButton("Load Dataset")
        self.load_video_button = QPushButton("Load Video")
        self.process_button = QPushButton("Generate LIME Explanation")
        self.clear_button = QPushButton("Clear Views")

        self.load_button.clicked.connect(self.load_dataset)
        self.load_video_button.clicked.connect(self.load_video)
        self.process_button.clicked.connect(self.generate_explanation)
        self.clear_button.clicked.connect(self.clear_views)

        actions_layout.addWidget(self.load_button)
        actions_layout.addWidget(self.load_video_button)
        actions_layout.addWidget(self.process_button)
        actions_layout.addWidget(self.clear_button)
        actions_group.setLayout(actions_layout)
        control_layout.addWidget(actions_group)

        # Navigation Controls
        nav_group = QGroupBox("Navigation")
        nav_layout = QVBoxLayout()

        # Image Navigation
        image_nav_layout = QHBoxLayout()
        self.prev_image_button = QPushButton("← Previous")
        self.next_image_button = QPushButton("Next →")
        self.prev_image_button.clicked.connect(self.previous_image)
        self.next_image_button.clicked.connect(self.next_image)
        image_nav_layout.addWidget(self.prev_image_button)
        image_nav_layout.addWidget(self.next_image_button)
        nav_layout.addLayout(image_nav_layout)

        # Explanation Navigation
        explanation_nav_layout = QHBoxLayout()
        self.prev_explanation_button = QPushButton("← Previous explanation")
        self.next_explanation_button = QPushButton("Next explanation →")
        self.prev_explanation_button.clicked.connect(self.previous_explanation)
        self.next_explanation_button.clicked.connect(self.next_explanation)
        explanation_nav_layout.addWidget(self.prev_explanation_button)
        explanation_nav_layout.addWidget(self.next_explanation_button)
        nav_layout.addLayout(explanation_nav_layout)

        nav_group.setLayout(nav_layout)
        control_layout.addWidget(nav_group)

        # Image Index Label
        self.image_index_label = QLabel("Image: 0/0")
        self.image_index_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(self.image_index_label)

        main_layout.addWidget(control_frame, 1)

        # Center Panel: Image Views with tabs
        self.image_views = ImageViews(
            self,
            self.model_selector,
            self.class_checkboxes,
        )
        
        # Explicitly set model_handler for object detection view
        self.image_views.object_detection_view.set_model_handler(self.model_handler)
        
        # Create a tabbed widget that includes the ImageViews and the NovelLIMEView
        self.tabbed_view = QTabWidget()
        self.tabbed_view.addTab(self.image_views, "Standard LIME")
        
        # Create Novel LIME View tab
        self.novel_lime_view = DPPLimeView(self)
        self.novel_lime_view.set_model_handler(self.model_handler)
        self.tabbed_view.addTab(self.novel_lime_view, "Novel LIME")
        
        main_layout.addWidget(self.tabbed_view, 3)
        
        # Initialize video detection view with model handler
        self.setup_video_detection()
        
        # Connect model selector to update classes
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        
    def update_model_selector(self):
        """Update the model selector with available models."""
        self.model_selector.clear()
        model_names = self.model_handler.get_model_display_names()
        if model_names:
            self.model_selector.addItems(model_names)
        else:
            # Create default models if none exist
            self.model_handler.config_manager.create_default_configs()
            model_names = self.model_handler.get_model_display_names()
            self.model_selector.addItems(model_names)

    def add_model(self):
        """Open dialog to add a new model."""
        dialog = ModelConfigDialog(self.model_handler, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.update_model_selector()

    def edit_model(self):
        """Open dialog to edit selected model."""
        model_name = self.model_selector.currentText()
        if not model_name:
            QMessageBox.warning(self, "Warning", "No model selected!")
            return
        
        model_id = self.model_handler.get_model_id_by_name(model_name)
        if not model_id:
            QMessageBox.warning(self, "Warning", "Model not found!")
            return
        
        model_config = self.model_handler.config_manager.get_model_by_id(model_id)
        dialog = ModelConfigDialog(self.model_handler, edit_model=model_config, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.update_model_selector()

    def remove_model(self):
        """Remove selected model from configuration."""
        model_name = self.model_selector.currentText()
        if not model_name:
            QMessageBox.warning(self, "Warning", "No model selected!")
            return
        
        model_id = self.model_handler.get_model_id_by_name(model_name)
        if not model_id:
            QMessageBox.warning(self, "Warning", "Model not found!")
            return
        
        response = QMessageBox.question(
            self, 
            "Confirm Removal", 
            f"Are you sure you want to remove model '{model_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            if self.model_handler.config_manager.remove_model(model_id):
                QMessageBox.information(self, "Success", f"Model '{model_name}' has been removed.")
                self.update_model_selector()
            else:
                QMessageBox.warning(self, "Warning", f"Failed to remove model '{model_name}'.")

    def add_class_set(self):
        """Open dialog to add a new class set."""
        dialog = ClassSetConfigDialog(self.model_handler, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.update_model_selector()
            
            # Refresh class selection if the current model uses this class set
            current_model_name = self.model_selector.currentText()
            if current_model_name:
                self.on_model_changed(current_model_name)

    def edit_class_set(self):
        """Open dialog to edit a class set."""
        class_sets = list(self.model_handler.config_manager.classes_config.keys())
        if not class_sets:
            QMessageBox.warning(self, "Warning", "No class sets available!")
            return
        
        selected_class_set, ok = QInputDialog.getItem(
            self, "Select Class Set", "Choose a class set to edit:", class_sets, 0, False
        )
        
        if ok and selected_class_set:
            dialog = ClassSetConfigDialog(self.model_handler, edit_class_set_id=selected_class_set, parent=self)
            if dialog.exec_() == QDialog.Accepted:
                # Refresh class selection if the current model uses this class set
                current_model_name = self.model_selector.currentText()
                if current_model_name:
                    model_id = self.model_handler.get_model_id_by_name(current_model_name)
                    if model_id:
                        model_config = self.model_handler.config_manager.get_model_by_id(model_id)
                        if model_config and model_config.get("output_classes") == selected_class_set:
                            self.update_class_selection_widget()

    def remove_class_set(self):
        """Remove a class set from configuration."""
        class_sets = list(self.model_handler.config_manager.classes_config.keys())
        if not class_sets:
            QMessageBox.warning(self, "Warning", "No class sets available!")
            return
        
        selected_class_set, ok = QInputDialog.getItem(
            self, "Select Class Set", "Choose a class set to remove:", class_sets, 0, False
        )
        
        if ok and selected_class_set:
            # Check if any models are using this class set
            models_using = []
            for model in self.model_handler.config_manager.get_models():
                if model.get("output_classes") == selected_class_set:
                    models_using.append(model.get("name"))
            
            if models_using:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    f"Cannot remove class set '{selected_class_set}' because it is used by models: {', '.join(models_using)}"
                )
                return
            
            response = QMessageBox.question(
                self, 
                "Confirm Removal", 
                f"Are you sure you want to remove class set '{selected_class_set}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if response == QMessageBox.Yes:
                if self.model_handler.config_manager.remove_class_set(selected_class_set):
                    QMessageBox.information(self, "Success", f"Class set '{selected_class_set}' has been removed.")
                else:
                    QMessageBox.warning(self, "Warning", f"Failed to remove class set '{selected_class_set}'.")

    def update_class_selection_widget(self):
        
        print("Updating class selection widget...")
        
        # Get current model info
        model_name = self.model_selector.currentText()
        if not model_name:
            print("No model selected, clearing class selection widget")
            # Clear existing layout and add placeholder
            self._clear_class_grid()
            self.class_placeholder.setText("Select a model to view available classes")
            self.class_grid.addWidget(self.class_placeholder, 0, 0, 1, 3)
            return
            
        # Get class names for the current model
        class_names = self.model_handler.get_class_names()
        print(f"Found {len(class_names)} classes for model {model_name}")
        
        # Clear existing checkboxes dictionary
        self.class_checkboxes.clear()
        
        # Clear the grid layout
        self._clear_class_grid()
        
        # If no classes found, show a message
        if not class_names:
            self.class_placeholder.setText("No classes found for this model")
            self.class_grid.addWidget(self.class_placeholder, 0, 0, 1, 3)
            return
            
        # Add class checkboxes to the grid layout
        row, col = 0, 0
        max_cols = 3
        
        # Sort class indices numerically for consistent display
        sorted_items = []
        for idx_str, name in class_names.items():
            try:
                idx = int(idx_str)
                sorted_items.append((idx, idx_str, name))
            except ValueError:
                # Handle non-integer class indices
                print(f"Warning: Non-integer class index: {idx_str}")
                # Still add the item, but use the string as index
                sorted_items.append((idx_str, idx_str, name))
        
        # Sort primarily by numeric indices, keep string indices at the end
        def sort_key(x):
            return (0, x[0]) if isinstance(x[0], int) else (1, str(x[0]))
        
        sorted_items.sort(key=sort_key)
        
        for idx, idx_str, name in sorted_items:
            checkbox = QCheckBox(name)
            checkbox.setChecked(True)  # Default to checked
            checkbox.setStyleSheet("QCheckBox { margin: 2px; padding: 2px; }")
            
            # Connect the stateChanged signal to update selected classes
            checkbox.stateChanged.connect(lambda state, idx=idx: self.on_class_checkbox_changed(state, idx))
            
            self.class_grid.addWidget(checkbox, row, col)
            self.class_checkboxes[idx] = checkbox
            
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # Force layout update
        self.class_groupbox.updateGeometry()
        self.class_container.updateGeometry()
        
        # Update related views with selected classes
        if hasattr(self, 'novel_lime_view'):
            selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]
            self.novel_lime_view.set_selected_classes(selected_classes)
            
        if hasattr(self.image_views, 'layer_lime_view'):
            selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]
            self.image_views.layer_lime_view.set_selected_classes(selected_classes)
    
    def on_class_checkbox_changed(self, state, class_idx):
        """Handle individual class checkbox state changes."""
        # Update Novel LIME view if it exists
        if hasattr(self, 'novel_lime_view'):
            selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]
            self.novel_lime_view.set_selected_classes(selected_classes)
            
        # Update Layer LIME view if it exists
        if hasattr(self.image_views, 'layer_lime_view'):
            selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]
            self.image_views.layer_lime_view.set_selected_classes(selected_classes)
        
        
        # Log for debugging
        print(f"Class {class_idx} checkbox changed to state {state}")
        print(f"Selected classes: {[idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]}")
        
    def _clear_class_grid(self):
        """Clear the class grid layout."""
        if self.class_grid is None:
            return
            
        while self.class_grid.count():
            item = self.class_grid.takeAt(0)
            if item is None:
                continue
                
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
                
    def _clear_layout(self, layout):
        """Recursively clear a layout."""
        if layout is None:
            return
            
        while layout.count():
            item = layout.takeAt(0)
            if item is None:
                continue
                
            if item.widget() is not None:
                item.widget().setParent(None)
                item.widget().deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())
                layout.removeItem(item)
                
    def setup_video_detection(self):
        """Set up the video detection view with the model handler."""
        # Ensure video_view exists
        if hasattr(self.image_views, 'video_view'):
            # Set model handler directly 
            self.image_views.video_view.set_model_handler(self.model_handler)
            print("Setup video detection: model_handler set successfully")


    
    def show_novel_lime_tab(self):
        """Switch to Novel LIME tab and update with current image."""
        # Switch to the Novel LIME tab
        self.tabbed_view.setCurrentWidget(self.novel_lime_view)
        
        # Update Novel LIME view with current image and selected classes
        if self.dataset and len(self.dataset) > 0:
            self.novel_lime_view.set_image_path(self.dataset[self.current_image_index])
            selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]
            self.novel_lime_view.set_selected_classes(selected_classes)
            
            # Show a tip to the user
            QMessageBox.information(
                self, 
                "Novel LIME", 
                "Novel LIME tab is now active. Configure parameters and click 'Generate Novel LIME' to create explanations."
            )
        else:
            QMessageBox.warning(
                self, 
                "Warning", 
                "Please load a dataset first before using Novel LIME."
            )

    def setup_video_detection(self):
        """Set up the video detection view with the model handler."""
        # The video view is already created in ImageViews
        self.image_views.video_view.set_model_handler(self.model_handler)

    def load_video(self):
        """Load a video file for video detection."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if file_path:
            # Switch to video detection tab
            for i in range(self.image_views.count()):
                if self.image_views.tabText(i) == "Video Detection":
                    self.image_views.setCurrentIndex(i)
                    break
                    
            # Set the video path and load it
            self.image_views.video_view.video_path = file_path
            
            # If load_video method exists in VideoDetectionView, call it
            if hasattr(self.image_views.video_view, 'load_video'):
                self.image_views.video_view.load_video()
            else:
                # Otherwise just call the existing load_video method
                self.image_views.video_view.load_button.click()

    def _handle_video_complete(self, frame_paths, progress_dialog):
        """Handle completion of video processing."""
        progress_dialog.close()
        
        # Add extracted frames to the dataset
        if frame_paths:
            self.dataset.extend(frame_paths)
            self.current_image_index = 0
            self.image_index_label.setText(f"Image: {self.current_image_index + 1}/{len(self.dataset)}")
            self.show_image(self.dataset[self.current_image_index])
            
            # Update the dataset in LimeExplanationGenerator
            self.explanation_generator.dataset = self.dataset
            self.explanation_generator.current_image_index = self.current_image_index
            
            QMessageBox.information(self, "Video Processing", f"Successfully extracted {len(frame_paths)} frames")
        else:
            QMessageBox.warning(self, "Warning", "No frames could be extracted from the video.")

    def _handle_video_error(self, error_message, progress_dialog):
        """Handle errors in video processing."""
        progress_dialog.close()
        QMessageBox.critical(self, "Error", f"Error processing video: {error_message}")

    def on_model_changed(self, model_name):
        """Handle model selection changes."""
        try:
            print(f"Model changed to: {model_name}")
            
            if not model_name:
                print("No model selected")
                return
                
            model_id = self.model_handler.get_model_id_by_name(model_name)
            if model_id:
                # Load the model
                print(f"Loading model ID: {model_id}")
                
                # Process events to ensure UI is responsive
                QApplication.processEvents()
                
                # Load the model
                success = self.model_handler.load_model(model_id)
                
                if not success:
                    QMessageBox.warning(self, "Warning", f"Failed to load model: {model_name}")
                    return
                
                # Process events to ensure UI is responsive
                QApplication.processEvents()
                
                # Update class selection widget
                self.update_class_selection_widget()
                
                # Ensure UI updates are processed
                QApplication.processEvents()
                
                # Update object detection view with the model handler
                if hasattr(self.image_views, 'object_detection_view'):
                    self.image_views.object_detection_view.set_model_handler(self.model_handler)
                    
                # Update Novel LIME view
                if hasattr(self, 'novel_lime_view'):
                    self.novel_lime_view.set_model_handler(self.model_handler)
                    
                    # Make sure class_checkboxes exists and has items before trying to use it
                    if hasattr(self, 'class_checkboxes') and self.class_checkboxes:
                        selected_classes = [
                            idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()
                        ]
                        self.novel_lime_view.set_selected_classes(selected_classes)
                
                # Update video view
                if hasattr(self.image_views, 'video_view'):
                    self.image_views.video_view.set_model_handler(self.model_handler)
                    
                # Update current image if dataset is loaded
                if self.dataset and len(self.dataset) > 0:
                    self.show_image(self.dataset[self.current_image_index])
            else:
                QMessageBox.warning(self, "Warning", f"Model '{model_name}' not found in configuration")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")

    def generate_explanation(self):
        """Generate LIME explanation for the current image."""
        selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]
        if not selected_classes:
            QMessageBox.warning(self, "Warning", "Please select at least one class!")
            return

        # Get LIME parameters from the input fields
        try:
            num_samples = int(self.num_samples_input.text())
            batch_size = int(self.batch_size_input.text())
            positive_only = self.positive_only_checkbox.isChecked()
            num_features = int(self.num_features_input.text())
            hide_rest = self.hide_rest_checkbox.isChecked()
            grey_color = np.array([float(c) for c in self.grey_color_input.text().split(',')])
            distance_metric = self.distance_metric_input.currentText()
            kernel_width = float(self.kernel_width_input.text())
            interpretable_model_type = self.interpretable_model_type_input.currentText()
            use_heatmap = self.heatmap_visualization_checkbox.isChecked()
            heatmap_opacity = float(self.heatmap_opacity_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Invalid LIME parameters!")
            return

        current_image_path = self.dataset[self.current_image_index]
        self.progress_dialog = ProgressDialog("Generating LIME Explanations...", len(selected_classes), self)

        # Create and start worker thread - pass LIME parameters
        self.explanation_worker = ExplanationWorker(
            self.explanation_generator,
            current_image_path,
            selected_classes,
            lime_params={
                "num_samples": num_samples,
                "batch_size": batch_size,
                "positive_only": positive_only,
                "num_features": num_features,
                "hide_rest": hide_rest,
                "grey_color": grey_color,
                "distance_metric": distance_metric,
                "kernel_width": kernel_width,
                "interpretable_model_type": interpretable_model_type,
                "use_heatmap": use_heatmap,
                "heatmap_opacity": heatmap_opacity
            }
        )

        self.explanation_worker.progress.connect(self.progress_dialog.update_progress)
        self.explanation_worker.finished.connect(self._handle_explanation_complete)
        self.explanation_worker.error.connect(self._handle_explanation_error)
        self.explanation_worker.finished.connect(self.progress_dialog.close)
        self.explanation_worker.error.connect(self.progress_dialog.close)
        self.explanation_worker.start()

    def _handle_explanation_complete(self, explanation_images, explanation_labels):
        """Handle completion of LIME explanation generation."""
        self.explanation_images = explanation_images
        self.explanation_labels = explanation_labels
        self.current_explanation_index = 0
        self.update_lime_view()

    def _handle_explanation_error(self, error_message):
        """Handle errors in LIME explanation generation."""
        QMessageBox.critical(self, "Error", error_message)

    def process_video(self, video_path):
        """Process video file and extract frames."""
        # Ask user if they want to extract frames or use video detection
        response = QMessageBox.question(
            self, 
            "Video Processing Options",
            "Do you want to extract frames or view the video with detection?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if response == QMessageBox.Yes:
            # Extract frames (original behavior)
            output_dir = os.path.join(os.path.dirname(video_path), "frames")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create and start video processing worker
            self.video_worker = VideoProcessingWorker(video_path, output_dir)
            progress_dialog = ProgressDialog("Processing Video...", 100, self)
            self.video_worker.progress.connect(progress_dialog.update_progress)
            self.video_worker.finished.connect(lambda frames: self._handle_video_complete(frames, progress_dialog))
            self.video_worker.error.connect(lambda e: self._handle_video_error(e, progress_dialog))
            self.video_worker.start()
        else:
            # Use video detection view instead
            self.load_video()

    def load_dataset(self):
        """Load a dataset of images or videos."""
        options = QFileDialog.Options()
        folder_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder", options=options)
        if folder_path:
            self.dataset = []
            for f in os.listdir(folder_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.dataset.append(os.path.join(folder_path, f))
                elif f.lower().endswith(('.mp4', '.avi', '.mov')):
                    video_path = os.path.join(folder_path, f)
                    self.process_video(video_path)

            if self.dataset:
                self.current_image_index = 0
                self.image_index_label.setText(f"Image: {self.current_image_index + 1}/{len(self.dataset)}")
                self.show_image(self.dataset[self.current_image_index])

                # Update the dataset in LimeExplanationGenerator
                self.explanation_generator.dataset = self.dataset
                self.explanation_generator.current_image_index = self.current_image_index
            else:
                QMessageBox.warning(self, "Warning", "No valid images or videos found in the selected folder.")

    def show_image(self, image_path):
        """Display the image in both original and object detection views."""
        pixmap = QPixmap(image_path)
        
        # Scale images to fit the view while maintaining aspect ratio
        target_size = self.image_views.original_image.size()
        scaled_pixmap = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.image_views.original_image.setPixmap(scaled_pixmap)
        self.image_views.object_detection_view.image_path = image_path
        self.image_views.object_detection_view.run_detection()
        
        # Update the Layer-wise LIME view with the current image path
        self.image_views.layer_lime_view.set_image_path(image_path)
        
        # Update selected classes in Layer-wise LIME view
        selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]
        self.image_views.layer_lime_view.set_selected_classes(selected_classes)
        
        # Also update the Novel LIME view if it exists
        if hasattr(self, 'novel_lime_view'):
            self.novel_lime_view.set_image_path(image_path)
            
            # Update selected classes in Novel LIME view
            self.novel_lime_view.set_selected_classes(selected_classes)

    def update_lime_view(self):
        """Update the LIME visualization canvas."""
        if not self.explanation_images:
            return

        self.image_views.lime_canvas.figure.clear()
        ax = self.image_views.lime_canvas.figure.add_subplot(111)
        
        explanation_image = self.explanation_images[self.current_explanation_index]
        class_label = self.explanation_labels[self.current_explanation_index]
        
        ax.imshow(explanation_image)
        ax.set_title(class_label)
        ax.axis('off')
        self.image_views.lime_canvas.draw()

    def clear_views(self):
        """Clear all views and reset the application state."""
        try:
            if hasattr(self.image_views, 'lime_canvas') and self.image_views.lime_canvas is not None:
                self.image_views.lime_canvas.figure.clear()
                self.image_views.lime_canvas.draw()
            
            # Clear Novel LIME view if it exists
            if hasattr(self, 'novel_lime_view'):
                if hasattr(self.novel_lime_view, 'figure') and self.novel_lime_view.figure is not None:
                    self.novel_lime_view.figure.clear()
                    self.novel_lime_view.canvas.draw()
                if hasattr(self.novel_lime_view, 'importance_canvas') and self.novel_lime_view.importance_canvas is not None:
                    self.novel_lime_view.importance_canvas.figure.clear()
                    self.novel_lime_view.importance_canvas.draw()
                self.novel_lime_view.explanation_images = []
                self.novel_lime_view.explanation_labels = []
                if hasattr(self.novel_lime_view, 'importance_values'):
                    self.novel_lime_view.importance_values = []
                self.novel_lime_view.current_explanation_index = 0
                
            # Clear Layer-wise LIME view
            if hasattr(self.image_views, 'layer_lime_view'):
                self.image_views.layer_lime_view.layer_explanations = []
                self.image_views.layer_lime_view.layer_names = []
                # Safely clear visualization - check if current_canvas exists AND is valid
                if (hasattr(self.image_views.layer_lime_view, 'current_canvas') and 
                    self.image_views.layer_lime_view.current_canvas is not None and
                    hasattr(self.image_views.layer_lime_view.current_canvas, 'figure')):
                    try:
                        self.image_views.layer_lime_view.current_canvas.figure.clear()
                        self.image_views.layer_lime_view.current_canvas.draw()
                    except RuntimeError:
                        print("Warning: Could not clear layer_lime_view canvas - it may have been deleted")
                # Update the display label if it exists
                if hasattr(self.image_views.layer_lime_view, 'current_explanation_label'):
                    self.image_views.layer_lime_view.current_explanation_label.setText("No explanation generated")
                # Clear any explanation layout if the method exists
                if hasattr(self.image_views.layer_lime_view, 'clear_explanation_layout'):
                    self.image_views.layer_lime_view.clear_explanation_layout()
                    
            self.explanation_images = []
            self.explanation_labels = []
            self.current_explanation_index = 0
            QMessageBox.information(self, "Views Cleared", "All views have been cleared.")
        except Exception as e:
            # Handle the exception more gracefully
            print(f"Error while clearing views: {str(e)}")
            QMessageBox.warning(self, "Warning", f"Some views could not be cleared: {str(e)}")
    # Navigation methods
    def previous_image(self):
        """Navigate to the previous image in the dataset."""
        if self.dataset:
            self.current_image_index = (self.current_image_index - 1) % len(self.dataset)
            self.image_index_label.setText(f"Image: {self.current_image_index + 1}/{len(self.dataset)}")
            self.show_image(self.dataset[self.current_image_index])
            
            # Clear previous explanations
            self.explanation_images = []
            self.explanation_labels = []
            self.current_explanation_index = 0
            self.update_lime_view()
            
            # Also clear Layer-wise LIME view
            if hasattr(self.image_views, 'layer_lime_view'):
                self.image_views.layer_lime_view.layer_explanations = []
                self.image_views.layer_lime_view.layer_names = []
            
            # Update Novel LIME view
            if hasattr(self, 'novel_lime_view'):
                self.novel_lime_view.explanation_images = []
                self.novel_lime_view.explanation_labels = []
                if hasattr(self.novel_lime_view, 'importance_values'):
                    self.novel_lime_view.importance_values = []
                self.novel_lime_view.current_explanation_index = 0
                self.novel_lime_view.update_display()

    def next_image(self):
        """Navigate to the next image in the dataset."""
        if self.dataset:
            self.current_image_index = (self.current_image_index + 1) % len(self.dataset)
            self.image_index_label.setText(f"Image: {self.current_image_index + 1}/{len(self.dataset)}")
            self.show_image(self.dataset[self.current_image_index])
            
            # Clear previous explanations
            self.explanation_images = []
            self.explanation_labels = []
            self.current_explanation_index = 0
            self.update_lime_view()
            
            # Also clear Layer-wise LIME view
            if hasattr(self.image_views, 'layer_lime_view'):
                self.image_views.layer_lime_view.layer_explanations = []
                self.image_views.layer_lime_view.layer_names = []
            
            # Update Novel LIME view
            if hasattr(self, 'novel_lime_view'):
                self.novel_lime_view.explanation_images = []
                self.novel_lime_view.explanation_labels = []
                if hasattr(self.novel_lime_view, 'importance_values'):
                    self.novel_lime_view.importance_values = []
                self.novel_lime_view.current_explanation_index = 0
                self.novel_lime_view.update_display()

    def previous_explanation(self):
        """Navigate to the previous LIME explanation."""
        if self.explanation_images:
            self.current_explanation_index = (self.current_explanation_index - 1) % len(self.explanation_images)
            self.update_lime_view()

    def next_explanation(self):
        """Navigate to the next LIME explanation."""
        if self.explanation_images:
            self.current_explanation_index = (self.current_explanation_index + 1) % len(self.explanation_images)
            self.update_lime_view()

    def select_all_classes(self):
        """Select all class checkboxes."""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(True)
            
        # Update selected classes in Novel LIME view
        if hasattr(self, 'novel_lime_view'):
            selected_classes = list(self.class_checkboxes.keys())
            self.novel_lime_view.set_selected_classes(selected_classes)

    def deselect_all_classes(self):
        """Deselect all class checkboxes."""
        for checkbox in self.class_checkboxes.values():
            checkbox.setChecked(False)
            
        # Update selected classes in Novel LIME view
        if hasattr(self, 'novel_lime_view'):
            self.novel_lime_view.set_selected_classes([])
            
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key_Left:
            self.previous_image()
        elif event.key() == Qt.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key_Up:
            # Handle differently based on current tab
            if self.tabbed_view.currentWidget() == self.novel_lime_view:
                self.novel_lime_view.previous_class()
            else:
                self.previous_explanation()
        elif event.key() == Qt.Key_Down:
            # Handle differently based on current tab
            if self.tabbed_view.currentWidget() == self.novel_lime_view:
                self.novel_lime_view.next_class()
            else:
                self.next_explanation()
        elif event.key() == Qt.Key_Escape:
            # Cancel ongoing operations
            if self.explanation_worker and self.explanation_worker.isRunning():
                self.explanation_worker.terminate()
                self.explanation_worker.wait()
                QMessageBox.information(self, "Cancelled", "LIME explanation generation cancelled.")
            if self.video_worker and self.video_worker.isRunning():
                self.video_worker.terminate()
                self.video_worker.wait()
                QMessageBox.information(self, "Cancelled", "Video processing cancelled.")
            if hasattr(self, 'novel_lime_view') and hasattr(self.novel_lime_view, 'worker') and self.novel_lime_view.worker and self.novel_lime_view.worker.isRunning():
                self.novel_lime_view.worker.terminate()
                self.novel_lime_view.worker.wait()
                QMessageBox.information(self, "Cancelled", "Novel LIME explanation generation cancelled.")
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        # Update image scaling
        if hasattr(self, 'image_views') and self.dataset:
            self.show_image(self.dataset[self.current_image_index])

    def update_progress_dialog(self, current, total):
        """Update the progress dialog."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.update_progress(current, total)

    def cleanup_temporary_files(self):
        """Clean up temporary files and directories."""
        if self.dataset:
            for path in self.dataset:
                if 'frames' in path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f"Error removing temporary file {path}: {e}")
            # Try to remove empty frames directory
            if len(self.dataset) > 0:
                frames_dir = os.path.dirname(self.dataset[0])
                if 'frames' in frames_dir and os.path.exists(frames_dir):
                    try:
                        os.rmdir(frames_dir)
                    except Exception as e:
                        print(f"Error removing frames directory {frames_dir}: {e}")
                        
    def closeEvent(self, event):
        """Handle application closure."""
        try:
            # Stop any running workers
            if self.explanation_worker and self.explanation_worker.isRunning():
                self.explanation_worker.terminate()
                self.explanation_worker.wait()

            if self.video_worker and self.video_worker.isRunning():
                self.video_worker.terminate()
                self.video_worker.wait()
                
            # Stop Novel LIME worker if running
            if hasattr(self, 'novel_lime_view') and hasattr(self.novel_lime_view, 'worker') and self.novel_lime_view.worker and self.novel_lime_view.worker.isRunning():
                self.novel_lime_view.worker.terminate()
                self.novel_lime_view.worker.wait()

            # Clean up resources
            if hasattr(self, 'explanation_generator'):
                self.explanation_generator.cleanup()

            # Clean up temporary files
            self.cleanup_temporary_files()

        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        event.accept()


