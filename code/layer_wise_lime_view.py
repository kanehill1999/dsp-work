import numpy as np
import torch
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QGridLayout, QScrollArea, QFrame, QLineEdit,
    QGroupBox, QCheckBox, QMessageBox, QProgressBar, QSizePolicy,
    QSlider, QSpinBox, QDoubleSpinBox, QToolButton
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries, quickshift
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io
from PIL import Image


class CollapsibleBox(QWidget):
    """
    A custom widget that can be collapsed/expanded to show/hide content.
    """
    def __init__(self, title="", parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton(text=title, checkable=True, checked=True)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.setFixedHeight(28)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.content_area = QWidget()
        
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.addWidget(self.toggle_button)
        self.main_layout.addWidget(self.content_area)

    def on_pressed(self):
        """Toggle collapse/expand when button is pressed."""
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if not checked else Qt.RightArrow)
        self.toggle_button.setChecked(not checked)
        self.content_area.setVisible(not checked)

    def setContentLayout(self, layout):
        """Set the layout of the content area."""
        self.content_area.setLayout(layout)
        self.content_area.setVisible(True)


class LayerWiseLIMEWorker(QThread):
    """Worker thread for generating layer-wise LIME explanations."""
    progress = pyqtSignal(int, int)  # Signal for progress updates (current, total)
    finished = pyqtSignal(list, list)  # Signal for completed explanations (images, layer_names)
    error = pyqtSignal(str)  # Signal for error messages

    def __init__(self, parent, image_path, selected_classes, model_handler, params=None, layers_to_explain=None):
        super().__init__()
        self.parent = parent
        self.image_path = image_path
        self.selected_classes = selected_classes
        self.model_handler = model_handler
        self.params = params or {}
        self.layers_to_explain = layers_to_explain or []

    def run(self):
        try:
            explanation_images, layer_names = self.parent.generate_layer_wise_explanations(
                self.image_path,
                self.selected_classes,
                self.model_handler,
                self.params,
                self.layers_to_explain,
                progress_callback=self.progress.emit
            )
            self.finished.emit(explanation_images, layer_names)
        except Exception as e:
            import traceback
            print(f"LayerWiseLIME Worker Error: {str(e)}")
            print(traceback.format_exc())
            self.error.emit(f"Error generating layer-wise explanations: {str(e)}")


class LayerWiseLIMEView(QWidget):
    """View for displaying layer-wise LIME explanations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.image_path = None
        self.selected_classes = []
        self.layer_explanations = []
        self.layer_names = []
        self.worker = None
        self.current_image_index = 0
        self.model_handler = None  # Direct reference to model handler
        self.explanation_canvases = []  # List to hold explanation canvases
        self.display_mode = "single"  # 'single' or 'grid'

        # Initialize UI components
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Parameters section in a collapsible box
        self.params_box = CollapsibleBox("Layer-wise LIME Parameters")
        params_layout = QGridLayout()
        
        # Layer selection section
        layer_group = QGroupBox("Layer Selection")
        layer_layout = QVBoxLayout()
        layer_group.setLayout(layer_layout)

        self.layer_combo = QComboBox()
        self.layer_combo.addItems(["Select layers..."])
        self.refresh_layers_button = QPushButton("Refresh Layers")
        self.refresh_layers_button.clicked.connect(self.refresh_layers)

        layer_control_layout = QHBoxLayout()
        layer_control_layout.addWidget(QLabel("Select Layer:"))
        layer_control_layout.addWidget(self.layer_combo)
        layer_control_layout.addWidget(self.refresh_layers_button)
        layer_layout.addLayout(layer_control_layout)

        self.selected_layers_list = QLabel("No layers selected")
        layer_layout.addWidget(self.selected_layers_list)

        add_layer_button = QPushButton("Add Layer")
        add_layer_button.clicked.connect(self.add_layer)
        clear_layers_button = QPushButton("Clear Layers")
        clear_layers_button.clicked.connect(self.clear_layers)

        layer_buttons_layout = QHBoxLayout()
        layer_buttons_layout.addWidget(add_layer_button)
        layer_buttons_layout.addWidget(clear_layers_button)
        layer_layout.addLayout(layer_buttons_layout)

        params_layout.addWidget(layer_group, 0, 0, 1, 2)

        # LIME parameters
        lime_params_group = QGroupBox("LIME Parameters")
        lime_params_layout = QGridLayout()
        lime_params_group.setLayout(lime_params_layout)

        # Sampling parameters
        lime_params_layout.addWidget(QLabel("Number of Samples:"), 0, 0)
        self.num_samples_input = QSpinBox()
        self.num_samples_input.setRange(100, 5000)
        self.num_samples_input.setValue(500)
        lime_params_layout.addWidget(self.num_samples_input, 0, 1)

        lime_params_layout.addWidget(QLabel("Number of Features:"), 1, 0)
        self.num_features_input = QSpinBox()
        self.num_features_input.setRange(1, 20)
        self.num_features_input.setValue(5)
        lime_params_layout.addWidget(self.num_features_input, 1, 1)

        lime_params_layout.addWidget(QLabel("Kernel Width:"), 2, 0)
        self.kernel_width_input = QDoubleSpinBox()
        self.kernel_width_input.setRange(0.1, 1.0)
        self.kernel_width_input.setSingleStep(0.05)
        self.kernel_width_input.setValue(0.25)
        self.kernel_width_input.setToolTip("Width of the kernel used in LIME")
        lime_params_layout.addWidget(self.kernel_width_input, 2, 1)

        lime_params_layout.addWidget(QLabel("Batch Size:"), 3, 0)
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 50)
        self.batch_size_input.setValue(10)
        lime_params_layout.addWidget(self.batch_size_input, 3, 1)

        # Segmentation parameters
        lime_params_layout.addWidget(QLabel("Segmentation Method:"), 4, 0)
        self.segmentation_method = QComboBox()
        self.segmentation_method.addItems(["quickshift", "felzenszwalb", "slic"])
        lime_params_layout.addWidget(self.segmentation_method, 4, 1)

        lime_params_layout.addWidget(QLabel("Kernel Size:"), 5, 0)
        self.kernel_size_input = QSpinBox()
        self.kernel_size_input.setRange(1, 20)
        self.kernel_size_input.setValue(4)
        lime_params_layout.addWidget(self.kernel_size_input, 5, 1)

        lime_params_layout.addWidget(QLabel("Max Distance:"), 6, 0)
        self.max_dist_input = QSpinBox()
        self.max_dist_input.setRange(1, 500)
        self.max_dist_input.setValue(200)
        lime_params_layout.addWidget(self.max_dist_input, 6, 1)

        lime_params_layout.addWidget(QLabel("Ratio:"), 7, 0)
        self.ratio_input = QDoubleSpinBox()
        self.ratio_input.setRange(0.1, 1.0)
        self.ratio_input.setSingleStep(0.1)
        self.ratio_input.setValue(0.2)
        lime_params_layout.addWidget(self.ratio_input, 7, 1)

        params_layout.addWidget(lime_params_group, 1, 0)

        # Visualization parameters
        viz_params_group = QGroupBox("Visualization Parameters")
        viz_params_layout = QGridLayout()
        viz_params_group.setLayout(viz_params_layout)

        viz_params_layout.addWidget(QLabel("Positive Only:"), 0, 0)
        self.positive_only_check = QCheckBox()
        self.positive_only_check.setChecked(True)
        viz_params_layout.addWidget(self.positive_only_check, 0, 1)

        viz_params_layout.addWidget(QLabel("Use Heatmap:"), 1, 0)
        self.heatmap_check = QCheckBox()
        self.heatmap_check.setChecked(True)
        viz_params_layout.addWidget(self.heatmap_check, 1, 1)

        viz_params_layout.addWidget(QLabel("Heatmap Opacity:"), 2, 0)
        self.opacity_input = QDoubleSpinBox()
        self.opacity_input.setRange(0.1, 1.0)
        self.opacity_input.setSingleStep(0.1)
        self.opacity_input.setValue(0.7)
        viz_params_layout.addWidget(self.opacity_input, 2, 1)

        viz_params_layout.addWidget(QLabel("Threshold for Visibility:"), 3, 0)
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 0.5)
        self.threshold_input.setSingleStep(0.01)
        self.threshold_input.setValue(0.05)
        self.threshold_input.setToolTip("Confidence threshold to skip low probability detections")
        viz_params_layout.addWidget(self.threshold_input, 3, 1)

        viz_params_layout.addWidget(QLabel("Display Mode:"), 4, 0)
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems(["Single (Cycle)", "Grid View"])
        self.display_mode_combo.setCurrentIndex(0)
        self.display_mode_combo.currentIndexChanged.connect(self.change_display_mode)
        viz_params_layout.addWidget(self.display_mode_combo, 4, 1)

        params_layout.addWidget(viz_params_group, 1, 1)

        # Generate button
        generate_button = QPushButton("Generate Layer-wise LIME")
        generate_button.clicked.connect(self.start_layer_wise_lime)
        params_layout.addWidget(generate_button, 2, 0, 1, 2)

        # Set the layout for the collapsible box
        self.params_box.setContentLayout(params_layout)
        main_layout.addWidget(self.params_box)

        # Controls for navigation (for single view mode)
        self.nav_layout = QHBoxLayout()
        
        self.prev_button = QPushButton("← Previous")
        self.prev_button.clicked.connect(self.previous_explanation)
        self.next_button = QPushButton("Next →")
        self.next_button.clicked.connect(self.next_explanation)
        
        self.toggle_params_button = QPushButton("Toggle Parameters")
        self.toggle_params_button.clicked.connect(self.toggle_parameters)
        
        self.nav_layout.addWidget(self.prev_button)
        self.nav_layout.addWidget(self.toggle_params_button)
        self.nav_layout.addWidget(self.next_button)
        
        main_layout.addLayout(self.nav_layout)

        # Visualization section - will be dynamic based on display mode
        self.explanation_frame = QFrame()
        self.explanation_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Use a layout with no margins for the explanation frame
        self.explanation_layout = QVBoxLayout()
        self.explanation_layout.setContentsMargins(0, 0, 0, 0)
        self.explanation_layout.setSpacing(0)  # Reduce spacing between items
        self.explanation_frame.setLayout(self.explanation_layout)

        # Create scroll area to contain the explanations
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.explanation_frame)
        self.scroll_area.setMinimumHeight(400)  # Set minimum height for better visibility
        main_layout.addWidget(self.scroll_area, 1)  # Add stretch factor for expansion

        # Initialize canvas for single view mode with proper size policy
        self.current_canvas = FigureCanvas(Figure(figsize=(8, 8), dpi=100))
        self.current_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.current_canvas.setMinimumHeight(300)  # Set minimum height
        self.explanation_layout.addWidget(self.current_canvas, 1)  # Add stretch factor
        
        # Current explanation label
        self.current_explanation_label = QLabel("No explanation generated")
        self.current_explanation_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.current_explanation_label)

        # Initialize selected layers list
        self.selected_layers = []
        
        # Initially disable navigation buttons
        self.prev_button.setEnabled(False)
        self.next_button.setEnabled(False)

    def toggle_parameters(self):
        """Toggle the visibility of the parameters section."""
        self.params_box.on_pressed()
        
        # Force layout update after toggle with a delay to let layout settle
        QTimer.singleShot(100, self.delayed_layout_update)
        
    def delayed_layout_update(self):
        """Update layouts after toggling parameters with a delay."""
        self.updateGeometry()
        self.scroll_area.updateGeometry()
        self.explanation_frame.updateGeometry()
        
        # If we have explanations, update the display
        if self.layer_explanations:
            self.update_display()
        
    def change_display_mode(self, index):
        """Change between single view and grid view modes."""
        modes = ["single", "grid"]
        self.display_mode = modes[index]
        
        # Clear the current layout
        self.clear_explanation_layout()
        
        # Update the display based on the new mode
        if self.layer_explanations:
            self.update_display()

    def clear_explanation_layout(self):
        """Clear all widgets from the explanation layout."""
        # Remove all widgets from the layout
        while self.explanation_layout.count():
            item = self.explanation_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        # Reset canvases list
        self.explanation_canvases = []
        
    def get_model_handler(self):
        """Get the model handler from the parent."""
        # First check direct reference
        if self.model_handler is not None:
            return self.model_handler
            
        # Then try direct parent
        if hasattr(self.parent, 'model_handler') and self.parent.model_handler is not None:
            return self.parent.model_handler
        
        # Then try parent's parent
        elif hasattr(self.parent, 'parent') and callable(getattr(self.parent, 'parent', None)):
            parent_app = self.parent.parent()
            if parent_app and hasattr(parent_app, 'model_handler') and parent_app.model_handler is not None:
                return parent_app.model_handler
        
        # Log the failure for debugging
        print("WARNING: LayerWiseLIMEView could not access model_handler")
        return None

    def refresh_layers(self):
        """Refresh the list of available layers from the current model."""
        # Get model handler
        model_handler = self.get_model_handler()
        print(f"refresh_layers: model_handler is {model_handler}")
        
        if not model_handler:
            QMessageBox.warning(self, "Warning", "Could not access model handler!")
            return
        
        if not model_handler.model:
            QMessageBox.warning(self, "Warning", "No model loaded. Please select a model first.")
            return
            
        try:
            # Get layer names from the model
            model = model_handler.model
            
            # For YOLO models, layers are in the model
            if hasattr(model, 'model') and hasattr(model.model, 'named_modules'):
                layer_names = []
                for name, module in model.model.named_modules():
                    if len(name) > 0 and '.' in name:  # Skip the root module
                        layer_names.append(name)
            else:
                # Generic approach for other models
                layer_names = []
                for name, _ in model.named_modules():
                    if len(name) > 0:  # Skip the root module
                        layer_names.append(name)
            
            # Update the combo box
            self.layer_combo.clear()
            self.layer_combo.addItems(layer_names)
            
            QMessageBox.information(self, "Success", f"Found {len(layer_names)} layers in the model.")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to get model layers: {str(e)}")
    
    def add_layer(self):
        """Add selected layer to the list of layers to explain."""
        current_layer = self.layer_combo.currentText()
        if current_layer and current_layer != "Select layers...":
            if current_layer not in self.selected_layers:
                self.selected_layers.append(current_layer)
                self.update_selected_layers_display()
    
    def clear_layers(self):
        """Clear the list of selected layers."""
        self.selected_layers = []
        self.update_selected_layers_display()
    
    def update_selected_layers_display(self):
        """Update the display of selected layers."""
        if not self.selected_layers:
            self.selected_layers_list.setText("No layers selected")
        else:
            self.selected_layers_list.setText(", ".join(self.selected_layers))
    
    def start_layer_wise_lime(self):
        """Start the layer-wise LIME explanation generation process."""
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
            
        if not self.selected_layers:
            QMessageBox.warning(self, "Warning", "Please select at least one layer to explain!")
            return
            
        if not self.selected_classes:
            QMessageBox.warning(self, "Warning", "Please select at least one class!")
            return
        
        # Get model handler
        model_handler = self.get_model_handler()
        print(f"LayerWiseLIMEView: model_handler is {model_handler}")
        
        if not model_handler:
            QMessageBox.warning(self, "Warning", "Could not access model handler!")
            return
        
        if not model_handler.model:
            QMessageBox.warning(self, "Warning", "No model loaded. Please select a model first.")
            return
        
        # Collect parameters
        params = {
            'num_samples': self.num_samples_input.value(),
            'num_features': self.num_features_input.value(),
            'kernel_width': self.kernel_width_input.value(),
            'batch_size': self.batch_size_input.value(),
            'segmentation_method': self.segmentation_method.currentText(),
            'kernel_size': self.kernel_size_input.value(),
            'max_dist': self.max_dist_input.value(),
            'ratio': self.ratio_input.value(),
            'positive_only': self.positive_only_check.isChecked(),
            'use_heatmap': self.heatmap_check.isChecked(),
            'opacity': self.opacity_input.value(),
            'threshold': self.threshold_input.value(),
            'display_mode': self.display_mode
        }
        
        # Create progress dialog
        try:
            from progress_dialog import ProgressDialog
            progress_dialog = ProgressDialog("Generating Layer-wise LIME Explanations...", 
                                           len(self.selected_layers) * len(self.selected_classes), 
                                           self.parent)
        except ImportError:
            # Fallback progress dialog if custom one is not available
            progress_dialog = QProgressBar(self)
            progress_dialog.setRange(0, len(self.selected_layers) * len(self.selected_classes))
            progress_dialog.setValue(0)
            progress_dialog.show()
        
        # Create and start worker thread
        self.worker = LayerWiseLIMEWorker(
            self,
            self.image_path,
            self.selected_classes,
            model_handler,
            params,
            self.selected_layers
        )
        
        # Connect signals
        if hasattr(progress_dialog, 'update_progress'):
            self.worker.progress.connect(progress_dialog.update_progress)
        else:
            self.worker.progress.connect(progress_dialog.setValue)
            
        self.worker.finished.connect(lambda images, names: self.handle_explanations_complete(images, names, progress_dialog))
        self.worker.error.connect(lambda msg: self.handle_explanations_error(msg, progress_dialog))
        
        # Start worker
        self.worker.start()
    
    def handle_explanations_complete(self, explanation_images, layer_names, progress_dialog):
        """Handle completion of layer-wise LIME explanations."""
        # Close progress dialog
        if hasattr(progress_dialog, 'close'):
            progress_dialog.close()
        else:
            progress_dialog.hide()
            
        self.layer_explanations = explanation_images
        self.layer_names = layer_names
        self.current_image_index = 0
        
        # Enable navigation buttons
        self.prev_button.setEnabled(True)
        self.next_button.setEnabled(True)
        
        # Auto-collapse parameters for better viewing
        if not self.params_box.content_area.isHidden():
            self.toggle_parameters()
            
        # Update the display
        self.update_display()

    def handle_explanations_error(self, error_message, progress_dialog):
        """Handle errors in layer-wise LIME explanation generation."""
        if hasattr(progress_dialog, 'close'):
            progress_dialog.close()
        else:
            progress_dialog.hide()
            
        QMessageBox.critical(self, "Error", error_message)
    
    def update_display(self):
        """Update the display based on current display mode."""
        if not self.layer_explanations:
            return
            
        # Clear previous display
        self.clear_explanation_layout()
        
        if self.display_mode == "single":
            # Create a single canvas for cycling through explanations with improved sizing
            self.current_canvas = FigureCanvas(Figure(figsize=(8, 8), dpi=100))
            self.current_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.current_canvas.setMinimumHeight(300)  # Set minimum height
            
            # Use a layout that allows proper expansion
            canvas_layout = QVBoxLayout()
            canvas_layout.setContentsMargins(0, 0, 0, 0)
            canvas_layout.setSpacing(0)
            canvas_layout.addWidget(self.current_canvas, 1)  # Add stretch factor
            self.explanation_layout.addLayout(canvas_layout, 1)  # Add stretch factor
            
            # Show the current explanation
            self.update_single_explanation()
        else:
            # Grid view mode - show all explanations
            self.create_grid_view()
            
    def update_single_explanation(self):
        """Update the single explanation view with current image."""
        if not self.layer_explanations or self.current_image_index >= len(self.layer_explanations):
            return
            
        # Get the current explanation
        explanation_image = self.layer_explanations[self.current_image_index]
        explanation_label = self.layer_names[self.current_image_index]
        
        # Update the canvas
        self.current_canvas.figure.clear()
        ax = self.current_canvas.figure.add_subplot(111)
        ax.imshow(explanation_image)
        ax.set_title(explanation_label, fontsize=10)
        ax.axis('off')
        
        # Adjust figure to fill canvas entirely and remove excess whitespace
        self.current_canvas.figure.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
        self.current_canvas.figure.tight_layout(pad=0)
        self.current_canvas.draw()
        
        # Update the explanation label
        self.current_explanation_label.setText(f"Explanation {self.current_image_index + 1}/{len(self.layer_explanations)}: {explanation_label}")
    
    def create_grid_view(self):
        """Create a grid view of all explanations."""
        if not self.layer_explanations:
            return
            
        import math
        
        # Calculate grid dimensions
        n_explanations = len(self.layer_explanations)
        grid_size = math.ceil(math.sqrt(n_explanations))
        rows = grid_size
        cols = math.ceil(n_explanations / rows)
        
        # Create grid layout with no margins
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(5)  # Small spacing between grid items
        self.explanation_layout.addLayout(grid_layout, 1)  # Add stretch factor
        
        # Add each explanation to the grid
        for i, (image, label) in enumerate(zip(self.layer_explanations, self.layer_names)):
            row = i // cols
            col = i % cols
            
            # Create a figure and canvas for this explanation with improved sizing
            fig = Figure(figsize=(5, 5), dpi=100)
            canvas = FigureCanvas(fig)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            canvas.setMinimumHeight(200)  # Set minimum height for grid items
            
            # Plot the explanation with improved layout
            ax = fig.add_subplot(111)
            ax.imshow(image)
            ax.set_title(label, fontsize=8)
            ax.axis('off')
            
            # Adjust figure to fill canvas entirely
            fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
            fig.tight_layout(pad=0)
            
            # Create a frame to hold the canvas with proper expansion
            frame = QFrame()
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.addWidget(canvas, 1)  # Add stretch factor
            
            # Add to grid
            grid_layout.addWidget(frame, row, col, 1, 1)
            self.explanation_canvases.append(canvas)
        
        # Update label
        self.current_explanation_label.setText(f"Showing all {n_explanations} explanations in grid view")
    
    def previous_explanation(self):
        """Show the previous explanation (in single view mode)."""
        if not self.layer_explanations or self.display_mode != "single":
            return
            
        self.current_image_index = (self.current_image_index - 1) % len(self.layer_explanations)
        self.update_single_explanation()
    
    def next_explanation(self):
        """Show the next explanation (in single view mode)."""
        if not self.layer_explanations or self.display_mode != "single":
            return
            
        self.current_image_index = (self.current_image_index + 1) % len(self.layer_explanations)
        self.update_single_explanation()
    
    def set_image_path(self, image_path):
        """Set the current image path."""
        self.image_path = image_path
    
    def set_selected_classes(self, selected_classes):
        """Set the selected classes for explanation."""
        self.selected_classes = [int(cls) if isinstance(cls, str) else cls for cls in selected_classes]
    
    def generate_layer_wise_explanations(self, image_path, selected_classes, model_handler, params, layers_to_explain, progress_callback=None):
        """Generate layer-wise LIME explanations for the selected layers and all selected classes."""
        if not image_path:
            raise ValueError("No image path provided!")
            
        if not selected_classes:
            raise ValueError("No classes selected!")
            
        if not layers_to_explain:
            raise ValueError("No layers selected for explanation!")
        
        try:
            from yolo_predict_fn import preprocess_image_for_yolo
            import torch
            import matplotlib.pyplot
            
            # Get class names from model_handler
            class_names = model_handler.get_class_names()
            if not class_names:
                raise ValueError("No class names available from model handler!")
            
            # Get parameters
            num_samples = params.get('num_samples', 500)
            num_features = params.get('num_features', 5)
            kernel_width = params.get('kernel_width', 0.25)
            batch_size = params.get('batch_size', 10)
            positive_only = params.get('positive_only', True)
            use_heatmap = params.get('use_heatmap', True)
            opacity = params.get('opacity', 0.7)
            threshold = params.get('threshold', 0.05)
            
            # Segmentation parameters
            segmentation_method = params.get('segmentation_method', 'quickshift')
            kernel_size = params.get('kernel_size', 4)
            max_dist = params.get('max_dist', 200)
            ratio = params.get('ratio', 0.2)
            
            print(f"LIME parameters: samples={num_samples}, features={num_features}, kernel_width={kernel_width}")
            print(f"Selected classes: {selected_classes}")
            print(f"Class names mapping: {class_names}")
            
            
            # Process the image
            img = preprocess_image_for_yolo(image_path)
            print(f"Image shape after preprocessing: {img.shape}")

            # Add normalization if needed (if img is a numpy array with values 0-255)
            if isinstance(img, np.ndarray) and img.max() > 1.0:
                img = img / 255.0
                print("Normalized image to 0.0-1.0 range")
            
            # Create a custom segmentation function based on parameters
            def segmentation_fn(img):
                if segmentation_method == 'quickshift':
                    return quickshift(img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
                elif segmentation_method == 'felzenszwalb':
                    from skimage.segmentation import felzenszwalb
                    return felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
                elif segmentation_method == 'slic':
                    from skimage.segmentation import slic
                    return slic(img, n_segments=100, compactness=10, sigma=1)
                else:
                    return quickshift(img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
            
            # Get model and initialize hooks for layer activation
            model = model_handler.model
            
            # Store activations from specified layers
            layer_activations = {}
            hooks = []
            
            # For each selected layer, add a forward hook to capture activations
            registered_layers = []
            for layer_name in layers_to_explain:
                # Function to create a hook function for a specific layer
                def get_hook_fn(layer_name):
                    def hook_fn(module, input, output):
                        # Store the output activations for this layer
                        layer_activations[layer_name] = output.detach()
                        print(f"Captured activation for layer {layer_name}, shape: {output.shape}")
                    return hook_fn
                
                # Get the layer by name and register hook
                try:
                    # Handle nested modules with dot notation
                    module = model
                    if hasattr(model, 'model'):  # For YOLO models
                        module = model.model
                        
                    parts = layer_name.split('.')
                    for part in parts:
                        if part.isdigit():  # If it's an index in a Sequential
                            module = module[int(part)]
                        else:
                            module = getattr(module, part)
                    
                    hook = module.register_forward_hook(get_hook_fn(layer_name))
                    hooks.append(hook)
                    registered_layers.append(layer_name)
                except Exception as e:
                    print(f"Error registering hook for layer {layer_name}: {str(e)}")
                    continue
            
            if not registered_layers:
                raise ValueError("No layers could be registered with hooks. Please check layer names.")
            
            # Run a forward pass to capture activations
            with torch.no_grad():
                # Convert image to tensor if needed
                if isinstance(img, np.ndarray):
                    # Ensure the tensor is in the correct format (BCHW)
                    if img.shape[-1] == 3:  # If last dimension is channels (HWC format)
                        tensor_img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Transpose to CHW
                    else:
                        tensor_img = torch.from_numpy(img).float()
                        
                    # Add batch dimension if needed
                    if len(tensor_img.shape) == 3:  # CHW -> BCHW
                        tensor_img = tensor_img.unsqueeze(0)
                else:
                    tensor_img = img
                
                if tensor_img.device != next(model.parameters()).device:
                    tensor_img = tensor_img.to(next(model.parameters()).device)
                
                # Forward pass
                _ = model(tensor_img)
            
            # Remove hooks after forward pass
            for hook in hooks:
                hook.remove()
            
            if not layer_activations:
                raise ValueError("No layer activations were captured. Please check layer names.")
            
            # Function to create LIME explanations for each layer
            explanation_images = []
            layer_explanation_names = []
            
            total_steps = len(layer_activations) * len(selected_classes)
            current_step = 0
            
            # First get the baseline prediction for the original image
            def baseline_predict_fn(images):
                base_preds = []
                for image in images:
                    # Forward pass
                    with torch.no_grad():
                        if isinstance(image, np.ndarray):
                            # Ensure the tensor is in the correct format (BCHW)
                            if image.shape[-1] == 3:  
                                tensor_img = torch.from_numpy(image.transpose(2, 0, 1)).float()  # Transpose to CHW
                            else:
                                tensor_img = torch.from_numpy(image).float()
                            
                            if len(tensor_img.shape) == 3:  
                                tensor_img = tensor_img.unsqueeze(0)
                        else:
                            tensor_img = image
                        
                        if tensor_img.device != next(model.parameters()).device:
                            tensor_img = tensor_img.to(next(model.parameters()).device)
                        
                        # Get model prediction
                        results = model(tensor_img)
                        
                        # Process prediction based on model type
                        pred = np.zeros(len(selected_classes))
                        for i, cls_idx in enumerate(selected_classes):
                            if hasattr(results, 'boxes'):
                                # For torchvision models
                                boxes = results.boxes
                                cls_indices = boxes.cls.cpu().numpy().astype(int)
                                confs = boxes.conf.cpu().numpy()
                                mask = (cls_indices == cls_idx)
                                if np.any(mask):
                                    pred[i] = np.max(confs[mask])
                            elif isinstance(results, list) and hasattr(results[0], 'boxes'):
                                # For YOLO models
                                result = results[0]
                                boxes = result.boxes
                                cls_indices = boxes.cls.cpu().numpy().astype(int)
                                confs = boxes.conf.cpu().numpy()
                                mask = (cls_indices == cls_idx)
                                if np.any(mask):
                                    pred[i] = np.max(confs[mask])
                    
                    base_preds.append(pred)
                
                return np.array(base_preds)
            
            # Get baseline prediction
            baseline_preds = baseline_predict_fn([img])[0]
            print(f"Baseline predictions: {baseline_preds}")
            
            # Now generate explanations for each layer and each class
            for layer_name, activations in layer_activations.items():
                print(f"Processing layer: {layer_name}")
                
                # MODIFIED: Use the full layer name instead of a shortened version
                # No need to create a short_name variable anymore
                
                # For each selected class, create an explanation
                for class_index, class_idx in enumerate(selected_classes):
                    if progress_callback:
                        current_step += 1
                        progress_callback(current_step, total_steps)
                    
                    # Get class name from model_handler
                    class_name = class_names.get(str(class_idx), f"Class {class_idx}")
                    print(f"  Processing class: {class_idx} ({class_name})")
                    
                    # Check confidence for this class
                    confidence = baseline_preds[class_index]
                    print(f"  Confidence for class {class_idx} ({class_name}): {confidence}")
                    
                    # If confidence is below threshold, we might skip or create a placeholder
                    if confidence < threshold:  # Adjustable threshold
                        print(f"  Low confidence for class {class_idx} - creating placeholder")
                        
                        # Create a placeholder image with a message
                        placeholder_img = np.copy(img)
                        if placeholder_img.shape[-1] != 3:  # Ensure HWC format for visualization
                            placeholder_img = placeholder_img.transpose(1, 2, 0)
                        
                        # Create a simple frame with text
                        placeholder_img = mark_boundaries(placeholder_img, np.zeros(placeholder_img.shape[:2], dtype=bool))
                        
                        # Add a note about low confidence directly in the image
                        plt.figure(figsize=(5, 5))
                        plt.imshow(placeholder_img)
                        
                        # MODIFIED: Use full layer name in title
                        plt.title(f"{layer_name}\n({class_name})\nLow confidence: {confidence:.3f}")
                        plt.axis('off')
                        
                        # Save to a buffer and reload
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png')
                        plt.close()
                        
                        buf.seek(0)
                        placeholder_img = np.array(Image.open(buf))
                        
                        # Add to explanations
                        explanation_images.append(placeholder_img)
                        
                        # MODIFIED: Use full layer name in explanation name
                        layer_explanation_names.append(f"{layer_name} - {class_name} (Conf: {confidence:.3f})")
                        continue
                    
                    try:
                        print(f"  Generating LIME explanation...")
                        
                        # Create a prediction function that focuses only on the current class
                        def class_predict_fn(images):
                            all_preds = baseline_predict_fn(images)
                            # Extract only the column for the current class and reshape to 2D
                            return all_preds[:, class_index].reshape(-1, 1)
                        
                        # Set up LIME explainer with kernel width from parameters
                        explainer = LimeImageExplainer(kernel_width=kernel_width)
                        
                        # Generate explanation for this class
                        explanation = explainer.explain_instance(
                            img,
                            class_predict_fn,
                            labels=(0,),  # Single label is 0
                            num_samples=num_samples,
                            batch_size=batch_size,
                            segmentation_fn=segmentation_fn
                        )
                        
                        # Create visualization
                        temp, mask = explanation.get_image_and_mask(
                            0,  # Label index is 0
                            positive_only=positive_only,
                            num_features=num_features,
                            hide_rest=False,
                            min_weight=0.0
                        )
                        
                        # Create explanation image
                        if use_heatmap:
                            # Use custom heatmap visualization
                            explanation_img = self._create_heatmap_visualization(
                                temp, explanation, 0,
                                num_features, positive_only, opacity
                            )
                        else:
                            # Use standard LIME visualization with boundaries
                            explanation_img = mark_boundaries(temp, mask, color=(0, 0, 0))
                        
                        # Add the explanation image to the list
                        explanation_images.append(explanation_img)
                        
                        # MODIFIED: Use full layer name in explanation name
                        layer_explanation_names.append(f"{layer_name} - {class_name} (Conf: {confidence:.3f})")
                        
                        print(f"  Successfully added explanation for {layer_name} ({class_name})")
                        
                    except Exception as e:
                        print(f"  Error generating explanation for layer {layer_name}, class {class_idx}: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        
                        try:
                            print(f"  Creating fallback visualization...")
                            # Create a placeholder visualization
                            placeholder_img = np.copy(img)
                            if placeholder_img.shape[-1] != 3:  # Ensure HWC format for visualization
                                placeholder_img = placeholder_img.transpose(1, 2, 0)
                            
                            # Create a simple frame with text
                            placeholder_img = mark_boundaries(placeholder_img, np.zeros(placeholder_img.shape[:2], dtype=bool))
                            
                            # Add a note about the error
                            plt.figure(figsize=(5, 5))
                            plt.imshow(placeholder_img)
                            
                            # MODIFIED: Use full layer name in title
                            plt.title(f"{layer_name}\n({class_name})\nError: {str(e)[:50]}")
                            plt.axis('off')
                            
                            # Save to a buffer and reload
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png')
                            plt.close()
                            
                            buf.seek(0)
                            fallback_img = np.array(Image.open(buf))
                            
                            # Add to explanations
                            explanation_images.append(fallback_img)
                            
                            # MODIFIED: Use full layer name in explanation name
                            layer_explanation_names.append(f"{layer_name} - {class_name} (Error)")
                            print(f"  Added fallback explanation for {layer_name} ({class_name})")
                        except Exception as fallback_err:
                            print(f"  Fallback approach also failed: {str(fallback_err)}")
                            continue
            
            if not explanation_images:
                raise ValueError("No explanations could be generated. Check the model and selected classes.")
                
            print(f"Returning {len(explanation_images)} explanations and {len(layer_explanation_names)} layer names")
            return explanation_images, layer_explanation_names
                
        except Exception as e:
            import traceback
            print(f"Exception in layer-wise LIME: {str(e)}")
            print(traceback.format_exc())
            raise RuntimeError(f"Failed to generate layer-wise explanation: {str(e)}")
        
    def _create_heatmap_visualization(self, original_image, explanation, label_idx, num_features, positive_only, opacity=0.7):
        """Create a heatmap visualization for the explanation."""
        try:
            from matplotlib.colors import LinearSegmentedColormap
            
            # Get the feature importances for this class
            ind = explanation.local_exp[label_idx]
            
            # Create a mask of the same size as the image
            mask = np.zeros(explanation.segments.shape, dtype=np.float64)
            
            # Get absolute weights for normalization
            abs_weights = [abs(weight) for _, weight in ind]
            if not abs_weights:
                print("No weights found in explanation, returning original image")
                return original_image.copy()  # Return original if no weights
            max_weight = max(abs_weights)
            
            # Filter by positive weights if needed
            if positive_only:
                ind = [(segment_id, weight) for segment_id, weight in ind if weight > 0]
            
            # Sort segments by absolute weight value
            sorted_ind = sorted(ind, key=lambda x: abs(x[1]), reverse=True)
            
            # Take only the top num_features if specified
            if num_features > 0:
                sorted_ind = sorted_ind[:num_features]
            
            # Create the mask with normalized weights
            for segment_id, weight in sorted_ind:
                # For heatmap visualization, we use absolute value for intensity
                # but keep the sign for color choice
                norm_weight = weight / max_weight
                mask[explanation.segments == segment_id] = norm_weight
            
            # Create separate masks for positive and negative weights
            pos_mask = (mask > 0)
            neg_mask = (mask < 0)
            
            # Initialize the heatmap with zeros and alpha=0
            heatmap = np.zeros((*mask.shape, 4))
            
            # Create custom colormaps for positive and negative weights
            pos_colors = [(0.0, (0.0, 0.5, 0.0)),      # Dark green
                         (0.3, (0.0, 0.8, 0.0)),      # Green
                         (0.5, (0.9, 0.9, 0.0)),      # Yellow
                         (0.7, (0.9, 0.4, 0.0)),      # Orange
                         (1.0, (1.0, 0.0, 0.0))]      # Red
            pos_cmap = LinearSegmentedColormap.from_list('green_yellow_red', pos_colors)
            
            neg_colors = [(0.0, (0.0, 0.0, 0.5)),      # Dark blue
                         (0.5, (0.0, 0.0, 0.9)),      # Blue
                         (1.0, (0.5, 0.0, 0.9))]      # Purple
            neg_cmap = LinearSegmentedColormap.from_list('blue_purple', neg_colors)
            
            # Apply positive colormap
            if np.any(pos_mask):
                pos_values = mask[pos_mask]
                pos_colors = pos_cmap(pos_values / np.max(pos_values))
                pos_colors[..., 3] = np.abs(pos_values) * opacity
                heatmap[pos_mask] = pos_colors
            
            # Apply negative colormap if not positive_only
            if np.any(neg_mask) and not positive_only:
                neg_values = -mask[neg_mask]  # Make positive for colormap
                neg_colors = neg_cmap(neg_values / np.max(neg_values))
                neg_colors[..., 3] = np.abs(neg_values) * opacity
                heatmap[neg_mask] = neg_colors
            
            # Make sure the original image is in the right format (float 0-1)
            if original_image.max() > 1.0:
                original_image = original_image.astype(np.float32) / 255.0
            
            # Convert to appropriate shape if needed
            if len(original_image.shape) == 2:
                # Grayscale image
                original_image = np.stack([original_image] * 3, axis=-1)
                
            # Overlay heatmap on original image
            result = original_image.copy()
            for i in range(3):  # RGB channels
                blend_mask = (heatmap[..., 3] > 0)
                result[..., i] = np.where(
                    blend_mask,
                    result[..., i] * (1 - heatmap[..., 3]) + heatmap[..., i] * heatmap[..., 3],
                    result[..., i]
                )
                
            return result
        except Exception as e:
            import traceback
            print(f"Error in heatmap visualization: {str(e)}")
            traceback.print_exc()
            # Return original image as fallback
            return original_image

    def generate_layer_wise_lime(self, model_handler):
        """Main entry point for generating layer-wise LIME explanations."""
        if not hasattr(self.parent, 'dataset') or not self.parent.dataset:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        
        # Set the model handler
        self.model_handler = model_handler
        
        # Get current image
        self.image_path = self.parent.dataset[self.parent.current_image_index]
        
        # Get selected classes
        self.selected_classes = [idx for idx, checkbox in self.parent.class_checkboxes.items() 
                                if checkbox.isChecked()]
        
        if not self.selected_classes:
            QMessageBox.warning(self, "Warning", "Please select at least one class!")
            return
            
        # If no layers have been selected, prompt user to refresh and select layers
        if not self.selected_layers:
            response = QMessageBox.question(
                self, 
                "No Layers Selected", 
                "No layers have been selected for explanation. Do you want to refresh the layer list?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            
            if response == QMessageBox.Yes:
                self.refresh_layers()
            return
        
        # Start generating explanations
        self.start_layer_wise_lime()
        
    def resizeEvent(self, event):
        """Handle widget resize events."""
        super().resizeEvent(event)
        
        # Add a slight delay to allow the layout to settle
        QTimer.singleShot(50, self.delayed_resize_update)
        
    def delayed_resize_update(self):
        """Update display after resize with a small delay to let layout settle."""
        # Update display if needed when widget is resized
        if self.layer_explanations and self.display_mode == "single":
            # Re-adjust figure to fill canvas
            self.current_canvas.figure.subplots_adjust(left=0, right=1, top=0.95, bottom=0)
            self.current_canvas.figure.tight_layout(pad=0)
            self.current_canvas.draw()

        # Force layout update
        self.scroll_area.updateGeometry()
        self.explanation_frame.updateGeometry()