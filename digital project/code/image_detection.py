# image_detection.py
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2


class ObjectDetectionView(QWidget):
    """View for displaying object detection results."""

    def __init__(self, parent=None, model_selector=None, class_checkboxes=None):
        super().__init__(parent)
        self.parent = parent
        self.model_selector = model_selector
        self.class_checkboxes = class_checkboxes
        self.image_path = None
        self.model_handler = None  

        self.initUI()

    
    def set_model_handler(self, model_handler):
        """Set the model handler directly."""
        self.model_handler = model_handler
        print(f"ObjectDetectionView: model_handler set to {model_handler}")

    def initUI(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.image_label = QLabel("Object Detection")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid black;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.image_label.setMinimumSize(320, 240)
        
        layout.addWidget(self.image_label, 1)

        self.detect_button = QPushButton("Run Object Detection")
        self.detect_button.clicked.connect(self.run_detection)
        layout.addWidget(self.detect_button)

    def get_model_handler(self):
        """Get the model handler from the parent."""
        # First check direct reference
        if self.model_handler is not None:
            return self.model_handler

        # Then try direct parent
        if hasattr(self.parent, 'model_handler') and self.parent.model_handler is not None:
            return self.parent.model_handler

        # Then try parent's parent
        elif hasattr(self.parent, 'parent') and self.parent.parent() is not None:
            parent_app = self.parent.parent()
            if hasattr(parent_app, 'model_handler') and parent_app.model_handler is not None:
                return parent_app.model_handler

        # Then try the model_selector if it has a reference to model_handler
        elif self.model_selector is not None and hasattr(self.model_selector, 'model_handler'):
            return self.model_selector.model_handler

        # Log the failure for debugging
        print("WARNING: ObjectDetectionView could not access model_handler")
        return None

    def run_detection(self):
        """Run object detection on the current image."""
        if not self.image_path:
            QMessageBox.warning(self, "Error", "No image loaded!")
            return

        # Get model handler
        model_handler = self.get_model_handler()

        if not model_handler:
            QMessageBox.warning(self, "Error", "Could not access model handler!")
            return

        if not model_handler.model:
            QMessageBox.warning(self, "Error", "No model loaded. Please select a model first.")
            return

        # Get selected classes
        selected_classes = []
        if isinstance(self.class_checkboxes, dict):
            selected_classes = [idx for idx, checkbox in self.class_checkboxes.items() if checkbox.isChecked()]

        if not selected_classes:
            QMessageBox.warning(self, "Error", "Please select at least one class!")
            return

        try:
            # Run detection
            detections, image_rgb = model_handler.detect(self.image_path, selected_classes)

            # Draw detections on the image
            annotated_image = self.draw_detections(image_rgb, detections, model_handler)

            # Convert to QImage
            height, width, channel = annotated_image.shape
            bytes_per_line = 3 * width
            qimage = QImage(annotated_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)

            # Get the current size of the label to scale the pixmap to fit
            label_width = self.image_label.width()
            label_height = self.image_label.height()

            # Display image at maximum size possible while maintaining aspect ratio
            self.update_image_display(pixmap)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run detection: {e}")

    def update_image_display(self, pixmap):
        """Update the image display to show the pixmap as large as possible."""
        if pixmap:
            # Scale the pixmap to fit the available space while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

    def draw_detections(self, image, detections, model_handler):
        
        annotated_image = image.copy()
        class_names = model_handler.get_class_names()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_idx = detection['class_idx']
            confidence = detection['confidence']
            class_name = class_names.get(str(class_idx), f"Class {class_idx}")
            label = f"{class_name}: {confidence:.2f}"

            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                annotated_image,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0], y1),
                (0, 255, 0),
                -1
            )
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )

        return annotated_image

    def resizeEvent(self, event):
        """Handle resize events to rescale the displayed image."""
        if self.image_label.pixmap():
            self.update_image_display(self.image_label.pixmap().copy())
        super().resizeEvent(event)