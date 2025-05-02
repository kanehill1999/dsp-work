# image_views.py
from PyQt5.QtWidgets import QTabWidget, QLabel, QVBoxLayout
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from image_detection import ObjectDetectionView
from video_detection_view import VideoDetectionView
from layer_wise_lime_view import LayerWiseLIMEView 



class ImageViews(QTabWidget):
    def __init__(self, parent=None, model_selector=None, class_checkboxes=None):
        super().__init__(parent)
        self.setTabPosition(QTabWidget.South)
        
        # Original Image View
        self.original_image = QLabel()
        self.original_image.setAlignment(Qt.AlignCenter)
        self.original_image.setMinimumSize(800, 600)
        self.addTab(self.original_image, "Original Image")
        
        # Object Detection View
        self.object_detection_view = ObjectDetectionView(
            parent=self, 
            model_selector=model_selector, 
            class_checkboxes=class_checkboxes
        )
        self.addTab(self.object_detection_view, "Object Detection")
        
        # Video Detection View
        self.video_view = VideoDetectionView()
        self.addTab(self.video_view, "Video Detection")
        
        # LIME View
        self.lime_canvas = FigureCanvas(Figure(figsize=(8, 8)))
        self.addTab(self.lime_canvas, "LIME Explanation")
        
        #Layer-wise LIME View
        self.layer_lime_view = LayerWiseLIMEView(parent)
        self.addTab(self.layer_lime_view, "Layer-wise LIME")

        
        
