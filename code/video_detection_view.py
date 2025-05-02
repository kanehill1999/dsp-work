# video_detection_view.py
import cv2
import os
import sys
import numpy as np
import traceback
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QComboBox, QProgressBar, QFileDialog, QMessageBox


class VideoProcessingThread(QThread):
    frame_ready = pyqtSignal(object, list)  # Signal for processed frame and detections
    progress_update = pyqtSignal(int, int)  # Signal for progress updates
    error_occurred = pyqtSignal(str)  # Signal for error reporting
    
    def __init__(self, video_path, model_handler, selected_classes, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model_handler = model_handler
        self.selected_classes = selected_classes
        self.is_running = True
        self.playback_speed = 1.0
        print(f"VideoProcessingThread initialized with video_path: {video_path}")
        print(f"VideoProcessingThread initialized with model_handler: {model_handler}")
        print(f"VideoProcessingThread initialized with selected_classes: {selected_classes}")
        
    def run(self):
        print(f"VideoProcessingThread.run() started")
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                error_msg = f"Could not open video file: {self.video_path}"
                print(f"ERROR: {error_msg}")
                self.error_occurred.emit(error_msg)
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Video has {total_frames} frames at {fps} FPS")
            
            current_frame = 0
            while self.is_running and current_frame < total_frames:
                ret, frame = cap.read()
                if not ret:
                    error_msg = f"Failed to read frame {current_frame}/{total_frames}"
                    print(f"ERROR: {error_msg}")
                    self.error_occurred.emit(error_msg)
                    break
                    
                # Print progress every 30 frames
                if current_frame % 30 == 0:
                    print(f"Processing frame {current_frame}/{total_frames}")
                    
                # Convert to RGB for processing
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run detection on the frame
                try:
                    print(f"Running detection on frame {current_frame}")
                    detections, _ = self.model_handler.detect(None, self.selected_classes, frame=frame_rgb)
                    print(f"Frame {current_frame}: Found {len(detections)} detections")
                    self.frame_ready.emit(frame_rgb, detections)
                    self.progress_update.emit(current_frame, total_frames)
                    
                    # Control playback speed
                    sleep_time = (1.0 / fps) / self.playback_speed
                    if sleep_time > 0:
                        self.msleep(int(sleep_time * 1000))
                except Exception as e:
                    error_msg = f"Error processing frame {current_frame}: {str(e)}"
                    print(error_msg)
                    traceback.print_exc()
                    self.error_occurred.emit(error_msg)
                
                current_frame += 1
                
            cap.release()
            print("Video processing completed")
            
        except Exception as e:
            error_msg = f"Critical error in VideoProcessingThread: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
        
    def stop(self):
        print("VideoProcessingThread.stop() called")
        self.is_running = False
        self.wait()
        print("VideoProcessingThread stopped")
        
    def set_playback_speed(self, speed):
        self.playback_speed = speed
        print(f"Playback speed set to {speed}x")


class VideoDetectionView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.model_handler = None
        self.video_path = None
        self.processing_thread = None
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        # Video name label
        self.video_name_label = QLabel("No video loaded")
        self.video_name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_name_label)
        
        # Video display
        self.video_frame = QLabel()
        self.video_frame.setAlignment(Qt.AlignCenter)
        self.video_frame.setMinimumSize(800, 600)
        layout.addWidget(self.video_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        # Time remaining
        self.time_label = QLabel("Time Remaining: --:--")
        layout.addWidget(self.time_label)
        
        # Controls
        controls = QHBoxLayout()
        
        # Load button
        self.load_button = QPushButton("Load Video")
        self.load_button.clicked.connect(self.load_video)
        controls.addWidget(self.load_button)
        
        # Play button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setEnabled(False)  # Disabled until video is loaded
        controls.addWidget(self.play_button)
        
        # Speed control
        self.speed_combo = QComboBox()
        speeds = ["0.25x", "0.5x", "1.0x", "1.5x", "2.0x"]
        self.speed_combo.addItems(speeds)
        self.speed_combo.setCurrentIndex(2)  # Default to 1.0x
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        controls.addWidget(self.speed_combo)
        
        layout.addLayout(controls)
    
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
        print("WARNING: VideoDetectionView could not access model_handler")
        return None
        
    def set_model_handler(self, model_handler):
        self.model_handler = model_handler
        print(f"VideoDetectionView: model_handler has been set: {self.model_handler is not None}")
        
    def load_video(self):
        if not hasattr(self, 'video_path') or not self.video_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
            )
            if file_path:
                self.video_path = file_path
        
        if self.video_path:
            try:
                # Display the video file name
                video_filename = os.path.basename(self.video_path)
                self.video_name_label.setText(f"Loaded: {video_filename}")
                
                # Enable the play button
                self.play_button.setEnabled(True)
                self.play_button.setText("Play")
                
                # Reset progress bar
                self.progress_bar.setValue(0)
                self.time_label.setText("Time Remaining: --:--")
                
                # Show a preview frame if possible
                try:
                    cap = cv2.VideoCapture(self.video_path)
                    ret, frame = cap.read()
                    if ret:
                        # Convert to RGB for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        h, w, c = frame_rgb.shape
                        q_img = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img)
                        self.video_frame.setPixmap(pixmap.scaled(
                            self.video_frame.size(),
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation
                        ))
                    cap.release()
                except Exception as e:
                    print(f"Error loading preview frame: {str(e)}")
                    
            except Exception as e:
                QMessageBox.warning(
                    self, 
                    "Video Loading Error", 
                    f"Failed to load video: {str(e)}"
                )
            
    def toggle_playback(self):
        # Get model handler
        model_handler = self.get_model_handler()
        print(f"VideoDetectionView toggle_playback: model_handler is {model_handler}")
        
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please load a video first!")
            return
            
        if not model_handler:
            QMessageBox.warning(self, "Warning", "Could not access model handler!")
            return
        
        if not model_handler.model:
            QMessageBox.warning(self, "Warning", "No model loaded. Please select a model first.")
            return
            
        if not self.processing_thread or not self.processing_thread.isRunning():
            # Start playback
            print("Starting video playback")
            self.play_button.setText("Pause")
            
            # Get selected classes
            selected_classes = []
            try:
                # Try to get selected classes from parent -> parent -> class_checkboxes
                parent = self.parent
                if parent and hasattr(parent, 'parent') and callable(getattr(parent, 'parent', None)):
                    parent_app = parent.parent()
                    if parent_app and hasattr(parent_app, 'class_checkboxes'):
                        class_checkboxes = parent_app.class_checkboxes
                        selected_classes = [idx for idx, checkbox in class_checkboxes.items() if checkbox.isChecked()]
                        print(f"Got selected classes from parent app: {selected_classes}")
                
                # If still no selected classes, try alternate path
                if not selected_classes and hasattr(parent, 'class_checkboxes'):
                    class_checkboxes = parent.class_checkboxes
                    selected_classes = [idx for idx, checkbox in class_checkboxes.items() if checkbox.isChecked()]
                    print(f"Got selected classes from parent: {selected_classes}")
                
                # If still no selected classes, try the model handler's class names
                if not selected_classes and model_handler:
                    class_names = model_handler.get_class_names()
                    if class_names:
                        # Convert string keys to integers where needed
                        selected_classes = [int(k) if isinstance(k, str) and k.isdigit() else k 
                                           for k in class_names.keys()]
                        print(f"Got selected classes from model handler: {selected_classes}")
                
                # If still no selected classes, default to range
                if not selected_classes:
                    selected_classes = list(range(20))  # Default to first 20 classes
                    print(f"Using default selected classes: {selected_classes}")
                
            except Exception as e:
                print(f"Error getting selected classes: {str(e)}")
                # Default to range if there's an error
                selected_classes = list(range(20))  # Default to first 20 classes
                print(f"Using fallback selected classes after error: {selected_classes}")
            
            # Verify that video path is valid and exists
            if not os.path.exists(self.video_path):
                print(f"ERROR: Video file does not exist: {self.video_path}")
                QMessageBox.warning(self, "Error", f"Video file not found: {self.video_path}")
                return
                
            print(f"Creating processing thread with video_path: {self.video_path}")
            print(f"Creating processing thread with model_handler: {model_handler}")
            print(f"Creating processing thread with selected_classes: {selected_classes}")
            
            # Start processing thread
            try:
                self.processing_thread = VideoProcessingThread(
                    self.video_path, model_handler, selected_classes
                )
                # Connect signals
                self.processing_thread.frame_ready.connect(self.update_frame)
                self.processing_thread.progress_update.connect(self.update_progress)
                self.processing_thread.error_occurred.connect(self.show_error)
                # Start thread
                self.processing_thread.start()
                print("Video processing thread started")
            except Exception as e:
                print(f"Error starting processing thread: {str(e)}")
                traceback.print_exc()
                QMessageBox.critical(self, "Error", f"Failed to start processing: {str(e)}")
        else:
            # Stop playback
            print("Stopping video playback")
            self.play_button.setText("Play")
            if self.processing_thread:
                self.processing_thread.stop()
            else:
                print("WARNING: No processing thread to stop")
    
    def show_error(self, error_message):
        """Show error message from the processing thread."""
        QMessageBox.critical(self, "Processing Error", error_message)
                
    def update_frame(self, frame, detections):
        # Draw bounding boxes
        display_frame = frame.copy()
        
        # Get the model handler to access class names
        model_handler = self.get_model_handler()
        
        # Get class names dictionary from the model handler
        class_names = {}
        if model_handler:
            class_names = model_handler.get_class_names()
        
        for detection in detections:
            bbox = detection['bbox']
            class_idx = detection['class_idx']
            confidence = detection['confidence']
            
            # Get class name from the model handler's class_names dictionary
            class_name = class_names.get(str(class_idx), f"Class {class_idx}")
                
            # Generate color based on class
            color = self.get_color(class_idx)
            
            # Draw box
            cv2.rectangle(
                display_frame,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                color,
                2
            )
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                display_frame,
                label,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
        # Convert to QImage and display
        h, w, c = display_frame.shape
        q_img = QImage(display_frame.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.video_frame.setPixmap(pixmap.scaled(
            self.video_frame.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
    def get_color(self, class_idx):
        colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Yellow
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Cyan
            (255, 128, 0),    # Orange
        ]
        return colors[class_idx % len(colors)]
        
    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        # Update time remaining
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            remaining_frames = total - current
            speed = float(self.speed_combo.currentText().replace('x', ''))
            remaining_time = remaining_frames / (fps * speed)
            
            minutes = int(remaining_time // 60)
            seconds = int(remaining_time % 60)
            self.time_label.setText(f"Time Remaining: {minutes:02d}:{seconds:02d}")
        
    def change_speed(self, index):
        if self.processing_thread and self.processing_thread.isRunning():
            speed = float(self.speed_combo.currentText().replace('x', ''))
            self.processing_thread.set_playback_speed(speed)