import os
import torch
import cv2
import numpy as np
from PIL import Image
from model_config import ModelConfig


class ModelHandler:
    """Class to handle loading and executing different object detection models."""
    
    def __init__(self):
        """Initialize the model handler."""
        self.model = None
        self.current_model_id = None
        self.config_manager = ModelConfig()
        
    def get_model_display_names(self):
        """Get list of model display names.
        
        Returns:
            List of model names for display
        """
        models = self.config_manager.get_models()
        return [model.get("name", model.get("id", "Unknown")) for model in models]
    
    def get_model_id_by_name(self, model_name):
        """Get model ID by name.
        
        Args:
            model_name: Display name of the model
            
        Returns:
            Model ID or None if not found
        """
        models = self.config_manager.get_models()
        for model in models:
            if model.get("name") == model_name:
                return model.get("id")
        return None
    
    def load_model(self, model_id_or_name):
        """Load a model by ID or name.
        
        Args:
            model_id_or_name: ID or name of the model to load
            
        Returns:
            True if model was loaded successfully, False otherwise
        """
        try:
            # First check if this is a model ID
            model_config = self.config_manager.get_model_by_id(model_id_or_name)
            
            # If not found by ID, try to find by name
            if not model_config:
                model_id = self.get_model_id_by_name(model_id_or_name)
                if model_id:
                    model_config = self.config_manager.get_model_by_id(model_id)
            
            if not model_config:
                print(f"Model not found: {model_id_or_name}")
                return False
                
            # Get model configuration details
            architecture_type = model_config.get("architecture_type", "yolo")
            model_path = model_config.get("model_path", "")
            weight_file = model_config.get("weight_file", "")
            
            # Construct the full path to the weights file
            weight_path = os.path.join(model_path, weight_file)
            print(f"Attempting to load model from: {weight_path}")
            
            # Check if the weight file exists
            if not os.path.exists(weight_path):
                print(f"Weight file not found: {weight_path}")
                return False
                
            # Load the model based on architecture type
            if architecture_type.lower() == "yolo":
                try:
                    from ultralytics import YOLO
                    self.model = YOLO(weight_path)
                    print(f"YOLO model loaded from: {weight_path}")
                    if hasattr(self.model, 'names'):
                        print(f"YOLO model classes: {self.model.names}")
                except ImportError:
                    print("Failed to import YOLO. Make sure ultralytics is installed.")
                    return False
            
            elif architecture_type.lower() == "faster_rcnn":
                try:
                    import torchvision
                    
                    # Get class names from the model's class set configuration
                    class_names = self.config_manager.get_model_classes(model_config.get("id"))
                    
                    # Get the number of classes from the class names dictionary
                    if not class_names:
                        print("No class names found in configuration for this model")
                        return False
                    
                    num_classes = len(class_names) + 1  # +1 for background class
                    print(f"Initializing Faster R-CNN with {num_classes} classes (including background)")
                    
                    # Initialize model with the correct number of classes
                    self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                        weights=None, 
                        num_classes=num_classes
                    )
                    
                    # Load weights
                    checkpoint = torch.load(weight_path, map_location='cpu')
                    
                    # Extract the model state dict - handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        elif 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Try loading the model
                    try:
                        self.model.load_state_dict(state_dict)
                        print("Successfully loaded Faster R-CNN model with strict=True")
                    except Exception as e:
                        print(f"Strict loading failed: {e}. Trying non-strict loading...")
                        # Some keys might not match, try non-strict loading
                        self.model.load_state_dict(state_dict, strict=False)
                        print("Successfully loaded Faster R-CNN model with strict=False")
                    
                    # Set model to evaluation mode
                    self.model.eval()
                    print(f"Faster R-CNN model loaded from: {weight_path}")
                    
                    # Print class mapping for debugging
                    print("Class mapping:")
                    for idx, name in class_names.items():
                        print(f"  {idx}: {name}")
                    
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error loading Faster R-CNN model: {str(e)}")
                    return False
            
            else:
                print(f"Unsupported architecture type: {architecture_type}")
                return False
                
            # Store the current model ID
            self.current_model_id = model_config.get("id")
            
            print(f"Successfully loaded model: {model_config.get('name', model_config.get('id'))}")
            return True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading model: {str(e)}")
            return False
    
    def detect(self, image_path, selected_classes, frame=None):
        """Run object detection on an image.
        
        Args:
            image_path: Path to the image file (or None if frame is provided)
            selected_classes: List of class indices to detect
            frame: Optional pre-loaded image frame
            
        Returns:
            Tuple of (detections, processed_image)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Convert selected classes to integers if they are strings
        selected_classes = [int(cls) if isinstance(cls, str) else cls for cls in selected_classes]
        
        # Process the image
        if frame is not None:
            # Use the provided frame
            image_rgb = frame.copy()
        else:
            # Load the image from path
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference based on the model type
        if hasattr(self.model, 'predict'):  # YOLO model
            results = self.model.predict(image_rgb, verbose=False)
            detections = self._process_yolo_detections(results, selected_classes)
        else:  # Assume torchvision model (Faster R-CNN)
            # Convert image to tensor
            image_tensor = self._prepare_image_for_torchvision(image_rgb)
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Run inference
            with torch.no_grad():
                outputs = self.model([image_tensor])
            
            detections = self._process_torchvision_detections(outputs, selected_classes, image_rgb.shape)
        
        return detections, image_rgb
    
    def _process_yolo_detections(self, results, selected_classes):
        """Process YOLO model detection results.
        
        Args:
            results: YOLO detection results
            selected_classes: List of class indices to include
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        for result in results:
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                for i in range(len(boxes)):
                    class_idx = int(classes[i])
                    
                    # Check if this class is in the selected classes
                    if class_idx in selected_classes:
                        detections.append({
                            'bbox': boxes[i].tolist(),
                            'class_idx': class_idx,
                            'confidence': float(confidences[i])
                        })
        
        return detections
    
    def _prepare_image_for_torchvision(self, image):
        """Prepare an image for inference with a torchvision model.
        
        Args:
            image: Input image (RGB format)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert numpy array to PIL Image
        from PIL import Image as PILImage
        image_pil = PILImage.fromarray(image)
        
        # Convert to tensor
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # No need to add batch dimension - the model expects [C, H, W]
        return transform(image_pil)
    
    def _process_torchvision_detections(self, outputs, selected_classes, image_shape):
        """Process torchvision model detection results.
        
        Args:
            outputs: Model outputs
            selected_classes: List of class indices to include
            image_shape: Shape of the original image
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Process first image in batch
        boxes = outputs[0]['boxes'].cpu().numpy().astype(int)
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        # Print detection info for debugging
        print(f"Raw detections: {len(boxes)} boxes found")
        print(f"Labels: {labels}")
        print(f"Scores: {scores}")
        print(f"Selected classes: {selected_classes}")
        
        # Apply a confidence threshold
        confidence_threshold = 0.5
        
        for i in range(len(boxes)):
            class_idx = int(labels[i])
            confidence = float(scores[i])
            
            # Check if this class is in the selected classes and if score is above threshold
            if class_idx in selected_classes and confidence > confidence_threshold:
                x1, y1, x2, y2 = boxes[i].tolist()
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image_shape[1], x2)
                y2 = min(image_shape[0], y2)
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'class_idx': class_idx,
                    'confidence': confidence
                })
        
        print(f"After filtering: {len(detections)} valid detections")
        return detections
    
    def get_class_names(self):
        """Get class names for the current model.
        
        Returns:
            Dictionary mapping class indices to class names
        """
        if not self.current_model_id:
            print("No model is currently loaded")
            return {}
        
        # Get class names from the configuration
        class_names = self.config_manager.get_model_classes(self.current_model_id)
        
        # If no class names found in configuration, return empty dict
        if not class_names:
            print(f"Warning: No class names found for model ID: {self.current_model_id}")
            return {}
        
        return class_names
    
    def cleanup(self):
        """Clean up resources."""
        # Set model to None to allow garbage collection
        self.model = None