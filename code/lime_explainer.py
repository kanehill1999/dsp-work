# lime_explainer.py
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
from ultralytics import YOLO
from yolo_predict_fn import preprocess_image_for_yolo
# Remove the import of constant CLASS_NAMES since we'll use the model_handler instead
from PyQt5.QtCore import QThread, pyqtSignal
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.kernel_ridge import KernelRidge
import cv2


class ExplanationWorker(QThread):
    """Worker thread for generating LIME explanations."""
    progress = pyqtSignal(int, int)  # Signal for progress updates (current, total)
    finished = pyqtSignal(list, list)  # Signal for completed explanations (images, labels)
    error = pyqtSignal(str)  # Signal for error messages

    def __init__(self, explanation_generator, image_path, selected_classes, lime_params=None):
        super().__init__()
        self.explanation_generator = explanation_generator
        self.image_path = image_path
        self.selected_classes = selected_classes
        self.lime_params = lime_params or {}

    def run(self):
        try:
            explanation_images, explanation_labels = self.explanation_generator.generate_explanation(
                self.image_path,
                self.selected_classes,
                self.lime_params,
                progress_callback=self.progress.emit
            )
            self.finished.emit(explanation_images, explanation_labels)
        except Exception as e:
            self.error.emit(str(e))


class LimeExplanationGenerator:
    def __init__(self, dataset=None, current_image_index=0, model_handler=None):
        self.dataset = dataset
        self.current_image_index = current_image_index
        self.model_handler = model_handler

    def _get_model_regressor(self, model_type='linear'):
        """Get the appropriate sklearn model based on the model type string."""
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif model_type == 'kernel_ridge':
            return KernelRidge(alpha=1.0)
        else:
            # Default to linear regression if unknown type
            return LinearRegression()

    def _create_heatmap_visualization(self, original_image, explanation, label_idx, num_features, positive_only, opacity=0.7):
        # Get the feature importances for this class
        ind = explanation.local_exp[label_idx]
        
        # Create a mask of the same size as the image
        mask = np.zeros(explanation.segments.shape, dtype=np.float64)
        
        # Filter by positive weights if needed
        min_weight = 0.0 if positive_only else None
        
        # Get absolute weights for normalization
        abs_weights = [abs(weight) for _, weight in ind]
        if not abs_weights:
            return original_image.copy()  # Return original if no weights
        max_weight = max(abs_weights)
        
        # Filter and sort importances
        if min_weight is not None:
            ind = [(segment_id, weight) for segment_id, weight in ind if weight >= min_weight]
        
        # Sort segments by absolute weight value
        sorted_ind = sorted(ind, key=lambda x: abs(x[1]), reverse=True)
        
        # Take only the top num_features if specified
        if num_features > 0:
            sorted_ind = sorted_ind[:num_features]
        
        # Create the mask with normalized weights
        for segment_id, weight in sorted_ind:
            # Normalize weight to [0,1] range
            norm_weight = abs(weight / max_weight)
            mask[explanation.segments == segment_id] = norm_weight
        
        # Create a custom colormap for smoother transitions
        from matplotlib.colors import LinearSegmentedColormap
        
        # Create a smooth transition from dark green to yellow to red
        colors = [(0.0, (0.0, 0.5, 0.0)),      # Dark green
                (0.3, (0.0, 0.8, 0.0)),      # Green
                (0.5, (0.9, 0.9, 0.0)),      # Yellow
                (0.7, (0.9, 0.4, 0.0)),      # Orange
                (1.0, (1.0, 0.0, 0.0))]      # Red
        
        cmap_name = 'green_yellow_red'
        custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
        
        # Apply the colormap to the mask
        heatmap = custom_cmap(mask)
        
        # Make areas with zero importance transparent
        # Add a small gradient of transparency even for zero values
        alpha_mask = np.clip(mask, 0.0, 1.0)  # No minimum clip to allow full transparency
        heatmap[..., 3] = alpha_mask * opacity  # Set alpha channel
        
        # Make sure the original image is in the right format (float 0-1)
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
        
        # Overlay heatmap on original image
        result = original_image.copy()
        for i in range(3):  # RGB channels
            result[..., i] = result[..., i] * (1 - heatmap[..., 3]) + heatmap[..., i] * heatmap[..., 3]
            
        return result

    def generate_explanation(self, image_path, selected_classes, lime_params=None, progress_callback=None):
        if not image_path:
            raise ValueError("No image path provided!")

        if not selected_classes:
            raise ValueError("Please select at least one class!")

        try:
            # Check if model_handler is available
            if self.model_handler is None:
                raise ValueError("Model handler is not available!")
                
            model = self.model_handler.model
            if model is None:
                raise ValueError("No model loaded!")
                
            # Get class names from model_handler
            class_names = self.model_handler.get_class_names()
            if not class_names:
                raise ValueError("No class names available from model handler!")

            img = preprocess_image_for_yolo(image_path)

            def filtered_predict_fn(images):
                predictions = []
                for image in images:
                    # Run inference
                    results = model(image, verbose=False)
                    
                    # Initialize confidences array for selected classes
                    filtered_confs = np.zeros(len(selected_classes))
                    
                    # Get boxes from the first (and only) image result
                    if len(results) > 0:
                        result = results[0]  # Get first result
                        if hasattr(result, 'boxes') and len(result.boxes) > 0:
                            boxes = result.boxes
                            class_indices = boxes.cls.cpu().numpy().astype(int)
                            confidences = boxes.conf.cpu().numpy()
                            
                            # Fill in confidence scores for selected classes
                            for i, class_idx in enumerate(selected_classes):
                                # Convert class_idx to int if it's a string
                                if isinstance(class_idx, str):
                                    class_idx = int(class_idx)
                                    
                                mask = (class_indices == class_idx)
                                if np.any(mask):
                                    filtered_confs[i] = np.max(confidences[mask])
                    
                    predictions.append(filtered_confs)
                
                return np.array(predictions)

            # Set up LIME parameters with defaults
            lime_params = lime_params or {}
            num_samples = lime_params.get('num_samples', 1000)
            num_features = lime_params.get('num_features', 5)
            hide_rest = lime_params.get('hide_rest', False)
            positive_only = lime_params.get('positive_only', True)
            batch_size = lime_params.get('batch_size', 10)
            distance_metric = lime_params.get('distance_metric', 'cosine')
            kernel_width = lime_params.get('kernel_width', 0.25)
            use_heatmap = lime_params.get('use_heatmap', False)
            heatmap_opacity = lime_params.get('heatmap_opacity', 0.7)
            grey_color = lime_params.get('grey_color', np.array([0.5, 0.5, 0.5]))
            
            # Get the appropriate model regressor
            model_type = lime_params.get('interpretable_model_type', 'linear')
            model_regressor = self._get_model_regressor(model_type)

            # Create and run explainer
            explainer = LimeImageExplainer(kernel_width=kernel_width)
            
            # IMPORTANT FIX: Use indices instead of actual class IDs for LIME
            explanation = explainer.explain_instance(
                img,
                filtered_predict_fn,
                labels=list(range(len(selected_classes))),  # Use indices instead of actual class IDs
                top_labels=len(selected_classes),
                num_samples=num_samples,
                batch_size=batch_size,
                distance_metric=distance_metric,
                model_regressor=model_regressor
            )

            explanation_images = []
            explanation_labels = []

            # Generate explanations for each selected class
            for i, class_idx in enumerate(selected_classes):
                # Report progress if callback provided
                if progress_callback:
                    progress_callback(i + 1, len(selected_classes))

                
                try:
                    # Generate visualization based on user selection
                    if use_heatmap:
                        # Use custom heatmap visualization
                        temp, mask = explanation.get_image_and_mask(
                            i,  # Use index i instead of class_idx
                            positive_only=positive_only,
                            num_features=num_features,
                            hide_rest=hide_rest,
                            min_weight=0.0
                        )
                        marked_image = self._create_heatmap_visualization(
                            temp, 
                            explanation, 
                            i,  # Use index i instead of class_idx
                            num_features, 
                            positive_only, 
                            heatmap_opacity
                        )
                    else:
                        # Use standard LIME visualization
                        temp, mask = explanation.get_image_and_mask(
                            i,  # Use index i instead of class_idx
                            positive_only=positive_only,
                            num_features=num_features,
                            hide_rest=hide_rest,
                            min_weight=0.0
                        )
                        
                        # Create visualization with black boundaries
                        marked_image = mark_boundaries(temp, mask, color=(0, 0, 0))
                except Exception as e:
                    print(f"Error generating visualization for class {class_idx}: {str(e)}")
                    # Fall back to original image if there's an error
                    marked_image = img.copy()
                    if marked_image.max() > 1.0:
                        marked_image = marked_image / 255.0
                
                # Get class name and prediction confidence
                pred_probs = filtered_predict_fn([img])[0]
                class_confidence = pred_probs[i]  # Use index i instead of selected_classes.index(class_idx)
                
                
                # Convert class_idx to string to match the format in the JSON config
                class_name = class_names.get(str(class_idx), f"Class {class_idx}")
                
                explanation_images.append(marked_image)
                explanation_labels.append(f"{class_name} (Confidence: {class_confidence:.2f})")

            return explanation_images, explanation_labels

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to generate explanation: {str(e)}")

    def cleanup(self):
        """Clean up any resources."""
        if self.model_handler:
            self.model_handler.cleanup()