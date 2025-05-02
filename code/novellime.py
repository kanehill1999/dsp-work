import numpy as np
import cv2
import os
import copy
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox,
    QGroupBox, QGridLayout, QFileDialog, QMessageBox, QProgressBar,
    QSizePolicy, QFrame, QToolButton
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from skimage import io, segmentation
from skimage.segmentation import mark_boundaries
from scipy import ndimage
from sklearn.metrics.pairwise import rbf_kernel


from lime.lime_image import LimeImageExplainer, ImageExplanation
from lime import explanation
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

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
        self.toggle_button.setIconSize(QSize(12, 12))
        self.toggle_button.setFixedHeight(28)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = None
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


class DPPImageExplainer(LimeImageExplainer):
    
    def __init__(self, kernel_width=0.25, kernel=None, verbose=False):
        super().__init__(kernel_width, kernel, verbose)
        self.kernel_width = kernel_width
        
    def explain_instance(self, image, classifier_fn, labels=None, hide_color=None,
                        top_labels=5, num_features=100000, num_samples=1000,
                        batch_size=10, segmentation_fn=None,
                        distance_metric='cosine', model_regressor=None,
                        random_seed=None, dpp_params=None, progress_callback=None):
        
        # Set defaults if parameters not provided
        if dpp_params is None:
            dpp_params = {
                'position_weight': 1.5,
                'color_weight': 1.0,
                'size_weight': 0.8,
                'texture_weight': 0.6,
                'redundancy_penalty': 0.3
            }

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Get segmentation
        if segmentation_fn is None:
            # Choose segmentation method based on parameters
            segmentation_method = dpp_params.get('segmentation_method', 'quickshift')
            
            if segmentation_method == 'quickshift':
                kernel_size = dpp_params.get('kernel_size', 4)
                max_dist = dpp_params.get('max_dist', 200)
                ratio = dpp_params.get('ratio', 0.2)
                
                segmentation_fn = lambda x: segmentation.quickshift(
                    x, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
                    
            elif segmentation_method == 'felzenszwalb':
                scale = dpp_params.get('scale', 100)
                sigma = dpp_params.get('sigma', 0.5)
                min_size = dpp_params.get('min_size', 50)
                
                segmentation_fn = lambda x: segmentation.felzenszwalb(
                    x, scale=scale, sigma=sigma, min_size=min_size)
                    
            elif segmentation_method == 'slic':
                n_segments = dpp_params.get('n_segments', 100)
                compactness = dpp_params.get('compactness', 10)
                sigma = dpp_params.get('sigma', 1)
                
                segmentation_fn = lambda x: segmentation.slic(
                    x, n_segments=n_segments, compactness=compactness, sigma=sigma)
            else:
                # Default to quickshift if unknown method
                segmentation_fn = lambda x: segmentation.quickshift(
                    x, kernel_size=4, max_dist=200, ratio=0.2)
                
        segments = segmentation_fn(image)
        
        # Verify segments are valid
        if np.max(segments) < 1:
            raise ValueError("Segmentation failed: No segments found in the image")
            
        # Generate fudged image
        fudged_image = image.copy()
        if hide_color is None:
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2])
                )
        else:
            fudged_image = np.ones(image.shape) * hide_color

        # Generate samples using DPP
        data, labels_matrix = self._dpp_data_labels(
            image, fudged_image, segments, classifier_fn, num_samples,
            batch_size=batch_size, dpp_params=dpp_params,
            progress_callback=progress_callback)
            
        # Verify data and labels have the same number of samples
        if data.shape[0] != labels_matrix.shape[0]:
            raise ValueError(f"Data and labels dimension mismatch: data={data.shape}, labels={labels_matrix.shape}")

        # Create the explanation with the actual provided labels
        return self._create_explanation(image, segments, data, labels_matrix,
                                labels, distances=None, 
                                top_labels=top_labels, num_features=num_features,
                                distance_metric=distance_metric,
                                model_regressor=model_regressor)
            
    def _create_explanation(self, image, segments, data, labels_matrix, labels=None, distances=None, 
                      top_labels=5, num_features=100000, distance_metric='cosine', model_regressor=None):
        """Create an explanation object from the sampled data and labels."""
        # Create an image explanation
        exp = ImageExplanation(image, segments)
        
        # Process all labels if labels is None
        if labels is None:
            if top_labels:
                # Get top labels based on average prediction
                top_indices = np.argsort(labels_matrix.mean(axis=0))[-top_labels:]
                labels = top_indices
            else:
                # Include all labels
                labels = range(labels_matrix.shape[1])
        
        # Make sure labels is iterable
        if not hasattr(labels, '__iter__'):
            labels = [labels]
        
        # Process each label
        for label_idx, label in enumerate(labels):
            try:
                # Check if labels_matrix has enough columns
                if labels_matrix.shape[1] <= label_idx:
                    print(f"Warning: Label index {label_idx} out of bounds for labels matrix with shape {labels_matrix.shape}")
                    continue
                    
                # Get label data - use the index position in the labels list
                label_data = labels_matrix[:, label_idx]
                
                # Fit the regressor model
                if model_regressor is None:
                    model_regressor = Ridge(alpha=1.0, fit_intercept=True)
                
                # Fit the model
                model_regressor.fit(data, label_data)
                
                # Get coefficients and intercept
                coefficients = model_regressor.coef_
                intercept = model_regressor.intercept_
                
                # Check coefficient dimensions - ensure it's the right length
                if len(coefficients) != data.shape[1]:
                    print(f"Warning: Coefficient dimension mismatch. Expected {data.shape[1]}, got {len(coefficients)}")
                    continue
                
                # Create a list of (feature_id, coefficient) tuples for non-zero coefficients
                feature_weights = [(i, coefficients[i]) for i in range(len(coefficients)) if coefficients[i] != 0]
                
                # Sort by absolute value
                feature_weights.sort(key=lambda x: np.abs(x[1]), reverse=True)
                
                # Limit to num_features if needed
                if num_features < len(feature_weights):
                    feature_weights = feature_weights[:num_features]
                
                
                exp.local_exp[label] = feature_weights
                exp.intercept[label] = intercept
                
                # Score is the R^2 score on training data
                score = r2_score(label_data, model_regressor.predict(data))
                exp.score = score
            except Exception as e:
                print(f"Error processing label {label}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return exp

    def _extract_segment_features(self, image, segments):
        """Extract features for each segment to compute diversity."""
        num_segments = np.max(segments) + 1

        # Initialize features: [position_x, position_y, r, g, b, size, texture]
        features = np.zeros((num_segments, 7))

        # Height and width for normalizing positions
        h, w = segments.shape[:2]

        for i in range(num_segments):
            mask = (segments == i)
            if np.any(mask):
                # Position (center of mass)
                y_indices, x_indices = np.where(mask)
                features[i, 0] = np.mean(x_indices) / w  # normalized x position
                features[i, 1] = np.mean(y_indices) / h  # normalized y position

                # Color (average RGB)
                segment_pixels = image[mask]
                features[i, 2:5] = np.mean(segment_pixels, axis=0) / 255.0  # normalized color

                # Size (proportion of image)
                features[i, 5] = np.sum(mask) / (h * w)
                
                # Improved texture feature (using both std dev and range of grayscale values)
                if segment_pixels.shape[0] > 1:  # Ensure we have enough pixels
                    gray_pixels = np.mean(segment_pixels, axis=1)  # Convert to grayscale
                    std_dev = np.std(gray_pixels) / 255.0
                    value_range = (np.max(gray_pixels) - np.min(gray_pixels)) / 255.0 if segment_pixels.shape[0] > 1 else 0
                    features[i, 6] = (std_dev + value_range) / 2  # combined texture feature
                else:
                    features[i, 6] = 0.0

        return features

    def _compute_similarity_matrix(self, features, position_weight=1.5, color_weight=1.0,
                                  size_weight=0.8, texture_weight=0.6, gamma=0.5):
        
       
        normalized_features = features.copy()
        for i in range(features.shape[1]):
            col_max = np.max(features[:, i])
            if col_max > 0:
                normalized_features[:, i] = features[:, i] / col_max

        # Feature weights vector
        feature_weights = np.array([
            position_weight, position_weight,  # x, y position
            color_weight, color_weight, color_weight,  # r, g, b color
            size_weight,   # size
            texture_weight  # texture
        ])

        # Compute weighted features
        weighted_features = normalized_features * feature_weights

        # Use RBF kernel for similarity
        similarity = rbf_kernel(weighted_features, gamma=gamma)

        alpha = 0.1  
        identity = np.eye(similarity.shape[0])
        similarity = similarity + alpha * identity

        return similarity

    def _sample_dpp_eigen(self, L, k=None):
       
        n = L.shape[0]
        
        
        if k is None:
            k = min(int(np.sqrt(n)), 10)
        k = min(k, n)
        
       
        L = (L + L.T) / 2
        L = L + 1e-4 * np.eye(n)  
        
        try:
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            
            # Sort eigenvalues in descending order
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Truncate to positive eigenvalues
            pos_mask = eigenvalues > 1e-8 
            eigenvalues = eigenvalues[pos_mask]
            eigenvectors = eigenvectors[:, pos_mask]
            
            
            if len(eigenvalues) == 0:
                return np.random.choice(n, size=min(k, n), replace=False).tolist()
            
            # Compute selection probabilities
            probs = np.zeros(n)
            for i in range(len(eigenvalues)):
                v = eigenvectors[:, i]
                probs += (v * v) * eigenvalues[i] / (1 + eigenvalues[i])
            
            # Ensure valid probability distribution
            if np.sum(probs) <= 0:
              
                probs = np.ones(n) / n
            else:
                probs = probs / np.sum(probs)
            
            # Sample indices according to probabilities
            selected_indices = np.random.choice(n, size=min(k, n), replace=False, p=probs)
            
            return selected_indices.tolist()
            
        except np.linalg.LinAlgError as e:
            print(f"Eigendecomposition error: {e}. Using random sampling as fallback.")
            # Handle errors by returning random selection instead of SVD fallback
            return np.random.choice(n, size=min(k, n), replace=False).tolist()

    def _dpp_data_labels(self, image, fudged_image, segments, classifier_fn, num_samples,
                        batch_size=10, dpp_params=None, progress_callback=None):
        """Generate samples using eigendecomposition-based DPP and get their predictions."""
        num_segments = np.max(segments) + 1

        # Extract segment features
        segment_features = self._extract_segment_features(image, segments)

        # Compute similarity matrix with tuned parameters
        position_weight = dpp_params.get('position_weight', 1.5)
        color_weight = dpp_params.get('color_weight', 1.0)
        size_weight = dpp_params.get('size_weight', 0.8)
        texture_weight = dpp_params.get('texture_weight', 0.6)
        
        similarity = self._compute_similarity_matrix(
            segment_features,
            position_weight=position_weight,
            color_weight=color_weight,
            size_weight=size_weight,
            texture_weight=texture_weight,
            gamma=0.5)  # Reduced gamma for smoother similarity

        # Create DPP kernel with better quality scores
       
        quality_scores = segment_features[:, 5] * num_segments * 2  # Size feature, scaled
        # Ensure all quality scores are positive and not too small
        quality_scores = np.maximum(quality_scores, 0.2)
        
        L = np.outer(quality_scores, quality_scores) * similarity

        # Generate diverse samples
        data = np.zeros((num_samples, num_segments))
        
        # Generate varied sets of samples
        for i in range(num_samples):
            # Vary the number of segments per sample (between 10% and 40% of total segments)
            min_segments = max(1, int(num_segments * 0.1))
            max_segments = max(min_segments + 1, min(int(num_segments * 0.4), 20))
            k = np.random.randint(min_segments, max_segments + 1)
            
            # Sample segments using eigendecomposition-based DPP
            selected_segments = self._sample_dpp_eigen(L, k)
            data[i, selected_segments] = 1

        # Generate perturbed images and get predictions
        labels = []
        imgs = []

        # For progress tracking
        total_batches = int(np.ceil(num_samples / batch_size))
        current_batch = 0

        for row in data:
            temp = copy.deepcopy(fudged_image)
            for j, r in enumerate(row):
                if r == 1:  # If the segment is present
                    temp[segments == j] = image[segments == j]
            imgs.append(temp)

            # Predict in batches for efficiency
            if len(imgs) == batch_size:
                try:
                    preds = classifier_fn(np.array(imgs))
                    # Verify predictions shape
                    if len(preds) != batch_size:
                        raise ValueError(f"Classifier returned {len(preds)} predictions for {batch_size} samples")
                    labels.extend(preds)
                except Exception as e:
                    print(f"Error in classifier_fn: {e}")
                    # Raise to propagate the error and stop execution
                    raise
                
                imgs = []

                # Update progress
                if progress_callback is not None:
                    current_batch += 1
                    progress_callback(current_batch, total_batches)

        # Handle remaining images
        if len(imgs) > 0:
            try:
                preds = classifier_fn(np.array(imgs))
                if len(preds) != len(imgs):
                    raise ValueError(f"Classifier returned {len(preds)} predictions for {len(imgs)} samples")
                labels.extend(preds)
            except Exception as e:
                print(f"Error in classifier_fn for final batch: {e}")
                raise

            # Final progress update
            if progress_callback is not None:
                progress_callback(total_batches, total_batches)

        return data, np.array(labels)


class DPPExplanationWorker(QThread):
    """Worker thread for generating DPP-LIME explanations."""
    progress = pyqtSignal(int, int)  # Signal for progress updates (current, total)
    finished = pyqtSignal(list, list)  # Signal for completed explanations (images, labels)
    error = pyqtSignal(str)  # Signal for error messages

    def __init__(self, explanation_generator, image_path, selected_classes, dpp_params=None):
        super().__init__()
        self.explanation_generator = explanation_generator
        self.image_path = image_path
        self.selected_classes = selected_classes
        self.dpp_params = dpp_params or {}

    def run(self):
        try:
            explanation_images, explanation_labels = self.explanation_generator.generate_explanation(
                self.image_path,
                self.selected_classes,
                self.dpp_params,
                progress_callback=self.progress.emit)
            self.finished.emit(explanation_images, explanation_labels)
        except Exception as e:
            import traceback
            print(f"DPP-LIME error: {str(e)}")
            print(traceback.format_exc())
            self.error.emit(f"Error generating explanation: {str(e)}")


class DPPLimeExplanationGenerator:
    """Generator for DPP-LIME explanations."""

    def __init__(self, dataset=None, current_image_index=0, model_handler=None):
        self.dataset = dataset
        self.current_image_index = current_image_index
        self.model_handler = model_handler

    def _get_model_regressor(self, model_type='linear'):
        """Get the appropriate sklearn model based on the model type string."""
        from sklearn.linear_model import Ridge, LinearRegression
        from sklearn.kernel_ridge import KernelRidge

        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(alpha=1.0)
        elif model_type == 'kernel_ridge':
            return KernelRidge(alpha=1.0)
        else:
            # Default to linear regression if unknown type
            return LinearRegression()

    def _create_matplotlib_heatmap(self, original_image, explanation, label_idx, num_features, positive_only, opacity=0.7):
        """Create a heatmap visualization using matplotlib's capabilities."""
        # Get the feature importances for this class
        ind = explanation.local_exp[label_idx]

        # Create a mask of the same size as the image
        mask = np.zeros(explanation.segments.shape, dtype=np.float64)

        # Get absolute weights for normalization
        abs_weights = [abs(weight) for _, weight in ind]
        if not abs_weights:
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
            # Normalize weight (allow negative values)
            norm_weight = weight / max_weight
            # For heatmap visualization, cap negative values for better visibility
            if norm_weight < 0:
                norm_weight = max(norm_weight, -1)  # Cap negative values at -1
            mask[explanation.segments == segment_id] = norm_weight

        # Make sure the original image is in the right format (float 0-1)
        if original_image.max() > 1.0:
            original_image = original_image / 255.0
            
        # Create a heatmap with a better colormap for explanation
        # Use viridis for positive weights, and hot_r for negative weights
        pos_mask = mask > 0
        neg_mask = mask < 0
        
        # Initialize the heatmap with zeros and alpha=0
        heatmap = np.zeros((*mask.shape, 4))
        
        # Apply positive colormap (viridis)
        if np.any(pos_mask):
            pos_cmap = plt.cm.get_cmap('viridis')
            # Scale positive values between 0 and 1
            pos_values = mask[pos_mask]
            pos_colors = pos_cmap(pos_values)
            # Set alpha value based on importance and opacity
            pos_colors[..., 3] = pos_values * opacity
            heatmap[pos_mask] = pos_colors
        
        # Apply negative colormap (hot_r)
        if np.any(neg_mask) and not positive_only:
            neg_cmap = plt.cm.get_cmap('hot_r')
            # Scale negative values between 0 and 1
            neg_values = -mask[neg_mask]  # Negate to get positive values for colormap
            neg_colors = neg_cmap(neg_values)
            # Set alpha value based on importance and opacity
            neg_colors[..., 3] = neg_values * opacity
            heatmap[neg_mask] = neg_colors
        
        # Overlay heatmap on original image
        result = original_image.copy()
        # Only blend where the mask is non-zero
        blend_mask = (mask != 0)
        for i in range(3):  # RGB channels
            result[..., i] = np.where(
                blend_mask,
                result[..., i] * (1 - heatmap[..., 3]) + heatmap[..., i] * heatmap[..., 3],
                result[..., i]
            )

        return result

    def generate_explanation(self, image_path, selected_classes, dpp_params=None, progress_callback=None):
        if not image_path:
            raise ValueError("No image path provided!")

        if not selected_classes:
            raise ValueError("Please select at least one class!")

        if not dpp_params:
            raise ValueError("DPP parameters must be provided!")

        try:
            model = self.model_handler.model
            if model is None:
                raise ValueError("No model loaded!")

            # Get class names from the model handler
            class_names = self.model_handler.get_class_names()

            # Preprocess the image
            from yolo_predict_fn import preprocess_image_for_yolo
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

            # Extract parameters from dpp_params
            num_samples = dpp_params['num_samples']
            num_features = dpp_params['num_features']
            hide_rest = dpp_params.get('hide_rest', False)
            positive_only = dpp_params['positive_only']
            batch_size = dpp_params['batch_size']
            distance_metric = dpp_params['distance_metric']
            kernel_width = dpp_params['kernel_width']
            use_heatmap = dpp_params['use_heatmap']
            heatmap_opacity = dpp_params.get('heatmap_opacity', 0.7)
            hide_color = dpp_params.get('hide_color', np.array([0.5, 0.5, 0.5]))

            # DPP specific parameters
            position_weight = dpp_params['position_weight']
            color_weight = dpp_params['color_weight']
            size_weight = dpp_params['size_weight']
            texture_weight = dpp_params.get('texture_weight', 0.6)
            redundancy_penalty = dpp_params['redundancy_penalty']

            # Get the appropriate model regressor
            model_type = dpp_params.get('interpretable_model_type', 'linear')
            model_regressor = self._get_model_regressor(model_type)

            # Create segmentation parameters dictionary
            segmentation_params = {
                'segmentation_method': dpp_params['segmentation_method'],
                # Parameters for quickshift
                'kernel_size': dpp_params.get('kernel_size', 4),
                'max_dist': dpp_params.get('max_dist', 200),
                'ratio': dpp_params.get('ratio', 0.2),
                # Parameters for felzenszwalb
                'scale': dpp_params.get('scale', 100),
                'sigma': dpp_params.get('sigma', 0.5),
                'min_size': dpp_params.get('min_size', 50),
                # Parameters for SLIC
                'n_segments': dpp_params.get('n_segments', 100),
                'compactness': dpp_params.get('compactness', 10)
            }

            # Set DPP parameters
            dpp_specific_params = {
                'position_weight': position_weight,
                'color_weight': color_weight,
                'size_weight': size_weight,
                'texture_weight': texture_weight,
                'redundancy_penalty': redundancy_penalty
            }

            # Merge all parameters
            all_params = {**dpp_params, **dpp_specific_params, **segmentation_params}

            # Create DPP-enhanced LIME explainer
            explainer = DPPImageExplainer(kernel_width=kernel_width)

            # IMPORTANT: Pass the actual selected classes to explain_instance
            explanation_obj = explainer.explain_instance(
                image=img,
                classifier_fn=filtered_predict_fn,
                labels=selected_classes,  # Pass actual class IDs
                top_labels=len(selected_classes),
                num_features=num_features,
                num_samples=num_samples,
                batch_size=batch_size,
                segmentation_fn=None,  # Let the explainer create based on parameters
                distance_metric=distance_metric,
                model_regressor=model_regressor,
                hide_color=hide_color,
                dpp_params=all_params,  # Pass all parameters
                progress_callback=progress_callback)

            explanation_images = []
            explanation_labels = []

            # Generate explanations for each selected class
            for i, class_idx in enumerate(selected_classes):
                # Report progress if callback provided
                if progress_callback:
                    progress_callback(i + 1, len(selected_classes))

                # Check if this class exists in the explanation
                if class_idx not in explanation_obj.local_exp:
                    print(f"Warning: Class {class_idx} not found in explanation. Available classes: {list(explanation_obj.local_exp.keys())}")
                    
                    # Create a simple visualization without explanation
                    if use_heatmap:
                        # Just show the original image without highlighting
                        marked_image = img.copy()
                        if marked_image.max() > 1.0:
                            marked_image = marked_image / 255.0
                    else:
                        # Show the original image with segment boundaries
                        marked_image = mark_boundaries(img, explanation_obj.segments, color=(0, 0, 0), mode='thick')
                else:
                    # Generate visualization based on user selection
                    try:
                        if use_heatmap:
                            # Use our matplotlib-based heatmap visualization
                            temp, mask = explanation_obj.get_image_and_mask(
                                class_idx,  # Use actual class ID
                                positive_only=positive_only,
                                num_features=num_features,
                                hide_rest=hide_rest,
                                min_weight=0.0)
                            
                            # Use the new matplotlib-based heatmap function
                            marked_image = self._create_matplotlib_heatmap(
                                temp,
                                explanation_obj,
                                class_idx,  # Use actual class ID
                                num_features,
                                positive_only,
                                heatmap_opacity)
                        else:
                            # Use standard LIME visualization with mark_boundaries
                            temp, mask = explanation_obj.get_image_and_mask(
                                class_idx,  # Use actual class ID
                                positive_only=positive_only,
                                num_features=num_features,
                                hide_rest=hide_rest,
                                min_weight=0.0)
                            
                            # Make sure mask is boolean
                            mask_bool = mask.astype(bool)
                            
                            # Use mark_boundaries from skimage.segmentation with explicit parameters
                            marked_image = mark_boundaries(temp, mask_bool, color=(0, 0, 0), mode='thick')
                    except Exception as viz_error:
                        print(f"Error generating visualization for class {class_idx}: {str(viz_error)}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to showing the original image with segments
                        marked_image = mark_boundaries(img, explanation_obj.segments, color=(0, 0, 0), mode='thick')

                # Get class name and prediction confidence
                pred_probs = filtered_predict_fn([img])[0]
                # Find the index of class_idx in selected_classes
                class_index = selected_classes.index(class_idx)
                class_confidence = pred_probs[class_index]

                # Get class name from the model handler's class dictionary
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


class DPPLimeView(QWidget):
    """View for DPP-enhanced LIME explanations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.model_handler = None
        self.current_image_path = None
        self.explanation_images = []
        self.explanation_labels = []
        self.current_image_index = 0
        self.selected_classes = []
        self.is_parameters_collapsed = False
        
        self.initUI()
    
    def initUI(self):
        """Initialize the user interface."""
        main_layout = QVBoxLayout()
        
        # Create collapsible box for parameters
        self.params_box = CollapsibleBox("DPP-LIME Parameters")
        params_layout = QGridLayout()
        
        # Sampling parameters
        params_layout.addWidget(QLabel("Number of Samples:"), 0, 0)
        self.num_samples_input = QSpinBox()
        self.num_samples_input.setRange(100, 5000)
        self.num_samples_input.setValue(1000)
        params_layout.addWidget(self.num_samples_input, 0, 1)
        
        params_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setRange(1, 100)
        self.batch_size_input.setValue(10)
        params_layout.addWidget(self.batch_size_input, 1, 1)
        
        # DPP parameters
        params_layout.addWidget(QLabel("Position Weight:"), 2, 0)
        self.position_weight_input = QDoubleSpinBox()
        self.position_weight_input.setRange(0.1, 5.0)
        self.position_weight_input.setSingleStep(0.1)
        self.position_weight_input.setValue(1.5)
        self.position_weight_input.setToolTip("Weight for position features in DPP diversity calculation")
        params_layout.addWidget(self.position_weight_input, 2, 1)
        
        params_layout.addWidget(QLabel("Color Weight:"), 3, 0)
        self.color_weight_input = QDoubleSpinBox()
        self.color_weight_input.setRange(0.1, 5.0)
        self.color_weight_input.setSingleStep(0.1)
        self.color_weight_input.setValue(1.0)
        self.color_weight_input.setToolTip("Weight for color features in DPP diversity calculation")
        params_layout.addWidget(self.color_weight_input, 3, 1)
        
        params_layout.addWidget(QLabel("Size Weight:"), 4, 0)
        self.size_weight_input = QDoubleSpinBox()
        self.size_weight_input.setRange(0.1, 5.0)
        self.size_weight_input.setSingleStep(0.1)
        self.size_weight_input.setValue(0.8)
        self.size_weight_input.setToolTip("Weight for size features in DPP diversity calculation")
        params_layout.addWidget(self.size_weight_input, 4, 1)
        
        # Texture Weight
        params_layout.addWidget(QLabel("Texture Weight:"), 5, 0)
        self.texture_weight_input = QDoubleSpinBox()
        self.texture_weight_input.setRange(0.1, 5.0)
        self.texture_weight_input.setSingleStep(0.1)
        self.texture_weight_input.setValue(0.6)
        self.texture_weight_input.setToolTip("Weight for texture features in DPP diversity calculation")
        params_layout.addWidget(self.texture_weight_input, 5, 1)
        
        # Redundancy Penalty
        params_layout.addWidget(QLabel("Redundancy Penalty:"), 6, 0)
        self.redundancy_penalty_input = QDoubleSpinBox()
        self.redundancy_penalty_input.setRange(0.0, 2.0)
        self.redundancy_penalty_input.setSingleStep(0.1)
        self.redundancy_penalty_input.setValue(0.3)
        self.redundancy_penalty_input.setToolTip("Penalty applied to redundant features")
        params_layout.addWidget(self.redundancy_penalty_input, 6, 1)
        
        # Visualization parameters
        params_layout.addWidget(QLabel("Number of Features:"), 7, 0)
        self.num_features_input = QSpinBox()
        self.num_features_input.setRange(1, 20)
        self.num_features_input.setValue(5)
        params_layout.addWidget(self.num_features_input, 7, 1)
        
        # Visualization options
        params_layout.addWidget(QLabel("Visualization Type:"), 8, 0)
        self.visualization_type = QComboBox()
        self.visualization_type.addItems(["Standard", "Heatmap"])
        self.visualization_type.setCurrentIndex(1)  # Set Heatmap as default
        params_layout.addWidget(self.visualization_type, 8, 1)
        
        # Heatmap Opacity
        params_layout.addWidget(QLabel("Heatmap Opacity:"), 9, 0)
        self.heatmap_opacity_input = QDoubleSpinBox()
        self.heatmap_opacity_input.setRange(0.1, 1.0)
        self.heatmap_opacity_input.setSingleStep(0.1)
        self.heatmap_opacity_input.setValue(0.7)
        self.heatmap_opacity_input.setToolTip("Opacity of the heatmap overlay")
        params_layout.addWidget(self.heatmap_opacity_input, 9, 1)
        
        # Positive only
        params_layout.addWidget(QLabel("Positive Only:"), 10, 0)
        self.positive_only_checkbox = QCheckBox()
        self.positive_only_checkbox.setChecked(True)
        self.positive_only_checkbox.setToolTip("Show only positive feature contributions")
        params_layout.addWidget(self.positive_only_checkbox, 10, 1)
        
        # Segmentation Method
        params_layout.addWidget(QLabel("Segmentation Method:"), 11, 0)
        self.segmentation_method_input = QComboBox()
        self.segmentation_method_input.addItems(["quickshift", "felzenszwalb", "slic"])
        self.segmentation_method_input.currentTextChanged.connect(self.update_segment_controls)
        params_layout.addWidget(self.segmentation_method_input, 11, 1)
        
        # Number of segments (primarily for SLIC)
        params_layout.addWidget(QLabel("Number of Segments:"), 12, 0)
        self.n_segments_input = QSpinBox()
        self.n_segments_input.setRange(5, 500)
        self.n_segments_input.setValue(100)
        self.n_segments_input.setToolTip("Number of segments to use (primarily for SLIC)")
        params_layout.addWidget(self.n_segments_input, 12, 1)
        
        # Segmentation Parameters
        # For Quickshift
        params_layout.addWidget(QLabel("Kernel Size:"), 13, 0)
        self.kernel_size_input = QSpinBox()
        self.kernel_size_input.setRange(1, 20)
        self.kernel_size_input.setValue(4)
        self.kernel_size_input.setToolTip("Controls size of segments in quickshift - larger values give fewer segments")
        params_layout.addWidget(self.kernel_size_input, 13, 1)
        
        params_layout.addWidget(QLabel("Max Distance:"), 14, 0)
        self.max_dist_input = QSpinBox()
        self.max_dist_input.setRange(1, 500)
        self.max_dist_input.setValue(200)
        self.max_dist_input.setToolTip("Controls size of segments in quickshift - smaller values give more segments")
        params_layout.addWidget(self.max_dist_input, 14, 1)
        
        # For Felzenszwalb
        params_layout.addWidget(QLabel("Scale:"), 15, 0)
        self.scale_input = QSpinBox()
        self.scale_input.setRange(1, 500)
        self.scale_input.setValue(100)
        self.scale_input.setToolTip("Controls size of segments in felzenszwalb - larger values give fewer segments")
        params_layout.addWidget(self.scale_input, 15, 1)
        
        params_layout.addWidget(QLabel("Min Size:"), 16, 0)
        self.min_size_input = QSpinBox()
        self.min_size_input.setRange(1, 200)
        self.min_size_input.setValue(50)
        self.min_size_input.setToolTip("Minimum component size in felzenszwalb - larger values give fewer segments")
        params_layout.addWidget(self.min_size_input, 16, 1)
        
        # For SLIC
        params_layout.addWidget(QLabel("Compactness:"), 17, 0)
        self.compactness_input = QSpinBox()
        self.compactness_input.setRange(1, 100)
        self.compactness_input.setValue(10)
        self.compactness_input.setToolTip("Compactness parameter for SLIC - shape of segments")
        params_layout.addWidget(self.compactness_input, 17, 1)
        
        # Kernel width parameter
        params_layout.addWidget(QLabel("Kernel Width:"), 18, 0)
        self.kernel_width_input = QDoubleSpinBox()
        self.kernel_width_input.setRange(0.1, 1.0)
        self.kernel_width_input.setSingleStep(0.05)
        self.kernel_width_input.setValue(0.25)
        self.kernel_width_input.setToolTip("Width of the kernel used in LIME")
        params_layout.addWidget(self.kernel_width_input, 18, 1)
        
        # Set the layout for the collapsible box
        self.params_box.setContentLayout(params_layout)
        main_layout.addWidget(self.params_box)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.generate_button = QPushButton("Generate DPP-LIME")
        self.generate_button.clicked.connect(self.generate_explanation)
        buttons_layout.addWidget(self.generate_button)
        
        self.save_button = QPushButton("Save Explanation")
        self.save_button.clicked.connect(self.save_explanation)
        buttons_layout.addWidget(self.save_button)
        
        # Toggle parameters visibility button
        self.toggle_params_button = QPushButton("Toggle Parameters")
        self.toggle_params_button.clicked.connect(self.toggle_parameters)
        buttons_layout.addWidget(self.toggle_params_button)
        
        main_layout.addLayout(buttons_layout)
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("← Previous Class")
        self.next_button = QPushButton("Next Class →")
        self.prev_button.clicked.connect(self.previous_class)
        self.next_button.clicked.connect(self.next_class)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        main_layout.addLayout(nav_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)
        
        # Figure for plotting - use SizePolicy to make it expand
        self.figure_container = QWidget()
        self.figure_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        figure_layout = QVBoxLayout(self.figure_container)
        figure_layout.setContentsMargins(0, 0, 0, 0)
        
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        figure_layout.addWidget(self.canvas)
        
        main_layout.addWidget(self.figure_container, 1)  # Add stretch factor for expansion
        
        # Class label
        self.class_label = QLabel("No explanation generated")
        self.class_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.class_label)
        
        self.setLayout(main_layout)
        
        # Initialize visibility of segment controls
        self.update_segment_controls()
    
    def update_segment_controls(self):
        """Update visibility of segmentation controls based on selected method"""
        method = self.segmentation_method_input.currentText()
        
        # Hide all controls first
        self.n_segments_input.setVisible(False)
        self.kernel_size_input.setVisible(False)
        self.max_dist_input.setVisible(False)
        self.scale_input.setVisible(False)
        self.min_size_input.setVisible(False)
        self.compactness_input.setVisible(False)
        
        # Show relevant controls for the selected method
        if method == "quickshift":
            self.kernel_size_input.setVisible(True)
            self.max_dist_input.setVisible(True)
        elif method == "felzenszwalb":
            self.scale_input.setVisible(True)
            self.min_size_input.setVisible(True)
        elif method == "slic":
            self.n_segments_input.setVisible(True)
            self.compactness_input.setVisible(True)
    
    def toggle_parameters(self):
        """Toggle the visibility of the parameters section."""
        # Use the collapsible box's own toggle mechanism
        self.params_box.on_pressed()
        
        # Update the canvas to use the available space
        self.canvas.updateGeometry()
        
    def set_model_handler(self, model_handler):
        """Set the model handler for predictions."""
        self.model_handler = model_handler
    
    def set_image_path(self, image_path):
        """Set the current image path."""
        self.current_image_path = image_path
        self.update_display()
    
    def set_selected_classes(self, selected_classes):
        """Set the selected classes for explanation."""
        # Convert class_idx to int if they are strings
        self.selected_classes = [int(cls) if isinstance(cls, str) else cls for cls in selected_classes]
        print(f"DPP-LIME View: Selected classes set to {self.selected_classes}")
        
    def update_display(self):
        """Update the display with current image."""
        if not self.explanation_images:
            if self.current_image_path and os.path.exists(self.current_image_path):
                self.figure.clear()
                ax = self.figure.add_subplot(111)
                
                try:
                    image = io.imread(self.current_image_path)
                    ax.imshow(image)
                    ax.axis('off')
                    ax.set_title(os.path.basename(self.current_image_path))
                    self.canvas.draw()
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error loading image: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center')
                    self.canvas.draw()
        else:
            self.update_explanation_view()
    
    def generate_explanation(self):
        """Generate DPP-LIME explanation for current image."""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No image selected!")
            return
        
        if not self.model_handler:
            QMessageBox.warning(self, "Warning", "Model handler not set!")
            return
        
        if not self.selected_classes:
            QMessageBox.warning(self, "Warning", "No classes selected!")
            return
        
        # Collect parameters from inputs
        params = {
            # Sampling parameters
            'num_samples': self.num_samples_input.value(),
            'batch_size': self.batch_size_input.value(),
            
            # DPP parameters
            'position_weight': self.position_weight_input.value(),
            'color_weight': self.color_weight_input.value(),
            'size_weight': self.size_weight_input.value(),
            'texture_weight': self.texture_weight_input.value(),
            'redundancy_penalty': self.redundancy_penalty_input.value(),
            
            # Visualization parameters
            'num_features': self.num_features_input.value(),
            'use_heatmap': self.visualization_type.currentText() == "Heatmap",
            'positive_only': self.positive_only_checkbox.isChecked(),
            'hide_rest': False,
            'heatmap_opacity': self.heatmap_opacity_input.value(),
            
            # Segmentation parameters
            'segmentation_method': self.segmentation_method_input.currentText(),
            'kernel_size': self.kernel_size_input.value(),
            'max_dist': self.max_dist_input.value(),
            'scale': self.scale_input.value(),
            'min_size': self.min_size_input.value(),
            'n_segments': self.n_segments_input.value(),
            'compactness': self.compactness_input.value(),
            
            # Additional parameters
            'kernel_width': self.kernel_width_input.value(),
            'distance_metric': 'cosine',
            'hide_color': np.array([0.5, 0.5, 0.5]),
            'interpretable_model_type': 'linear'
        }
        
        # Configure UI for processing
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.generate_button.setEnabled(False)
        
        # Create explanation generator
        self.explanation_generator = DPPLimeExplanationGenerator(
            model_handler=self.model_handler
        )
        
        # Create and start worker thread
        self.worker = DPPExplanationWorker(
            self.explanation_generator,
            self.current_image_path,
            self.selected_classes,
            params
        )
        
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.handle_explanation)
        self.worker.error.connect(self.handle_error)
        
        # Start worker
        self.worker.start()
    
    def update_progress(self, current, total):
        """Update progress bar during processing."""
        progress_percent = int(100 * current / max(1, total))
        self.progress_bar.setValue(progress_percent)
    
    def handle_explanation(self, explanation_images, explanation_labels):
        """Handle the completed DPP-LIME explanation."""
        self.explanation_images = explanation_images
        self.explanation_labels = explanation_labels
        self.current_image_index = 0
        
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        
        if not explanation_images:
            QMessageBox.warning(self, "Warning", "No explanation was generated!")
            return
        
        # Auto-collapse parameters panel when explanation is ready
        if not self.params_box.content_area.isHidden():
            self.toggle_parameters()
            
        self.update_explanation_view()
    
    def handle_error(self, error_message):
        """Handle errors during DPP-LIME processing."""
        self.progress_bar.setVisible(False)
        self.generate_button.setEnabled(True)
        QMessageBox.critical(self, "Error", error_message)
    
    def update_explanation_view(self):
        """Update the display with the current explanation."""
        if not self.explanation_images or self.current_image_index >= len(self.explanation_images):
            return
        
        # Get the current explanation image and label
        explanation_image = self.explanation_images[self.current_image_index]
        explanation_label = self.explanation_labels[self.current_image_index]
        
        # Display the explanation
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Show the explanation image
        ax.imshow(explanation_image)
        ax.axis('off')
        ax.set_title(explanation_label)
        
        # Update class label
        self.class_label.setText(explanation_label)
        
        # Draw with tight layout to use available space
        self.figure.tight_layout()
        self.canvas.draw()
    
    def previous_class(self):
        """Navigate to the previous class explanation."""
        if self.explanation_images:
            self.current_image_index = (self.current_image_index - 1) % len(self.explanation_images)
            self.update_explanation_view()
    
    def next_class(self):
        """Navigate to the next class explanation."""
        if self.explanation_images:
            self.current_image_index = (self.current_image_index + 1) % len(self.explanation_images)
            self.update_explanation_view()
    
    def save_explanation(self):
        """Save the current explanation image."""
        if not self.explanation_images:
            QMessageBox.warning(self, "Warning", "No explanation to save!")
            return
        
        try:
            # Open file dialog to get save path
            filename, _ = QFileDialog.getSaveFileName(
                self, 
                "Save Explanation", 
                "", 
                "Images (*.png *.jpg *.jpeg)"
            )
            
            if filename:
                # Save the current figure
                self.figure.savefig(filename, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"Explanation saved to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save explanation: {str(e)}")
            
    def resizeEvent(self, event):
        """Handle resize events to update the figure's size."""
        super().resizeEvent(event)
        # Update the figure when the widget is resized
        self.canvas.updateGeometry()
        if self.explanation_images:
            # Redraw with proper layout
            self.figure.tight_layout()
            self.canvas.draw()