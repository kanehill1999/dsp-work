# model_hooks.py
"""
Helper functions for hooking into model layers for visualizations.
"""
import torch
import numpy as np

class LayerHooks:
    """Class for managing hooks into model layers."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.activations = {}
        self.gradients = {}
        
    def register_hooks(self, layer_names):
        """Register forward hooks for the specified layers."""
        self.clear_hooks()
        self.activations = {}
        
        for layer_name in layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                # Define the hook function to capture activations
                def get_activation(name):
                    def hook(module, input, output):
                        self.activations[name] = output.detach()
                    return hook
                
                # Register the hook
                hook = layer.register_forward_hook(get_activation(layer_name))
                self.hooks.append(hook)
    
    def register_backward_hooks(self, layer_names):
        """Register backward hooks for the specified layers."""
        for layer_name in layer_names:
            layer = self._get_layer_by_name(layer_name)
            if layer is not None:
                # Define the hook function to capture gradients
                def get_gradient(name):
                    def hook(module, grad_input, grad_output):
                        self.gradients[name] = grad_output[0].detach()
                    return hook
                
                # Register the hook
                hook = layer.register_backward_hook(get_gradient(layer_name))
                self.hooks.append(hook)
    
    def _get_layer_by_name(self, name):
        """Get a layer from the model by its name."""
        try:
            # Handle nested modules with dot notation
            if hasattr(self.model, 'model'):  # For YOLO models
                module = self.model.model
            else:
                module = self.model
                
            # Split the name into parts and navigate the module hierarchy
            parts = name.split('.')
            for part in parts:
                if part.isdigit():  # Handle numeric indices for list-like modules
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            
            return module
        except Exception as e:
            print(f"Error getting layer {name}: {e}")
            return None
    
    def clear_hooks(self):
        """Clear all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_all_layers(self):
        """Get all available layers in the model."""
        layer_names = []
        
        def get_layers_recursive(module, prefix=''):
            for name, child in module.named_children():
                new_prefix = f"{prefix}.{name}" if prefix else name
                layer_names.append(new_prefix)
                get_layers_recursive(child, new_prefix)
        
        if hasattr(self.model, 'model'):  # For YOLO models
            get_layers_recursive(self.model.model)
        else:
            get_layers_recursive(self.model)
        
        return layer_names
    
    def get_activations(self, layer_name):
        """Get the activations for a specific layer."""
        return self.activations.get(layer_name, None)
    
    def get_all_activations(self):
        """Get all activations."""
        return self.activations
    
    def get_layer_importance(self, layer_name, class_idx):
        
        activations = self.get_activations(layer_name)
        if activations is None:
            return None
        
        # Get the average activation for this layer
        if isinstance(activations, torch.Tensor):
            # For convolutional layers, take the mean across channels
            if len(activations.shape) == 4:  # [batch, channels, height, width]
                importance = activations.mean(dim=1).squeeze().cpu().numpy()
            else:
                importance = activations.mean(dim=1).cpu().numpy()
        else:
            importance = np.mean(activations, axis=0)
        
        return importance
    
    def __del__(self):
        """Ensure hooks are cleared when the object is deleted."""
        self.clear_hooks()


def visualize_layer_activation(activation, image_shape=None):
   
    if isinstance(activation, torch.Tensor):
        activation = activation.detach().cpu().numpy()
    
    # Handle different activation shapes
    if len(activation.shape) == 4:  # [batch, channels, height, width]
        # Take the first batch and mean across channels
        activation = np.mean(activation[0], axis=0)
    elif len(activation.shape) == 3:  # [channels, height, width]
        # Mean across channels
        activation = np.mean(activation, axis=0)
    elif len(activation.shape) == 2:  # [height, width]
        pass  # Already in the right format
    else:
        # For fully connected layers or other shapes, try to reshape into a square
        size = int(np.sqrt(activation.size))
        try:
            activation = activation.reshape(size, size)
        except:
            # If reshaping fails, create a dummy heatmap
            activation = np.ones((10, 10)) * np.mean(activation)
    
    # Normalize to [0, 1] range
    activation -= activation.min()
    activation_max = activation.max()
    if activation_max > 0:
        activation /= activation_max
    
    # If target shape is provided, resize the activation
    if image_shape is not None:
        import cv2
        activation = cv2.resize(activation, (image_shape[1], image_shape[0]))
    
    # Create a heatmap using matplotlib's viridis colormap
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create a custom colormap (similar to matplotlib's viridis)
    colors = [
        (0, 0, 0.5),      # Dark blue for low values
        (0, 0.5, 1),      # Light blue
        (0, 0.75, 0.5),   # Teal
        (0.5, 1, 0),      # Light green
        (1, 1, 0),        # Yellow
        (1, 0, 0)         # Red for high values
    ]
    cmap = LinearSegmentedColormap.from_list("activation_cmap", colors)
    
    # Apply colormap to create RGB image
    colored_activation = cmap(activation)
    
    return colored_activation

def overlay_activation_on_image(image, activation, alpha=0.5):
    
    if image.max() > 1.0:
        image = image.astype(float) / 255.0
    
    # Resize activation to match image dimensions if needed
    if activation.shape[:2] != image.shape[:2]:
        import cv2
        activation = cv2.resize(activation, (image.shape[1], image.shape[0]))
    
    # Create the overlay
    overlay = image.copy()
    for i in range(3):  # RGB channels
        overlay[..., i] = (1 - alpha) * image[..., i] + alpha * activation[..., i]
    
    return overlay